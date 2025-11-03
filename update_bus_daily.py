#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_bus_daily.py — daily_crime_03.csv + (sf_bus_stops_with_geoid.csv, blocks GeoJSON)
#                      → daily_crime_04.csv (bus_stop_count, distance_to_bus, *_range)

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# KD-Tree: SciPy varsa onu, yoksa NumPy fallback
try:
    from scipy.spatial import cKDTree
    _TREE_IMPL = "scipy"
except Exception:
    cKDTree = None
    _TREE_IMPL = "numpy"

pd.options.mode.copy_on_write = True

# ========= helpers =========
def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
def normalize_geoid(s: pd.Series, L: int = DEFAULT_GEOID_LEN) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)", expand=False)
    return x.str[:L].str.zfill(L)

def freedman_diaconis_bin_count(data: np.ndarray, max_bins: int = 10) -> int:
    data = np.asarray(data)
    if len(data) < 2 or np.all(data == data[0]):
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    bw = 2 * iqr / (len(data) ** (1 / 3))
    if bw <= 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    return max(2, min(max_bins, int(np.ceil((data.max() - data.min()) / bw))))

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"💾 Kaydedildi: {path} (satır={len(df):,}, sütun={df.shape[1]})")

# ========= config (indirimsiz) =========
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)

# IO
DAILY_IN   = Path(os.getenv("DAILY_IN",  str(BASE_DIR / "daily_crime_03.csv")))
DAILY_OUT  = Path(os.getenv("DAILY_OUT", str(BASE_DIR / "daily_crime_04.csv")))

# Canonical BUS cache (GEOID eşlenmiş duraklar)
BUS_CANON  = Path(os.getenv("BUS_CANON_RAW", str(BASE_DIR / "sf_bus_stops_with_geoid.csv")))

# Blocks GeoJSON (centroid → en yakın durak mesafesi için)
BLOCKS_CANDIDATES = [
    Path(os.getenv("SF_BLOCKS_GEOJSON", str(BASE_DIR / "sf_census_blocks_with_population.geojson"))),
    Path("sf_census_blocks_with_population.geojson"),
]

# ========= 1) crime oku =========
if not DAILY_IN.exists() or not DAILY_IN.is_file():
    raise FileNotFoundError(f"❌ Girdi yok: {DAILY_IN}")
crime = pd.read_csv(DAILY_IN, low_memory=False)
if "GEOID" not in crime.columns:
    raise KeyError("❌ daily_crime_03.csv içinde 'GEOID' yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime_geoids = pd.Series(crime["GEOID"].unique(), name="GEOID")
log(f"📥 daily_crime_03.csv okundu — satır={len(crime):,}, uniq GEOID={crime_geoids.size:,}")

# ========= 2) bus cache oku (indirimsiz) =========
if not BUS_CANON.exists() or not BUS_CANON.is_file():
    raise FileNotFoundError(f"❌ BUS cache yok: {BUS_CANON}\n"
                            f"Bu dosya aylık işte üretilmeli: sf_bus_stops_with_geoid.csv")
bus = pd.read_csv(BUS_CANON, low_memory=False)

# beklenen: en azından stop_lat/stop_lon ya da lat/lon ve GEOID (tercihen)
lat_col = next((c for c in ["stop_lat","latitude","lat","y"] if c in bus.columns), None)
lon_col = next((c for c in ["stop_lon","longitude","long","lon","x"] if c in bus.columns), None)
if lat_col is None or lon_col is None:
    raise KeyError("❌ BUS cache içinde lat/lon benzeri kolonlar yok (stop_lat/stop_lon vb.).")

bus["stop_lat"] = pd.to_numeric(bus[lat_col], errors="coerce")
bus["stop_lon"] = pd.to_numeric(bus[lon_col], errors="coerce")
bus = bus.dropna(subset=["stop_lat","stop_lon"]).copy()

if "GEOID" in bus.columns:
    bus["GEOID"] = normalize_geoid(bus["GEOID"], DEFAULT_GEOID_LEN)

log(f"🚌 BUS cache okundu — satır={len(bus):,}, GEOID kolonlu={ 'GEOID' in bus.columns }")

# ========= 3) Blocks GeoJSON bul =========
blocks_path = next((p for p in BLOCKS_CANDIDATES if p and p.exists() and p.is_file()), None)
if blocks_path is None:
    raise FileNotFoundError("❌ Blocks GeoJSON bulunamadı (sf_census_blocks_with_population.geojson).")

gdf_blocks = gpd.read_file(blocks_path)
gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next(
    (c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None
)
if not gcol:
    raise KeyError("❌ Blocks GeoJSON içinde GEOID benzeri kolon yok.")
gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)

# CRS → WGS84
if gdf_blocks.crs is None:
    gdf_blocks.set_crs("EPSG:4326", inplace=True)
elif str(gdf_blocks.crs).lower() not in ("epsg:4326","wgs84","wgs 84"):
    gdf_blocks = gdf_blocks.to_crs(epsg=4326)

# ========= 4) BUS → GEOID sayımı =========
if "GEOID" not in bus.columns or bus["GEOID"].isna().all():
    # GEOID yoksa: nokta → polygon sjoin ile atayıp sonra say
    gdf_bus = gpd.GeoDataFrame(
        bus,
        geometry=gpd.points_from_xy(bus["stop_lon"], bus["stop_lat"]),
        crs="EPSG:4326"
    )
    try:
        # En doğrusu: durak poligon içinde mi?
        gdf_bus = gpd.sjoin(gdf_bus, gdf_blocks[["GEOID","geometry"]], how="left", predicate="within")
    except Exception:
        # WGS84'te 'max_distance' derece olur → metrik CRS'e geçip metre kullan
        gdf_bus_m = gdf_bus.to_crs(3857)
        gdf_blocks_m = gdf_blocks.to_crs(3857)[["GEOID","geometry"]]
        # 1000 m eşik (yakın poligon seçimi için mantıklı)
        gdf_bus = gpd.sjoin_nearest(gdf_bus_m, gdf_blocks_m, how="left", max_distance=1000).to_crs(4326)

    gdf_bus = gdf_bus.drop(columns=["index_right"], errors="ignore")
    bus_geo = pd.DataFrame(gdf_bus.drop(columns=["geometry"], errors="ignore"))
else:
    bus_geo = bus.copy()

bus_geo["GEOID"] = normalize_geoid(bus_geo["GEOID"], DEFAULT_GEOID_LEN)
bus_count = (
    bus_geo.dropna(subset=["GEOID"])
           .groupby("GEOID", as_index=False)
           .agg(bus_stop_count=("stop_lat", "size"))
)

# ========= 5) distance_to_bus (m) =========
# Block centroid’lerini METRİK CRS’te al (EPSG:3857)
gdf_blocks_xy = gdf_blocks[["GEOID","geometry"]].copy().to_crs(3857)
gdf_blocks_xy["cx"] = gdf_blocks_xy.geometry.centroid.x
gdf_blocks_xy["cy"] = gdf_blocks_xy.geometry.centroid.y

# Durakları METRİK CRS’e geçir
_bus_pts = bus_geo.dropna(subset=["stop_lat","stop_lon"]).copy()
gdf_bus_xy = gpd.GeoDataFrame(
    _bus_pts[["stop_lat","stop_lon"]],
    geometry=gpd.points_from_xy(_bus_pts["stop_lon"], _bus_pts["stop_lat"]),
    crs="EPSG:4326"
).to_crs(3857)

if len(gdf_bus_xy) == 0:
    bus_dist = gdf_blocks_xy[["GEOID"]].copy()
    bus_dist["distance_to_bus"] = np.nan
else:
    centroids = np.vstack([gdf_blocks_xy["cx"].values, gdf_blocks_xy["cy"].values]).T
    bus_coords = np.vstack([gdf_bus_xy.geometry.x.values, gdf_bus_xy.geometry.y.values]).T
    if cKDTree is not None:
        tree = cKDTree(bus_coords)
        distances, _ = tree.query(centroids, k=1)
    else:
        # NumPy fallback — O(N×M) ama SF için makul
        diff = centroids[:, None, :] - bus_coords[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
    bus_dist = gdf_blocks_xy[["GEOID"]].copy()
    bus_dist["distance_to_bus"] = distances.astype(float)

# ========= 6) feature tablosu + binler =========
bus_feat = bus_dist.merge(bus_count, on="GEOID", how="left")
bus_feat["bus_stop_count"] = bus_feat["bus_stop_count"].fillna(0).astype(int)
bus_feat["GEOID"] = normalize_geoid(bus_feat["GEOID"], DEFAULT_GEOID_LEN)

# Sadece crime’da olan GEOID’leri tut
bus_feat = bus_feat.merge(crime_geoids.to_frame(), on="GEOID", how="right")

# Binleme (opsiyonel)
d = bus_feat["distance_to_bus"].replace([np.inf,-np.inf], np.nan).dropna()
if len(d) >= 2 and d.max() > d.min():
    n_bins = freedman_diaconis_bin_count(d.to_numpy(), max_bins=10)
    # qcut ile kantil bazlı kenarlar; uygun değilse duplicates drop
    _, edges = pd.qcut(d, q=n_bins, retbins=True, duplicates="drop")
    labels = [f"{int(round(edges[i]))}–{int(round(edges[i+1]))}m" for i in range(len(edges)-1)]
    bus_feat["distance_to_bus_range"] = pd.cut(bus_feat["distance_to_bus"], bins=edges,
                                               labels=labels, include_lowest=True)
else:
    bus_feat["distance_to_bus_range"] = pd.Series(["0–0m"] * len(bus_feat), index=bus_feat.index)

cnt = bus_feat["bus_stop_count"].fillna(0)
if cnt.nunique() > 1:
    n_c_bins = freedman_diaconis_bin_count(cnt.to_numpy(), max_bins=8)
    _, c_edges = pd.qcut(cnt, q=n_c_bins, retbins=True, duplicates="drop")
    c_labels = [f"{int(round(c_edges[i]))}–{int(round(c_edges[i+1]))}" for i in range(len(c_edges)-1)]
    bus_feat["bus_stop_count_range"] = pd.cut(cnt, bins=c_edges, labels=c_labels, include_lowest=True)
else:
    bus_feat["bus_stop_count_range"] = pd.Series([f"{int(cnt.min())}–{int(cnt.max())}"] * len(cnt), index=cnt.index)

bus_feat = bus_feat.sort_values("GEOID").drop_duplicates("GEOID", keep="first")
assert bus_feat["GEOID"].is_unique, "BUS: GEOID unique olmalı"

# ========= 7) crime ile merge =========
before = crime.shape
crime = crime.merge(bus_feat, on="GEOID", how="left", validate="many_to_one")
crime["bus_stop_count"] = crime["bus_stop_count"].fillna(0).astype(int)
log(f"🔗 CRIME ⨯ BUS: {before} → {crime.shape} (KDTree={_TREE_IMPL})")

# ========= 8) yaz =========
safe_save_csv(crime, str(DAILY_OUT))
try:
    log(crime.head(5).to_string(index=False))
except Exception:
    pass
