#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_train_daily.py — daily_crime_04.csv + (sf_train_stops_with_geoid.csv, blocks GeoJSON)
#                         → daily_crime_05.csv  (distance_to_train, train_stop_count, *_range)

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

# ---------- helpers ----------
def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
def normalize_geoid(s: pd.Series, L: int = DEFAULT_GEOID_LEN) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)", expand=False)
    return x.str[:L].str.zfill(L)

def freedman_diaconis_bin_count(data: np.ndarray, max_bins: int = 10) -> int:
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if data.size < 2 or np.allclose(data.min(), data.max()):
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
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

# ---------- config (no network) ----------
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)

DAILY_IN  = Path(os.getenv("DAILY_IN",  str(BASE_DIR / "daily_crime_04.csv")))
DAILY_OUT = Path(os.getenv("DAILY_OUT", str(BASE_DIR / "daily_crime_05.csv")))

# Hazır cache: GTFS durakları GEOID’e eşlenmiş
TRAIN_STOPS_WITH_GEOID = Path(os.getenv("TRAIN_STOPS_WITH_GEOID", str(BASE_DIR / "sf_train_stops_with_geoid.csv")))
TRAIN_LEGACY_RAW_Y     = Path(os.getenv("TRAIN_LEGACY_RAW_Y",     str(BASE_DIR / "train_y.csv")))  # alternatif cache
TRAIN_SUMMARY_NAME     = Path(os.getenv("TRAIN_SUMMARY_NAME",     str(BASE_DIR / "train.csv")))    # GEOID-level özet

BLOCKS_CANDIDATES = [
    Path(os.getenv("SF_BLOCKS_GEOJSON", str(BASE_DIR / "sf_census_blocks_with_population.geojson"))),
    Path("sf_census_blocks_with_population.geojson"),
]

# ---------- 1) crime oku ----------
if not DAILY_IN.exists() or not DAILY_IN.is_file():
    raise FileNotFoundError(f"❌ Girdi yok: {DAILY_IN}")
crime = pd.read_csv(DAILY_IN, low_memory=False)
if "GEOID" not in crime.columns:
    raise KeyError("❌ daily_crime_04.csv içinde 'GEOID' yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime_geoids = pd.Series(crime["GEOID"].unique(), name="GEOID")
log(f"📥 daily_crime_04.csv okundu — satır={len(crime):,}, uniq GEOID={crime_geoids.size:,}")

# ---------- 2) GTFS durak cache (indirimsiz) ----------
stops_path = TRAIN_STOPS_WITH_GEOID if (TRAIN_STOPS_WITH_GEOID.exists() and TRAIN_STOPS_WITH_GEOID.is_file()) else (
    TRAIN_LEGACY_RAW_Y if (TRAIN_LEGACY_RAW_Y.exists() and TRAIN_LEGACY_RAW_Y.is_file()) else None
)
if stops_path is None:
    raise FileNotFoundError("❌ GTFS durak cache'i bulunamadı (sf_train_stops_with_geoid.csv veya train_y.csv).")

stops = pd.read_csv(stops_path, low_memory=False)
# Kolonları normalize et
lat_col = next((c for c in ["stop_lat","latitude","lat","y"] if c in stops.columns), None)
lon_col = next((c for c in ["stop_lon","longitude","long","lon","x"] if c in stops.columns), None)
if lat_col is None or lon_col is None:
    raise KeyError("❌ Train cache içinde lat/lon benzeri kolonlar yok (stop_lat/stop_lon vb.).")

stops["stop_lat"] = pd.to_numeric(stops[lat_col], errors="coerce")
stops["stop_lon"] = pd.to_numeric(stops[lon_col], errors="coerce")
stops = stops.dropna(subset=["stop_lat","stop_lon"]).copy()
if "GEOID" in stops.columns:
    stops["GEOID"] = normalize_geoid(stops["GEOID"], DEFAULT_GEOID_LEN)
log(f"🚉 Train stops cache: {stops_path} — satır={len(stops):,}, GEOID kolonlu={ 'GEOID' in stops.columns }")

# ---------- 3) Blocks GeoJSON ----------
blocks_path = next((p for p in BLOCKS_CANDIDATES if p and p.exists() and p.is_file()), None)
if blocks_path is None:
    raise FileNotFoundError("❌ Blocks GeoJSON yok: sf_census_blocks_with_population.geojson")
gdf_blocks = gpd.read_file(blocks_path)
gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next((c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None)
if not gcol:
    raise KeyError("❌ Blocks GeoJSON içinde GEOID benzeri kolon yok.")
gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)
if gdf_blocks.crs is None:
    gdf_blocks.set_crs("EPSG:4326", inplace=True)
elif str(gdf_blocks.crs).lower() not in ("epsg:4326","wgs84","wgs 84"):
    gdf_blocks = gdf_blocks.to_crs(epsg=4326)

# ---------- 4) GEOID sayımı ----------
if "GEOID" not in stops.columns or stops["GEOID"].isna().all():
    gdf_stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326")
    try:
        # Poligon içinde mi?
        gdf_stops = gpd.sjoin(gdf_stops, gdf_blocks[["geometry","GEOID"]], how="left", predicate="within")
    except Exception:
        # WGS84'te max_distance derece olur → metrik CRS'e geç
        gdf_stops_m = gdf_stops.to_crs(3857)
        gdf_blocks_m = gdf_blocks.to_crs(3857)[["geometry","GEOID"]]
        gdf_stops = gpd.sjoin_nearest(gdf_stops_m, gdf_blocks_m, how="left", max_distance=1000).to_crs(4326)
    gdf_stops = gdf_stops.drop(columns=["index_right"], errors="ignore")
    stops_geo = pd.DataFrame(gdf_stops.drop(columns=["geometry"], errors="ignore"))
else:
    stops_geo = stops.copy()

stops_geo["GEOID"] = normalize_geoid(stops_geo["GEOID"], DEFAULT_GEOID_LEN)
train_count = (
    stops_geo.dropna(subset=["GEOID"])
             .groupby("GEOID", as_index=False)
             .agg(train_stop_count=("stop_lat","size"))
)

# ---------- 5) distance_to_train (m) ----------
gdf_blocks_3857 = gdf_blocks[["GEOID","geometry"]].copy().to_crs(epsg=3857)
gdf_blocks_3857["cx"] = gdf_blocks_3857.geometry.centroid.x
gdf_blocks_3857["cy"] = gdf_blocks_3857.geometry.centroid.y

_stops_pts = stops_geo.dropna(subset=["stop_lat","stop_lon"]).copy()
gdf_train_xy = gpd.GeoDataFrame(
    _stops_pts[["stop_lat","stop_lon"]],
    geometry=gpd.points_from_xy(_stops_pts["stop_lon"], _stops_pts["stop_lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

if len(gdf_train_xy) == 0:
    dist_tbl = gdf_blocks_3857[["GEOID"]].copy()
    dist_tbl["distance_to_train"] = np.nan
else:
    train_xy = np.vstack([gdf_train_xy.geometry.x.values, gdf_train_xy.geometry.y.values]).T
    centroids = np.vstack([gdf_blocks_3857["cx"].values, gdf_blocks_3857["cy"].values]).T
    if cKDTree is not None:
        tree = cKDTree(train_xy)
        d, _ = tree.query(centroids, k=1)
    else:
        # NumPy fallback — O(N×M)
        diff = centroids[:, None, :] - train_xy[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
    dist_tbl = gdf_blocks_3857[["GEOID"]].copy()
    dist_tbl["distance_to_train"] = d.astype(float)

# ---------- 6) GEOID-level feature + binler ----------
geo_metrics = dist_tbl.merge(train_count, on="GEOID", how="left")
geo_metrics["train_stop_count"] = pd.to_numeric(geo_metrics["train_stop_count"], errors="coerce").fillna(0).astype(int)
geo_metrics["GEOID"] = normalize_geoid(geo_metrics["GEOID"], DEFAULT_GEOID_LEN)

# sadece crime GEOID evreni
geo_metrics = geo_metrics.merge(crime_geoids.to_frame(), on="GEOID", how="right")

dist = pd.to_numeric(geo_metrics["distance_to_train"], errors="coerce").replace([np.inf, -np.inf], np.nan)
finite_dist = dist.dropna()
if len(finite_dist) >= 2 and finite_dist.max() > finite_dist.min():
    n_bins = freedman_diaconis_bin_count(finite_dist.to_numpy(), max_bins=10)
    _, edges = pd.qcut(finite_dist, q=n_bins, retbins=True, duplicates="drop")
    labels = [f"{int(round(edges[i]))}–{int(round(edges[i+1]))}m" for i in range(len(edges) - 1)]
    geo_metrics["distance_to_train_range"] = pd.cut(dist, bins=edges, labels=labels, include_lowest=True)
else:
    geo_metrics["distance_to_train_range"] = "0–0m"

cnt = pd.to_numeric(geo_metrics["train_stop_count"], errors="coerce").fillna(0)
if cnt.nunique() > 1:
    n_c_bins = freedman_diaconis_bin_count(cnt.to_numpy(), max_bins=8)
    _, c_edges = pd.qcut(cnt, q=n_c_bins, retbins=True, duplicates="drop")
    c_labels = [f"{int(round(c_edges[i]))}–{int(round(c_edges[i+1]))}" for i in range(len(c_edges) - 1)]
    geo_metrics["train_stop_count_range"] = pd.cut(cnt, bins=c_edges, labels=c_labels, include_lowest=True)
else:
    geo_metrics["train_stop_count_range"] = f"{int(cnt.min())}–{int(cnt.max())}"

# GEOID tekilleştir
geo_metrics = geo_metrics.sort_values("GEOID").drop_duplicates("GEOID", keep="first")
assert geo_metrics["GEOID"].is_unique, "TRAIN: GEOID unique olmalı"

# ---------- 7) GEOID-level özeti yaz (train.csv) ----------
safe_save_csv(geo_metrics, str(TRAIN_SUMMARY_NAME))
print(f"✅ TRAIN özet (GEOID-level) yazıldı → {TRAIN_SUMMARY_NAME}")

# ---------- 8) crime ile merge ----------
_before = crime.shape
crime_enriched = crime.merge(geo_metrics, on="GEOID", how="left", validate="many_to_one")
log(f"🔗 CRIME ⨯ TRAIN: {_before} → {crime_enriched.shape} (KDTree={_TREE_IMPL})")

# ---------- 9) yaz ----------
safe_save_csv(crime_enriched, str(DAILY_OUT))
try:
    print("daily_crime_05.csv — ilk 5 satır")
    print(crime_enriched.head(5).to_string(index=False))
except Exception:
    pass
