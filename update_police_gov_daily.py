#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_police_gov_daily.py
# IN : daily_crime_06.csv (GEOID zorunlu; tercihen latitude/longitude veya centroid_lat/centroid_lon)
# OUT: daily_crime_07.csv — GEOID-level polis & government mesafeleri + aralık & yakınlık bayrakları

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

# BallTree opsiyonel (yoksa NumPy haversine fallback)
try:
    from sklearn.neighbors import BallTree
    _BALLTREE_OK = True
except Exception:
    BallTree = None
    _BALLTREE_OK = False

pd.options.mode.copy_on_write = True

# ───────────────── helpers ─────────────────
def log(msg: str): print(msg, flush=True)

def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    log(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    log(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"💾 Kaydedildi: {path} | satır={len(df):,}, sütun={df.shape[1]}")

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

def make_quantile_ranges(series: pd.Series, max_bins: int = 5, fallback_label: str = "Unknown") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = s.notna()
    s_valid = s[mask]
    if s_valid.nunique() <= 1 or len(s_valid) < 2:
        return pd.Series([fallback_label] * len(series), index=series.index, dtype="object")
    q = min(max_bins, max(3, s_valid.nunique()))
    try:
        _, edges = pd.qcut(s_valid, q=q, retbins=True, duplicates="drop")
    except Exception:
        return pd.Series([fallback_label] * len(series), index=series.index, dtype="object")
    if len(edges) < 3:
        return pd.Series([fallback_label] * len(series), index=series.index, dtype="object")
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        labels.append(f"Q{i+1} ({(lo if i>0 else float('-inf')):.1f}-{hi:.1f})")
    out = pd.Series(fallback_label, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(s_valid, bins=edges, labels=labels, include_lowest=True).astype(str)
    return out

def prep_points(df_points: pd.DataFrame) -> pd.DataFrame:
    """Polis/Gov noktaları için lat/lon kolonlarını standartlaştırır."""
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=["latitude", "longitude"])
    lat_col = find_col(df_points.columns, ["latitude", "lat", "y"])
    lon_col = find_col(df_points.columns, ["longitude", "lon", "x"])
    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["latitude", "longitude"])
    out = df_points.rename(columns={lat_col: "latitude", lon_col: "longitude"}).copy()
    out["latitude"]  = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"])
    return out[["latitude", "longitude"]]

def pick_existing(paths):
    for p in paths:
        if p:
            fp = Path(p)
            if fp.exists() and fp.is_file():
                return str(fp)
    return None

# Haversine (radyan koordinatlar, çıktı: metre)
_EARTH_R = 6_371_000.0
def haversine_m(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * _EARTH_R * np.arcsin(np.sqrt(a))

def nearest_distances(centroids_deg: np.ndarray, points_deg: np.ndarray) -> np.ndarray:
    """
    centroids_deg: (N,2) deg, points_deg: (M,2) deg → returns (N,) distances in meters to nearest point
    """
    if points_deg.size == 0:
        return np.full((centroids_deg.shape[0],), np.nan, dtype=float)
    # BallTree varsa hızlı
    if _BALLTREE_OK:
        tree = BallTree(np.radians(points_deg), metric="haversine")
        d, _ = tree.query(np.radians(centroids_deg), k=1)
        return (d[:, 0] * _EARTH_R).astype(float)
    # Fallback: vektörize NumPy
    c_rad = np.radians(centroids_deg)
    p_rad = np.radians(points_deg)
    # (N,1,2) vs (1,M,2) yayınla, haversine hesapla
    d = haversine_m(c_rad[:, None, 0], c_rad[:, None, 1], p_rad[None, :, 0], p_rad[None, :, 1])
    return d.min(axis=1).astype(float)

# ───────────────── config ─────────────────
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(exist_ok=True, parents=True)

# IN / OUT
CRIME_IN  = os.path.join(BASE_DIR, os.getenv("DAILY_IN",  "daily_crime_06.csv"))
CRIME_OUT = os.path.join(BASE_DIR, os.getenv("DAILY_OUT", "daily_crime_07.csv"))

# Opsiyonel: Blocks GeoJSON (centroid fallback için)
BLOCKS_GEOJSON = os.path.join(BASE_DIR, os.getenv("SF_BLOCKS_GEOJSON", "sf_census_blocks_with_population.geojson"))

# Polis & Gov dosyaları (yerel)
POLICE_CANDIDATES = [
    os.path.join(BASE_DIR, os.getenv("POLICE_CSV", "sf_police_stations.csv")),
    os.path.join(".",      "sf_police_stations.csv"),
]
GOV_CANDIDATES = [
    os.path.join(BASE_DIR, os.getenv("GOV_CSV", "sf_government_buildings.csv")),
    os.path.join(".",      "sf_government_buildings.csv"),
]

NEAR_THRESHOLD_M = float(os.getenv("NEAR_THRESHOLD_M", "300"))

# ───────────────── load ─────────────────
if not Path(CRIME_IN).exists():
    raise FileNotFoundError(f"❌ Girdi yok: {CRIME_IN}")

df = pd.read_csv(CRIME_IN, low_memory=False)
log_shape(df, "CRIME (yükleme)")

if "GEOID" not in df.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")
df["GEOID"] = normalize_geoid(df["GEOID"], target_len=11)

# GEOID konumu (centroid proxy)
lat_pref = find_col(df.columns, ["centroid_lat", "latitude", "lat", "y"])
lon_pref = find_col(df.columns, ["centroid_lon", "longitude", "lon", "x"])

geo = pd.DataFrame(columns=["GEOID","centroid_lat","centroid_lon"])
if lat_pref and lon_pref:
    df["_lat_"] = pd.to_numeric(df[lat_pref], errors="coerce")
    df["_lon_"] = pd.to_numeric(df[lon_pref], errors="coerce")
    geo = (
        df.dropna(subset=["_lat_", "_lon_"])
          .groupby("GEOID", as_index=False)[["_lat_", "_lon_"]]
          .mean()
          .rename(columns={"_lat_": "centroid_lat", "_lon_": "centroid_lon"})
    )

# Eğer halen boş ise ve Blocks GeoJSON mevcutsa, oradan centroid üret
if geo.empty and Path(BLOCKS_GEOJSON).exists() and Path(BLOCKS_GEOJSON).is_file():
    try:
        import geopandas as gpd
        gdf_blocks = gpd.read_file(BLOCKS_GEOJSON)
        gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next(
            (c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None
        )
        if gcol:
            gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], 11)
            if gdf_blocks.crs is None:
                gdf_blocks.set_crs("EPSG:4326", inplace=True)
            ctr = gdf_blocks.to_crs(3857).copy()
            ctr["centroid"] = ctr.geometry.centroid
            ctr_xy = ctr[["GEOID","centroid"]].copy().to_crs(4326)
            ctr_xy["centroid_lat"] = ctr_xy["centroid"].y
            ctr_xy["centroid_lon"] = ctr_xy["centroid"].x
            geo = ctr_xy.drop(columns=["centroid"])
    except Exception as e:
        log(f"⚠️ Blocks centroid üretimi başarısız: {e}")

if geo.empty:
    raise KeyError("❌ GEOID centroid üretilemedi: 'latitude/longitude' ya da Blocks GeoJSON gerek.")

geo = geo.dropna(subset=["centroid_lat","centroid_lon"]).copy()
geo["centroid_lat"]  = pd.to_numeric(geo["centroid_lat"], errors="coerce")
geo["centroid_lon"]  = pd.to_numeric(geo["centroid_lon"], errors="coerce")
geo = geo.dropna(subset=["centroid_lat","centroid_lon"])
geo = geo.sort_values("GEOID").drop_duplicates("GEOID", keep="first")
log_shape(geo, "GEOID centroid (hazır)")

# Polis / Government noktaları
police_path = pick_existing(POLICE_CANDIDATES)
gov_path    = pick_existing(GOV_CANDIDATES)

if police_path is None:
    log("⚠️ sf_police_stations.csv bulunamadı; polis mesafeleri NaN.")
    df_police = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_police = pd.read_csv(police_path, low_memory=False)

if gov_path is None:
    log("⚠️ sf_government_buildings.csv bulunamadı; government mesafeleri NaN.")
    df_gov = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_gov = pd.read_csv(gov_path, low_memory=False)

df_police = prep_points(df_police)
df_gov    = prep_points(df_gov)

# ───────────────── nearest distances ─────────────────
centroids_deg = geo[["centroid_lat","centroid_lon"]].to_numpy(dtype=float)

if not df_police.empty:
    police_deg = df_police[["latitude","longitude"]].to_numpy(dtype=float)
    geo["distance_to_police"] = nearest_distances(centroids_deg, police_deg).round(1)
else:
    geo["distance_to_police"] = np.nan

if not df_gov.empty:
    gov_deg = df_gov[["latitude","longitude"]].to_numpy(dtype=float)
    geo["distance_to_government_building"] = nearest_distances(centroids_deg, gov_deg).round(1)
else:
    geo["distance_to_government_building"] = np.nan

# Yakınlık bayrakları (ENV ile ayarlanabilir)
geo["is_near_police"] = (geo["distance_to_police"] <= NEAR_THRESHOLD_M).astype("Int64")
geo["is_near_government"] = (geo["distance_to_government_building"] <= NEAR_THRESHOLD_M).astype("Int64")

# Dinamik aralık etiketleri
geo["distance_to_police_range"] = make_quantile_ranges(geo["distance_to_police"], max_bins=5, fallback_label="Unknown")
geo["distance_to_government_building_range"] = make_quantile_ranges(
    geo["distance_to_government_building"], max_bins=5, fallback_label="Unknown"
)

log_shape(geo, "GEOID metrikleri (polis+gov)")

# ───────────────── merge (GEOID-only) ─────────────────
_before = df.shape
keep_cols = [
    "GEOID",
    "distance_to_police", "distance_to_police_range",
    "distance_to_government_building", "distance_to_government_building_range",
    "is_near_police", "is_near_government",
]
df = df.merge(geo[keep_cols], on="GEOID", how="left", validate="many_to_one")
log_delta(_before, df.shape, "CRIME ⨯ GEOID(polis+gov)")

# ───────────────── save ─────────────────
safe_save_csv(df, CRIME_OUT)

# Kısa önizleme
try:
    show_cols = ["GEOID","distance_to_police","distance_to_government_building","is_near_police","is_near_government"]
    print(df[show_cols].head(5).to_string(index=False))
except Exception:
    pass
