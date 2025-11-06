#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_police_gov.py
# Amaç: Suç verisini POLICE & GOVERNMENT noktalarıyla YALNIZCA GEOID üzerinden zenginleştirmek.
# Not: 'date' kullanılmaz; GEOID → centroid (veya lat/lon ortalaması) → en yakın polis/gov mesafesi hesaplanır.
# Çıktı, giriş dosyasına göre 06→07 olarak belirlenir; aksi halde *_pg.csv uzantısı kullanılır.

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# -----------------------------------------------------------------------------
# LOG/YARDIMCI
# -----------------------------------------------------------------------------
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        try:
            df.to_csv(path + ".bak", index=False)
            print(f"📁 Yedek oluşturuldu: {path}.bak")
        except Exception as e2:
            print(f"❌ Yedek de kaydedilemedi: {e2}")

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len)

def make_quantile_ranges(series: pd.Series, max_bins: int = 5, fallback_label: str = "Unknown") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = s.notna()
    s_valid = s[mask]
    if s_valid.nunique() <= 1 or len(s_valid) < 2:
        return pd.Series([fallback_label] * len(series), index=series.index)
    q = min(max_bins, max(3, s_valid.nunique()))
    try:
        _, edges = pd.qcut(s_valid, q=q, retbins=True, duplicates="drop")
    except Exception:
        return pd.Series([fallback_label] * len(series), index=series.index)
    if len(edges) < 3:
        return pd.Series([fallback_label] * len(series), index=series.index)
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            labels.append(f"Q{i+1} (≤{hi:.1f})")
        else:
            labels.append(f"Q{i+1} ({lo:.1f}-{hi:.1f})")
    out = pd.Series(fallback_label, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(s_valid, bins=edges, labels=labels, include_lowest=True).astype(str)
    return out

def prep_points(df_points: pd.DataFrame) -> pd.DataFrame:
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

# -----------------------------------------------------------------------------
# GİRİŞ/ÇIKIŞ YOLLARI (sağlam)
# -----------------------------------------------------------------------------
def here(*p): return Path.cwd().joinpath(*p)
def pexists(p): return Path(p).expanduser().resolve().exists()

CRIME_DATA_DIR = os.getenv("CRIME_DATA_DIR", "").strip()

BASE_CANDIDATES = [
    CRIME_DATA_DIR,                       # env ile gelen tam yol
    str(here("crime_prediction_data")),   # repo kökünde tipik klasör
    str(here()),                          # doğrudan CWD
]

INPUT_FILE = "sf_crime_06.csv"
CRIME_INPUT_CANDIDATES = []
for base in BASE_CANDIDATES:
    if base:
        CRIME_INPUT_CANDIDATES.append(str(Path(base) / INPUT_FILE))
CRIME_INPUT_CANDIDATES.append(str(here(INPUT_FILE)))  # kökte olabilir

print("🔎 sf_crime_06.csv aday yollar:")
for p in CRIME_INPUT_CANDIDATES:
    print("  -", p, "✅" if pexists(p) else "❌")

CRIME_IN = next((p for p in CRIME_INPUT_CANDIDATES if pexists(p)), None)
if CRIME_IN is None:
    raise FileNotFoundError(
        "❌ Suç girdisi bulunamadı. Şunlardan en az biri olmalı: "
        + ", ".join(CRIME_INPUT_CANDIDATES)
    )

in_dir = Path(CRIME_IN).parent
CRIME_OUT = str(in_dir / ("sf_crime_07.csv" if Path(CRIME_IN).name == "sf_crime_06.csv" else f"{Path(CRIME_IN).stem}_pg.csv"))

# Polis/Gov adayları: aynı kök + yaygın yerler
POLICE_CANDIDATES = [
    str(in_dir / "sf_police_stations.csv"),
    str(Path(CRIME_DATA_DIR) / "sf_police_stations.csv") if CRIME_DATA_DIR else "",
    str(here("crime_prediction_data", "sf_police_stations.csv")),
    str(here("sf_police_stations.csv")),
]
GOV_CANDIDATES = [
    str(in_dir / "sf_government_buildings.csv"),
    str(Path(CRIME_DATA_DIR) / "sf_government_buildings.csv") if CRIME_DATA_DIR else "",
    str(here("crime_prediction_data", "sf_government_buildings.csv")),
    str(here("sf_government_buildings.csv")),
]
POLICE_CANDIDATES = [p for p in POLICE_CANDIDATES if p]
GOV_CANDIDATES    = [p for p in GOV_CANDIDATES if p]

def pick_existing(paths):
    for p in paths:
        if pexists(p):
            return p
    return None

print(f"📂 Seçilen giriş: {CRIME_IN}")
print(f"📂 Yazılacak çıkış: {CRIME_OUT}")

# -----------------------------------------------------------------------------
# VERİ OKU
# -----------------------------------------------------------------------------
df = pd.read_csv(CRIME_IN, low_memory=False)
log_shape(df, "CRIME (yükleme)")

if "GEOID" not in df.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")
df["GEOID"] = normalize_geoid(df["GEOID"], target_len=11)

lat_pref = find_col(df.columns, ["centroid_lat", "latitude", "lat", "y"])
lon_pref = find_col(df.columns, ["centroid_lon", "longitude", "lon", "x"])
if lat_pref is None or lon_pref is None:
    raise KeyError("❌ 'latitude/longitude' veya 'centroid_lat/centroid_lon' benzeri kolonlar bulunamadı.")

df["_lat_"] = pd.to_numeric(df[lat_pref], errors="coerce")
df["_lon_"] = pd.to_numeric(df[lon_pref], errors="coerce")

geo = (
    df.dropna(subset=["_lat_", "_lon_"])
      .groupby("GEOID", as_index=False)[["_lat_", "_lon_"]]
      .mean()
      .rename(columns={"_lat_": "centroid_lat", "_lon_": "centroid_lon"})
)
log_shape(geo, "GEOID centroid (hazır)")

# -----------------------------------------------------------------------------
# POLICE / GOVERNMENT
# -----------------------------------------------------------------------------
police_path = pick_existing(POLICE_CANDIDATES)
gov_path    = pick_existing(GOV_CANDIDATES)

if police_path is None:
    print("⚠️ sf_police_stations.csv bulunamadı; polis mesafeleri NaN olacak.")
    df_police = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_police = pd.read_csv(police_path, low_memory=False)

if gov_path is None:
    print("⚠️ sf_government_buildings.csv bulunamadı; government mesafeleri NaN olacak.")
    df_gov = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_gov = pd.read_csv(gov_path, low_memory=False)

df_police = prep_points(df_police)
df_gov    = prep_points(df_gov)

# -----------------------------------------------------------------------------
# BALLTREE (Haversine)
# -----------------------------------------------------------------------------
EARTH_R = 6_371_000.0  # metre
centroids_rad = np.radians(geo[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float))

if not df_police.empty:
    police_rad = np.radians(df_police[["latitude", "longitude"]].to_numpy(dtype=float))
    police_tree = BallTree(police_rad, metric="haversine")
    dist_police, _ = police_tree.query(centroids_rad, k=1)
    geo["distance_to_police"] = (dist_police[:, 0] * EARTH_R).round(1)
else:
    geo["distance_to_police"] = np.nan

if not df_gov.empty:
    gov_rad = np.radians(df_gov[["latitude", "longitude"]].to_numpy(dtype=float))
    gov_tree = BallTree(gov_rad, metric="haversine")
    dist_gov, _ = gov_tree.query(centroids_rad, k=1)
    geo["distance_to_government_building"] = (dist_gov[:, 0] * EARTH_R).round(1)
else:
    geo["distance_to_government_building"] = np.nan

geo["is_near_police"] = (geo["distance_to_police"] <= 300).astype("Int64")
geo["is_near_government"] = (geo["distance_to_government_building"] <= 300).astype("Int64")

geo["distance_to_police_range"] = make_quantile_ranges(geo["distance_to_police"], max_bins=5, fallback_label="Unknown")
geo["distance_to_government_building_range"] = make_quantile_ranges(
    geo["distance_to_government_building"], max_bins=5, fallback_label="Unknown"
)

log_shape(geo, "GEOID metrikleri (polis+gov)")

# -----------------------------------------------------------------------------
# MERGE (sadece GEOID)
# -----------------------------------------------------------------------------
_before = df.shape
keep_cols = [
    "GEOID",
    "distance_to_police", "distance_to_police_range",
    "distance_to_government_building", "distance_to_government_building_range",
    "is_near_police", "is_near_government",
]
df = df.merge(geo[keep_cols], on="GEOID", how="left")
log_delta(_before, df.shape, "CRIME ⨯ GEOID(polis+gov)")

# -----------------------------------------------------------------------------
# KAYDET
# -----------------------------------------------------------------------------
safe_save_csv(df, CRIME_OUT)
print(f"✅ Kaydedildi: {CRIME_OUT} | Satır: {len(df):,} | Sütun: {df.shape[1]}")

try:
    preview = pd.read_csv(CRIME_OUT, nrows=3)
    print(f"📄 {CRIME_OUT} — ilk 3 satır:")
    print(preview.to_string(index=False))
except Exception as e:
    print(f"⚠️ Önizleme okunamadı: {e}")
