# update_police_gov.py
# Amaç: OLAY BAZLI (latitude/longitude) en yakın POLICE & GOV mesafeleri.
# Fallback: lat/lon eksikse GEOID centroid kullanılır.
# Girdi:  crime_prediction_data/fr_crime_06.csv
# Çıktı:  crime_prediction_data/fr_crime_07.csv

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

pd.options.mode.copy_on_write = True

# -------------------------- LOG/YARDIMCI --------------------------
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape; print(f"📊 {label}: {r} satır × {c} sütun")

def ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path); df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        print(f"📁 Yedek oluşturuldu: {path}.bak")

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m: return m[cand.lower()]
    return None

def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len).str[:target_len]

def make_quantile_ranges(series: pd.Series, max_bins: int = 5, fallback_label: str = "Unknown") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mask = s.notna(); s_valid = s[mask]
    if s_valid.nunique() <= 1 or len(s_valid) < 2:
        return pd.Series([fallback_label]*len(series), index=series.index)
    q = min(max_bins, max(3, s_valid.nunique()))
    try:
        _, edges = pd.qcut(s_valid, q=q, retbins=True, duplicates="drop")
    except Exception:
        return pd.Series([fallback_label]*len(series), index=series.index)
    if len(edges) < 3:
        return pd.Series([fallback_label]*len(series), index=series.index)

    labels = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        labels.append(f"Q{i+1} (≤{hi:.1f})" if i==0 else f"Q{i+1} ({lo:.1f}-{hi:.1f})")

    out = pd.Series(fallback_label, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(s_valid, bins=edges, labels=labels, include_lowest=True).astype(str)
    return out

def prep_points(df_points: pd.DataFrame) -> pd.DataFrame:
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=["latitude","longitude"])
    lat_col = find_col(df_points.columns, ["latitude","lat","y"])
    lon_col = find_col(df_points.columns, ["longitude","lon","x"])
    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["latitude","longitude"])
    out = df_points.rename(columns={lat_col:"latitude", lon_col:"longitude"}).copy()
    out["latitude"]  = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude","longitude"])
    return out[["latitude","longitude"]]

# -------------------------- GİRİŞ/ÇIKIŞ --------------------------
BASE_DIR = "crime_prediction_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_IN = os.path.join(BASE_DIR, "fr_crime_06.csv")
if not os.path.exists(CRIME_IN):
    raise FileNotFoundError("❌ Girdi bulunamadı: crime_prediction_data/fr_crime_06.csv")

# 06 → 07 (sabit kural)
CRIME_OUT = os.path.join(BASE_DIR, "fr_crime_07.csv")

POLICE_CANDS = [os.path.join(BASE_DIR,"sf_police_stations.csv"), "sf_police_stations.csv"]
GOV_CANDS    = [os.path.join(BASE_DIR,"sf_government_buildings.csv"), "sf_government_buildings.csv"]

# -------------------------- YÜKLE --------------------------
df = pd.read_csv(CRIME_IN, low_memory=False)
log_shape(df, f"CRIME (yükleme: {Path(CRIME_IN).name})")

# GEOID normalize
if "GEOID" not in df.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' yok.")
df["GEOID"] = normalize_geoid(df["GEOID"], 11)

# Olay LAT/LON kolonları
lat_col = find_col(df.columns, ["latitude","lat","y"])
lon_col = find_col(df.columns, ["longitude","lon","x"])
clat_col = find_col(df.columns, ["centroid_lat"])
clon_col = find_col(df.columns, ["centroid_lon"])

df["_lat_evt_"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
df["_lon_evt_"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan

if clat_col and clon_col:
    df["_lat_evt_"] = df["_lat_evt_"].fillna(pd.to_numeric(df[clat_col], errors="coerce"))
    df["_lon_evt_"] = df["_lon_evt_"].fillna(pd.to_numeric(df[clon_col], errors="coerce"))

miss_mask = df["_lat_evt_"].isna() | df["_lon_evt_"].isna()
if miss_mask.any():
    geo_mean = (df.loc[~(df["_lat_evt_"].isna() | df["_lon_evt_"].isna())]
                  .groupby("GEOID")[["_lat_evt_","_lon_evt_"]].mean())
    df.loc[miss_mask, "_lat_evt_"] = df.loc[miss_mask, "GEOID"].map(geo_mean["_lat_evt_"])
    df.loc[miss_mask, "_lon_evt_"] = df.loc[miss_mask, "GEOID"].map(geo_mean["_lon_evt_"])

n_before = len(df)
df = df.dropna(subset=["_lat_evt_","_lon_evt_"]).copy()
if len(df) < n_before:
    print(f"ℹ️ Koordinatı olmayan {n_before - len(df):,} satır atlandı (mesafe hesaplayamadık).")

log_shape(df, "CRIME (olay koordinatları hazır)")

# -------------------------- POLICE/GOV NOKTALARI --------------------------
police_path = next((p for p in POLICE_CANDS if os.path.exists(p)), None)
gov_path    = next((p for p in GOV_CANDS if os.path.exists(p)), None)

if police_path is None:
    print("⚠️ sf_police_stations.csv bulunamadı; polis mesafeleri NaN olacak.")
    df_police = pd.DataFrame(columns=["latitude","longitude"])
else:
    df_police = pd.read_csv(police_path, low_memory=False)

if gov_path is None:
    print("⚠️ sf_government_buildings.csv bulunamadı; government mesafeleri NaN olacak.")
    df_gov = pd.DataFrame(columns=["latitude","longitude"])
else:
    df_gov = pd.read_csv(gov_path, low_memory=False)

df_police = prep_points(df_police)
df_gov    = prep_points(df_gov)

# -------------------------- BALLTREE (Haversine) --------------------------
EARTH_R = 6_371_000.0  # metre
events_rad = np.radians(df[["_lat_evt_","_lon_evt_"]].to_numpy(dtype=float))

# POLICE
if not df_police.empty:
    police_rad = np.radians(df_police[["latitude","longitude"]].to_numpy(dtype=float))
    police_tree = BallTree(police_rad, metric="haversine")
    dist_police, _ = police_tree.query(events_rad, k=1)
    df["distance_to_police"] = (dist_police[:,0] * EARTH_R).round(1)
else:
    df["distance_to_police"] = np.nan

# GOVERNMENT
if not df_gov.empty:
    gov_rad = np.radians(df_gov[["latitude","longitude"]].to_numpy(dtype=float))
    gov_tree = BallTree(gov_rad, metric="haversine")
    dist_gov, _ = gov_tree.query(events_rad, k=1)
    df["distance_to_government_building"] = (dist_gov[:,0] * EARTH_R).round(1)
else:
    df["distance_to_government_building"] = np.nan

# Yakınlık bayrakları (300 m)
df["is_near_police"] = (df["distance_to_police"] <= 300).astype("Int64")
df["is_near_government"] = (df["distance_to_government_building"] <= 300).astype("Int64")

# Dinamik aralık etiketleri
df["distance_to_police_range"] = make_quantile_ranges(df["distance_to_police"])
df["distance_to_government_building_range"] = make_quantile_ranges(df["distance_to_government_building"])

# Geçici kolonları temizle
df = df.drop(columns=["_lat_evt_","_lon_evt_"], errors="ignore")

# -------------------------- KAYDET --------------------------
safe_save_csv(df, CRIME_OUT)
print(f"✅ Kaydedildi: {CRIME_OUT} | Satır: {len(df):,} | Sütun: {df.shape[1]}")

# Hızlı önizleme
try:
    preview = pd.read_csv(CRIME_OUT, nrows=3)
    print(f"📄 {CRIME_OUT} — ilk 3 satır:")
    print(preview.to_string(index=False))
except Exception as e:
    print(f"⚠️ Önizleme okunamadı: {e}")
