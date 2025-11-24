# update_police_gov_fr.py
# Amaç:
#  - OLAY BAZLI (latitude/longitude) en yakın POLICE & GOV mesafeleri
#  - Eğer event coords yoksa GEOID centroid üzerinden distance üret
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
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        print(f"📁 Yedek oluşturuldu: {path}.bak")

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len).str[:target_len]

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
        labels.append(
            f"Q{i+1} (≤{hi:.1f})" if i == 0 else f"Q{i+1} ({lo:.1f}-{hi:.1f})"
        )

    out = pd.Series(fallback_label, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(
        s_valid, bins=edges, labels=labels, include_lowest=True
    ).astype(str)
    return out

def prep_points(df_points: pd.DataFrame) -> pd.DataFrame:
    """
    Girdi df'de lat/lon varyasyonlarını bulur ve
    ["latitude","longitude"] standardına çeker.
    """
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=["latitude", "longitude"])

    lat_col = find_col(df_points.columns, ["latitude", "lat", "y", "centroid_lat"])
    lon_col = find_col(df_points.columns, ["longitude", "lon", "x", "centroid_lon"])

    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["latitude", "longitude"])

    out = df_points.rename(columns={lat_col: "latitude", lon_col: "longitude"}).copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"])
    return out[["latitude", "longitude"]]

def load_centroids(path: str) -> pd.DataFrame:
    """
    Centroid dosyasını okur ve GEOID, latitude, longitude kolonlarını döndürür.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["GEOID", "latitude", "longitude"])

    try:
        cdf = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"⚠️ Centroid dosyası okunamadı: {path}\n{e}")
        return pd.DataFrame(columns=["GEOID", "latitude", "longitude"])

    if "GEOID" not in cdf.columns:
        geoid_col = find_col(cdf.columns, ["geoid"])
        if geoid_col:
            cdf = cdf.rename(columns={geoid_col: "GEOID"})
        else:
            print(f"⚠️ Centroid dosyasında GEOID yok: {path}")
            return pd.DataFrame(columns=["GEOID", "latitude", "longitude"])

    cdf["GEOID"] = normalize_geoid(cdf["GEOID"], 11)

    lat_col = find_col(cdf.columns, ["centroid_lat", "latitude", "lat", "y"])
    lon_col = find_col(cdf.columns, ["centroid_lon", "longitude", "lon", "x"])

    if lat_col is None or lon_col is None:
        print(f"⚠️ Centroid dosyasında lat/lon kolonları yok: {path}")
        return pd.DataFrame(columns=["GEOID", "latitude", "longitude"])

    cdf = cdf.rename(columns={lat_col: "latitude", lon_col: "longitude"}).copy()
    cdf["latitude"] = pd.to_numeric(cdf["latitude"], errors="coerce")
    cdf["longitude"] = pd.to_numeric(cdf["longitude"], errors="coerce")
    cdf = cdf.dropna(subset=["latitude", "longitude"])

    return cdf[["GEOID", "latitude", "longitude"]]


# -------------------------- GİRİŞ/ÇIKIŞ --------------------------
BASE_DIR = "crime_prediction_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_IN = os.path.join(BASE_DIR, "fr_crime_06.csv")
if not os.path.exists(CRIME_IN):
    raise FileNotFoundError(f"❌ Girdi bulunamadı: {CRIME_IN}")

CRIME_OUT = os.path.join(BASE_DIR, "fr_crime_07.csv")

POLICE_CANDS = [
    os.path.join(BASE_DIR, "sf_police_stations.csv"),
    "sf_police_stations.csv",
]
GOV_CANDS = [
    os.path.join(BASE_DIR, "sf_government_buildings.csv"),
    "sf_government_buildings.csv",
]

# Centroid adayları (ENV + default)
CENTROIDS_ENV = os.getenv("FR_CENTROIDS_PATH", "").strip()
CENTROID_CANDS = [CENTROIDS_ENV] if CENTROIDS_ENV else []
CENTROID_CANDS += [
    os.path.join(BASE_DIR, "sf_blocks_centroids.csv"),
    os.path.join(BASE_DIR, "sf_centroids.csv"),
    "sf_blocks_centroids.csv",
    "sf_centroids.csv",
]


# -------------------------- YÜKLE --------------------------
df = pd.read_csv(CRIME_IN, low_memory=False)
log_shape(df, f"CRIME (yükleme: {Path(CRIME_IN).name})")

if "GEOID" not in df.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' yok.")

df["GEOID"] = normalize_geoid(df["GEOID"], 11)


# -------------------------- EVENT veya CENTROID COORD HAZIRLA --------------------------
# 1) Event coords kolonlarını ara
evt_lat_col = find_col(df.columns, ["latitude", "lat", "y"])
evt_lon_col = find_col(df.columns, ["longitude", "lon", "x"])

# 2) Mevcut centroid kolonları var mı?
clat_col = find_col(df.columns, ["centroid_lat", "cent_lat"])
clon_col = find_col(df.columns, ["centroid_lon", "cent_lon"])

# Olay coords varsa al
if evt_lat_col and evt_lon_col:
    df["_lat_evt_"] = pd.to_numeric(df[evt_lat_col], errors="coerce")
    df["_lon_evt_"] = pd.to_numeric(df[evt_lon_col], errors="coerce")
else:
    df["_lat_evt_"] = np.nan
    df["_lon_evt_"] = np.nan

# Centroid kolonları varsa event boşlarını doldur
if clat_col and clon_col:
    df["_lat_evt_"] = df["_lat_evt_"].fillna(pd.to_numeric(df[clat_col], errors="coerce"))
    df["_lon_evt_"] = df["_lon_evt_"].fillna(pd.to_numeric(df[clon_col], errors="coerce"))

# 3) GEOID içinden ortalama event coords varsa doldur (nadiren işine yarar)
miss_mask = df["_lat_evt_"].isna() | df["_lon_evt_"].isna()
if miss_mask.any():
    geo_mean = (
        df.loc[~miss_mask]
        .groupby("GEOID")[["_lat_evt_", "_lon_evt_"]]
        .mean()
    )
    if not geo_mean.empty:
        df.loc[miss_mask, "_lat_evt_"] = df.loc[miss_mask, "GEOID"].map(geo_mean["_lat_evt_"])
        df.loc[miss_mask, "_lon_evt_"] = df.loc[miss_mask, "GEOID"].map(geo_mean["_lon_evt_"])

valid_evt_mask = df["_lat_evt_"].notna() & df["_lon_evt_"].notna()
n_valid_evt = int(valid_evt_mask.sum())
print(f"ℹ️ Event koordinatı bulunan satır: {n_valid_evt:,} / {len(df):,}")

# Eğer event yoksa centroid tablosu yükle
centroids_df = pd.DataFrame(columns=["GEOID", "latitude", "longitude"])
if n_valid_evt == 0:
    cent_path = next((p for p in CENTROID_CANDS if p and os.path.exists(p)), None)
    if cent_path:
        print(f"📍 Centroid dosyası bulundu → {cent_path}")
        centroids_df = load_centroids(cent_path)
        print(f"📊 Centroids: {len(centroids_df):,} GEOID")
    else:
        print("⚠️ Event coords yok + centroid dosyası bulunamadı → distance NaN kalacak.")


# -------------------------- POLICE/GOV NOKTALARI --------------------------
police_path = next((p for p in POLICE_CANDS if os.path.exists(p)), None)
gov_path = next((p for p in GOV_CANDS if os.path.exists(p)), None)

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
df_gov = prep_points(df_gov)

# -------------------------- BALLTREE (Haversine) --------------------------
EARTH_R = 6_371_000.0  # metre

# distance kolonlarını baştan yarat
df["distance_to_police"] = np.nan
df["distance_to_government_building"] = np.nan

# 1) EVENT-BASED distance (varsa)
if n_valid_evt > 0:
    events_rad = np.radians(df.loc[valid_evt_mask, ["_lat_evt_", "_lon_evt_"]].to_numpy(dtype=float))

    # POLICE
    if not df_police.empty:
        police_rad = np.radians(df_police[["latitude", "longitude"]].to_numpy(dtype=float))
        police_tree = BallTree(police_rad, metric="haversine")
        dist_police, _ = police_tree.query(events_rad, k=1)
        df.loc[valid_evt_mask, "distance_to_police"] = (dist_police[:, 0] * EARTH_R).round(1)

    # GOV
    if not df_gov.empty:
        gov_rad = np.radians(df_gov[["latitude", "longitude"]].to_numpy(dtype=float))
        gov_tree = BallTree(gov_rad, metric="haversine")
        dist_gov, _ = gov_tree.query(events_rad, k=1)
        df.loc[valid_evt_mask, "distance_to_government_building"] = (dist_gov[:, 0] * EARTH_R).round(1)

    print("✅ Event-based distance hesaplandı.")

# 2) CENTROID-BASED distance (event yoksa)
elif not centroids_df.empty:
    # centroid noktalarını radyana çevir
    cen_rad = np.radians(centroids_df[["latitude", "longitude"]].to_numpy(dtype=float))
    centroids_df = centroids_df.copy()

    if not df_police.empty:
        police_rad = np.radians(df_police[["latitude", "longitude"]].to_numpy(dtype=float))
        police_tree = BallTree(police_rad, metric="haversine")
        dist_police, _ = police_tree.query(cen_rad, k=1)
        centroids_df["distance_to_police"] = (dist_police[:, 0] * EARTH_R).round(1)
    else:
        centroids_df["distance_to_police"] = np.nan

    if not df_gov.empty:
        gov_rad = np.radians(df_gov[["latitude", "longitude"]].to_numpy(dtype=float))
        gov_tree = BallTree(gov_rad, metric="haversine")
        dist_gov, _ = gov_tree.query(cen_rad, k=1)
        centroids_df["distance_to_government_building"] = (dist_gov[:, 0] * EARTH_R).round(1)
    else:
        centroids_df["distance_to_government_building"] = np.nan

    # GEOID bazlı merge
    df = df.merge(
        centroids_df[["GEOID", "distance_to_police", "distance_to_government_building"]],
        on="GEOID",
        how="left",
        suffixes=("", "_cen"),
    )

    # merge sonrası kolonları konsolide et
    if "distance_to_police_cen" in df.columns:
        df["distance_to_police"] = df["distance_to_police"].fillna(df["distance_to_police_cen"])
        df.drop(columns=["distance_to_police_cen"], inplace=True)

    if "distance_to_government_building_cen" in df.columns:
        df["distance_to_government_building"] = df["distance_to_government_building"].fillna(
            df["distance_to_government_building_cen"]
        )
        df.drop(columns=["distance_to_government_building_cen"], inplace=True)

    print("✅ Centroid-based distance hesaplandı.")

else:
    print("⚠️ Distance hesaplanamadı (event yok, centroid yok). Kolonlar NaN kalacak.")


# -------------------------- YAKINLIK BAYRAKLARI & RANGE --------------------------
df["is_near_police"] = (df["distance_to_police"] <= 300).astype("Int64")
df["is_near_government"] = (df["distance_to_government_building"] <= 300).astype("Int64")

df["distance_to_police_range"] = make_quantile_ranges(df["distance_to_police"])
df["distance_to_government_building_range"] = make_quantile_ranges(df["distance_to_government_building"])

# Geçici kolonları temizle
df.drop(columns=["_lat_evt_", "_lon_evt_"], inplace=True, errors="ignore")


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
