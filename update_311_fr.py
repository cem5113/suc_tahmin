#!/usr/bin/env python
# -*- coding: utf-8 -*-
# update_311_fr.py
#
# Amaç:
# - 311 verisini *varsa* _y.csv (ham, nokta bazlı) dosyasından, yoksa orijinal 311 CSV'den oku
# - fr_crime_01.csv içindeki SUÇ noktalarının etrafında 300/600/900m tampon (buffer) alanlarına göre 311 öznitelikleri üret
# - Birleştirme SUÇ koordinatlarına (lat/lon) göre yapılır; GEOID'e bağımlı değildir
# - Çıktı: crime_prediction_data/fr_crime_02.csv
#
# Üretilen 311 alanları:
#   311_cnt_300m, 311_cnt_600m, 311_cnt_900m
#   311_dist_min_m, 311_dist_min_range

from __future__ import annotations
import os, re, sys
import pandas as pd
import geopandas as gpd
from pathlib import Path

SAVE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# 311 dosya adayları (öncelik: *_y.csv)
_311_CANDIDATES = [
    # yeni adlandırmalar
    "sf_311_last_5_years_y.csv",
    # legacy adlandırmalar
    "sf_311_last_5_year_y.csv",
    # bazı repolarda düz ad
    "sf_311_last_5_years.csv",         # dikkat: bazen özet dosyadır (lat/lon olmayabilir)
    "sf_311_last_5_year.csv",
]

CRIME_IN  = Path(SAVE_DIR) / "fr_crime_01.csv"
CRIME_OUT = Path(SAVE_DIR) / "fr_crime_02.csv"

# Varsayılan tampon yarıçapları (metre)
BUFFERS_M = [300, 600, 900]

# ----------------- yardımcılar -----------------
def log(msg: str): print(msg, flush=True)

def _find_existing(paths, base_dir=SAVE_DIR) -> Path | None:
    for name in paths:
        p1 = Path(base_dir) / name
        p2 = Path(".") / name
        if p1.exists(): return p1
        if p2.exists(): return p2
    return None

def _pick_lat_lon(df: pd.DataFrame):
    lat_col = next((c for c in ["latitude","lat","y","_lat_"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude","long","lon","x","_lon_"] if c in df.columns), None)
    return lat_col, lon_col

def _ensure_point_gdf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf

def _project_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # SF için metrik uygun projeksiyon: EPSG:3857 pratik ve hızlı
    try:
        return gdf.to_crs(3857)
    except Exception:
        # son çare: UTM Zone 10N (California kıyıları)
        return gdf.to_crs(32610)

def _distance_features(crime_gdf_m: gpd.GeoDataFrame, pts_gdf_m: gpd.GeoDataFrame) -> pd.Series:
    """
    En yakın 311 noktasına mesafe (metre). pts yoksa NaN.
    """
    if pts_gdf_m.empty:
        return pd.Series([pd.NA]*len(crime_gdf_m), index=crime_gdf_m.index, dtype="float")
    # hızlı KNN için sjoin_nearest (geopandas >=0.10)
    try:
        j = gpd.sjoin_nearest(
            crime_gdf_m[["geometry"]],
            pts_gdf_m[["geometry"]].assign(_idx_pts=pts_gdf_m.index),
            how="left",
            distance_col="_dist"
        )
        return j["_dist"]
    except Exception:
        # fallback: geometri.distance() (daha yavaş)
        dmin = []
        pts = pts_gdf_m.geometry.values
        for geom in crime_gdf_m.geometry.values:
            if len(pts) == 0:
                dmin.append(pd.NA)
            else:
                dmin.append(min(geom.distance(p) for p in pts))
        return pd.Series(dmin, index=crime_gdf_m.index)

def _bin_distance(m):
    if pd.isna(m): return "none"
    try:
        m = float(m)
    except Exception:
        return "none"
    if m <= 300: return "≤300m"
    if m <= 600: return "300–600m"
    if m <= 900: return "600–900m"
    return ">900m"

def _count_within(crime_gdf_m: gpd.GeoDataFrame, pts_gdf_m: gpd.GeoDataFrame, radius_m: int) -> pd.Series:
    """
    Her suç noktası için belirtilen yarıçapta kaç 311 noktası var?
    """
    if pts_gdf_m.empty:
        return pd.Series([0]*len(crime_gdf_m), index=crime_gdf_m.index, dtype="int")
    # Buffer ve sjoin ile say
    buf = crime_gdf_m.buffer(radius_m)
    gdf_buf = gpd.GeoDataFrame(crime_gdf_m[[]], geometry=buf, crs=crime_gdf_m.crs)
    j = gpd.sjoin(pts_gdf_m[["geometry"]], gdf_buf.rename_geometry("geometry"), how="left", predicate="within")
    # j: 311 satırları + index_right (crime index)
    counts = j.groupby("index_right").size()
    out = pd.Series(0, index=crime_gdf_m.index, dtype="int")
    out.loc[counts.index] = counts.values
    return out

# ----------------- ana akış -----------------
def main():
    # 1) fr_crime_01.csv oku
    if not CRIME_IN.exists():
        log(f"❌ Bulunamadı: {CRIME_IN}")
        sys.exit(1)
    crime = pd.read_csv(CRIME_IN, low_memory=False)
    if crime.empty:
        log("❌ fr_crime_01.csv boş.")
        sys.exit(1)

    lat_c, lon_c = _pick_lat_lon(crime)
    if not lat_c or not lon_c:
        log("❌ fr_crime_01.csv içinde latitude/longitude bulunamadı (örn. latitude/longitude veya _lat_/_lon_).")
        sys.exit(1)
    log(f"📥 fr_crime_01.csv yüklendi: {len(crime):,} satır | lat='{lat_c}' lon='{lon_c}'")

    # satır kimliği korumak için index'i sakla
    crime = crime.reset_index(drop=False).rename(columns={"index":"__row_id"})
    crime_gdf = _ensure_point_gdf(crime, lat_c, lon_c)
    if crime_gdf.empty:
        log("❌ Suç noktaları için geçerli koordinat yok.")
        sys.exit(1)
    crime_gdf_m = _project_metric(crime_gdf)

    # 2) 311 CSV seç (öncelik _y.csv)
    _311_path = _find_existing(_311_CANDIDATES, base_dir=SAVE_DIR)
    if _311_path is None:
        _311_path = _find_existing(_311_CANDIDATES, base_dir=".")
    if _311_path is None:
        log("❌ 311 CSV bulunamadı. Adaylar:")
        for n in _311_CANDIDATES: log(f"   - {n}")
        sys.exit(1)

    df311 = pd.read_csv(_311_path, low_memory=False)
    log(f"📥 311 kaynağı: {os.path.abspath(_311_path)} | satır={len(df311):,}")

    lat311, lon311 = _pick_lat_lon(df311)
    if not lat311 or not lon311:
        # bazı özet dosyaları lat/lon içermez → bu durumda işlenemez
        log("❌ Seçilen 311 dosyası nokta (latitude/longitude) içermiyor. Ham (_y.csv) dosyaya ihtiyaç var.")
        log("   Lütfen _y.csv (ham nokta bazlı) 311 dosyasını sağlayın.")
        sys.exit(1)

    pts311 = _ensure_point_gdf(df311, lat311, lon311)
    # SF dışını kaba ele:  lat 37~38, lon -123~-122 gibi; isterseniz ek filtrenizi ekleyin
    pts311 = pts311[(pts311[lat311].between(37.5, 38.2)) & (pts311[lon311].between(-123.2, -122.0))]
    pts311_m = _project_metric(pts311)

    log(f"🗺️ Suç noktası: {len(crime_gdf_m):,} | 311 noktası: {len(pts311_m):,}")

    # 3) mesafe & buffer sayıları
    # — en yakın 311 mesafesi
    dist_min = _distance_features(crime_gdf_m, pts311_m)
    crime_gdf_m["311_dist_min_m"] = pd.to_numeric(dist_min, errors="coerce")
    crime_gdf_m["311_dist_min_range"] = crime_gdf_m["311_dist_min_m"].apply(_bin_distance)

    # — 300/600/900m tampon sayıları
    for r in BUFFERS_M:
        crime_gdf_m[f"311_cnt_{r}m"] = _count_within(crime_gdf_m, pts311_m, r)

    # 4) GeoDataFrame → DataFrame & orijinal suç satırlarıyla birleştir
    features_cols = ["__row_id", "311_dist_min_m", "311_dist_min_range"] + [f"311_cnt_{r}m" for r in BUFFERS_M]
    feat = pd.DataFrame(crime_gdf_m[features_cols].copy())

    # Merge: __row_id üzerinden, suffix oluşumu yok
    before_shape = crime.shape
    merged = crime.merge(feat, on="__row_id", how="left")
    log(f"🔗 Birleştirme: {before_shape} → {merged.shape}")

    # 5) Tip & doldurma
    for c in [f"311_cnt_{r}m" for r in BUFFERS_M]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("int32")
    merged["311_dist_min_m"] = pd.to_numeric(merged["311_dist_min_m"], errors="coerce")
    # aralık etiketi boşsa "none"
    merged["311_dist_min_range"] = merged["311_dist_min_range"].fillna("none")

    # 6) Kaydet
    # __row_id yardımcı kolonu temizleyip yazalım
    out = merged.drop(columns=["__row_id"], errors="ignore")
    out.to_csv(CRIME_OUT, index=False)
    log(f"✅ Yazıldı: {CRIME_OUT} | satır={len(out):,}")

    # kısa önizleme
    try:
        preview_cols = ["311_cnt_300m","311_cnt_600m","311_cnt_900m","311_dist_min_m","311_dist_min_range"]
        show_cols = [c for c in preview_cols if c in out.columns]
        log(out[show_cols].head(5).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
