#!/usr/bin/env python
# -*- coding: utf-8 -*-
# update_311_fr.py (revised)
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

# SF kabaca BBOX (min_lon, min_lat, max_lon, max_lat)
SF_BBOX = (-123.2, 37.6, -122.3, 37.9)


# ----------------- yardımcılar -----------------
def log(msg: str): print(msg, flush=True)

def _find_existing(paths, base_dir=SAVE_DIR) -> Path | None:
    for name in paths:
        p1 = Path(base_dir) / name
        p2 = Path(".") / name
        if p1.exists(): return p1
        if p2.exists(): return p2
    return None

def _normalize_headers(df: pd.DataFrame) -> dict:
    """Map of normalized header -> original header (strip spaces, lower)."""
    return {re.sub(r"\s+", "", str(c)).lower(): c for c in df.columns}

def _detect_point_columns(df: pd.DataFrame):
    normmap = _normalize_headers(df)
    lat_alias = ("latitude","lat","y","_lat_")
    lon_alias = ("longitude","long","lon","x","_lon_")
    lat_col = next((normmap[a] for a in lat_alias if a in normmap), None)
    lon_col = next((normmap[a] for a in lon_alias if a in normmap), None)
    return lat_col, lon_col

def _coerce_decimal(series: pd.Series) -> pd.Series:
    """Virgüllü ondalıkları da güvenle sayıya çevir (\"37,77\" -> 37.77)."""
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _ensure_point_gdf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    df = df.copy()
    # Sayısal çeviri (virgül vs.)
    df["__lat"] = _coerce_decimal(df[lat_col])
    df["__lon"] = _coerce_decimal(df[lon_col])

    # BBOX öncesi sayım
    nn_lat0, nn_lon0 = df["__lat"].notna().sum(), df["__lon"].notna().sum()
    log(f"[coords] numeric non-null lat/lon: {nn_lat0}/{nn_lon0}")

    # SF BBOX filtresi
    min_lon, min_lat, max_lon, max_lat = SF_BBOX
    mask_bbox = df["__lat"].between(min_lat, max_lat) & df["__lon"].between(min_lon, max_lon)
    kept = int(mask_bbox.sum())
    log(f"[coords] BBOX içinde kalan: {kept}/{len(df)}")

    use = df.loc[mask_bbox & df["__lat"].notna() & df["__lon"].notna()].copy()

    # BBOX sonrası hiç kalmadıysa lat/lon swap dene
    if use.empty:
        log("[coords] BBOX sonrası satır kalmadı; lat/lon swap deneniyor…")
        df_sw = df.copy()
        df_sw["__lat"], df_sw["__lon"] = df_sw["__lon"], df_sw["__lat"]
        mask_sw = df_sw["__lat"].between(min_lat, max_lat) & df_sw["__lon"].between(min_lon, max_lon)
        kept_sw = int(mask_sw.sum())
        log(f"[coords] swap sonrası BBOX içinde kalan: {kept_sw}/{len(df_sw)}")
        if kept_sw > 0:
            log("[coords] enlem-boylam sütunları yer değiştirmişti, düzeltildi.")
            use = df_sw.loc[mask_sw & df_sw["__lat"].notna() & df_sw["__lon"].notna()].copy()
        else:
            # BBOX devre dışı brak: şehir dışı veri veya BBOX yanlışlığı
            log("[coords] swap da başarısız; BBOX devre dışı bırakılıyor.")
            use = df.loc[df["__lat"].notna() & df["__lon"].notna()].copy()
            if use.empty:
                return gpd.GeoDataFrame(use, geometry=[])

    gdf = gpd.GeoDataFrame(
        use,
        geometry=gpd.points_from_xy(use["__lon"], use["__lat"]),
        crs="EPSG:4326"
    )
    return gdf

def _pick_lat_lon(df: pd.DataFrame):
    # Eski fonksiyon, geriye dönük uyumluluk için normalize'lı versiyona yönlendiriyoruz
    return _detect_point_columns(df)

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
        # daha esnek: ham dosya zorunluluğu yerine gerçek başlıkları göster
        log(f"❌ Seçilen 311 dosyasında latitude/longitude başlığı tespit edilemedi. Mevcut kolonlar: {list(df311.columns)}")
        sys.exit(1)

    # Noktaları güvenli biçimde hazırla (comma decimal, BBOX, swap)
    pts311 = _ensure_point_gdf(df311, lat311, lon311)
    if pts311.empty:
        log("❌ 311 nokta kümesi boş (başlık/dönüşüm/BBOX sorunları olabilir).")
        sys.exit(1)
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
