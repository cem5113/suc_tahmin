#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911_daily.py — Girdi: sf_crime.csv → Çıktılar:
#   1) fr_crime_events_daily.csv  (event-level + günlük 911 özellikleri)
#   2) fr_crime_grid_daily.csv    (GEOID×date grid + günlük 911 özellikleri)
#
# Notlar:
# - 911 ham/özetten datetime → date; GEOID×date sayımı (n_911_day) + last1d/3d/7d (shift(1))
# - Join yalnızca (GEOID, date) ile yapılır (saat önemsiz)
# - GEOID yoksa lat/lon → GEOID eşlemesi yapılır (census geojson gerekir)

from __future__ import annotations
import os, zipfile
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd

# =========================
# BASIC UTILS
# =========================
def log(msg: str): print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        log(f"💾 Kaydedildi: {path} (satır={len(df):,})")
    except Exception as e:
        b = path + ".bak"
        try:
            df.to_csv(b, index=False)
        except Exception:
            pass
        log(f"❌ Kaydetme hatası: {path} — Yedek: {b}\n{e}")

def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

def first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

# =========================
# CONFIG
# =========================
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

# GİRİŞ
INPUT_CRIME_FILENAME = os.getenv("FR_CRIME_FILE", "sf_crime.csv")

# ÇIKIŞLAR
OUTPUT_EVENTS_DAILY  = os.getenv("FR_EVENTS_DAILY_OUT", "fr_crime_events_daily.csv")
OUTPUT_GRID_DAILY    = os.getenv("FR_GRID_DAILY_OUT",   "fr_crime_grid_daily.csv")
OUTPUT_DIR = Path(os.getenv("FR_OUTPUT_DIR", "crime_prediction_data"))

# Aday yollar
def build_candidates():
    return {
        "FR_911": [
            ARTIFACT_DIR / "sf_911_last_5_year_y.csv",
            ARTIFACT_DIR / "sf_911_last_5_year.csv",
            Path("crime_prediction_data") / "sf_911_last_5_year_y.csv",
            Path("crime_prediction_data") / "sf_911_last_5_year.csv",
        ],
        "CENSUS": [
            ARTIFACT_DIR / "sf_census_blocks_with_population.geojson",
            Path("crime_prediction_data") / "sf_census_blocks_with_population.geojson",
            Path("./sf_census_blocks_with_population.geojson"),
        ],
        "CRIME": [
            ARTIFACT_DIR / INPUT_CRIME_FILENAME,
            Path("crime_prediction_data") / INPUT_CRIME_FILENAME,
            Path(INPUT_CRIME_FILENAME),
        ],
    }

# =========================
# ZIP HELPERS
# =========================
def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))
    except Exception:
        return False

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        log(f"ℹ️ Artifact ZIP bulunamadı: {zip_path} — klasörlerden denenecek.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in zf.infolist():
            out_path = dest_dir / m.filename
            if not _is_within_directory(dest_dir, out_path.parent):
                raise RuntimeError(f"Zip path outside target dir engellendi: {m.filename}")
            if m.is_dir():
                out_path.mkdir(parents=True, exist_ok=True); continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, 'r') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

# =========================
# GEO HELPERS
# =========================
def _load_blocks(CENSUS_GEOJSON_CANDIDATES) -> tuple[gpd.GeoDataFrame, int]:
    census_path = first_existing(CENSUS_GEOJSON_CANDIDATES)
    if census_path is None:
        raise FileNotFoundError("❌ GEOID poligon dosyası yok: sf_census_blocks_with_population.geojson")

    gdf_blocks = gpd.read_file(census_path)
    if "GEOID" not in gdf_blocks.columns:
        cand = [c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID benzeri sütun yok.")
        gdf_blocks = gdf_blocks.rename(columns={cand[0]: "GEOID"})

    tlen = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
    gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], int(tlen))

    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True)
    elif gdf_blocks.crs.to_epsg() != 4326:
        gdf_blocks = gdf_blocks.to_crs(4326)
    return gdf_blocks, int(tlen)

def ensure_geoid_from_latlon(df: pd.DataFrame, CENSUS_GEOJSON_CANDIDATES) -> pd.DataFrame:
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df

    lat_col = next((c for c in ["latitude", "lat", "y"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude", "lon", "x"] if c in df.columns), None)
    if not lat_col or not lon_col:
        raise ValueError("❌ Veride GEOID yok ve lat/lon bulunamadı.")

    gdf_blocks, tlen = _load_blocks(CANDS["CENSUS"])

    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col]).copy()

    pts = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    return out

# =========================
# 911 DAILY SUMMARY (NO HOURS)
# =========================
def read_911_daily(FR_911_CANDIDATES, CENSUS_GEOJSON_CANDIDATES) -> pd.DataFrame:
    src = first_existing(FR_911_CANDIDATES)
    if src is None:
        raise FileNotFoundError("❌ 911 verisi bulunamadı (zip veya klasör).")
    log(f"📥 911 kaynağı yükleniyor: {src}")

    df = pd.read_csv(src, low_memory=False, dtype={"GEOID":"string"})
    # Zaman kolonu tespiti
    ts_col = next((c for c in ["received_time","received_datetime","datetime","timestamp",
                               "call_received_datetime","date"] if c in df.columns), None)
    if ts_col is None:
        raise ValueError("911 verisinde datetime/içeren bir zaman kolonu bulunamadı.")

    # GEOID yoksa lat/lon → GEOID
    if "GEOID" not in df.columns or df["GEOID"].isna().all():
        log("ℹ️ 911 verisinde GEOID yok; lat/lon → GEOID hesaplanacak.")
        df = ensure_geoid_from_latlon(df, CANDS["CENSUS"])

    # date üret
    df["date"] = to_date(df[ts_col])
    df = df.dropna(subset=["GEOID","date"]).copy()

    # Günlük sayım (GEOID × date)
    day = (df.groupby(["GEOID","date"], as_index=False)
             .size()
             .rename(columns={"size":"n_911_day"}))

    # Rolling (sızıntısız: shift(1))
    day = day.sort_values(["GEOID","date"]).reset_index(drop=True)

    day["n_911_last1d"] = (
        day.groupby("GEOID")["n_911_day"]
           .transform(lambda s: s.shift(1).fillna(0))
    ).astype("float32")

    def roll_sum(s: pd.Series, W: int) -> pd.Series:
        return s.shift(1).rolling(W, min_periods=1).sum()

    day["n_911_last3d"] = (
        day.groupby("GEOID")["n_911_day"].transform(lambda s: roll_sum(s, 3)).fillna(0)
    ).astype("float32")

    day["n_911_last7d"] = (
        day.groupby("GEOID")["n_911_day"].transform(lambda s: roll_sum(s, 7)).fillna(0)
    ).astype("float32")

    return day

# =========================
# MAIN
# =========================
def main():
    global CANDS  # ensure_geoid_from_latlon içinde kullanılıyor

    # 0) ZIP varsa çıkar
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) Aday yollar
    CANDS = build_candidates()

    # 2) 911 günlük özet
    fr911_daily = read_911_daily(CANDS["FR_911"], CANDS["CENSUS"])
    log(f"📊 911 günlük özet: {fr911_daily.shape[0]:,} satır × {fr911_daily.shape[1]} sütun")

    # 3) sf_crime.csv (girdi)
    crime_path = first_existing(CANDS["CRIME"])
    if crime_path is None:
        raise FileNotFoundError("❌ sf_crime.csv bulunamadı.")
    crime = pd.read_csv(crime_path, low_memory=False, dtype={"GEOID":"string"})
    log(f"📥 sf_crime.csv: {crime_path} — satır: {len(crime):,}")

    # 4) GEOID normalize / gerekirse lat-lon → GEOID
    try:
        gdf_blocks, tlen = _load_blocks(CANDS["CENSUS"])
    except Exception:
        tlen = 11
    if "GEOID" in crime.columns and crime["GEOID"].notna().any():
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)
    else:
        log("ℹ️ sf_crime: GEOID yok; lat/lon → GEOID hesaplanacak.")
        crime = ensure_geoid_from_latlon(crime, CANDS["CENSUS"])
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)

    # 5) DATE kolonu (olay saati ÖNEMSİZ; datetime → date)
    if "date" in crime.columns:
        crime["date"] = to_date(crime["date"])
    else:
        dt_col = next((c for c in ["datetime","event_datetime","occurred_at","timestamp"] if c in crime.columns), None)
        if dt_col is None:
            raise ValueError("❌ sf_crime.csv içinde 'date' veya 'datetime' benzeri bir kolon yok.")
        crime["date"] = to_date(crime[dt_col])

    # 6) EVENTS_DAILY: (GEOID, date) ile 911 join
    keys = ["GEOID","date"]
    events_daily = crime.merge(fr911_daily, on=keys, how="left")
    for c in ["n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]:
        if c in events_daily.columns:
            events_daily[c] = pd.to_numeric(events_daily[c], errors="coerce").fillna(0)

    # 7) GRID_DAILY: olayları GEOID×date topla + 911 özetlerini eşleştir
    #    - crime_count_day: o gün GEOID içindeki olay sayısı
    #    - Y_day: (crime_count_day > 0) ikili etiket (işine yarıyorsa)
    agg_crime = (events_daily
                 .groupby(keys, as_index=False)
                 .size()
                 .rename(columns={"size":"crime_count_day"}))
    grid_daily = agg_crime.merge(fr911_daily, on=keys, how="left")
    grid_daily["crime_count_day"] = pd.to_numeric(grid_daily["crime_count_day"], errors="coerce").fillna(0).astype(int)
    grid_daily["Y_day"] = (grid_daily["crime_count_day"] > 0).astype("int8")

    for c in ["n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]:
        if c in grid_daily.columns:
            grid_daily[c] = pd.to_numeric(grid_daily[c], errors="coerce").fillna(0)

    # 8) Yaz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_events = OUTPUT_DIR / OUTPUT_EVENTS_DAILY
    out_grid   = OUTPUT_DIR / OUTPUT_GRID_DAILY

    safe_save_csv(events_daily, str(out_events))
    safe_save_csv(grid_daily,   str(out_grid))

    # 9) Kısa önizleme
    try:
        log("—— fr_crime_events_daily.csv — örnek —")
        cols = ["GEOID","date","n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]
        log(events_daily[[c for c in cols if c in events_daily.columns]].head(8).to_string(index=False))
    except Exception:
        pass
    try:
        log("—— fr_crime_grid_daily.csv — örnek —")
        cols = ["GEOID","date","crime_count_day","Y_day","n_911_day","n_911_last1d","n_911_last3d","n_911_last7d"]
        log(grid_daily[[c for c in cols if c in grid_daily.columns]].head(8).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
