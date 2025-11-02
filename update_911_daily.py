#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911_daily.py — 911 günlük özet (GEOID × date)
from __future__ import annotations
import os, re, zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import geopandas as gpd

# -------------------------
# LOG & IO HELPERS
# -------------------------
def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)
def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"💾 Kaydedildi: {path} (satır={len(df):,})")
def to_date(s): return pd.to_datetime(s, errors="coerce").dt.date
def first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

# -------------------------
# CONFIG
# -------------------------
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

OUTPUT_DIR   = Path(os.getenv("FR_OUTPUT_DIR", str(ARTIFACT_DIR)))
OUT_FILENAME = os.getenv("FR_911_DAILY_FILE", "fr_911_daily.csv")

# TZ (tarih üretimi için gerekirse raporlama; ham veriler günlük date içeriyorsa UTC kalır)
LOCAL_TZ     = os.getenv("FR_DAILY_TZ", "UTC")

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
    }

# -------------------------
# ZIP
# -------------------------
def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve(); target = target.resolve()
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
                raise RuntimeError(f"Zip path outside target dir: {m.filename}")
            if m.is_dir():
                out_path.mkdir(parents=True, exist_ok=True); continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, 'r') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

# -------------------------
# GEO
# -------------------------
def _load_blocks(cand) -> tuple[gpd.GeoDataFrame, int]:
    census_path = first_existing(cand)
    if census_path is None:
        raise FileNotFoundError("❌ GEOID poligon dosyası yok: sf_census_blocks_with_population.geojson")
    gdf = gpd.read_file(census_path)
    if "GEOID" not in gdf.columns:
        alt = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
        if not alt: raise ValueError("GeoJSON içinde GEOID benzeri sütun yok.")
        gdf = gdf.rename(columns={alt[0]: "GEOID"})
    tlen = gdf["GEOID"].astype(str).str.len().mode().iat[0]
    gdf["GEOID"] = normalize_geoid(gdf["GEOID"], int(tlen))
    if gdf.crs is None: gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(4326)
    return gdf, int(tlen)

def ensure_geoid_from_latlon(df: pd.DataFrame, CENSUS_GEOJSON_CANDIDATES) -> pd.DataFrame:
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df
    lat_col = next((c for c in ["latitude","lat","y"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude","lon","x"] if c in df.columns), None)
    if not lat_col or not lon_col:
        raise ValueError("❌ 911 verisinde GEOID yok ve lat/lon bulunamadı.")
    gdf_blocks, tlen = _load_blocks(CENSUS_GEOJSON_CANDIDATES)
    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col]).copy()
    pts = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, gdf_blocks[["GEOID","geometry"]], how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns=["geometry","index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    return out

# -------------------------
# 911 → GÜNLÜK
# -------------------------
HR_PAT = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
def _fmt_hr_range(hr):
    m = HR_PAT.match(str(hr)); 
    if not m: return None
    a = int(m.group(1)) % 24; b = int(m.group(2)); b = b if b > a else min(a+3, 24)
    return f"{a:02d}-{b:02d}"

def read_911(FR_911_CANDIDATES, CENSUS_GEOJSON_CANDIDATES) -> pd.DataFrame:
    src = first_existing(FR_911_CANDIDATES)
    if src is None:
        raise FileNotFoundError("❌ 911 veri/özet bulunamadı.")
    log(f"📥 911 verisi: {src}")
    df = pd.read_csv(src, low_memory=False, dtype={"GEOID":"string"})

    # Tarih ve saat
    ts_col = next((c for c in ["received_time","received_datetime","date","datetime",
                               "timestamp","call_received_datetime"] if c in df.columns), None)
    if ts_col is None:
        # Summary olabilir: date + hour_range varsa kabul
        if {"date","hour_range"}.issubset(df.columns):
            df["date"] = to_date(df["date"])
        else:
            raise ValueError("911 ham veride zaman kolonu (veya date+hour_range) bulunamadı.")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["date"] = df[ts_col].dt.date
        df["event_hour"] = df[ts_col].dt.hour.fillna(0).astype(int) % 24
        start = (df["event_hour"] // 3) * 3
        df["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")

    # GEOID zorunlu; yoksa lat/lon'dan üret
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        try:
            _, tlen = _load_blocks(CENSUS_GEOJSON_CANDIDATES)
        except Exception:
            tlen = 11
        df["GEOID"] = normalize_geoid(df["GEOID"], tlen)
    else:
        df = ensure_geoid_from_latlon(df, CENSUS_GEOJSON_CANDIDATES)

    # Saatlik özet varsa günlük toplama
    if "hour_range" in df.columns:
        # hour_range sayacı kolonunu belirle
        cnt_col = next((c for c in ["911_request_count_hour_range","call_count","count","requests","n"] if c in df.columns), None)
        if cnt_col is None:
            # yoksa satır sayısından kur
            hr_agg = (df.groupby(["GEOID","date","hour_range"], dropna=False)
                        .size().reset_index(name="hr_cnt"))
        else:
            # varsa onu topla
            # hour_range formatını normalize et
            df["hour_range"] = df["hour_range"].apply(_fmt_hr_range)
            hr_agg = (df.groupby(["GEOID","date","hour_range"], dropna=False)[cnt_col]
                        .sum().reset_index(name="hr_cnt"))

        # günlük toplam
        day_agg = (hr_agg.groupby(["GEOID","date"], dropna=False)["hr_cnt"]
                         .sum().reset_index(name="911_request_count_daily(before_24_hours)"))
    else:
        # hour_range yoksa doğrudan günlük satır adedi
        day_agg = (df.groupby(["GEOID","date"], dropna=False)
                     .size().reset_index(name="911_request_count_daily(before_24_hours)"))

    # rolling (sızıntısız) — önce tarih sıralı unique
    daily = (day_agg.sort_values(["GEOID","date"]).reset_index(drop=True).copy())
    for W in (3,7):
        daily[f"911_geo_last{W}d"] = (
            daily.groupby("GEOID")["911_request_count_daily(before_24_hours)"]
                 .transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

    # kalender etiketleri
    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.weekday.astype("int8")
    daily["month"] = pd.to_datetime(daily["date"]).dt.month.astype("int8")
    _season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    daily["season"] = daily["month"].map(_season_map).astype("category")

    # iz bilgisi
    daily["fr_daily_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    daily["fr_daily_tz"] = LOCAL_TZ
    return daily

# -------------------------
# MAIN
# -------------------------
def main():
    # 0) ZIP varsa aç
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) Adaylar
    CANDS = build_candidates()

    # 2) 911 günlük üret
    fr911_daily = read_911(CANDS["FR_911"], CANDS["CENSUS"])
    log(f"📊 911 günlük boyut: {fr911_daily.shape[0]:,} × {fr911_daily.shape[1]}")

    # 3) Yaz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUT_FILENAME
    safe_save_csv(fr911_daily, str(out_path))

    # 4) Kısa kontrol
    try:
        log(fr911_daily.head(8).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
