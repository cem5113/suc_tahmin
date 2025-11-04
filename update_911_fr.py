#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_911_fr.py — 911 verisini gün-bazlı özetleyip rolling pencereler (1/3/7g) üretir.
İsteğe bağlı olarak fr_crime.csv ile tarih tabanlı birleştirir.

ÇIKTILAR (ENV ile değiştirilebilir):
- FR_911_DAILY_FILE   (default: fr_911_daily.csv)
- FR_911_ROLLUP_FILE  (default: fr_911_rollups.csv)
- FR_CRIME_01_FILE    (default: fr_crime_01.csv)  # sadece merge açıksa

ÖNEMLİ: Rolling pencereler sızıntıyı önlemek için "dün dahil" hesaplanır (shift(1)).
"""

from __future__ import annotations
import os, re, zipfile
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd

# =========================
# LOG & I/O HELPERS
# =========================
def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)
def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"💾 Kaydedildi: {path} (satır={len(df):,})")

def to_date(s): return pd.to_datetime(s, errors="coerce").dt.date
def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

def first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

# =========================
# CONFIG (ZIP → extract → read)
# =========================
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))
OUTPUT_DIR   = Path(os.getenv("FR_OUTPUT_DIR", str(ARTIFACT_DIR)))

FR_911_DAILY_FILE  = os.getenv("FR_911_DAILY_FILE",  "fr_911_daily.csv")
FR_911_ROLLUP_FILE = os.getenv("FR_911_ROLLUP_FILE", "fr_911_rollups.csv")

INPUT_CRIME_FILENAME    = os.getenv("FR_CRIME_FILE",     "fr_crime.csv")
OUTPUT_MERGED_FILENAME  = os.getenv("FR_CRIME_01_FILE",  "fr_crime_01.csv")
FR_MERGE_WITH_CRIME     = os.getenv("FR_MERGE_WITH_CRIME", "1") == "1"

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
# ZIP
# =========================
def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        return str(target.resolve()).startswith(str(directory.resolve()))
    except Exception:
        return False

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        log(f"ℹ️ Artifact ZIP yok: {zip_path} — klasörlerden denenecek.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in zf.infolist():
            out_path = dest_dir / m.filename
            if not _is_within_directory(dest_dir, out_path.parent):
                raise RuntimeError(f"Zip-Slip engellendi: {m.filename}")
            if m.is_dir():
                out_path.mkdir(parents=True, exist_ok=True); continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, 'r') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

# =========================
# GEO / GEOID
# =========================
def _load_blocks(cands) -> tuple[gpd.GeoDataFrame, int]:
    census_path = first_existing(cands)
    if census_path is None:
        raise FileNotFoundError("❌ GEOID poligon dosyası yok: sf_census_blocks_with_population.geojson")
    gdf = gpd.read_file(census_path)
    if "GEOID" not in gdf.columns:
        cand = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
        if not cand: raise ValueError("GeoJSON içinde GEOID benzeri sütun yok.")
        gdf = gdf.rename(columns={cand[0]: "GEOID"})
    tlen = gdf["GEOID"].astype(str).str.len().mode().iat[0]
    gdf["GEOID"] = normalize_geoid(gdf["GEOID"], int(tlen))
    if gdf.crs is None: gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(4326)
    return gdf, int(tlen)

# =========================
# 911 → Günlük + Rolling
# =========================
HR_PAT = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
def _fmt_hr_range(hr):
    m = HR_PAT.match(str(hr))
    if not m: return None
    a = int(m.group(1)) % 24
    b = int(m.group(2)); b = b if b > a else min(a + 3, 24)
    return f"{a:02d}-{b:02d}"

def _hr_key_from_range(hr):
    m = HR_PAT.match(str(hr))
    return int(m.group(1)) % 24 if m else None

def make_daily_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["GEOID","date","hour_range",
                                     "911_request_count_hour_range",
                                     "911_request_count_daily(before_24_hours)"])
    df = raw.copy()
    ts_col = next((c for c in ["received_time","received_datetime","date","datetime",
                               "timestamp","call_received_datetime"] if c in df.columns), None)
    if ts_col is None:
        raise ValueError("911 ham veride zaman kolonu bulunamadı.")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour.fillna(0).astype(int) % 24
    start = (df["event_hour"] // 3) * 3
    df["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")

    grp_hr  = (["GEOID"] if "GEOID" in df.columns else []) + ["date","hour_range"]
    grp_day = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]

    hr_agg  = df.groupby(grp_hr,  dropna=False).size().reset_index(name="911_request_count_hour_range")
    day_agg = df.groupby(grp_day, dropna=False).size().reset_index(name="911_request_count_daily(before_24_hours)")
    out = hr_agg.merge(day_agg, on=grp_day, how="left")

    # Takvim sütunları
    out["date"] = to_date(out["date"])
    out["hour_range"] = out["hour_range"].apply(_fmt_hr_range)
    out["hr_key"] = out["hour_range"].apply(_hr_key_from_range).astype("int16")
    out["day_of_week"] = pd.to_datetime(out["date"]).dt.weekday.astype("int8")
    out["month"] = pd.to_datetime(out["date"]).dt.month.astype("int8")
    _season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    out["season"] = out["month"].map(_season_map).astype("category")
    return out

def compute_rollups(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    GEOID×date için:
      - daily_cnt = o gün toplam 911
      - 911_last1d / last3d / last7d = önceki 1/3/7 günlerin toplamı (shift(1)) — sızıntısız
    Ayrıca GEOID×hr_key×date için:
      - hr_cnt ve 1/3/7g rolling'ler (shift(1))
    """
    df = daily_df.copy()
    # --- Günlük (GEOID×date)
    day = (df.groupby((["GEOID"] if "GEOID" in df.columns else []) + ["date"], dropna=False)
             ["911_request_count_daily(before_24_hours)"]
             .sum().reset_index(name="daily_cnt")
             .sort_values((["GEOID"] if "GEOID" in df.columns else []) + ["date"]))
    keys_day = ["GEOID"] if "GEOID" in day.columns else []
    for W in (1,3,7):
        day[f"911_last{W}d"] = (
            day.groupby(keys_day)["daily_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

    # --- Saat aralığı (GEOID×hr_key×date)
    hr = (df.groupby((["GEOID"] if "GEOID" in df.columns else []) + ["hr_key","date"], dropna=False)
            ["911_request_count_hour_range"].sum()
            .reset_index(name="hr_cnt")
            .sort_values((["GEOID"] if "GEOID" in df.columns else []) + ["hr_key","date"]))
    keys_hr = (["GEOID","hr_key"] if "GEOID" in hr.columns else ["hr_key"])
    for W in (1,3,7):
        hr[f"911_hr_last{W}d"] = (
            hr.groupby(keys_hr)["hr_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

    # GEOID×date seviyesine indirgeme (hr toplamları opsiyonel)
    hr_day = hr.groupby((["GEOID"] if "GEOID" in hr.columns else []) + ["date"], dropna=False)["hr_cnt"].sum().reset_index(name="hr_cnt_sum")
    roll = day.merge(hr_day, on=(["GEOID"] if "GEOID" in hr_day.columns else []) + ["date"], how="left")
    return roll, hr

# =========================
# CRIME + 911 MERGE (opsiyonel)
# =========================
def merge_with_crime(roll: pd.DataFrame, daily_df: pd.DataFrame, CANDS) -> pd.DataFrame:
    crime_path = first_existing(CANDS["CRIME"])
    if crime_path is None:
        raise FileNotFoundError("❌ fr_crime.csv bulunamadı (zip/klasör).")
    crime = pd.read_csv(crime_path, low_memory=False, dtype={"GEOID":"string"})
    log(f"📥 fr_crime.csv: {crime_path} — satır: {len(crime):,}")

    # GEOID normalize
    try:
        _, tlen = _load_blocks(CANDS["CENSUS"])
    except Exception:
        tlen = 11
    if "GEOID" in crime.columns and crime["GEOID"].notna().any():
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)

    # tarih
    if "date" not in crime.columns:
        # olay bazlı dosyada incident_datetime varsa tarihle
        dt_col = next((c for c in ["incident_datetime","datetime","occurred_at","timestamp","date"] if c in crime.columns), None)
        if dt_col:
            crime["date"] = to_date(crime[dt_col])
        else:
            raise ValueError("❌ fr_crime.csv içinde 'date' veya datetime benzeri kolonu yok.")

    # JOIN: GEOID + date (gün-bazlı)
    keys = ["GEOID","date"] if "GEOID" in crime.columns else ["date"]
    merged = crime.merge(roll, on=keys, how="left")

    # varsayılan sayı kolonu doldurma
    for c in ["daily_cnt","911_last1d","911_last3d","911_last7d","hr_cnt_sum"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    # örnek gösterim
    try:
        log(merged[["GEOID","date","daily_cnt","911_last1d","911_last3d","911_last7d"]].head(8).to_string(index=False))
    except Exception:
        pass
    return merged

# =========================
# MAIN
# =========================
def main():
    # ZIP'i aç (varsa)
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    CANDS = build_candidates()

    # 911 kaynağını bul
    src_911 = first_existing(CANDS["FR_911"])
    if src_911 is None:
        raise FileNotFoundError("❌ 911 kaynağı yok (zip/klasör).")
    log(f"📥 911 kaynak: {src_911}")

    # 911 oku
    df911_raw = pd.read_csv(src_911, low_memory=False)
    # GEOID normalizasyonu (varsa)
    try:
        _, tlen = _load_blocks(CANDS["CENSUS"])
        if "GEOID" in df911_raw.columns:
            df911_raw["GEOID"] = normalize_geoid(df911_raw["GEOID"], tlen)
    except Exception:
        pass

    # Günlük özet
    daily = make_daily_summary(df911_raw)
    safe_save_csv(daily, str(OUTPUT_DIR / FR_911_DAILY_FILE))

    # Rolling özetler (GEOID×date ve hr düzeyleri)
    roll, hr_roll = compute_rollups(daily)
    safe_save_csv(roll,    str(OUTPUT_DIR / FR_911_ROLLUP_FILE))

    # Opsiyonel: suç verisi ile günlük birleşim
    if FR_MERGE_WITH_CRIME:
        merged = merge_with_crime(roll, daily, CANDS)
        safe_save_csv(merged, str(OUTPUT_DIR / OUTPUT_MERGED_FILENAME))
        log("🔗 Birleşim tamam (GEOID+date).")
    else:
        log("ℹ️ FR_MERGE_WITH_CRIME=0 — birleşim atlandı.")

if __name__ == "__main__":
    main()
