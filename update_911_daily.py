#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911_daily.py — sf_crime.csv → (1) fr_crime_events_daily.csv, (2) fr_crime_grid_daily.csv
# Bu sade sürüm, *yalnızca* RELEASE'ten indirilen sf_911_last_5_year.csv dosyasını okur.
# - ENV override: FR_911=/path/to/sf_911_last_5_year.csv python update_911_daily.py

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import pandas as pd

# ========================== Utils ==========================
def log(msg: str): print(msg, flush=True)
def ensure_parent(path: str): Path(path).parent.mkdir(parents=True, exist_ok=True)

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
    return pd.to_datetime(
        s,
        errors="coerce",
        utc=False,
        infer_datetime_format=True
    ).dt.date

def normalize_geoid(s: pd.Series, n: int = 11) -> pd.Series:
    s = s.astype("string")
    # rakamları çek → ilk n hane → soluna 0 doldur
    s = s.str.extract(r"(\d+)", expand=False)
    return s.str[:n].str.zfill(n)

def first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

# ========================== ENV / I/O ==========================
CRIME_DATA_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()

# Girdi/Çıktı dosya adları (değiştirilebilir)
INPUT_CRIME_FILENAME = os.getenv("FR_CRIME_FILE", "sf_crime.csv")               # olay-bazlı suç verisi
OUTPUT_EVENTS_DAILY  = os.getenv("FR_EVENTS_DAILY_OUT", "fr_crime_events_daily.csv")
OUTPUT_GRID_DAILY    = os.getenv("FR_GRID_DAILY_OUT",   "fr_crime_grid_daily.csv")

# 🔧 ÖNEMLİ: Tüm çıktıları artifact’a girmesi için DOĞRUDAN CRIME_DATA_DIR altına yaz.
OUTPUT_DIR           = CRIME_DATA_DIR

# 911 kaynağı (RELEASE dosyası)
ENV_911 = os.getenv("FR_911", "").strip()
CANDIDATES_911 = [
    Path(ENV_911) if ENV_911 else None,
    CRIME_DATA_DIR / "sf_911_last_5_year.csv",
    Path("./sf_911_last_5_year.csv"),
]

# ========================== 911 Oku & Günlük Özelle ==========================
def read_911_daily() -> pd.DataFrame:
    src = first_existing(CANDIDATES_911)
    if src is None:
        raise FileNotFoundError(
            "❌ 911 kaynağı bulunamadı. Beklenen dosya: sf_911_last_5_year.csv\n"
            "• ENV ile geçersiz kıl: FR_911=/path/to/sf_911_last_5_year.csv"
        )
    log(f"📥 911 kaynağı: {src}")

    df = pd.read_csv(src, low_memory=False, dtype={"GEOID": "string"})

    # Zaman kolonu autodetect
    ts_col = next(
        (c for c in [
            "received_time", "received_datetime", "datetime", "timestamp",
            "call_received_datetime", "date"
        ] if c in df.columns),
        None
    )
    if ts_col is None:
        raise ValueError("❌ 911 verisinde zaman kolonu bulunamadı (örn. received_datetime, datetime, date).")

    # GEOID kontrol — zorunlu
    if "GEOID" not in df.columns or df["GEOID"].isna().all():
        raise ValueError("❌ 911 verisinde GEOID yok. Bu sade sürüm lat/lon→GEOID eşlemesi yapmaz.")

    df["GEOID"] = normalize_geoid(df["GEOID"], 11)
    df["date"] = to_date(df[ts_col])

    df = df.dropna(subset=["GEOID", "date"]).copy()

    # Günlük sayımlar + kaydırmalı pencereler (sızıntı yok: shift(1))
    day = (
        df.groupby(["GEOID", "date"], as_index=False)
          .size()
          .rename(columns={"size": "n_911_day"})
          .sort_values(["GEOID", "date"])
          .reset_index(drop=True)
    )

    def roll_sum(s: pd.Series, W: int) -> pd.Series:
        return s.shift(1).rolling(W, min_periods=1).sum()

    day["n_911_last1d"] = day.groupby("GEOID")["n_911_day"].transform(lambda s: s.shift(1).fillna(0)).astype("float32")
    day["n_911_last3d"] = day.groupby("GEOID")["n_911_day"].transform(lambda s: roll_sum(s, 3)).fillna(0).astype("float32")
    day["n_911_last7d"] = day.groupby("GEOID")["n_911_day"].transform(lambda s: roll_sum(s, 7)).fillna(0).astype("float32")

    return day

# ========================== Suç Verisini Oku ==========================
def read_crime_events() -> pd.DataFrame:
    crime_path = first_existing([
        CRIME_DATA_DIR / INPUT_CRIME_FILENAME,
        Path(INPUT_CRIME_FILENAME),
    ])
    if crime_path is None:
        raise FileNotFoundError(
            f"❌ Suç verisi bulunamadı: {INPUT_CRIME_FILENAME} "
            f"(CRIME_DATA_DIR={CRIME_DATA_DIR})"
        )
    crime = pd.read_csv(crime_path, low_memory=False, dtype={"GEOID": "string"})
    log(f"📥 sf_crime.csv: {crime_path} — satır: {len(crime):,}")

    # GEOID zorunlu (bu sade sürüm GEOID üretmez)
    if "GEOID" not in crime.columns or crime["GEOID"].isna().all():
        raise ValueError("❌ sf_crime.csv içinde GEOID kolonu yok ya da tamamen boş.")

    crime["GEOID"] = normalize_geoid(crime["GEOID"], 11)

    # Tarih üretimi
    if "date" in crime.columns:
        crime["date"] = to_date(crime["date"])
    else:
        dt_col = next(
            (c for c in ["datetime", "event_datetime", "occurred_at", "timestamp"] if c in crime.columns),
            None
        )
        if dt_col is None:
            raise ValueError("❌ sf_crime.csv içinde 'date' veya 'datetime' benzeri bir kolon yok.")
        crime["date"] = to_date(crime[dt_col])

    crime = crime.dropna(subset=["GEOID", "date"]).copy()
    return crime

# ========================== Main ==========================
def main():
    log(f"📂 CRIME_DATA_DIR = {CRIME_DATA_DIR}")
    log(f"📂 OUTPUT_DIR     = {OUTPUT_DIR}")

    # 911 günlük özet
    fr911_daily = read_911_daily()
    log(f"📊 911 günlük özet: {fr911_daily.shape[0]:,} satır × {fr911_daily.shape[1]} sütun")

    # Suç olayları
    crime = read_crime_events()

    # —— EVENTS (olay-bazlı + 911 günlük özellikleri)
    keys = ["GEOID", "date"]
    events_daily = crime.merge(fr911_daily, on=keys, how="left")
    for c in ["n_911_day", "n_911_last1d", "n_911_last3d", "n_911_last7d"]:
        if c in events_daily.columns:
            events_daily[c] = pd.to_numeric(events_daily[c], errors="coerce").fillna(0)

    # —— GRID (GEOID×date günlük suç sayımı + 911)
    agg_crime = (
        events_daily.groupby(keys, as_index=False)
        .size()
        .rename(columns={"size": "crime_count_day"})
    )
    grid_daily = agg_crime.merge(fr911_daily, on=keys, how="left")
    grid_daily["crime_count_day"] = (
        pd.to_numeric(grid_daily["crime_count_day"], errors="coerce").fillna(0).astype(int)
    )
    grid_daily["Y_day"] = (grid_daily["crime_count_day"] > 0).astype("int8")
    for c in ["n_911_day", "n_911_last1d", "n_911_last3d", "n_911_last7d"]:
        if c in grid_daily.columns:
            grid_daily[c] = pd.to_numeric(grid_daily[c], errors="coerce").fillna(0)

    # —— Yaz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_save_csv(events_daily, str(OUTPUT_DIR / OUTPUT_EVENTS_DAILY))
    safe_save_csv(grid_daily,   str(OUTPUT_DIR / OUTPUT_GRID_DAILY))

    # —— Önizleme
    try:
        log("—— fr_crime_events_daily.csv — örnek —")
        cols = ["GEOID", "date", "n_911_day", "n_911_last1d", "n_911_last3d", "n_911_last7d"]
        log(events_daily[[c for c in cols if c in events_daily.columns]].head(8).to_string(index=False))
    except Exception:
        pass
    try:
        log("—— fr_crime_grid_daily.csv — örnek —")
        cols = ["GEOID", "date", "crime_count_day", "Y_day", "n_911_day", "n_911_last1d", "n_911_last3d", "n_911_last7d"]
        log(grid_daily[[c for c in cols if c in grid_daily.columns]].head(8).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
