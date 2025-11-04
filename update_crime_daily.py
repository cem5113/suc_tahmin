#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_crime_daily.py
Olay bazlı kaynak (FR_EVENTS_PATH) -> (1) olay-bazlı günlük çıktı (events_daily)
                              -> (2) GEOID x date full-grid günlük çıktı (grid_daily)

Özellikler:
- Olay 'id' korunur.
- Datetime'ten yalnızca date alınır (yerel/UTC farkı varsa incident_datetime içinde).
- En az 5 yıl (5*365 gün) tarih penceresi sağlanır; eğer olay verisi daha genişse ona göre genişler.
- Grid: tüm GEOID'ler × tüm tarihler (date_range) — events_count ve Y_label (0/1).
- ENV değişkenleri ile özelleştirilebilir.

ENV:
- FR_EVENTS_PATH   : olay bazlı giriş CSV (varsayılan: sf_crime.csv)
- FR_OUT_EVENTS    : olay-bazlı günlük çıktı (varsayılan: fr_crime_events_daily.csv)
- FR_OUT_GRID      : GEOID x date grid çıktı (varsayılan: fr_crime_grid_daily.csv)
- FR_MIN_YEARS     : en az kaç yıl (varsayılan: 5)
- FR_DATE_COL      : olay zaman sütunu (varsayılan: incident_datetime) -> otomatik tespit denemesi
- FR_GEOID_COL     : geoid sütunu (varsayılan: GEOID)
- FR_ID_COL        : olay id sütunu (varsayılan: id)
- FR_AUTOFIND_LABEL: (kullanılmıyor grid üretiminde; eski koddan ayrıldı)
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import shutil
import warnings

warnings.simplefilter("ignore", FutureWarning)

# -----------------------
# ENV / PATHS & SETTINGS
# -----------------------
BASE_DIR    = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).expanduser().resolve()
EVENTS_PATH = Path(os.getenv("FR_EVENTS_PATH", "sf_crime.csv"))
OUT_EVENTS  = Path(os.getenv("FR_OUT_EVENTS",  "fr_crime_events_daily.csv"))
OUT_GRID    = Path(os.getenv("FR_OUT_GRID",    "fr_crime_grid_daily.csv"))

DATE_COL    = os.getenv("FR_DATE_COL", "incident_datetime")  # fallback detection used
GEOID_COL   = os.getenv("FR_GEOID_COL", "GEOID")
ID_COL      = os.getenv("FR_ID_COL", "id")

MIN_YEARS   = int(os.getenv("FR_MIN_YEARS", "5"))
MIN_DAYS    = MIN_YEARS * 365

# -----------------------
# Helpers
# -----------------------
def _abs(p: Path) -> Path:
    p = p.expanduser()
    # Eğer verilen path göreliyse CRIME_DATA_DIR altına yaz/oku
    return (p if p.is_absolute() else (BASE_DIR / p)).resolve()

def safe_read_csv(p: Path) -> pd.DataFrame:
    p = _abs(p)
    if not p.exists():
        print(f"❌ Bulunamadı: {p}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, low_memory=False)
        print(f"📖 Okundu: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")
        return df
    except Exception as e:
        print(f"⚠️ Okunamadı: {p} → {e}")
        return pd.DataFrame()

def safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Kaydedildi: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

def normalize_geoid(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)")[0]
         .fillna("")
         .apply(lambda x: x.zfill(11) if x != "" else "")
    )

def detect_datetime_column(df: pd.DataFrame, hint: str = "incident_datetime") -> str | None:
    # tercih sırası: hint, 'datetime', 'date_time', 'occurred_at', 'incident_date'+'incident_time'
    candidates = [hint, "datetime", "date_time", "occurred_at", "occurred_datetime", "event_datetime",
                  "incident_date", "incident_date_time", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    # try combinations
    if "incident_date" in df.columns and "incident_time" in df.columns:
        return "incident_date+incident_time"
    return None

# -----------------------
# Main flow
# -----------------------
def main() -> int:
    print("📂 CWD:", Path.cwd())
    print("🔧 ENV → FR_EVENTS_PATH:", _abs(EVENTS_PATH))
    print("🔧 ENV → FR_OUT_EVENTS :", _abs(OUT_EVENTS))
    print("🔧 ENV → FR_OUT_GRID   :", _abs(OUT_GRID))
    print("🔧 ENV → FR_MIN_YEARS  :", MIN_YEARS)
    print("🔧 ENV → FR_DATE_COL   :", DATE_COL)
    print("🔧 ENV → FR_GEOID_COL  :", GEOID_COL)
    print("🔧 ENV → FR_ID_COL     :", ID_COL)

    events = safe_read_csv(EVENTS_PATH)
    if events.empty:
        print("❌ Olay verisi boş. İşlem sonlandırıldı.")
        return 1

    # normalize GEOID column (if varsa)
    if GEOID_COL in events.columns:
        events[GEOID_COL] = normalize_geoid(events[GEOID_COL])
    else:
        # attempt to find geoid-like column
        found = None
        for c in events.columns:
            if "geoid" in c.lower():
                found = c
                break
        if found:
            print(f"🔎 GEOID sütunu otomatik bulundu: {found} -> kullanılıyor.")
            events[GEOID_COL] = normalize_geoid(events[found])
        else:
            print("⚠️ GEOID sütunu bulunamadı. Grid üretimi için GEOID listesi gerekli.")
            # We still continue and will produce events_daily but grid will require GEOID list.
    # detect datetime
    dt_col = detect_datetime_column(events, DATE_COL)
    if dt_col is None:
        print("⚠️ Tarih/saat sütunu otomatik bulunamadı. Lütfen FR_DATE_COL ENV ile belirtin.")
        # still try to proceed if there is a 'date' column
        if "date" in events.columns:
            dt_col = "date"
        else:
            return 2

    # create canonical datetime and date columns
    df = events.copy()
    if dt_col == "incident_date+incident_time":
        df["incident_datetime"] = pd.to_datetime(
            df["incident_date"].astype(str).str.strip() + " " + df["incident_time"].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        # if the selected column is 'date' without time, parse it as date
        if dt_col in df.columns:
            df["incident_datetime"] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        else:
            df["incident_datetime"] = pd.NaT

    # fallback: try some other common names if parsing failed massively
    if df["incident_datetime"].isna().mean() > 0.5:
        for alt in ("datetime", "occurred_at", "timestamp"):
            if alt in df.columns:
                df["incident_datetime"] = pd.to_datetime(df[alt], errors="coerce", utc=True)
                break

    # convert to naive date (UTC date) — user asked "datetime baz alınsın" so we'll use the parsed datetime's date
    df["date"] = df["incident_datetime"].dt.date.fillna(pd.NaT)

    # preserve id column if exists; else generate a synthetic id to keep event uniqueness
    if ID_COL not in df.columns:
        print(f"⚠️ '{ID_COL}' sütunu bulunamadı — synthetic id oluşturulacak.")
        df.insert(0, ID_COL, pd.Series([f"synt_{i}" for i in range(len(df))]))

    # --- 1) EVENT-LEVEL DAILY OUTPUT: preserve all events, normalized date column, Y_label_event=1 ---
    events_daily = df.copy()
    # Y_label_event: any event row is considered 1 for that date & geoid (but we'll set explicit 1)
    events_daily["Y_label_event"] = 1

    safe_save_csv(events_daily, OUT_EVENTS)

    # --- 2) GRID: GEOID x DATE full range with events_count and Y_label ---
    # determine GEOID list: union of event GEOIDs (if none, try to load from existing label files)
    geoids = []
    if GEOID_COL in events_daily.columns and events_daily[GEOID_COL].notna().any():
        geoids = sorted(events_daily[events_daily[GEOID_COL] != "" ][GEOID_COL].unique().tolist())
    else:
        # try to find candidate label files (like sf_crime_grid_full_labeled)
        candidates = [
            Path("crime_prediction_data/sf_crime_grid_full_labeled.csv"),
            Path("crime_prediction_data/sf_crime_grid_full_labeled.parquet"),
            Path("sf_crime_grid_full_labeled.csv"),
            Path("sf_crime_grid_full_labeled.parquet"),
            Path("crime_prediction_data/sf_crime_y.csv"),
            Path("sf_crime_y.csv"),
        ]
        for c in candidates:
            c = _abs(c)
            if c.exists():
                try:
                    if c.suffix == ".parquet":
                        tmp = pd.read_parquet(c)
                    else:
                        tmp = pd.read_csv(c, low_memory=False)
                    if "GEOID" in tmp.columns:
                        geoids = sorted(tmp["GEOID"].astype(str).str.zfill(11).unique().tolist())
                        print(f"🔎 GEOID listesi {c} dosyasından alındı ({len(geoids)} geoid).")
                        break
                except Exception:
                    continue

    if not geoids:
        print("⚠️ GEOID listesi oluşturulamadı — grid üretimi atlandı.")
        return 0

    # date range: at least MIN_DAYS long, and covering events date span
    min_event_date = events_daily["date"].dropna().min()
    max_event_date = events_daily["date"].dropna().max()
    if pd.isna(min_event_date) or pd.isna(max_event_date):
        print("⚠️ Olaylarda kullanılabilir bir tarih bulunamadı — grid üretimi atlandı.")
        return 0

    # ensure min range covers MIN_DAYS ending at max_event_date
    desired_start = min_event_date
    # ensure at least MIN_DAYS up to max_event_date
    if (max_event_date - min_event_date).days + 1 < MIN_DAYS:
        desired_start = max_event_date - pd.Timedelta(days=(MIN_DAYS - 1))

    all_dates = pd.date_range(start=desired_start, end=max_event_date, freq="D").date
    print(f"📅 Tarih aralığı: {desired_start} → {max_event_date} ({len(all_dates)} gün)")

    # build full grid
    grid = pd.MultiIndex.from_product([geoids, all_dates], names=[GEOID_COL, "date"]).to_frame(index=False)
    # events_count aggregation per geoid-date
    agg = events_daily.groupby([GEOID_COL, "date"], as_index=False).size().rename(columns={"size": "events_count"})
    # merge
    grid = grid.merge(agg, on=[GEOID_COL, "date"], how="left")
    grid["events_count"] = grid["events_count"].fillna(0).astype(int)
    grid["Y_label"] = (grid["events_count"] > 0).astype(int)

    # optional: keep min event id if multiple (useful to trace one example) — not strictly required
    # we won't collapse event ids here; events are preserved in events_daily file.

    # save grid
    safe_save_csv(grid, OUT_GRID)

    # mirror copies (optional) — copy into MIRROR_DIR if exists
    MIRROR_DIR = Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data"))
    try:
        MIRROR_DIR = _abs(MIRROR_DIR)
        MIRROR_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_abs(OUT_EVENTS), MIRROR_DIR / _abs(OUT_EVENTS).name)
        shutil.copy2(_abs(OUT_GRID), MIRROR_DIR / _abs(OUT_GRID).name)
        print(f"📦 Mirror kopyalar: {MIRROR_DIR}")
    except Exception as e:
        print(f"ℹ️ Mirror kopya atlandı/başarısız: {e}")

    # quick summary
    print("\n📊 Özet:")
    print(f"  Olay (events_daily): {len(events_daily):,} satır — id korunuyor: {ID_COL in events_daily.columns}")
    print(f"  Grid (geo × date): {len(grid):,} satır ({len(geoids):,} GEOID × {len(all_dates):,} gün)")
    vc = grid["Y_label"].value_counts(normalize=True).mul(100).round(2)
    print("  Grid Y_label dağılımı (%):")
    for k, v in vc.items():
        print(f"    {k}: {v}%")

    return 0

if __name__ == "__main__":
    try:
        rc = main()
        raise SystemExit(rc if isinstance(rc, int) else 0)
    except Exception as exc:
        print(f"⚠️ Hata yakalandı: {exc}")
        raise SystemExit(1)
