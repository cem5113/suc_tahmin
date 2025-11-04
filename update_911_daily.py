#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_crime_daily.py  —  Event-level → (1) events_daily  (2) GEOID×date grid_daily

ENV:
- CRIME_DATA_DIR : kök (default: crime_prediction_data)
- FR_EVENTS_PATH : olay bazlı giriş (default: sf_crime.csv)
- FR_OUT_EVENTS  : olay-bazlı günlük çıktı (default: fr_crime_events_daily.csv)
- FR_OUT_GRID    : GEOID×date grid çıktı (default: fr_crime_grid_daily.csv)
- FR_MIN_YEARS   : min yıl (default: 5)
- FR_DATE_COL    : zaman sütunu ipucu (default: incident_datetime)
- FR_GEOID_COL   : geoid sütunu (default: GEOID)
- FR_ID_COL      : olay id sütunu (default: id)
"""

from __future__ import annotations
import os
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import shutil
import warnings

warnings.simplefilter("ignore", FutureWarning)
pd.options.mode.copy_on_write = True

# ----------------------- ENV / PATHS -----------------------
BASE_DIR    = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).expanduser().resolve()
EVENTS_PATH = Path(os.getenv("FR_EVENTS_PATH", "sf_crime.csv"))
OUT_EVENTS  = Path(os.getenv("FR_OUT_EVENTS",  "fr_crime_events_daily.csv"))
OUT_GRID    = Path(os.getenv("FR_OUT_GRID",    "fr_crime_grid_daily.csv"))

DATE_COL  = os.getenv("FR_DATE_COL", "incident_datetime")
GEOID_COL = os.getenv("FR_GEOID_COL", "GEOID")
ID_COL    = os.getenv("FR_ID_COL", "id")

MIN_YEARS = int(os.getenv("FR_MIN_YEARS", "5"))
MIN_DAYS  = max(1, MIN_YEARS * 365)

# ----------------------- Helpers -----------------------
def _abs(p: Path) -> Path:
    p = p.expanduser()
    return (p if p.is_absolute() else (BASE_DIR / p)).resolve()

def _read_table(p: Path) -> pd.DataFrame:
    p = _abs(p)
    if not p.exists():
        print(f"❌ Bulunamadı: {p}")
        return pd.DataFrame()
    try:
        suf = "".join(p.suffixes).lower()
        if suf.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif suf.endswith(".csv.gz"):
            df = pd.read_csv(p, low_memory=False, compression="gzip")
        else:
            df = pd.read_csv(p, low_memory=False)
        print(f"📖 Okundu: {p}  ({len(df):,}×{df.shape[1]})")
        return df
    except Exception as e:
        print(f"⚠️ Okunamadı: {p} → {e}")
        return pd.DataFrame()

def _safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Kaydedildi: {p}  ({len(df):,}×{df.shape[1]})")

def _norm_geoid(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .apply(lambda x: x.zfill(11) if x else "")
    )

def _detect_dt_col(df: pd.DataFrame, hint: str = "incident_datetime") -> str | None:
    cand = [
        hint, "datetime", "date_time", "occurred_at", "occurred_datetime",
        "event_datetime", "time", "date"
    ]
    for c in cand:
        if c in df.columns:
            return c
    # birleşik tarih+saat
    if "incident_date" in df.columns and "incident_time" in df.columns:
        return "incident_date+incident_time"
    return None

def _ensure_date(df: pd.DataFrame, dt_col_hint: str) -> pd.Series:
    use = _detect_dt_col(df, dt_col_hint)
    if use is None:
        # son çare: sadece 'date' varsa parse et
        if "date" in df.columns:
            return pd.to_datetime(df["date"], errors="coerce").dt.date
        raise ValueError("Tarih/saat sütunu bulunamadı (FR_DATE_COL ile belirtin).")
    if use == "incident_date+incident_time":
        dt = pd.to_datetime(
            df["incident_date"].astype(str).str.strip() + " " +
            df["incident_time"].astype(str).str.strip(),
            errors="coerce", utc=True
        )
    else:
        dt = pd.to_datetime(df[use], errors="coerce", utc=True)
    return dt.dt.date

# ----------------------- Main -----------------------
def main() -> int:
    print("📂 CWD:", Path.cwd())
    print("🔧 BASE_DIR       :", BASE_DIR)
    print("🔧 FR_EVENTS_PATH :", _abs(EVENTS_PATH))
    print("🔧 FR_OUT_EVENTS  :", _abs(OUT_EVENTS))
    print("🔧 FR_OUT_GRID    :", _abs(OUT_GRID))
    print("🔧 FR_MIN_YEARS   :", MIN_YEARS)
    print("🔧 FR_DATE_COL    :", DATE_COL)
    print("🔧 FR_GEOID_COL   :", GEOID_COL)
    print("🔧 FR_ID_COL      :", ID_COL)

    ev = _read_table(EVENTS_PATH)
    if ev.empty:
        print("❌ Olay verisi boş. Çıkılıyor.")
        return 1

    # GEOID normalize (varsa / yoksa otomatik bul)
    if GEOID_COL in ev.columns:
        ev[GEOID_COL] = _norm_geoid(ev[GEOID_COL])
    else:
        alt = next((c for c in ev.columns if "geoid" in c.lower()), None)
        if alt:
            print(f"🔎 GEOID otomatik bulundu: {alt}")
            ev[GEOID_COL] = _norm_geoid(ev[alt])
        else:
            print("⚠️ GEOID sütunu yok. Grid için GEOID listesi çıkmayabilir.")

    # DATE üret
    try:
        ev_date = _ensure_date(ev, DATE_COL)
    except Exception as e:
        print(f"❌ Tarih oluşturulamadı: {e}")
        return 2

    df = ev.copy()
    df["date"] = ev_date

    # id yoksa yarat
    if ID_COL not in df.columns:
        print(f"ℹ️ '{ID_COL}' yok → sentetik id üretilecek.")
        df.insert(0, ID_COL, pd.RangeIndex(0, len(df)).map(lambda i: f"synt_{i}"))

    # ---------- (1) EVENTS_DAILY ----------
    events_daily = df.copy()
    events_daily["Y_label_event"] = 1
    _safe_save_csv(events_daily, OUT_EVENTS)

    # ---------- (2) GRID (GEOID×date) ----------
    # GEOID listesi
    geoids: list[str] = []
    if GEOID_COL in events_daily.columns and events_daily[GEOID_COL].notna().any():
        geoids = sorted(events_daily.loc[events_daily[GEOID_COL] != "", GEOID_COL].unique().tolist())
    else:
        # yedek kaynaklardan dene
        for c in [
            "crime_prediction_data/sf_crime_grid_full_labeled.csv",
            "crime_prediction_data/sf_crime_grid_full_labeled.parquet",
            "sf_crime_grid_full_labeled.csv",
            "sf_crime_grid_full_labeled.parquet",
            "crime_prediction_data/sf_crime_y.csv",
            "sf_crime_y.csv",
        ]:
            src = _abs(Path(c))
            if not src.exists():
                continue
            try:
                tmp = pd.read_parquet(src) if str(src).lower().endswith(".parquet") else pd.read_csv(src, low_memory=False)
                if "GEOID" in tmp.columns:
                    geoids = sorted(tmp["GEOID"].astype(str).str.zfill(11).unique().tolist())
                    print(f"🔎 GEOID listesi {src} kaynağından alındı ({len(geoids)}).")
                    break
            except Exception:
                continue

    if not geoids:
        print("⚠️ GEOID listesi yok → grid üretimi atlanıyor.")
        return 0

    # tarih aralığı (min 5 yıl)
    dmin = events_daily["date"].dropna().min()
    dmax = events_daily["date"].dropna().max()
    if pd.isna(dmin) or pd.isna(dmax):
        print("⚠️ Kullanılabilir tarih bulunamadı → grid atlandı.")
        return 0

    if (dmax - dmin).days + 1 < MIN_DAYS:
        desired_start = dmax - timedelta(days=MIN_DAYS - 1)
    else:
        desired_start = dmin
    all_days = pd.date_range(start=desired_start, end=dmax, freq="D").date
    print(f"📅 Tarih aralığı: {desired_start} → {dmax} ({len(all_days)} gün)")

    # tam grid
    grid = pd.MultiIndex.from_product([geoids, all_days], names=[GEOID_COL, "date"]).to_frame(index=False)

    # günlük olay sayısı
    agg = (events_daily
           .dropna(subset=["date"])
           .groupby([GEOID_COL, "date"], as_index=False)
           .size()
           .rename(columns={"size": "events_count"}))

    grid = grid.merge(agg, on=[GEOID_COL, "date"], how="left")
    grid["events_count"] = pd.to_numeric(grid["events_count"], errors="coerce").fillna(0).astype("int32")
    grid["Y_label"] = (grid["events_count"] > 0).astype("int8")

    _safe_save_csv(grid, OUT_GRID)

    # opsiyonel mirror (CRIME_DATA_DIR içine)
    try:
        mirror = _abs(Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data")))
        mirror.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_abs(OUT_EVENTS), mirror / _abs(OUT_EVENTS).name)
        shutil.copy2(_abs(OUT_GRID),   mirror / _abs(OUT_GRID).name)
        print(f"📦 Mirror kopya: {mirror}")
    except Exception as e:
        print(f"ℹ️ Mirror atlandı: {e}")

    # özet
    print("\n📊 Özet:")
    print(f"  Events_daily: {len(events_daily):,} satır — id korunuyor: {ID_COL in events_daily.columns}")
    print(f"  Grid: {len(grid):,} satır ({len(geoids):,} GEOID × {len(all_days):,} gün)")
    try:
        vc = grid["Y_label"].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"  Y_label dağılımı (%): {vc}")
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"⚠️ Hata: {exc}")
        raise SystemExit(1)
