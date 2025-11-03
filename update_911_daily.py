#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_911_daily.py — 911 günlük özetleri GEOID×tarih gridine ekler.

Girdi(ler):
- DAILY_IN : daily_crime_00.csv (en az: ['GEOID','date'])
- N911_IN  : sf_911_last_5_year.csv (en az: ['GEOID','datetime'] veya ['GEOID','date'])

Çıktı:
- DAILY_OUT: daily_crime_02.csv  (daily_crime_00 + n_911_today + 911_request_count_daily(before_24_hours))

ENV:
- CRIME_DATA_DIR  (default: "crime_prediction_data")
- DAILY_IN        (default: "daily_crime_00.csv")
- DAILY_OUT       (default: "daily_crime_02.csv")
- N911_IN         (default: "sf_911_last_5_year.csv")
- FR_DAILY_TZ     (default: "UTC")  # örn: "America/Los_Angeles"
- VERBOSE         (default: "1")    # "0" yaparsan daha sessiz

Path güvenliği:
- Absolute verilirse direkt kullanır.
- Relative verilirse CRIME_DATA_DIR altına koyar.
- base klasör ismi zaten baştaysa ikinci kez eklemez (triple-prefix'i önler).
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------- ENV ----------
CRIME_DATA_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").strip()
DAILY_IN  = os.getenv("DAILY_IN",  "daily_crime_00.csv").strip()
DAILY_OUT = os.getenv("DAILY_OUT", "daily_crime_02.csv").strip()
N911_IN   = os.getenv("N911_IN",   "sf_911_last_5_year.csv").strip()
LOCAL_TZ  = os.getenv("FR_DAILY_TZ", "UTC").strip()
VERBOSE   = os.getenv("VERBOSE", "1").strip() != "0"

BASE = Path(CRIME_DATA_DIR)

def resolve_path(p: str | Path, base: Path) -> Path:
    """
    Triple-prefix'i engeller:
    - absolute ise dokunma
    - relative ise base altına koy
    - zaten base ile başlıyorsa tekrar ekleme
    """
    p = Path(p)
    if p.is_absolute():
        return p
    # "crime_prediction_data/..." gibi
    try:
        # normalize string karşılaştırma (./ vs none)
        p_norm = str(p).lstrip("./")
        base_norm = str(base).lstrip("./")
        if p_norm.startswith(base_norm + "/") or p_norm == base_norm:
            return Path(p_norm)  # zaten base ile başlıyor
    except Exception:
        pass
    return base / p

def ls_hint(path: Path, depth: int = 1) -> str:
    try:
        if path.is_file():
            return f"(file exists) {path}"
        if path.is_dir():
            entries = list(path.iterdir())
            names = [e.name + ("/" if e.is_dir() else "") for e in entries[:50]]
            more = "" if len(entries) <= 50 else f" (+{len(entries)-50} more)"
            return f"dir: {path}\n  -> {', '.join(names)}{more}"
        # parent varsa onu listele
        parent = path.parent
        if parent.exists():
            entries = [e.name + ("/" if e.is_dir() else "") for e in parent.iterdir()][:50]
            return f"(missing) parent dir: {parent}\n  -> {', '.join(entries)}"
    except Exception as e:
        return f"(ls_hint error: {e})"
    return "(path not found and no parent)"

def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)

def safe_read_csv(p: Path, **read_csv_kwargs) -> pd.DataFrame:
    if not p.exists():
        hint = ls_hint(p)
        raise FileNotFoundError(f"❌ Girdi bulunamadı: {p}\n{hint}")
    try:
        df = pd.read_csv(p, **read_csv_kwargs)
        return df
    except Exception as e:
        raise RuntimeError(f"❌ CSV okunamadı: {p}\nHata: {e}")

def ensure_date_col(df: pd.DataFrame, date_col: str = "date", tz: Optional[str] = None) -> pd.DataFrame:
    """
    - Eğer 'date' yoksa:
        * 'datetime' varsa: datetime→tz→dt.date
        * Yoksa hata
    - 'date' varsa: pandas datetime'e çevirip sadece date kısmını alır
    """
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
        if out[date_col].isna().all():
            # tümü NaT olduysa tekrar dene (olası string formatlar)
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
        return out

    # date yok -> datetime dene
    dt_col = None
    for cand in ["datetime", "incident_datetime", "created_datetime"]:
        if cand in out.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError("⛔ 'date' kolonu yok ve datetime türevi kolon bulunamadı (örn: 'datetime').")

    dt = pd.to_datetime(out[dt_col], errors="coerce", utc=True)
    if tz and tz.upper() != "UTC":
        try:
            dt = dt.dt.tz_convert(tz)
        except Exception:
            # önce UTC-aware yap
            dt = pd.to_datetime(out[dt_col], errors="coerce", utc=True).dt.tz_convert(tz)
    out[date_col] = dt.dt.date
    return out

def main() -> int:
    # ---------- Paths (güvenli) ----------
    p_in   = resolve_path(DAILY_IN, BASE)
    p_911  = resolve_path(N911_IN, BASE)
    p_out  = resolve_path(DAILY_OUT, BASE)

    log("🚀 update_911_daily.py")
    log(f"🔧 CRIME_DATA_DIR: {BASE}")
    log(f"🔧 DAILY_IN      : {p_in}")
    log(f"🔧 N911_IN       : {p_911}")
    log(f"🔧 DAILY_OUT     : {p_out}")
    log(f"🔧 FR_DAILY_TZ   : {LOCAL_TZ}")

    # ---------- Load inputs ----------
    grid = safe_read_csv(p_in)
    log(f"📖 Okundu GRID: {p_in}  ({len(grid):,} satır, {grid.shape[1]} sütun)")

    n911 = safe_read_csv(p_911)
    log(f"📖 Okundu 911 : {p_911}  ({len(n911):,} satır, {n911.shape[1]} sütun)")

    # ---------- Normalizations ----------
    # GRID: date zorunlu, GEOID zorunlu
    if "GEOID" not in grid.columns:
        raise KeyError("⛔ GRID dosyasında 'GEOID' kolonu yok.")
    grid = ensure_date_col(grid, date_col="date")  # grid zaten date içeriyor olmalı
    if grid["date"].isna().any():
        raise ValueError("⛔ GRID 'date' üretilemedi (NaT oluştu).")

    # 911: GEOID ve date üret
    if "GEOID" not in n911.columns:
        raise KeyError("⛔ 911 dosyasında 'GEOID' kolonu yok.")
    n911 = ensure_date_col(n911, date_col="date", tz=LOCAL_TZ)
    if n911["date"].isna().any():
        # NaT'leri at
        n_before = len(n911)
        n911 = n911.dropna(subset=["date"])
        log(f"⚠️  911 'date' NaT olan {n_before - len(n911)} satır atıldı.")

    # ---------- Aggregate 911 per GEOID×date ----------
    grp = (
        n911.groupby(["GEOID", "date"], as_index=False)
            .size()
            .rename(columns={"size": "n_911_today"})
    )
    log(f"🧮 911 günlük özet: {len(grp):,} satır (GEOID×date)")

    # ---------- Merge to GRID ----------
    out = grid.merge(grp, on=["GEOID", "date"], how="left")
    out["n_911_today"] = out["n_911_today"].fillna(0).astype("int64")

    # ---------- Lag(1d): previous day 911 by GEOID ----------
    # grid tarafında tüm GEOID×tarih kombinasyonları olduğundan, lag(1) düzgün çalışır.
    out = out.sort_values(["GEOID", "date"])
    out["911_request_count_daily(before_24_hours)"] = (
        out.groupby("GEOID", sort=False)["n_911_today"].shift(1).fillna(0).astype("int64")
    )

    # ---------- Write ----------
    p_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p_out, index=False)
    y1 = int((out.get("Y_day", pd.Series([0])).astype("int64") == 1).sum()) if "Y_day" in out.columns else None

    log(f"💾 Yazıldı: {p_out}  ({len(out):,} satır, {out.shape[1]} sütun)")
    if "Y_day" in out.columns:
        ratio = 100.0 * y1 / len(out) if len(out) else 0
        log(f"📊 Y_day=1: {y1:,}  (%{ratio:.2f})")

    # kısa bir ön-izleme
    log("---- Örnek satırlar (ilk 5) ----")
    log(out.head().to_string())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
