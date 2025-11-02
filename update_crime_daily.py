#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_crime_day.py — Günlük özet (GEOID × tarih)
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# ========= Ayarlar (ENV ile değiştirilebilir) =========
IN_PATH   = Path(os.getenv("FR_DAILY_IN",  "fr_crime.csv"))        # update_crime_fr.py çıktısı
OUT_PATH  = Path(os.getenv("FR_DAILY_OUT", "fr_crime_daily.csv"))  # günlük çıktı
LOCAL_TZ  = os.getenv("FR_DAILY_TZ", "UTC")                         # örn: Europe/Paris, America/Los_Angeles

# Zaman kolonu adayları (ilk bulunan kullanılır)
DT_CANDS = ["dt","datetime","timestamp","occurred_at","event_time","t0","t"]

# Adet sayımı için aday kolon (varsa sum, yoksa satır sayısı)
COUNT_CANDS = ["crime_count","count","n"]

# Label kolonu (olay bazlı Y)
YCOL = os.getenv("FR_YCOL", "Y_label")


def _abs(p: Path) -> Path:
    return p.expanduser().resolve()

def _read_csv(p: Path) -> pd.DataFrame:
    p = _abs(p)
    if not p.exists():
        print(f"❌ Girdi bulunamadı: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    print(f"📖 Okundu: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")
    return df

def _detect_col(cands: list[str], cols: pd.Index) -> str | None:
    for c in cands:
        if c in cols:
            return c
    return None

def _ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns:
        df = df.copy()
        df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(11)
    else:
        raise SystemExit("❌ 'GEOID' kolonu zorunlu ve bulunamadı.")
    return df

def _to_local_date(s: pd.Series, tz: str) -> pd.Series:
    # Giriş UTC aware da olabilir, naive da olabilir: güvenli dönüştürme
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert(tz)
    except Exception:
        dt = pd.to_datetime(s, errors="coerce").dt.tz_localize("UTC").dt.tz_convert(tz)
    return dt.dt.date.astype("string")

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_geoid(df)

    dcol = _detect_col(DT_CANDS, df.columns)
    if dcol is None:
        raise SystemExit(f"❌ Zaman kolonu bulunamadı. Adaylar: {DT_CANDS}")

    ccol = _detect_col(COUNT_CANDS, df.columns)  # opsiyonel
    if YCOL not in df.columns:
        # Y yoksa günlük Y_day’ı olay var/yok (satır) üzerinden kuramayız, 0 kabul edelim.
        print(f"ℹ️ Uyarı: '{YCOL}' yok. Y_day hesaplanırken 0 kabul edilecek.")
        df = df.copy()
        df[YCOL] = 0

    df = df.copy()
    df["event_date"] = _to_local_date(df[dcol], LOCAL_TZ)

    grp_keys = ["GEOID", "event_date"]

    if ccol:
        daily = (df
                 .groupby(grp_keys, as_index=False)
                 .agg(daily_count=(ccol, "sum"),
                      Y_day=(YCOL, lambda s: int((s.fillna(0) > 0).any()))))
    else:
        daily = (df
                 .groupby(grp_keys, as_index=False)
                 .agg(daily_count=("GEOID", "size"),
                      Y_day=(YCOL, lambda s: int((s.fillna(0) > 0).any()))))

    # Tipler + iz bilgisi
    daily["daily_count"] = pd.to_numeric(daily["daily_count"], errors="coerce").fillna(0).astype("int32")
    daily["Y_day"] = pd.to_numeric(daily["Y_day"], errors="coerce").fillna(0).astype("int8")
    daily["fr_daily_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    daily["fr_daily_tz"] = LOCAL_TZ

    # Güvenlik: kopya kolon isimleri olmasın
    daily = daily.loc[:, ~daily.columns.duplicated()].copy()
    return daily

def _save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Yazıldı: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

def main() -> int:
    print("📂 CWD:", Path.cwd())
    print("🔧 FR_DAILY_IN :", _abs(IN_PATH))
    print("🔧 FR_DAILY_OUT:", _abs(OUT_PATH))
    print("🔧 FR_DAILY_TZ :", LOCAL_TZ)

    src = _read_csv(IN_PATH)
    if src.empty:
        return 0

    daily = build_daily(src)
    _save_csv(daily, OUT_PATH)

    # Kısa özet
    y1 = int((daily["Y_day"] == 1).sum())
    y0 = int((daily["Y_day"] == 0).sum())
    tot = len(daily)
    pct1 = round(100 * y1 / tot, 2) if tot else 0.0
    print(f"📊 Günlük satır: {tot:,} | Y_day=1: {y1:,} (%{pct1}) | Y_day=0: {y0:,}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
