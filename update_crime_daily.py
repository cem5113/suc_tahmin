#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_crime_day.py — Günlük özet (GEOID × tarih) + eksik günleri 0 ile doldurma

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, date
import pandas as pd

# ========= Ayarlar (ENV ile değiştirilebilir) =========
IN_PATH   = Path(os.getenv("FR_DAILY_IN",  "fr_crime.csv"))          # olay bazlı giriş
OUT_PATH  = Path(os.getenv("FR_DAILY_OUT", "fr_crime_daily.csv"))    # günlük çıktı
LOCAL_TZ  = os.getenv("FR_DAILY_TZ", "UTC")                           # örn: Europe/Paris

# İsteğe bağlı: tarih penceresi zorlaması (YYYY-MM-DD)
FORCE_START = os.getenv("FR_DAILY_START", "").strip()  # boş ise otomatik min
FORCE_END   = os.getenv("FR_DAILY_END", "").strip()    # boş ise otomatik max

# Zaman kolonu adayları (ilk bulunan kullanılır)
DT_CANDS = ["dt", "datetime", "timestamp", "occurred_at", "event_time", "t0", "t"]

# Adet sayımı için aday kolon (varsa sum, yoksa satır sayısı)
COUNT_CANDS = ["crime_count", "count", "n"]

# Label kolonu (olay bazlı)
YCOL = os.getenv("FR_YCOL", "Y_label")


# ========= Yardımcılar =========
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
    if "GEOID" not in df.columns:
        raise SystemExit("❌ 'GEOID' kolonu zorunlu ve bulunamadı.")
    out = df.copy()
    out["GEOID"] = (
        out["GEOID"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(11)
    )
    return out

def _to_local_date(s: pd.Series, tz: str) -> pd.Series:
    """
    s UTC-aware da olabilir; naive de olabilir. Güvenli şekilde yerel TZ'ye çevirip sadece tarihi döndürür.
    """
    # Önce UTC varsayımı ile dene
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert(tz)
    except Exception:
        # Eğer yukarıdaki başarısızsa: önce naive → UTC kabul et → tz convert
        dt = pd.to_datetime(s, errors="coerce").dt.tz_localize("UTC").dt.tz_convert(tz)
    return dt.dt.date.astype("string")

def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


# ========= Çekirdek işlev =========
def build_daily(df_src: pd.DataFrame) -> pd.DataFrame:
    # GEOID zorunlu ve normalize
    df = _ensure_geoid(df_src)

    # Zaman ve count kolonlarını bul
    dcol = _detect_col(DT_CANDS, df.columns)
    if dcol is None:
        raise SystemExit(f"❌ Zaman kolonu bulunamadı. Adaylar: {DT_CANDS}")

    ccol = _detect_col(COUNT_CANDS, df.columns)  # opsiyonel

    # Y yoksa 0 kabul (var/yok üretimi için)
    if YCOL not in df.columns:
        print(f"ℹ️ Uyarı: '{YCOL}' yok. Y_day hesaplanırken 0 kabul edilecek.")
        df = df.copy()
        df[YCOL] = 0

    # Yerel tarihe indir
    df = df.copy()
    df["event_date"] = _to_local_date(df[dcol], LOCAL_TZ)

    # Günlük agregasyon (yalnızca mevcut satırlar)
    grp_keys = ["GEOID", "event_date"]
    if ccol:
        daily = (
            df.groupby(grp_keys, as_index=False)
              .agg(
                  daily_count=(ccol, "sum"),
                  Y_day=(YCOL, lambda s: int((s.fillna(0) > 0).any())),
              )
        )
    else:
        daily = (
            df.groupby(grp_keys, as_index=False)
              .agg(
                  daily_count=("GEOID", "size"),
                  Y_day=(YCOL, lambda s: int((s.fillna(0) > 0).any())),
              )
        )

    # --------- Eksik günleri 0'la doldurmak için tam ızgara ---------
    # Tüm GEOID’ler
    all_geoids = daily["GEOID"].dropna().unique()

    # Tarih aralığı (otomatik min..max veya FORCE_* ile)
    existing_dates = pd.to_datetime(daily["event_date"], errors="coerce")
    auto_start = existing_dates.min().date() if not existing_dates.isna().all() else None
    auto_end   = existing_dates.max().date() if not existing_dates.isna().all() else None

    d_start = _parse_date(FORCE_START) or auto_start
    d_end   = _parse_date(FORCE_END)   or auto_end
    if d_start is None or d_end is None:
        raise SystemExit("❌ Tarih aralığı tespit edilemedi (veride hiç geçerli tarih yok).")

    all_dates = pd.date_range(start=d_start, end=d_end, freq="D").date.astype("string")

    # Tam ızgara
    full = (
        pd.MultiIndex.from_product([all_geoids, all_dates], names=["GEOID", "event_date"])
          .to_frame(index=False)
    )

    # Left join & boşları doldur
    daily_full = (
        full.merge(daily, on=["GEOID", "event_date"], how="left")
            .fillna({"daily_count": 0, "Y_day": 0})
    )

    # Tipler + meta
    daily_full["daily_count"] = pd.to_numeric(daily_full["daily_count"], errors="coerce").fillna(0).astype("int32")
    daily_full["Y_day"] = pd.to_numeric(daily_full["Y_day"], errors="coerce").fillna(0).astype("int8")
    daily_full["fr_daily_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    daily_full["fr_daily_tz"] = LOCAL_TZ

    # Güvenlik: kopya kolon isimleri olmasın
    daily_full = daily_full.loc[:, ~daily_full.columns.duplicated()].copy()

    return daily_full


# ========= Kaydet =========
def _save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Yazıldı: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")


# ========= CLI =========
def main() -> int:
    print("📂 CWD:", Path.cwd())
    print("🔧 FR_DAILY_IN :", _abs(IN_PATH))
    print("🔧 FR_DAILY_OUT:", _abs(OUT_PATH))
    print("🔧 FR_DAILY_TZ :", LOCAL_TZ)
    if FORCE_START or FORCE_END:
        print(f"🔧 FORCE window: start={FORCE_START or 'auto'} end={FORCE_END or 'auto'}")

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
