#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911_daily.py — Input: update_911_fr.py çıktısı (örn. fr_crime_01.csv)
#                       Output: daily_crime_01.csv (GEOID × date günlük 911 özet)

from __future__ import annotations
import os, re
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

# ========= IO =========
IN_PATH   = Path(os.getenv("FR_911_IN",  "artifact_unzipped/fr_crime_01.csv"))
OUT_PATH  = Path(os.getenv("FR_911_OUT", "daily_crime_01.csv"))  
LOCAL_TZ  = os.getenv("FR_DAILY_TZ", "UTC")  # gerekirse raporlama için

# Saat aralığı formatı (00-03 gibi)
HR_PAT = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")

def log(msg: str): print(msg, flush=True)
def _abs(p: Path) -> Path: return p.expanduser().resolve()

def safe_read_csv(p: Path) -> pd.DataFrame:
    p = _abs(p)
    if not p.exists():
        raise FileNotFoundError(f"❌ Girdi bulunamadı: {p}")
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")
    return df

def safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

def to_date_series(s: pd.Series) -> pd.Series:
    # 'date' kolonu string/date olabilir; yoksa datetime'dan türetiriz
    return pd.to_datetime(s, errors="coerce").dt.date

def _fmt_hr_range(hr: str | int | float) -> Optional[str]:
    m = HR_PAT.match(str(hr))
    if not m: return None
    a = int(m.group(1)) % 24
    b = int(m.group(2))
    b = b if b > a else min(a + 3, 24)
    return f"{a:02d}-{b:02d}"

def detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("datetime","dt","timestamp","occurred_at","received_datetime","call_received_datetime"):
        if c in df.columns: return c
    return None

def main() -> int:
    log("🚀 update_911_daily.py")
    log(f"🔧 FR_911_IN  : {_abs(IN_PATH)}")
    log(f"🔧 FR_911_OUT : {_abs(OUT_PATH)}")
    log(f"🔧 FR_DAILY_TZ: {LOCAL_TZ}")

    src = safe_read_csv(IN_PATH)

    # GEOID zorunlu
    if "GEOID" not in src.columns:
        raise SystemExit("❌ Girdi dosyasında 'GEOID' kolonu yok.")
    src = src.copy()
    src["GEOID"] = src["GEOID"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(11)

    # Tarih türetme
    if "date" in src.columns:
        src["date"] = to_date_series(src["date"])
    else:
        dt_col = detect_datetime_col(src)
        if dt_col is None:
            raise SystemExit("❌ 'date' veya datetime benzeri bir kolon bulunamadı.")
        src[dt_col] = pd.to_datetime(src[dt_col], errors="coerce")
        src["date"] = src[dt_col].dt.date

    # hour_range veya event_hour varsa normalize et
    if "hour_range" in src.columns:
        src["hour_range"] = src["hour_range"].apply(_fmt_hr_range)
    elif "event_hour" in src.columns:
        hr = pd.to_numeric(src["event_hour"], errors="coerce").fillna(0).astype(int) % 24
        start = (hr // 3) * 3
        src["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")
    # değilse saatlik yok, doğrudan günlük sayacağız

    # Saatlik sayaç kolonunu belirle (varsa onu topla; yoksa satır sayısı)
    hr_cnt_col = None
    for c in ("911_request_count_hour_range","call_count","requests","count","n","hr_cnt"):
        if c in src.columns:
            hr_cnt_col = c
            break

    # Günlük toplama (GEOID × date)
    if "hour_range" in src.columns:
        if hr_cnt_col is None:
            # Saatlik sayaç yok → saatlik satır adedinden günlük
            hr_agg = (src.groupby(["GEOID","date","hour_range"], dropna=False)
                        .size().reset_index(name="hr_cnt"))
        else:
            hr_agg = (src.groupby(["GEOID","date","hour_range"], dropna=False)[hr_cnt_col]
                        .sum().reset_index(name="hr_cnt"))
        daily = (hr_agg.groupby(["GEOID","date"], dropna=False)["hr_cnt"]
                       .sum().reset_index(name="911_request_count_daily(before_24_hours)"))
    else:
        # hour_range hiç yok → doğrudan günlük satır adedi
        daily = (src.groupby(["GEOID","date"], dropna=False)
                    .size().reset_index(name="911_request_count_daily(before_24_hours)"))

    # Sızıntısız rolling (last3d/last7d) — 1 gün geriden
    daily = daily.sort_values(["GEOID","date"]).reset_index(drop=True)
    for W in (3, 7):
        daily[f"911_geo_last{W}d"] = (
            daily.groupby("GEOID")["911_request_count_daily(before_24_hours)"]
                 .transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

    # Kalender etiketleri (opsiyonel ama faydalı)
    dts = pd.to_datetime(daily["date"])
    daily["day_of_week"] = dts.dt.weekday.astype("int8")
    daily["month"] = dts.dt.month.astype("int8")
    _season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    daily["season"] = daily["month"].map(_season_map).astype("category")

    # İz bilgisi
    daily["fr_daily_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    daily["fr_daily_tz"] = LOCAL_TZ

    # Kaydet
    safe_save_csv(daily, OUT_PATH)

    # Kısa özet
    try:
        log(daily.head(10).to_string(index=False))
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
