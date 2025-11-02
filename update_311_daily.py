#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_311_daily.py — Input: daily_crime_01.csv  → Output: daily_crime_02.csv
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# ====== IO & AYARLAR ======
SAVE_DIR        = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
DAILY_IN        = Path(os.getenv("DAILY_IN",  str(SAVE_DIR / "daily_crime_01.csv")))
DAILY_OUT       = Path(os.getenv("DAILY_OUT", str(SAVE_DIR / "daily_crime_02.csv")))

# 311 giriş adayları
FR_311_DAILY_IN = Path(os.getenv("FR_311_DAILY_IN", ""))  # varsa doğrudan günlük (GEOID×date)
AGG_311_NAME    = os.getenv("AGG_311_NAME", "sf_311_last_5_years.csv")  # 3-saatlik özet (GEOID×date×hour_range)
AGG_311_CANDIDATES = [
    SAVE_DIR / AGG_311_NAME,
    Path("./") / AGG_311_NAME,
    SAVE_DIR / "sf_311_last_5_years_3h.csv",
]

def log(msg: str): print(msg, flush=True)
def _abs(p: Path) -> Path: return p.expanduser().resolve()

def _read_csv(path: Path) -> pd.DataFrame:
    p = _abs(path)
    if not p.exists():
        raise FileNotFoundError(f"❌ Dosya yok: {p}")
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p}  ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    p = _abs(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p}  ({len(df):,}×{df.shape[1]})")

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)", expand=False)
    return x.str[:L].str.zfill(L)

def _find_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None

def load_311_daily() -> pd.DataFrame:
    """
    1) FR_311_DAILY_IN varsa doğrudan oku (GEOID×date, içerikte 311 günlük sayı olmalı)
    2) Yoksa 3-saatlik özet dosyasını bul → GEOID×date bazında SUM → daily üret
    """
    # 1) Doğrudan günlük varsa
    if FR_311_DAILY_IN and _abs(FR_311_DAILY_IN).exists():
        df = _read_csv(FR_311_DAILY_IN)
        # kolon adı uyumları
        if "311_request_count_daily" not in df.columns:
            # farklı bir ad kullanılmış olabilir → mantıklı adaylar
            cand = [c for c in df.columns if "daily" in c and "311" in c]
            if cand:
                df = df.rename(columns={cand[0]: "311_request_count_daily"})
        # tarih kolonunu normalize et
        if "date" not in df.columns:
            for c in ("event_date", "dt", "day"):
                if c in df.columns:
                    df["date"] = df[c]
                    break
        df["date"] = _to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = _norm_geoid(df["GEOID"])
        # eksik sayaç → 0
        if "311_request_count_daily" not in df.columns:
            df["311_request_count_daily"] = 0
        return df[["GEOID","date","311_request_count_daily"]].copy()

    # 2) 3-saatlik özetten günlük üret
    src = _find_existing(AGG_311_CANDIDATES)
    if src is None:
        log("ℹ️ 311 özet bulunamadı; günlük 311 sıfır kabul edilecek.")
        return pd.DataFrame(columns=["GEOID","date","311_request_count_daily"])

    df = _read_csv(src)
    # beklenen kolonlar: GEOID, date, hour_range, 311_request_count
    if "date" not in df.columns:
        raise SystemExit("❌ 311 özetinde 'date' yok.")
    if "311_request_count" not in df.columns:
        # bazı pipeline'larda isimlenme farklı olabilir
        cand = [c for c in df.columns if c.lower() in ("count","requests","n")]
        if not cand:
            # saatlik satır adedinden kur
            df["311_request_count"] = 1
        else:
            df = df.rename(columns={cand[0]: "311_request_count"})
    if "GEOID" in df.columns:
        df["GEOID"] = _norm_geoid(df["GEOID"])
    df["date"] = _to_date(df["date"])
    daily = (df.groupby(["GEOID","date"], dropna=False)["311_request_count"]
               .sum()
               .reset_index(name="311_request_count_daily"))
    return daily

def main() -> int:
    log("🚀 update_311_daily.py")
    log(f"🔧 DAILY_IN : {_abs(DAILY_IN)}")
    log(f"🔧 DAILY_OUT: {_abs(DAILY_OUT)}")

    # 1) daily_crime_01.csv
    crime = _read_csv(DAILY_IN)
    # tarih kolon adı: date / event_date → date'e indir
    if "date" not in crime.columns:
        for c in ("event_date","dt","day"):
            if c in crime.columns:
                crime["date"] = crime[c]
                break
    crime["date"] = _to_date(crime["date"])
    if "GEOID" in crime.columns:
        crime["GEOID"] = _norm_geoid(crime["GEOID"])

    # 2) 311 günlük
    d311 = load_311_daily()

    # 3) Merge (GEOID + date)
    keys = ["GEOID","date"]
    before = crime.shape
    out = crime.merge(d311, on=keys, how="left")
    out["311_request_count_daily"] = pd.to_numeric(out["311_request_count_daily"], errors="coerce").fillna(0).astype("int32")
    log(f"🔗 Join: {before} → {out.shape}")

    # 4) İz bilgisi
    out["daily_311_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # 5) Yaz
    _safe_write_csv(out, DAILY_OUT)

    # 6) Kısa önizleme
    try:
        log(out.head(8).to_string(index=False))
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
