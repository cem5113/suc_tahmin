#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_311_daily.py — Input: daily_crime_01.csv  → Output: daily_crime_02.csv

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional

# ====== IO & AYARLAR ======
CRIME_DATA_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").strip()
BASE = Path(CRIME_DATA_DIR)

DAILY_IN  = os.getenv("DAILY_IN",  "daily_crime_01.csv").strip()
DAILY_OUT = os.getenv("DAILY_OUT", "daily_crime_02.csv").strip()

# 311 giriş adayları
FR_311_DAILY_IN = os.getenv("FR_311_DAILY_IN", "").strip()  # (opsiyonel) doğrudan günlük 311 dosyası
AGG_311_NAME    = os.getenv("AGG_311_NAME", "sf_311_last_5_years.csv").strip()

def log(msg: str): 
    print(msg, flush=True)

def resolve_path(p: str | Path, base: Path) -> Path:
    """Absolute ise dokunma; relative ise base altına koy.
       Zaten base ile başlıyorsa ikinci kez ekleme (triple-prefix'i engeller)."""
    p = Path(p)
    if p.is_absolute():
        return p
    p_norm = str(p).lstrip("./")
    base_norm = str(base).lstrip("./")
    if p_norm.startswith(base_norm + "/") or p_norm == base_norm:
        return Path(p_norm)
    return base / p

def ls_hint(path: Path) -> str:
    try:
        if path.exists():
            if path.is_file():
                return f"(file exists) {path}"
            if path.is_dir():
                items = [x.name + ("/" if x.is_dir() else "") for x in list(path.iterdir())[:50]]
                more = "" if len(items) < 50 else " (…)"
                return f"dir: {path}\n  -> {', '.join(items)}{more}"
        parent = path.parent
        if parent.exists():
            items = [x.name + ("/" if x.is_dir() else "") for x in list(parent.iterdir())[:50]]
            return f"(missing) parent: {parent}\n  -> {', '.join(items)}"
    except Exception as e:
        return f"(ls_hint error: {e})"
    return "(path not found)"

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"❌ Dosya yok: {path}\n{ls_hint(path)}")
    df = pd.read_csv(path, low_memory=False)
    log(f"📖 Okundu: {path}  ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    log(f"💾 Yazıldı: {path}  ({len(df):,}×{df.shape[1]})")

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)", expand=False)
    return x.str[:L].str.zfill(L)

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = _to_date(out["date"])
        return out
    for c in ("event_date","dt","day","datetime","timestamp","created_datetime"):
        if c in out.columns:
            out["date"] = _to_date(out[c])
            return out
    raise SystemExit("❌ Grid dosyasında tarih kolonu yok (date/event_date/dt/day/datetime/timestamp).")

def _find_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists() and p.is_file():
            return p
    return None

def load_311_daily(base: Path) -> pd.DataFrame:
    """1) FR_311_DAILY_IN verilmişse doğrudan onu oku (GEOID×date, 311_request_count_daily).
       2) Aksi halde aggregated (saatlik/3s) dosyadan günlük üret."""
    # 1) Doğrudan günlük dosya
    if FR_311_DAILY_IN:
        p = resolve_path(FR_311_DAILY_IN, base)
        df = _read_csv(p)
        if "311_request_count_daily" not in df.columns:
            cand = [c for c in df.columns if "daily" in c.lower() and "311" in c]
            if cand:
                df = df.rename(columns={cand[0]: "311_request_count_daily"})
        if "date" not in df.columns:
            for c in ("event_date","dt","day","datetime","timestamp"):
                if c in df.columns:
                    df["date"] = df[c]
                    break
        df["date"] = _to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = _norm_geoid(df["GEOID"])
        if "311_request_count_daily" not in df.columns:
            df["311_request_count_daily"] = 0
        return df[["GEOID","date","311_request_count_daily"]].copy()

    # 2) Aggregated kaynaktan günlük üret
    agg_candidates = [
        resolve_path(AGG_311_NAME, base),
        base / "sf_311_last_5_years_3h.csv",
        Path("./") / AGG_311_NAME,  # yedek
    ]
    src = _find_existing(agg_candidates)
    if src is None:
        log("ℹ️ 311 özet bulunamadı; günlük 311 sıfır kabul edilecek.")
        return pd.DataFrame(columns=["GEOID","date","311_request_count_daily"])

    df = _read_csv(src)
    if "date" not in df.columns:
        raise SystemExit("❌ 311 özetinde 'date' kolonu yok.")
    if "311_request_count" not in df.columns:
        cand = [c for c in df.columns if c.lower() in ("count","requests","n","311_count")]
        if cand:
            df = df.rename(columns={cand[0]: "311_request_count"})
        else:
            df["311_request_count"] = 1
    if "GEOID" in df.columns:
        df["GEOID"] = _norm_geoid(df["GEOID"])
    df["date"] = _to_date(df["date"])
    daily = (df.groupby(["GEOID","date"], dropna=False)["311_request_count"]
               .sum()
               .reset_index(name="311_request_count_daily"))
    return daily

def main() -> int:
    log("🚀 update_311_daily.py")

    p_in  = resolve_path(DAILY_IN,  BASE)
    p_out = resolve_path(DAILY_OUT, BASE)
    log(f"🔧 CRIME_DATA_DIR: {BASE}")
    log(f"🔧 DAILY_IN      : {p_in}")
    log(f"🔧 DAILY_OUT     : {p_out}")

    # 1) Grid (daily_crime_01.csv)
    crime = _read_csv(p_in)
    crime = _ensure_date_column(crime)
    if "GEOID" in crime.columns:
        crime["GEOID"] = _norm_geoid(crime["GEOID"])

    # 2) 311 günlük
    d311 = load_311_daily(BASE)

    # 3) Merge (GEOID + date)
    keys = ["GEOID","date"]
    before = crime.shape
    out = crime.merge(d311, on=keys, how="left")
    out["311_request_count_daily"] = pd.to_numeric(
        out.get("311_request_count_daily", 0), errors="coerce"
    ).fillna(0).astype("int32")
    log(f"🔗 Join: {before} → {out.shape}")

    # 4) İz bilgisi
    out["daily_311_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # 5) Yaz
    _safe_write_csv(out, p_out)

    # 6) Kısa önizleme
    try:
        log(out.head(8).to_string(index=False))
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
