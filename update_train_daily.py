#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_train_daily.py
- Tren durağı verisini (GEOID eşleşmeli) hem GRID (GEOID×date) hem EVENTS'e ekler.
- Günlük akış: saatlik yok. Tren özellikleri GEOID bazında sabittir.
- GEOID normalize (11 hane), akıllı agregasyon (count/sum/min), downcast ve hızlı önizleme.

ENV (varsayılanlar):
  CRIME_DATA_DIR         (crime_prediction_data)

  FR_GRID_DAILY_IN       (fr_crime_grid_daily.csv)
  FR_GRID_DAILY_OUT      (fr_crime_grid_daily.csv)

  FR_EVENTS_DAILY_IN     (fr_crime_events_daily.csv)
  FR_EVENTS_DAILY_OUT    (fr_crime_events_daily.csv)

  TRAIN_PATH             (sf_train_stops_with_geoid.csv)
  ARTIFACT_ZIP           (artifact/sf-crime-pipeline-output.zip)   # varsa açar
  ARTIFACT_DIR           (artifact_unzipped)
"""

from __future__ import annotations
import os, re, zipfile
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ============================== Utils ==============================
def log(msg: str): print(msg, flush=True)

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in zf.infolist():
            out = dest_dir / m.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if m.is_dir():
                out.mkdir(parents=True, exist_ok=True); continue
            with zf.open(m, "r") as src, open(out, "wb") as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

def _digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")

def _key_11(series: pd.Series) -> pd.Series:
    return _digits_only(series).str.replace(" ", "", regex=False).str.zfill(11).str[:11]

def _find_geoid_col(df: pd.DataFrame) -> str | None:
    cands = ["GEOID","geoid","geo_id","GEOID10","geoid10","GeoID",
             "tract","TRACT","tract_geoid","TRACT_GEOID","geography_id","GEOID2"]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        if "geoid" in c.lower(): return c
    return None

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"ℹ️ Dosya yok: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p}  ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    # downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p}  ({len(df):,}×{df.shape[1]})")

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        return out
    for c in ("event_date","dt","day","incident_datetime","datetime","timestamp","created_datetime"):
        if c in out.columns:
            out["date"] = pd.to_datetime(out[c], errors="coerce").dt.date
            return out
    return out

# ============================== Config ==============================
BASE_DIR     = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

GRID_IN   = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT  = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN     = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT    = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

TRAIN_PATH_ENV = os.getenv("TRAIN_PATH", "sf_train_stops_with_geoid.csv")
TRAIN_PATHS = [
    ARTIFACT_DIR / TRAIN_PATH_ENV,
    BASE_DIR / TRAIN_PATH_ENV,
    Path(TRAIN_PATH_ENV),
]

# Hangi tren özelliklerini arayalım? (dosyada ne varsa alır)
TRAIN_FEATS_CANDIDATES = [
    "train_stop_count", "distance_to_train_m",
    "train_within_250m","train_within_300m","train_within_500m","train_within_600m",
    "train_within_750m","train_within_900m","train_within_1000m",
    "train_0_300m","train_300_600m","train_600_900m",
]

# ============================== Core ==============================
def aggregate_train_by_geoid(train: pd.DataFrame) -> pd.DataFrame:
    """Satır=durak ise GEOID bazında özet üret:
       - *count/within* → sum
       - *distance*     → min
       - en azından train_stop_count üret.
    """
    t_geoid = _find_geoid_col(train)
    if not t_geoid:
        raise RuntimeError("Tren verisinde GEOID kolonu tespit edilemedi.")
    train["_key"] = _key_11(train[t_geoid])

    present = [c for c in TRAIN_FEATS_CANDIDATES if c in train.columns]
    if not present:
        agg = (train.groupby("_key", as_index=False)
                     .size().rename(columns={"size":"train_stop_count"}))
        return agg.rename(columns={"_key":"GEOID"})

    agg_dict = {}
    for c in present:
        if re.search(r"(count|within)", c, flags=re.I):
            agg_dict[c] = "sum"
        elif re.search(r"(dist|distance)", c, flags=re.I):
            agg_dict[c] = "min"
        else:
            agg_dict[c] = "sum"

    tr_num = train[["_key"] + present].copy()
    for c in present:
        tr_num[c] = pd.to_numeric(tr_num[c], errors="coerce")

    agg = tr_num.groupby("_key", as_index=False).agg(agg_dict)

    # güvenlik: stop sayısı yoksa ekle
    if "train_stop_count" not in agg.columns:
        add_cnt = (train.groupby("_key", as_index=False)
                         .size().rename(columns={"size":"train_stop_count"}))
        agg = agg.merge(add_cnt, on="_key", how="outer")

    agg = agg.rename(columns={"_key":"GEOID"})
    agg["GEOID"] = agg["GEOID"].astype("string")

    # tip düzeltmeleri
    if "distance_to_train_m" in agg.columns:
        agg["distance_to_train_m"] = pd.to_numeric(agg["distance_to_train_m"], errors="coerce")
    for c in [col for col in agg.columns if col.startswith("train_within_")] + ["train_stop_count",
                                                                                "train_0_300m","train_300_600m","train_600_900m"]:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0).astype("Int64")
    return agg

def enrich_one(df: pd.DataFrame, train_agg: pd.DataFrame, is_grid: bool) -> pd.DataFrame:
    """GEOID merge; GRID ise date kolonunu normalize eder."""
    if df.empty: return df
    c_geoid = _find_geoid_col(df)
    if not c_geoid:
        raise RuntimeError("Hedef tabloda GEOID kolonu bulunamadı.")
    out = df.copy()

    if is_grid:
        out = _ensure_date(out)

    # hedef GEOID → resmi tek GEOID (string, 11 hane)
    out.insert(0, "GEOID", _key_11(out[c_geoid]).astype("string"))
    drop_candidates = [c for c in out.columns if c != "GEOID" and "geoid" in c.lower()]
    out.drop(columns=drop_candidates, inplace=True, errors="ignore")

    out = out.merge(train_agg, on="GEOID", how="left", validate="many_to_one")

    # doldurma
    if "distance_to_train_m" in out.columns:
        out["distance_to_train_m"] = pd.to_numeric(out["distance_to_train_m"], errors="coerce")
    for c in ["train_stop_count","train_0_300m","train_300_600m","train_600_900m"] + \
             [col for col in out.columns if col.startswith("train_within_")]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64")

    return out

# ============================== Run ==============================
def main() -> int:
    # 0) artifact varsa aç
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) girişleri oku
    grid = _read_csv(BASE_DIR / GRID_IN) if not GRID_IN.is_absolute() else _read_csv(GRID_IN)
    ev   = _read_csv(BASE_DIR / EV_IN)   if not EV_IN.is_absolute()  else _read_csv(EV_IN)

    train_path = next((p for p in TRAIN_PATHS if Path(p).exists()), None)
    if not train_path:
        raise FileNotFoundError(f"❌ Tren dosyası yok: {TRAIN_PATH_ENV} (artifact/BASE/local)")
    train = _read_csv(Path(train_path))

    # 2) GEOID bazında tren özetlerini hazırla
    train_agg = aggregate_train_by_geoid(train)

    # 3) enrich & yaz
    if not grid.empty:
        grid2 = enrich_one(grid, train_agg, is_grid=True)
        _safe_write_csv(grid2, BASE_DIR / GRID_OUT if not GRID_OUT.is_absolute() else GRID_OUT)
    else:
        log("ℹ️ GRID dosyası bulunamadı/boş, atlandı.")

    if not ev.empty:
        ev2 = enrich_one(ev, train_agg, is_grid=False)
        _safe_write_csv(ev2, BASE_DIR / EV_OUT if not EV_OUT.is_absolute() else EV_OUT)
    else:
        log("ℹ️ EVENTS dosyası bulunamadı/boş, atlandı.")

    # 4) kısa önizleme
    try:
        cols = ["GEOID","train_stop_count","distance_to_train_m",
                "train_within_300m","train_within_600m","train_within_900m",
                "train_0_300m","train_300_600m","train_600_900m"]
        if not grid.empty:
            log("— GRID preview —")
            log(grid2[[c for c in cols if c in grid2.columns]].head(10).to_string(index=False))
        if not ev.empty:
            log("— EVENTS preview —")
            log(ev2[[c for c in cols if c in ev2.columns]].head(10).to_string(index=False))
    except Exception:
        pass

    log("✅ enrich_with_train_daily.py tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
