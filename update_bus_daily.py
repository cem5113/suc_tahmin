#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_bus_daily.py
- Otobüs durak verisini (GEOID eşleşmeli) hem GRID (GEOID×date) hem de EVENTS (olay satırları) dosyalarına ekler.
- Günlük veri: saatlik YOK. Bus özellikleri GEOID seviyesinde sabittir, tüm günlere aynen taşınır.
- Otomatik GEOID normalize (11 hane), mevcut kolonlardan akıllı agregasyon (count/sum/min).
- Üzerine yazar (in-place), istersen ENV ile farklı OUT verebilirsin.

ENV (varsayılanlar):
  CRIME_DATA_DIR         (crime_prediction_data)

  FR_GRID_DAILY_IN       (fr_crime_grid_daily.csv)
  FR_GRID_DAILY_OUT      (fr_crime_grid_daily.csv)

  FR_EVENTS_DAILY_IN     (fr_crime_events_daily.csv)
  FR_EVENTS_DAILY_OUT    (fr_crime_events_daily.csv)

  BUS_PATH               (sf_bus_stops_with_geoid.csv)
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
    # hafif downcast
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
    # EVENTS’te sadece incident_datetime olabilir; yoksa daily değil demektir.
    return out

# ============================== Config ==============================
BASE_DIR     = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

GRID_IN   = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT  = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN     = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT    = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

BUS_PATH_ENV = os.getenv("BUS_PATH", "sf_bus_stops_with_geoid.csv")
BUS_PATHS = [
    ARTIFACT_DIR / BUS_PATH_ENV,
    BASE_DIR / BUS_PATH_ENV,
    Path(BUS_PATH_ENV),
]

# ============================== Core ==============================
def aggregate_bus_by_geoid(bus: pd.DataFrame) -> pd.DataFrame:
    """Bus girişinde satır=durak ise GEOID bazında özet çıkar.
       Aşağıdaki kolonları varsa akıllıca toplar:
       - sayım:  bus_stop_count (yoksa üret)
       - bayrak/sayım: bus_within_XXXm  → sum
       - mesafe: distance_to_bus_m      → min
    """
    b_geoid = _find_geoid_col(bus)
    if not b_geoid:
        raise RuntimeError("Otobüs verisinde GEOID kolonu tespit edilemedi.")
    bus["_key"] = _key_11(bus[b_geoid])

    feat_candidates = [
        "bus_stop_count", "distance_to_bus_m",
        "bus_within_250m","bus_within_300m","bus_within_500m","bus_within_600m",
        "bus_within_750m","bus_within_900m","bus_within_1000m",
    ]
    present = [c for c in feat_candidates if c in bus.columns]

    if not present:
        # en azından her GEOID için durak say
        agg = (bus.groupby("_key", as_index=False)
                  .size().rename(columns={"size":"bus_stop_count"}))
        return agg.rename(columns={"_key":"GEOID"})

    agg_dict = {}
    for c in present:
        if re.search(r"(count|within)", c, flags=re.I):
            agg_dict[c] = "sum"
        elif re.search(r"(dist|distance)", c, flags=re.I):
            agg_dict[c] = "min"
        else:
            agg_dict[c] = "sum"

    bus_num = bus.copy()
    for c in present:
        bus_num[c] = pd.to_numeric(bus_num[c], errors="coerce")

    agg = (bus_num.groupby("_key", as_index=False).agg(agg_dict))
    # güvenlik: stop sayısı yoksa ekle
    if "bus_stop_count" not in agg.columns:
        cnt = (bus.groupby("_key", as_index=False)
                 .size().rename(columns={"size":"bus_stop_count"}))
        agg = agg.merge(cnt, on="_key", how="outer")

    agg = agg.rename(columns={"_key":"GEOID"})
    agg["GEOID"] = agg["GEOID"].astype("string")
    # tip düzeltmeleri
    if "distance_to_bus_m" in agg.columns:
        agg["distance_to_bus_m"] = pd.to_numeric(agg["distance_to_bus_m"], errors="coerce")
    for c in [col for col in agg.columns if c.startswith("bus_within_")] + ["bus_stop_count"]:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0).astype("Int64")
    return agg

def enrich_one(df: pd.DataFrame, bus_agg: pd.DataFrame, is_grid: bool) -> pd.DataFrame:
    """GEOID merge; GRID ise date kolonunu korur."""
    if df.empty: return df
    c_geoid = _find_geoid_col(df)
    if not c_geoid:
        raise RuntimeError("Hedef tabloda GEOID kolonu bulunamadı.")
    out = df.copy()

    # GRID: date kolonunu normalize et (sadece görünürlük)
    if is_grid:
        out = _ensure_date(out)

    # hedef GEOID -> resmi (tek) GEOID string(11)
    out.insert(0, "GEOID", _key_11(out[c_geoid]).astype("string"))
    # orijinal GEOID benzer kolonları kaldır
    drop_candidates = [c for c in out.columns if c != "GEOID" and "geoid" in c.lower()]
    out.drop(columns=drop_candidates, inplace=True, errors="ignore")

    out = out.merge(bus_agg, on="GEOID", how="left")

    # doldur
    if "distance_to_bus_m" in out.columns:
        out["distance_to_bus_m"] = pd.to_numeric(out["distance_to_bus_m"], errors="coerce")
    for c in ["bus_stop_count"] + [col for col in out.columns if col.startswith("bus_within_")]:
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

    bus_path = next((p for p in BUS_PATHS if Path(p).exists()), None)
    if not bus_path:
        raise FileNotFoundError(f"❌ Otobüs dosyası yok: {BUS_PATH_ENV} (artifact/BASE/local)")
    bus = _read_csv(Path(bus_path))

    # 2) GEOID bazında otobüs özetlerini hazırla
    bus_agg = aggregate_bus_by_geoid(bus)

    # 3) enrich & yaz
    if not grid.empty:
        grid2 = enrich_one(grid, bus_agg, is_grid=True)
        _safe_write_csv(grid2, BASE_DIR / GRID_OUT if not GRID_OUT.is_absolute() else GRID_OUT)
    else:
        log("ℹ️ GRID dosyası bulunamadı/boş, atlandı.")

    if not ev.empty:
        ev2 = enrich_one(ev, bus_agg, is_grid=False)
        _safe_write_csv(ev2, BASE_DIR / EV_OUT if not EV_OUT.is_absolute() else EV_OUT)
    else:
        log("ℹ️ EVENTS dosyası bulunamadı/boş, atlandı.")

    # 4) kısa önizleme
    try:
        cols = ["GEOID","bus_stop_count","distance_to_bus_m","bus_within_300m","bus_within_600m","bus_within_900m"]
        if not grid.empty:
            log("— GRID preview —")
            log(grid2[[c for c in cols if c in grid2.columns]].head(10).to_string(index=False))
        if not ev.empty:
            log("— EVENTS preview —")
            log(ev2[[c for c in cols if c in ev2.columns]].head(10).to_string(index=False))
    except Exception:
        pass

    log("✅ enrich_with_bus_daily.py tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
