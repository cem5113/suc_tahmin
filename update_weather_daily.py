#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_weather_daily.py
- Şehir-geneli günlük hava verisini GRID (GEOID×date) ve EVENTS (olay satırları) dosyalarına, sadece 'date' üzerinden ekler.
- Saatlik veri yok; her gün için tek satır weather varsayımı.

ENV (varsayılanlar):
  CRIME_DATA_DIR          (crime_prediction_data)
  ARTIFACT_ZIP            (artifact/sf-crime-pipeline-output.zip)
  ARTIFACT_DIR            (artifact_unzipped)

  WEATHER_PATH            (sf_weather_5years.csv)

  FR_GRID_DAILY_IN        (fr_crime_grid_daily.csv)
  FR_GRID_DAILY_OUT       (fr_crime_grid_daily.csv)   # üzerine yazar
  FR_EVENTS_DAILY_IN      (fr_crime_events_daily.csv)
  FR_EVENTS_DAILY_OUT     (fr_crime_events_daily.csv) # üzerine yazar
"""
from __future__ import annotations
import os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True

# -------- utils --------
def log(msg: str): print(msg, flush=True)

def _read_csv(p: Path) -> pd.DataFrame:
    if not p or not p.exists():
        log(f"ℹ️ Dosya yok: {p}"); return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p} ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    # ufak downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p} ({len(df):,}×{df.shape[1]})")

def _first_existing(cols, *cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if isinstance(cand, (list, tuple)):
            for c in cand:
                if c.lower() in low: return low[c.lower()]
        else:
            if cand.lower() in low: return low[cand.lower()]
    return None

def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cand = _first_existing(
        d.columns,
        ["date"],
        ["incident_date", "incident_datetime", "reported_date", "reported_datetime"],
        ["datetime", "time", "timestamp", "Timestamp"]
    )
    if cand is None:
        y = _first_existing(d.columns, "year", "Year")
        m = _first_existing(d.columns, "month", "Month")
        da = _first_existing(d.columns, "day", "Day")
        if y and m and da:
            d["date"] = pd.to_datetime(
                d[[y, m, da]].rename(columns={y:"year", m:"month", da:"day"}),
                errors="coerce"
            ).dt.date
            return d
        d["date"] = pd.NaT
        return d
    d["date"] = pd.to_datetime(d[cand], errors="coerce").dt.date
    return d

def normalize_weather_columns(dfw: pd.DataFrame) -> pd.DataFrame:
    """
    Weather kolonlarını standardize eder:
      date, tavg/tmin/tmax/prcp → wx_ prefiksiyle eklenir.
      Türev: wx_temp_range, wx_is_rainy, wx_is_hot_day
    """
    w = normalize_date_column(dfw)
    lower = {c.lower(): c for c in w.columns}
    def has(k): return k in lower
    def col(k): return lower[k]

    rename = {}
    if has("temp_min") and not has("tmin"): rename[col("temp_min")] = "tmin"
    if has("temp_max") and not has("tmax"): rename[col("temp_max")] = "tmax"
    if has("precipitation_mm") and not has("prcp"): rename[col("precipitation_mm")] = "prcp"
    if has("prcp_mm") and not has("prcp"): rename[col("prcp_mm")] = "prcp"
    if has("taverage") and not has("tavg"): rename[col("taverage")] = "tavg"
    w.rename(columns=rename, inplace=True)

    for c in ["tavg","tmin","tmax","prcp"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    HOT_DAY_THRESHOLD_C = 25.0
    w["temp_range"] = (w["tmax"] - w["tmin"]) if {"tmax","tmin"}.issubset(w.columns) else np.nan
    w["is_rainy"]   = (pd.to_numeric(w.get("prcp", np.nan), errors="coerce").fillna(0) > 0).astype("Int64")
    w["is_hot_day"] = (pd.to_numeric(w.get("tmax", np.nan), errors="coerce") > HOT_DAY_THRESHOLD_C).astype("Int64")

    keep = ["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"]
    for c in keep:
        if c not in w.columns: w[c] = np.nan
    w = w[keep].dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")

    # prefix (date hariç)
    wx = w.rename(columns={c: (f"wx_{c}" if c != "date" else c) for c in w.columns})
    return wx

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists(): return
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

# -------- config --------
BASE_DIR     = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

WEATHER_FILE = os.getenv("WEATHER_PATH", "sf_weather_5years.csv")

GRID_IN   = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT  = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN     = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT    = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

WX_CANDS = [ARTIFACT_DIR / WEATHER_FILE, BASE_DIR / WEATHER_FILE, Path(WEATHER_FILE)]

# -------- run --------
def main() -> int:
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    wx_p = next((p for p in WX_CANDS if p.exists()), None)
    if wx_p is None:
        raise FileNotFoundError("❌ Weather CSV bulunamadı (WEATHER_PATH / sf_weather_5years.csv).")

    wx_raw = _read_csv(wx_p)
    if wx_raw.empty:
        raise SystemExit("❌ Weather CSV boş.")
    wx = normalize_weather_columns(wx_raw)

    # ---- GRID ----
    grid_path_in  = BASE_DIR / GRID_IN  if not GRID_IN.is_absolute()  else GRID_IN
    grid_path_out = BASE_DIR / GRID_OUT if not GRID_OUT.is_absolute() else GRID_OUT
    grid = _read_csv(grid_path_in)
    if not grid.empty:
        grid = normalize_date_column(grid)
        if "date" not in grid.columns:
            raise RuntimeError("GRID’de 'date' oluşturulamadı.")
        before = grid.shape
        out_g = grid.merge(wx, on="date", how="left", validate="m:1")
        log(f"🔗 GRID join: {before} → {out_g.shape}")
        _safe_write_csv(out_g, grid_path_out)
    else:
        log("ℹ️ GRID bulunamadı → atlandı.")

    # ---- EVENTS ----
    ev_path_in  = BASE_DIR / EV_IN  if not EV_IN.is_absolute()  else EV_IN
    ev_path_out = BASE_DIR / EV_OUT if not EV_OUT.is_absolute() else EV_OUT
    ev = _read_csv(ev_path_in)
    if not ev.empty:
        ev = normalize_date_column(ev)
        if "date" not in ev.columns:
            raise RuntimeError("EVENTS’te 'date' oluşturulamadı.")
        before = ev.shape
        out_e = ev.merge(wx, on="date", how="left", validate="m:1")
        log(f"🔗 EVENTS join: {before} → {out_e.shape}")
        _safe_write_csv(out_e, ev_path_out)
    else:
        log("ℹ️ EVENTS bulunamadı → atlandı.")

    # küçük önizleme
    try:
        cols = ["date","wx_tmin","wx_tmax","wx_prcp","wx_temp_range","wx_is_rainy","wx_is_hot_day"]
        if not grid.empty:
            log("— GRID wx preview —")
            log(out_g[[c for c in cols if c in out_g.columns]].head(6).to_string(index=False))
        if not ev.empty:
            log("— EVENTS wx preview —")
            log(out_e[[c for c in cols if c in out_e.columns]].head(6).to_string(index=False))
    except Exception:
        pass

    log("✅ enrich_with_weather_daily.py tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
