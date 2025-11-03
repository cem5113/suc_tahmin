#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_crime_weather_daily.py — date-bazlı citywide hava durumu zenginleştirme
# IN : daily_crime_07.csv
# OUT: daily_crime_08.csv

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ── helpers ─────────────────────────────────────────────────────────────────
def log(msg: str): print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log(f"✅ saved → {path}  (rows={len(df):,}, cols={df.shape[1]})")

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def num(s: pd.Series, dtype="float32") -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(dtype)

# ── config ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
CRIME_IN   = Path(os.getenv("DAILY_IN",  str(BASE_DIR / "daily_crime_07.csv")))
WEATHER_IN = Path(os.getenv("WEATHER_IN", str(BASE_DIR / "sf_weather_5years.csv")))
CRIME_OUT  = Path(os.getenv("DAILY_OUT", str(BASE_DIR / "daily_crime_08.csv")))

# ── load ────────────────────────────────────────────────────────────────────
log("⏳ load…")
if not CRIME_IN.exists() or not CRIME_IN.is_file():
    raise FileNotFoundError(f"❌ input missing: {CRIME_IN}")
if not WEATHER_IN.exists() or not WEATHER_IN.is_file():
    raise FileNotFoundError(f"❌ weather missing: {WEATHER_IN}")

df = pd.read_csv(CRIME_IN, low_memory=False)
wx = pd.read_csv(WEATHER_IN, low_memory=False)
log(f"📥 crime:  {CRIME_IN}  {df.shape[0]:,}×{df.shape[1]}")
log(f"📥 weather:{WEATHER_IN} {wx.shape[0]:,}×{wx.shape[1]}")

# ── date normalization ──────────────────────────────────────────────────────
if "date" not in df.columns:
    # bazen event_date / dt olabilir
    cand = [c for c in ("date","event_date","dt","day") if c in df.columns]
    if not cand:
        raise KeyError("❌ crime verisinde 'date' türetilecek bir kolon yok.")
    df["date"] = df[cand[0]]
df["date"] = to_date(df["date"])
wx["date"] = to_date(wx.get("date", pd.NaT))

# ── weather de-dup (aynı güne çok kayıt varsa) ─────────────────────────────
# Hava durumu citywide → date tekil olmalı. Aynı güne çok satır varsa agregasyon.
if wx["date"].isna().all():
    raise ValueError("❌ weather dosyasında geçerli 'date' yok.")
# numeric/sözde-numeric kolonları topla/ortalama et
wx_cols = [c for c in wx.columns if c != "date"]
num_cols = [c for c in wx_cols if pd.api.types.is_numeric_dtype(wx[c])]

# Numerik olmayanlardan bazıları sayısal olabilir (string tipte sayılar)
for c in [c for c in wx_cols if c not in num_cols]:
    try:
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(wx[c]):
            num_cols.append(c)
    except Exception:
        pass

agg_dict = {c: "mean" for c in num_cols}
# Tamamen kategorik kalan kolonlar (örn. koşul/description) → en çok görülen
cat_cols = [c for c in wx_cols if c not in num_cols]
def _mode(s):
    s = s.dropna()
    return s.mode().iat[0] if not s.empty else np.nan
for c in cat_cols:
    agg_dict[c] = _mode

wx_agg = (wx.groupby("date", as_index=False)
            .agg(agg_dict)
            .sort_values("date")
            .reset_index(drop=True))

# ── rename with wx_ prefix (date hariç) ─────────────────────────────────────
rename_map = {c: f"wx_{c}" for c in wx_agg.columns if c != "date"}
wx_agg = wx_agg.rename(columns=rename_map)

# Önerilen çekirdek alanlar varsa tür/dolgu yap
for c in ("wx_precipitation_mm","wx_temp_max","wx_temp_min","wx_wind_speed","wx_humidity"):
    if c in wx_agg.columns:
        wx_agg[c] = pd.to_numeric(wx_agg[c], errors="coerce")

# Bazı pratik türevler (varsa)
if {"wx_temp_max","wx_temp_min"}.issubset(wx_agg.columns):
    wx_agg["wx_temp_range"] = (wx_agg["wx_temp_max"] - wx_agg["wx_temp_min"]).astype("float32")

# Yağış boşsa 0 (genelde mantıklı), diğerleri NaN bırakılabilir
if "wx_precipitation_mm" in wx_agg.columns:
    wx_agg["wx_precipitation_mm"] = wx_agg["wx_precipitation_mm"].fillna(0).astype("float32")

# ── merge ───────────────────────────────────────────────────────────────────
before = df.shape
log("🔗 merge (left)…")
out = df.merge(wx_agg, on="date", how="left", validate="many_to_one")

log(f"Δ rows: {before[0]} → {out.shape[0]}")
log(f"Δ cols: {before[1]} → {out.shape[1]}")

# ── write ───────────────────────────────────────────────────────────────────
safe_save_csv(out, str(CRIME_OUT))

# ── short preview ───────────────────────────────────────────────────────────
try:
    keep = ["date"] + [c for c in out.columns if c.startswith("wx_")][:6]
    log(out[keep].head(8).to_string(index=False))
except Exception:
    pass
