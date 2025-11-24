#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_weather_fr.py  (artifact-first download + date-merge)

Amaç:
- Şehir-geneli günlük hava verisini suç verisine sadece "tarih" üzerinden eklemek.
- Weather girdi kaynağı önceliği:
    1) Son başarılı Actions artifact'ı içinden sf_weather_5years.csv
    2) GitHub releases/latest download sf_weather_5years.csv
    3) Yerel path adayları
Girdi:
- fr_crime_07.csv / sf_crime_07.csv / fr_crime.csv / sf_crime.csv
Çıkış:
- fr_crime_08.csv / sf_crime_08.csv
"""

import os, io, zipfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

pd.options.mode.copy_on_write = True


# =============================================================================
# ENV / CONFIG
# =============================================================================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").rstrip("/")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Crime input candidates
CRIME_IN_CANDS = [
    os.path.join(BASE_DIR, "fr_crime_07.csv"),
]

# ---- GitHub artifact settings (update_crime.py ile aynı mantık) ----
GITHUB_REPO   = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")  # owner/repo
GH_TOKEN      = os.getenv("GH_TOKEN", "") or os.getenv("GITHUB_TOKEN", "")
ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output")
ARTIFACT_MAX_RUNS = int(os.getenv("ARTIFACT_MAX_RUNS", "20"))

# ---- Releases/latest fallback ----
WEATHER_RELEASE_URL = os.getenv(
    "WEATHER_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_weather_5years.csv"
)

# Local fallback candidates
WEATHER_LOCAL_CANDS = [
    os.path.join(BASE_DIR, "sf_weather_5years.csv"),
    "sf_weather_5years.csv",
    os.path.join(BASE_DIR, "weather.csv"),
    "weather.csv",
]

HOT_DAY_THRESHOLD_C = float(os.getenv("HOT_DAY_THRESHOLD_C", "25.0"))


# =============================================================================
# LOG HELPERS
# =============================================================================
def log_shape(df, label):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before, after, label):
    br, bc = before
    ar, ac = after
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        print(f"📁 Yedek oluşturuldu: {path}.bak")

def pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# =============================================================================
# DATE NORMALIZATION
# =============================================================================
def _first_existing(cols, *cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if isinstance(cand, (list, tuple)):
            for c in cand:
                if c.lower() in low:
                    return low[c.lower()]
        else:
            if cand.lower() in low:
                return low[cand.lower()]
    return None

def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """df içine 'date' (datetime.date) kolonu üretir."""
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
                d[[y, m, da]].rename(columns={y: "year", m: "month", da: "day"}),
                errors="coerce"
            ).dt.date
            return d
        d["date"] = pd.NaT
        return d

    d["date"] = pd.to_datetime(d[cand], errors="coerce").dt.date
    return d


# =============================================================================
# WEATHER NORMALIZATION
# =============================================================================
def normalize_weather_columns(dfw: pd.DataFrame) -> pd.DataFrame:
    """
    Weather kolonlarını standardize eder:
    - date/time/datetime → date
    - temp_min/temp_max/prcp_mm → tmin/tmax/prcp
    - türevler: temp_range, is_rainy, is_hot_day
    """
    w = dfw.copy()
    w = normalize_date_column(w)

    lower = {c.lower(): c for c in w.columns}
    def has(k): return k in lower
    def col(k): return lower[k]

    rename = {}
    if has("temp_min") and not has("tmin"):
        rename[col("temp_min")] = "tmin"
    if has("temp_max") and not has("tmax"):
        rename[col("temp_max")] = "tmax"
    if has("precipitation_mm") and not has("prcp"):
        rename[col("precipitation_mm")] = "prcp"
    if has("prcp_mm") and not has("prcp"):
        rename[col("prcp_mm")] = "prcp"
    if has("taverage") and not has("tavg"):
        rename[col("taverage")] = "tavg"
    w.rename(columns=rename, inplace=True)

    for c in ["tavg", "tmin", "tmax", "prcp"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    if {"tmax", "tmin"}.issubset(w.columns):
        w["temp_range"] = (w["tmax"] - w["tmin"])
    else:
        w["temp_range"] = np.nan

    w["is_rainy"] = (
        pd.to_numeric(w.get("prcp", np.nan), errors="coerce").fillna(0) > 0
    ).astype("Int64")

    w["is_hot_day"] = (
        pd.to_numeric(w.get("tmax", np.nan), errors="coerce") > HOT_DAY_THRESHOLD_C
    ).astype("Int64")

    keep = ["date", "tavg", "tmin", "tmax", "prcp", "temp_range", "is_rainy", "is_hot_day"]
    for c in keep:
        if c not in w.columns:
            w[c] = np.nan

    w = w[keep].dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return w


# =============================================================================
# GitHub ARTIFACT DOWNLOAD (update_crime.py benzeri)
# =============================================================================
def _gh_headers():
    if not GH_TOKEN:
        return None
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def fetch_file_from_latest_artifact(pick_names: List[str], artifact_name: str = ARTIFACT_NAME) -> Optional[bytes]:
    """
    Son başarılı Actions run’ının artifact’ından pick_names’teki ilk eşleşeni döndürür.
    GH_TOKEN yoksa None döner.
    """
    hdr = _gh_headers()
    if not hdr:
        return None

    try:
        runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page={ARTIFACT_MAX_RUNS}"
        runs = requests.get(runs_url, headers=hdr, timeout=30).json()
        run_ids = [
            r["id"] for r in runs.get("workflow_runs", [])
            if r.get("conclusion") == "success"
        ]

        for rid in run_ids:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=hdr, timeout=30).json().get("artifacts", [])

            for a in arts:
                if a.get("name") == artifact_name and not a.get("expired", False):
                    dl = requests.get(a["archive_download_url"], headers=hdr, timeout=60)

                    zf = zipfile.ZipFile(io.BytesIO(dl.content))
                    names = zf.namelist()

                    # tam ad + crime_prediction_data/ altı için dene
                    for pick in pick_names:
                        for c in (pick, f"crime_prediction_data/{pick}", f"sf-crime-pipeline-output/{pick}"):
                            if c in names:
                                return zf.read(c)

                    # suffix eşleşmesi
                    for n in names:
                        if any(n.endswith(p) for p in pick_names):
                            return zf.read(n)

        return None
    except Exception:
        return None


# =============================================================================
# WEATHER DOWNLOAD ORDER (artifact → release → local)
# =============================================================================
def ensure_local_weather_csv() -> Optional[Path]:
    """
    Tercih sırası:
      1) Actions artifact içinden sf_weather_5years.csv indir
      2) releases/latest download
      3) local fallback candidates
    """
    # 1) artifact-first
    print("📦 Weather artifact kontrolü...")
    art_bytes = fetch_file_from_latest_artifact(["sf_weather_5years.csv"])
    if art_bytes:
        outp = Path(BASE_DIR) / "sf_weather_5years.csv"
        outp.write_bytes(art_bytes)
        print(f"✅ Weather artifact indirildi → {outp}")
        return outp

    # 2) releases/latest fallback
    if WEATHER_RELEASE_URL:
        try:
            outp = Path(BASE_DIR) / "sf_weather_5years.csv"
            print(f"⬇️ Weather releases/latest indiriliyor → {WEATHER_RELEASE_URL}")
            r = requests.get(WEATHER_RELEASE_URL, timeout=60)
            r.raise_for_status()
            outp.write_bytes(r.content)
            print(f"✅ Weather releases indirildi → {outp}")
            return outp
        except Exception as e:
            print(f"⚠️ Weather releases indirilemedi: {e}")

    # 3) local fallback
    lp = pick_existing(WEATHER_LOCAL_CANDS)
    if lp:
        print(f"📂 Weather yerelden bulundu → {lp}")
        return Path(lp)

    print("❌ Weather bulunamadı (artifact/release/local).")
    return None


# =============================================================================
# MAIN
# =============================================================================
# Crime input seç
CRIME_IN = pick_existing(CRIME_IN_CANDS)
if not CRIME_IN:
    raise FileNotFoundError("❌ Suç girdisi bulunamadı: fr_crime_07.csv / sf_crime_07.csv / fr_crime.csv / sf_crime.csv")

# Weather input indir / seç
WEATHER_PATH = ensure_local_weather_csv()
if WEATHER_PATH is None:
    raise FileNotFoundError("❌ Weather dosyası indirilemedi/ bulunamadı.")

# Çıkış dosyası prefix
fname = os.path.basename(CRIME_IN)
prefix = "fr" if fname.startswith("fr_") else ("sf" if fname.startswith("sf_") else "fr")
CRIME_OUT = os.path.join(BASE_DIR, f"{prefix}_crime_08.csv")

# Load crime
print(f"📥 Suç: {CRIME_IN}")
crime = pd.read_csv(CRIME_IN, low_memory=False)
log_shape(crime, "CRIME (yükleme)")

crime = normalize_date_column(crime)
if "date" not in crime.columns:
    raise KeyError("❌ Crime verisinde tarih alanı türetilemedi.")

# Load & normalize weather
print(f"📥 Weather: {WEATHER_PATH}")
weather_raw = pd.read_csv(WEATHER_PATH, low_memory=False)
weather = normalize_weather_columns(weather_raw)
log_shape(weather, "WEATHER (normalize)")

# Prefix
wx = weather.rename(columns={c: (f"wx_{c}" if c != "date" else c) for c in weather.columns})

# Merge
before = crime.shape
out = crime.merge(wx, on="date", how="left", validate="m:1")
log_delta(before, out.shape, "CRIME ⨯ WEATHER (date-merge)")

# Debug last45
try:
    out_dt = pd.to_datetime(out["date"], errors="coerce")
    last45 = out.loc[out_dt >= (out_dt.max() - pd.Timedelta(days=45))]
    if len(last45) > 0:
        wx_cols = [c for c in out.columns if c.startswith("wx_")]
        null_rate = last45[wx_cols].isna().mean().mean() if wx_cols else 1.0
        print(f"🧪 Son 45 gün wx null oranı: {null_rate:.3f}")
        print("🧪 Son 3 tarih wx preview:")
        show_cols = ["date"] + [c for c in ["wx_tavg","wx_tmin","wx_tmax","wx_prcp"] if c in out.columns]
        print(out.sort_values("date").tail(3)[show_cols].to_string(index=False))
except Exception as e:
    print(f"(info) Son 45 gün debug atlandı: {e}")

# Save
safe_save_csv(out, CRIME_OUT)
log_shape(out, "CRIME (weather enrich sonrası)")
print(f"✅ Yazıldı: {CRIME_OUT} | Satır: {len(out):,} | Sütun: {out.shape[1]}")

# kısa örnek
try:
    cols = ["date","wx_tmin","wx_tmax","wx_prcp","wx_temp_range","wx_is_rainy","wx_is_hot_day"]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].head(3).to_string(index=False))
except Exception as e:
    print(f"(info) Önizleme yazdırılamadı: {e}")
