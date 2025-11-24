# update_weather_fr.py
# Amaç: Şehir-geneli günlük hava verisini suç verisine sadece "tarih" üzerinden eklemek.
# Girdi: fr_crime_07.csv / sf_crime_07.csv (veya alternatif adaylar)
# Weather girdi kaynağı: Öncelik artifact içindeki sf_weather_5years.csv
# Çıkış: fr_crime_08.csv / sf_crime_08.csv

import os
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True


# ---------------- LOG HELPERS ----------------
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


# ---------------- DATE NORMALIZATION ----------------
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


# ---------------- WEATHER NORMALIZATION ----------------
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

    HOT_DAY_THRESHOLD_C = 25.0

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

    w = w[keep].dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return w


# ---------------- PATHS (artifact öncelikli) ----------------
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(exist_ok=True)

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", os.path.join(BASE_DIR, "artifact"))
ARTIFACT_ZIP = os.getenv("ARTIFACT_ZIP", "").strip()
FALLBACK_DIRS = [p for p in os.getenv("FALLBACK_DIRS", "").split(",") if p.strip()]

# Suç girdi adayları
CRIME_IN_CANDS = [
    os.path.join(BASE_DIR, "fr_crime_07.csv"),
    os.path.join(BASE_DIR, "sf_crime_07.csv"),
    os.path.join(BASE_DIR, "fr_crime.csv"),
    os.path.join(BASE_DIR, "sf_crime.csv"),
]

def pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

CRIME_IN = pick_existing(CRIME_IN_CANDS)
if not CRIME_IN:
    raise FileNotFoundError(
        "❌ Suç girdisi bulunamadı: fr_crime_07.csv / sf_crime_07.csv / fr_crime.csv / sf_crime.csv"
    )

# Weather adayları (ÖNCELİK: artifact/sf-crime-pipeline-output)
WEATHER_CANDS = [
    os.path.join(BASE_DIR, "artifact", "sf-crime-pipeline-output", "sf_weather_5years.csv"),
]

# ARTIFACT_DIR içinde olası yerler
WEATHER_CANDS += [
    os.path.join(ARTIFACT_DIR, "sf-crime-pipeline-output", "sf_weather_5years.csv"),
    os.path.join(ARTIFACT_DIR, "sf_weather_5years.csv"),
]

# FALLBACK_DIRS taraması
for d in FALLBACK_DIRS:
    WEATHER_CANDS += [
        os.path.join(d, "artifact", "sf-crime-pipeline-output", "sf_weather_5years.csv"),
        os.path.join(d, "sf_weather_5years.csv"),
    ]

# klasik adaylar
WEATHER_CANDS += [
    os.path.join(BASE_DIR, "sf_weather_5years.csv"),
    "sf_weather_5years.csv",
    os.path.join(BASE_DIR, "weather.csv"),
    "weather.csv",
]

WEATHER_IN = pick_existing(WEATHER_CANDS)
if not WEATHER_IN:
    raise FileNotFoundError(
        "❌ Weather dosyası bulunamadı (artifact dahil tüm adaylar denendi)."
    )

# Çıkış dosya adı: girdinin prefix'ine göre
fname = os.path.basename(CRIME_IN)
prefix = "fr" if fname.startswith("fr_") else ("sf" if fname.startswith("sf_") else "fr")
CRIME_OUT = os.path.join(BASE_DIR, f"{prefix}_crime_08.csv")


# ---------------- LOAD & MERGE ----------------
print(f"📥 Suç: {CRIME_IN}")
crime = pd.read_csv(CRIME_IN, low_memory=False)
log_shape(crime, "CRIME (yükleme)")

crime = normalize_date_column(crime)
if "date" not in crime.columns:
    raise KeyError("❌ Crime verisinde tarih alanı türetilemedi.")

print(f"📥 Weather: {WEATHER_IN}")
weather_raw = pd.read_csv(WEATHER_IN, low_memory=False)
weather = normalize_weather_columns(weather_raw)
log_shape(weather, "WEATHER (normalize)")

# Weather kolonlarını wx_ ile prefixle
wx = weather.rename(columns={c: (f"wx_{c}" if c != "date" else c) for c in weather.columns})

before = crime.shape
out = crime.merge(wx, on="date", how="left", validate="m:1")
log_delta(before, out.shape, "CRIME ⨯ WEATHER (date-merge)")

# ---------------- DEBUG: son 45 gün doluluk ----------------
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

# ---------------- SAVE ----------------
safe_save_csv(out, CRIME_OUT)
log_shape(out, "CRIME (weather enrich sonrası)")
print(f"✅ Yazıldı: {CRIME_OUT} | Satır: {len(out):,} | Sütun: {out.shape[1]}")

# kısa örnek
try:
    cols = ["date","wx_tmin","wx_tmax","wx_prcp","wx_temp_range","wx_is_rainy","wx_is_hot_day"]
    cols = ["date"] + [c for c in cols if c in out.columns]
    print(out[cols].head(3).to_string(index=False))
except Exception as e:
    print(f"(info) Önizleme yazdırılamadı: {e}")
