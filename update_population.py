# update_population_simple.py
# Amaç: crime_prediction_data/sf_population.csv içindeki nüfusu,
#       suç CSV'sine (GEOID) göre ekleyip sf_crime_03.csv olarak yazmak.

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import re

pd.options.mode.copy_on_write = True

def log(msg: str): print(msg, flush=True)

def _clean_geoid_scalar(x: str) -> str:
    if x is None: return ""
    s = str(x).strip()
    try:
        # 6.0755980501E10, 60755980501.0 gibi görünümleri düzelt
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?([eE][+\-]?[0-9]+)?", s):
            return str(int(float(s)))
    except Exception:
        pass
    return re.sub(r"\D+", "", s)

def _key(series: pd.Series, L: int = 11) -> pd.Series:
    s = series.astype(str).map(_clean_geoid_scalar).fillna("")
    return s.str.zfill(L).str[:L]

# ---- Yollar (gerekirse env ile değiştir)
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

CRIME_IN  = Path(os.getenv("CRIME_IN",  BASE_DIR / "sf_crime_02.csv"))       # giriş suç CSV
CRIME_OUT = Path(os.getenv("CRIME_OUT", BASE_DIR / "sf_crime_03.csv"))    # çıkış (nüfus eklenmiş)
POP_PATH  = Path(os.getenv("POP_PATH",  BASE_DIR / "sf_population.csv"))  # tek kaynak

# ---- Oku
if not CRIME_IN.exists():
    raise FileNotFoundError(f"❌ Suç CSV yok: {CRIME_IN}")
if not POP_PATH.exists():
    raise FileNotFoundError(f"❌ Nüfus CSV yok: {POP_PATH}")

log(f"📥 crime: {CRIME_IN}")
log(f"📥 population: {POP_PATH}")

crime = pd.read_csv(CRIME_IN, low_memory=False, dtype=str)
pop   = pd.read_csv(POP_PATH, low_memory=False, dtype=str)

# ---- GEOID kolonlarını bul
def _find_geoid_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "geoid" in c.lower() or c.upper().startswith("GEOID"):
            return c
    return "GEOID" if "GEOID" in df.columns else None

crime_geoid = _find_geoid_col(crime)
if not crime_geoid:
    raise RuntimeError("❌ Suç CSV içinde GEOID kolonu bulunamadı.")
if "GEOID" not in pop.columns:
    raise RuntimeError("❌ Nüfus CSV 'GEOID' kolonu içermiyor (beklenen: GEOID,population).")
if "population" not in pop.columns:
    raise RuntimeError("❌ Nüfus CSV 'population' kolonu içermiyor.")

# ---- GEOID’leri 11 haneye normalize et
crime["_key"] = _key(crime[crime_geoid], 11)
pop["_key"]   = _key(pop["GEOID"], 11)

# ---- Sadece gerekli nüfus kolonları
pop_slim = pop[["_key", "population"]].copy()
# population numerik yap (string kalsın istersen bu satır kaldırılabilir)
pop_slim["population"] = pd.to_numeric(pop_slim["population"], errors="coerce")

# ---- Join
before = len(crime)
out = crime.merge(pop_slim, on="_key", how="left")

# Çıkışta tek resmi GEOID kolonu olsun (11 hane)
out.insert(0, "GEOID", out["_key"].astype("string"))
out.drop(columns=["_key"], inplace=True)

# ---- Kaydet
CRIME_OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(CRIME_OUT, index=False)

# ---- Log & örnek satırlar
match_rate = 1.0 - out["population"].isna().mean()
log(f"✅ Kaydedildi → {CRIME_OUT}")
log(f"📊 satır: in={before:,} | out={len(out):,} | match_rate={match_rate:.2%}")
log("🧾 Kolonlar: " + ", ".join(list(out.columns)))

with pd.option_context("display.max_columns", 80, "display.width", 1600):
    log("\n---- HEAD (in-memory) sf_crime_03.csv ----")
    log(out.head(5).to_string(index=False))

# Diskten tekrar okuyup head (isteğe bağlı, sağlaması)
try:
    df_disk = pd.read_csv(CRIME_OUT, low_memory=False)
    with pd.option_context("display.max_columns", 80, "display.width", 1600):
        log("\n---- HEAD (disk) sf_crime_03.csv ----")
        log(df_disk.head(5).to_string(index=False))
except Exception as e:
    log(f"ℹ️ Disk HEAD okunamadı: {e}")
