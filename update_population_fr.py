# update_population_fr.py — fr_crime_02 + sf_population → fr_crime_03 (GEOID=11, string)
# Basit GEOID-bazlı merge; artifact’teki sf_population.csv doğrudan kullanılır.

import os
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ----------------- helpers -----------------
def log(msg: str): print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def normalize_geoid11(s: pd.Series) -> pd.Series:
    # sadece rakamları al, SOL’dan 11 hane tut, zfill → STRING döner
    out = s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    return out.str[:11].str.zfill(11)

def find_geoid_col(df: pd.DataFrame) -> str | None:
    cands = ["GEOID","geoid","geo_id","GEOID10","geoid10","GeoID","geography_id","TRACT","tract","tract_geoid"]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        if "geoid" in c.lower(): return c
    return None

def find_pop_col(df: pd.DataFrame) -> str | None:
    cands = ["population","pop","total_population","B01003_001E","estimate","total","value"]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        if any(k in c.lower() for k in ["pop","population"]): return c
    return None

# ----------------- paths -----------------
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

CRIME_INPUT  = (os.getenv("CRIME_INPUT")  or str(BASE_DIR / "fr_crime_02.csv")).strip()
CRIME_OUTPUT = (os.getenv("CRIME_OUTPUT") or str(BASE_DIR / "fr_crime_03.csv")).strip()

# artifact’teki hazır dosya
POPULATION_PATH = (os.getenv("POPULATION_PATH") or str(BASE_DIR / "sf_population.csv")).strip()
if not Path(POPULATION_PATH).exists() and Path("sf_population.csv").exists():
    POPULATION_PATH = "sf_population.csv"

if not Path(CRIME_INPUT).exists():
    raise FileNotFoundError(f"fr_crime_02.csv bulunamadı: {CRIME_INPUT}")
if not Path(POPULATION_PATH).exists():
    raise FileNotFoundError(f"sf_population.csv bulunamadı: {POPULATION_PATH}")

# ----------------- read -----------------
# GEOID’in float’a dönüşmemesi için dtype=str
crime = pd.read_csv(CRIME_INPUT, low_memory=False, dtype=str)
pop   = pd.read_csv(POPULATION_PATH, low_memory=False, dtype=str)

# kolonları tespit et
g_crime = find_geoid_col(crime); 
if not g_crime: raise RuntimeError("Suç verisinde GEOID kolonu bulunamadı.")
g_pop   = find_geoid_col(pop);   
if not g_pop:   raise RuntimeError("Nüfus CSV’de GEOID kolonu bulunamadı.")
c_pop   = find_pop_col(pop);     
if not c_pop:   raise RuntimeError("Nüfus CSV’de nüfus değeri kolonu bulunamadı.")

log(f"📥 CRIME: {CRIME_INPUT}  | satır={len(crime):,}")
log(f"📥 POP  : {POPULATION_PATH}  | satır={len(pop):,}")
log(f"🔎 cols → crime[{g_crime}], pop[{g_pop}], population[{c_pop}]")

# ----------------- normalize & aggregate -----------------
crime["_GEOID11"] = normalize_geoid11(crime[g_crime])

pp = pop[[g_pop, c_pop]].copy()
pp["_GEOID11"] = normalize_geoid11(pp[g_pop])

# nüfusu numeriğe çevir
pp["population"] = (
    pp[c_pop].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(" ", "", regex=False)
)
pp["population"] = pd.to_numeric(pp["population"], errors="coerce").fillna(0)

# Aynı 11 hane için (blockgroup vs) topla
pop11 = pp.groupby("_GEOID11", as_index=False)["population"].sum()

# ----------------- merge -----------------
before = crime.shape
out = crime.drop(columns=["population"], errors="ignore").merge(pop11, on="_GEOID11", how="left")
out.drop(columns=["_GEOID11"], inplace=True)

# GEOID kolonu STRING kalsın (float .0 olmasın)
out[g_crime] = normalize_geoid11(out[g_crime])

# ----------------- save -----------------
ensure_parent(CRIME_OUTPUT)
out.to_csv(CRIME_OUTPUT, index=False)
after = out.shape

log(f"🔗 Merge: {before} → {after}")
log(f"✅ Kaydedildi → {CRIME_OUTPUT}")

# küçük önizleme
try:
    with pd.option_context("display.max_columns", 40, "display.width", 160):
        print(out[[g_crime, "population"]].head(8).to_string(index=False))
except Exception:
    pass
