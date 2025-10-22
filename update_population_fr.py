# update_population_fr.py — fr_crime_02 + sf_population → fr_crime_03 (GEOID=11, string)
import os, zipfile
from pathlib import Path
import pandas as pd

pd.options.mode.copy_on_write = True

# ---------- helpers ----------
def log(msg: str): print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def normalize_geoid11(s: pd.Series) -> pd.Series:
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
        if ("pop" in c.lower()) or ("population" in c.lower()): return c
    return None

def first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None

def safe_unzip_single(zip_path: str, member_name: str, dest_dir: str) -> str | None:
    """ZIP içinden yalnızca 'member_name' dosyasını güvenle çıkar."""
    zp, dd = Path(zip_path), Path(dest_dir)
    if not zp.exists(): return None
    dd.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp, 'r') as zf:
        names = zf.namelist()
        cand = next((n for n in names if n.endswith(member_name)), None)
        if not cand: return None
        out = dd / Path(cand).name
        with zf.open(cand, 'r') as src, open(out, 'wb') as dst:
            dst.write(src.read())
        return str(out)

# ---------- paths ----------
BASE_DIR      = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
ARTIFACT_DIR  = Path(os.getenv("ARTIFACT_DIR", "artifact"))
ARTIFACT_ZIP  = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))

BASE_DIR.mkdir(parents=True, exist_ok=True)

CRIME_INPUT  = (os.getenv("CRIME_INPUT")  or str(BASE_DIR / "fr_crime_02.csv")).strip()
CRIME_OUTPUT = (os.getenv("CRIME_OUTPUT") or str(BASE_DIR / "fr_crime_03.csv")).strip()

if not Path(CRIME_INPUT).exists():
    alt = Path("fr_crime_02.csv")
    if alt.exists(): CRIME_INPUT = str(alt)
if not Path(CRIME_INPUT).exists():
    raise FileNotFoundError(f"fr_crime_02.csv bulunamadı: {CRIME_INPUT}")

# population dosyasını bul
pop_candidates = [
    BASE_DIR / "sf_population.csv",
    ARTIFACT_DIR / "sf_population.csv",
    Path("sf_population.csv"),
]
POPULATION_PATH = first_existing(pop_candidates)

# zip’ten çıkar (gerekirse)
if not POPULATION_PATH and ARTIFACT_ZIP.exists():
    extracted = safe_unzip_single(str(ARTIFACT_ZIP), "sf_population.csv", str(ARTIFACT_DIR))
    if extracted and Path(extracted).exists():
        POPULATION_PATH = extracted

if not POPULATION_PATH:
    raise FileNotFoundError("sf_population.csv bulunamadı (BASE_DIR, ARTIFACT_DIR, ZIP).")

# ---------- read ----------
crime = pd.read_csv(CRIME_INPUT, low_memory=False, dtype=str)
pop   = pd.read_csv(POPULATION_PATH, low_memory=False, dtype=str)

g_crime = find_geoid_col(crime)
if not g_crime: raise RuntimeError("Suç verisinde GEOID kolonu yok.")
g_pop   = find_geoid_col(pop)
if not g_pop:   raise RuntimeError("Nüfus CSV’de GEOID kolonu yok.")
c_pop   = find_pop_col(pop)
if not c_pop:   raise RuntimeError("Nüfus CSV’de nüfus değeri kolonu yok.")

log(f"📥 CRIME: {CRIME_INPUT}  ({len(crime):,} satır)")
log(f"📥 POP  : {POPULATION_PATH}  ({len(pop):,} satır)")
log(f"🔎 cols → crime[{g_crime}], pop[{g_pop}], population[{c_pop}]")

# ---------- normalize & aggregate ----------
crime["_GEOID11"] = normalize_geoid11(crime[g_crime])

pp = pop[[g_pop, c_pop]].copy()
pp["_GEOID11"] = normalize_geoid11(pp[g_pop])

pp["population"] = (
    pp[c_pop].astype(str)
      .str.replace(",", "", regex=False)
      .str.replace(" ", "", regex=False)
)
pp["population"] = pd.to_numeric(pp["population"], errors="coerce").fillna(0)

# 12→11 kırpma durumunda aynı tract’ta topla
pop11 = pp.groupby("_GEOID11", as_index=False)["population"].sum()

# ---------- merge ----------
before = crime.shape
out = crime.drop(columns=["population"], errors="ignore").merge(pop11, on="_GEOID11", how="left")
out.drop(columns=["_GEOID11"], inplace=True)
out[g_crime] = normalize_geoid11(out[g_crime])  # GEOID kesin string

# ---------- diagnostics ----------
match_ratio = float(out["population"].notna().mean()) if len(out) else 0.0
log(f"✅ Merge tamam. Eşleşen satır oranı: {match_ratio:.2%}")
if match_ratio == 0.0:
    # İpucu için ilk 5 GEOID örneği
    samp_crime = crime[g_crime].head(5).tolist()
    samp_pop   = pop[g_pop].head(5).tolist()
    log(f"🧪 Örnek CRIME GEOID’ler: {samp_crime}")
    log(f"🧪 Örnek POP GEOID’ler  : {samp_pop}")
    log("💡 Not: GEOID 11 hane ile kırpılıyor; sf_population blockgroup (12) ise tract (11) toplamı alındı.")

# ---------- save ----------
ensure_parent(CRIME_OUTPUT)
out.to_csv(CRIME_OUTPUT, index=False)
log(f"💾 Yazıldı → {CRIME_OUTPUT}")

try:
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(out[[g_crime, "population"]].head(8).to_string(index=False))
except Exception:
    pass
