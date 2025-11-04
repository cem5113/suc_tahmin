# update_population_fr.py — fr_crime_02 + (sf_crime_03.csv | sf_population.csv) → fr_crime_03
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
        cl = c.lower()
        if ("pop" in cl) or ("population" in cl) or (cl == "value"): return c
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

# 1) Birincil kaynak: sf_crime_03.csv (update_population.py çıktısı)
PRIMARY_POP_SOURCE = first_existing([
    BASE_DIR / "sf_crime_03.csv",
    Path("sf_crime_03.csv"),
])

# 2) İkincil (fallback): sf_population.csv (BASE_DIR / artifact / cwd)
POPULATION_CSV = None
if not PRIMARY_POP_SOURCE:
    pop_candidates = [
        BASE_DIR / "sf_population.csv",
        ARTIFACT_DIR / "sf_population.csv",
        Path("sf_population.csv"),
    ]
    POPULATION_CSV = first_existing(pop_candidates)
    if not POPULATION_CSV and ARTIFACT_ZIP.exists():
        extracted = safe_unzip_single(str(ARTIFACT_ZIP), "sf_population.csv", str(ARTIFACT_DIR))
        if extracted and Path(extracted).exists():
            POPULATION_CSV = extracted

if not PRIMARY_POP_SOURCE and not POPULATION_CSV:
    raise FileNotFoundError("Nüfus kaynağı bulunamadı: sf_crime_03.csv ya da sf_population.csv gerekli.")

# ---------- read crime ----------
crime = pd.read_csv(CRIME_INPUT, low_memory=False, dtype=str)
g_crime = find_geoid_col(crime)
if not g_crime:
    raise RuntimeError("Suç verisinde GEOID kolonu yok.")

log(f"📥 CRIME: {CRIME_INPUT}  ({len(crime):,} satır)")

# ---------- build population mapping ----------
if PRIMARY_POP_SOURCE:
    # sf_crime_03.csv içinden (zaten population birleşmiş) GEOID11 → population haritasını çıkar
    src = PRIMARY_POP_SOURCE
    dfp = pd.read_csv(src, low_memory=False, dtype=str)
    g_src = find_geoid_col(dfp)
    if not g_src:
        raise RuntimeError("sf_crime_03.csv içinde GEOID kolonu bulunamadı.")
    if "population" not in dfp.columns:
        raise RuntimeError("sf_crime_03.csv içinde 'population' kolonu bekleniyordu fakat yok.")
    log(f"📥 POP (primary): {src}  ({len(dfp):,} satır)")

    dfp["_GEOID11"] = normalize_geoid11(dfp[g_src])
    # Nüfus sayısal
    pop_num = (
        dfp["population"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .astype(float)
        .astype("Int64")
    )
    dfp["_population_num"] = pop_num

    # GEOID11 bazında ilk dolu nüfus değerini al (aynı tract birden çok satır olabilir)
    pop_map = (
        dfp.loc[~dfp["_population_num"].isna(), ["_GEOID11", "_population_num"]]
        .drop_duplicates("_GEOID11")
        .rename(columns={"_population_num": "population"})
    )
else:
    # Eski davranış: sf_population.csv'den oku ve 12→11 kırpıp topla
    src = POPULATION_CSV
    pop   = pd.read_csv(src, low_memory=False, dtype=str)
    g_pop = find_geoid_col(pop)
    if not g_pop:
        raise RuntimeError("Nüfus CSV’de GEOID kolonu yok.")
    c_pop = find_pop_col(pop)
    if not c_pop:
        raise RuntimeError("Nüfus CSV’de nüfus değeri kolonu yok.")
    log(f"📥 POP (fallback CSV): {src}  ({len(pop):,} satır)")
    pp = pop[[g_pop, c_pop]].copy()
    pp["_GEOID11"] = normalize_geoid11(pp[g_pop])
    pp["population"] = (
        pp[c_pop].astype(str)
          .str.replace(",", "", regex=False)
          .str.replace(" ", "", regex=False)
    )
    pp["population"] = pd.to_numeric(pp["population"], errors="coerce").fillna(0)
    pop_map = pp.groupby("_GEOID11", as_index=False)["population"].sum()

# ---------- merge ----------
crime["_GEOID11"] = normalize_geoid11(crime[g_crime])
before = crime.shape
out = crime.drop(columns=["population"], errors="ignore").merge(pop_map, on="_GEOID11", how="left")
out.drop(columns=["_GEOID11"], inplace=True)
out[g_crime] = normalize_geoid11(out[g_crime])  # GEOID kesin string

# ---------- diagnostics ----------
match_ratio = float(out["population"].notna().mean()) if len(out) else 0.0
log(f"✅ Merge tamam. Eşleşen satır oranı: {match_ratio:.2%}  |  şekil: {before} → {out.shape}")

if match_ratio == 0.0:
    samp_crime = crime[g_crime].head(5).tolist()
    log(f"🧪 Örnek CRIME GEOID’ler: {samp_crime}")
    log("💡 Not: GEOID 11 haneye normalize ediliyor; kaynak 12 haneliyse tract düzeyinde toplanır.")

# ---------- save ----------
ensure_parent(CRIME_OUTPUT)
out.to_csv(CRIME_OUTPUT, index=False)
log(f"💾 Yazıldı → {CRIME_OUTPUT}")

try:
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(out[[g_crime, "population"]].head(8).to_string(index=False))
except Exception:
    pass
