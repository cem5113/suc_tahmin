# update_population.py — GEOID (11 hane, tract) zenginleştirme → sf_crime_03.csv
from __future__ import annotations
import os, re, zipfile, csv
from pathlib import Path
import pandas as pd

pd.options.mode.copy_on_write = True

# ============================== Utils ==============================
def log(msg: str): print(msg, flush=True)

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        log(f"ℹ️ Artifact ZIP yok: {zip_path}")
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

def _mode_len(series: pd.Series) -> int:
    if series.empty: return 11
    L = series.astype(str).str.len()
    m = L.mode(dropna=True)
    return int(m.iloc[0]) if not m.empty else int(L.dropna().median())

def _key(series: pd.Series, L: int) -> pd.Series:
    # Yalnızca rakamları al, boşlukları at, L haneye zfill, fazla ise kes
    s = _digits_only(series).str.replace(" ", "", regex=False)
    return s.str.zfill(L).str[:L]

def _find_geoid_col(df: pd.DataFrame) -> str | None:
    cands = ["GEOID","geoid","geo_id","GEOID10","geoid10","GeoID",
             "tract","TRACT","tract_geoid","TRACT_GEOID",
             "geography_id","GEOID2"]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        if "geoid" in c.lower(): return c
    return None

def _find_population_col(df: pd.DataFrame) -> str | None:
    cands = ["population","pop","total_population","B01003_001E","estimate","total","value"]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low: return low[n.lower()]
    for c in df.columns:
        if re.fullmatch(r"(pop.*|.*population.*|value)", c, flags=re.I): return c
    return None

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

# ============================== Config ==============================
BASE_DIR      = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_ZIP  = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR  = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))
CRIME_OUTPUT  = str(BASE_DIR / "sf_crime_03.csv")
JOIN_LEN      = 11  # TRACT

# ZIP varsa aç
safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

# Aday dosyalar (öncelik: artifact_unzipped)
CRIME_CANDS = [
    ARTIFACT_DIR / "fr_crime_02.csv",
    ARTIFACT_DIR / "fr_crime.csv",
    BASE_DIR / "sf_crime_02.csv",
    BASE_DIR / "sf_crime.csv",
    Path("sf_crime_02.csv"),
    Path("sf_crime.csv"),
]
POP_CANDS = [
    Path(os.getenv("POPULATION_PATH")) if os.getenv("POPULATION_PATH") else None,
    ARTIFACT_DIR / "sf_population.csv",
    BASE_DIR / "sf_population.csv",
    Path("sf_population.csv"),
]

def pick(paths): 
    for p in paths:
        if p and Path(p).exists(): return str(p)
    return None

crime_path = pick(CRIME_CANDS)
pop_path   = pick(POP_CANDS)
if not crime_path: raise FileNotFoundError("❌ CRIME CSV bulunamadı (fr_crime_02.csv / sf_crime_02.csv / fr_crime.csv / sf_crime.csv).")
if not pop_path:   raise FileNotFoundError("❌ POPULATION CSV bulunamadı (sf_population.csv).")
log(f"📥 crime: {crime_path}")
log(f"📥 population: {pop_path}")

# ============================== Read (STRING!) ==============================
# Tüm kolonları string olarak okuyalım; böylece otomatik float cast olmaz.
crime = pd.read_csv(crime_path, low_memory=False, dtype=str)
pop   = pd.read_csv(pop_path,   low_memory=False, dtype=str)

crime_geoid_col = _find_geoid_col(crime)
if not crime_geoid_col: raise RuntimeError("Suç veri setinde GEOID kolonu yok.")
pop_geoid_col = _find_geoid_col(pop)
if not pop_geoid_col:  raise RuntimeError("Nüfus CSV’de GEOID kolonu yok.")
pop_val_col   = _find_population_col(pop)
if not pop_val_col:    raise RuntimeError("Nüfus CSV’de nüfus değer kolonu yok (population/B01003_001E/estimate/...).")

crime_len = _mode_len(_digits_only(crime[crime_geoid_col]))
pop_len   = _mode_len(_digits_only(pop[pop_geoid_col]))
log(f"[info] crime GEO len≈{crime_len} | pop GEO len≈{pop_len} | join_len={JOIN_LEN} (tract)")

# ============================== Prepare POP ==============================
pp = pop[[pop_geoid_col, pop_val_col]].copy()
pp["_key"] = _key(pp[pop_geoid_col], JOIN_LEN)
pp["population"] = _num(pp[pop_val_col]).fillna(0)

# Pop seviyesi 12/15 ise 11'e aggregate (sum)
if pop_len > JOIN_LEN:
    pp = pp.groupby("_key", as_index=False)["population"].sum()
else:
    pp = pp[["_key","population"]].drop_duplicates("_key", keep="last")

# ============================== Prepare CRIME ==============================
cc = crime.copy()
cc["_key"] = _key(cc[crime_geoid_col], JOIN_LEN)

# Merge ve GEOID temizliği:
out = cc.merge(pp, how="left", on="_key")

# Tüm GEOID türevlerini at ve en başa 11 hanelik string GEOID koy
geoid_like_cols = [c for c in out.columns if c.lower().startswith("geoid")]
out.drop(columns=[c for c in geoid_like_cols if c != "_key"], errors="ignore", inplace=True)

# Tek resmi GEOID sütunu:
out.insert(0, "GEOID", out["_key"].astype("string"))
out.drop(columns=["_key"], inplace=True)

# Tip güvenliği: GEOID -> string (11 hane), population -> float (veya int)
out["GEOID"] = out["GEOID"].astype("string")

# Ek güvence: tüm GEOID'ler 11 hane mi?
bad = out["GEOID"].fillna("").str.fullmatch(r"\d{11}") == False
if bad.any():
    n_bad = int(bad.sum())
    log(f"⚠️ Uyarı: {n_bad} satırda GEOID 11 hane değil (örn: {out.loc[bad, 'GEOID'].head(3).tolist()})")

# ============================== Save ==============================
Path(CRIME_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
# CSV yazarken sayıları tırnak içine alma; ama GEOID zaten string olduğu için .0 olmaz.
out.to_csv(CRIME_OUTPUT, index=False, na_rep="")

log(f"✅ Kaydedildi → {CRIME_OUTPUT}")
try:
    null_rate = out["population"].isna().mean()
    log(f"📊 satır: crime={len(crime):,} | pop={len(pp):,} | out={len(out):,} | population NaN oranı={null_rate:.2%}")
    with pd.option_context("display.max_columns", 60, "display.width", 1600):
        log(out[["GEOID","population"]].head(10).to_string(index=False))
except Exception as e:
    log(f"ℹ️ Önizleme atlandı: {e}")
