# update_population_daily.py
from __future__ import annotations
import os, re, zipfile
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
    s = _digits_only(series).str.replace(" ", "", regex=False)
    return s.str.zfill(L).str[:L]

def _find_geoid_col(df: pd.DataFrame) -> str | None:
    cands = ["GEOID","geoid","geo_id","GEOID10","geoid10","GeoID",
             "tract","TRACT","tract_geoid","TRACT_GEOID",
             "geography_id","GEOID2"]
    lower = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in lower: return lower[n.lower()]
    for c in df.columns:
        if "geoid" in c.lower(): return c
    return None

def _find_population_col(df: pd.DataFrame) -> str | None:
    cands = ["population","pop","total_population","B01003_001E","estimate","total","value"]
    lower = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in lower: return lower[n.lower()]
    for c in df.columns:
        if re.fullmatch(r"(pop.*|.*population.*|value)", c, flags=re.I): return c
    return None

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

def _normalize_geoid11(s: pd.Series) -> pd.Series:
    return _digits_only(s).str[:11].str.zfill(11)

def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists(): return str(p)
    return None

# ============================== Config ==============================
BASE_DIR      = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_ZIP  = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR  = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))
JOIN_LEN      = int(os.getenv("POP_JOIN_LEN", "11"))      # tract
POP_FILL_ZERO = os.getenv("POP_FILL_ZERO", "0") == "1"    # eşleşmeyenlerde 0 yaz

# ZIP varsa aç
safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

# Aday dosyalar — günlük akış öncelikli (CRIME)
CRIME_CANDS = [
    BASE_DIR / "daily_crime_02.csv",             # <— öncelik: 311 eklenmiş GÜNLÜK
    ARTIFACT_DIR / "daily_crime_02.csv",
    ARTIFACT_DIR / "fr_crime_02.csv",
    ARTIFACT_DIR / "fr_crime.csv",
    BASE_DIR / "sf_crime_02.csv",
    BASE_DIR / "sf_crime.csv",
    Path("sf_crime_02.csv"),
    Path("sf_crime.csv"),
]

# Nüfus kaynağı adayları
PRIMARY_POP_CANDS = [                          # <— ÖNCE sf_crime_03.csv (update_population.py çıktısı)
    BASE_DIR / "sf_crime_03.csv",
    ARTIFACT_DIR / "sf_crime_03.csv",
    Path("sf_crime_03.csv"),
]
FALLBACK_POP_CANDS = [
    Path(os.getenv("POPULATION_PATH")) if os.getenv("POPULATION_PATH") else None,
    ARTIFACT_DIR / "sf_population.csv",
    BASE_DIR / "sf_population.csv",
    Path("sf_population.csv"),
]

OUT_PATH = BASE_DIR / "daily_crime_03.csv"     # <— günlük çıktı adı (değişmedi)

crime_path = _first_existing(CRIME_CANDS)
if not crime_path: raise FileNotFoundError("❌ CRIME CSV bulunamadı (daily_crime_02.csv / *_crime_02.csv / *_crime.csv).")

primary_pop_path = _first_existing(PRIMARY_POP_CANDS)
fallback_pop_path = None if primary_pop_path else _first_existing(FALLBACK_POP_CANDS)

if not (primary_pop_path or fallback_pop_path):
    raise FileNotFoundError("❌ Nüfus kaynağı yok: sf_crime_03.csv veya sf_population.csv bulunamadı.")

log(f"📥 crime: {crime_path}")
if primary_pop_path:
    log(f"📥 population (primary): {primary_pop_path}  [sf_crime_03.csv]")
else:
    log(f"📥 population (fallback CSV): {fallback_pop_path}  [sf_population.csv]")

# ============================== Read (STRING!) ==============================
crime = pd.read_csv(crime_path, low_memory=False, dtype=str)
crime_geoid_col = _find_geoid_col(crime)
if not crime_geoid_col: raise RuntimeError("Suç veri setinde GEOID kolonu yok.")

# --- Nüfus haritasını kur ---
if primary_pop_path:
    dfp = pd.read_csv(primary_pop_path, low_memory=False, dtype=str)
    g_src = _find_geoid_col(dfp)
    if not g_src:
        raise RuntimeError("sf_crime_03.csv içinde GEOID kolonu bulunamadı.")
    if "population" not in dfp.columns:
        raise RuntimeError("sf_crime_03.csv içinde 'population' kolonu bekleniyordu fakat yok.")
    dfp["_GEOID11"] = _normalize_geoid11(dfp[g_src])

    # numerik yap + ilk dolu değeri seç
    dfp["_pop_num"] = _num(dfp["population"])
    pop_map = (dfp.loc[~dfp["_pop_num"].isna(), ["_GEOID11","_pop_num"]]
                    .drop_duplicates("_GEOID11", keep="first")
                    .rename(columns={"_pop_num":"population"}))
else:
    pop = pd.read_csv(fallback_pop_path, low_memory=False, dtype=str)
    pop_geoid_col = _find_geoid_col(pop)
    if not pop_geoid_col:  raise RuntimeError("Nüfus CSV’de GEOID kolonu yok.")
    pop_val_col   = _find_population_col(pop)
    if not pop_val_col:    raise RuntimeError("Nüfus CSV’de nüfus değer kolonu yok (population/B01003_001E/estimate/...).")

    pop_len   = _mode_len(_digits_only(pop[pop_geoid_col]))
    log(f"[info] pop GEO len≈{pop_len} | join_len={JOIN_LEN} (tract)")
    pp = pop[[pop_geoid_col, pop_val_col]].copy()
    pp["_GEOID11"]   = _key(pp[pop_geoid_col], JOIN_LEN)
    pp["population"] = _num(pp[pop_val_col]).clip(lower=0)
    if pop_len > JOIN_LEN:
        # block(15) / bg(12) → tract(11) topla
        pop_map = pp.groupby("_GEOID11", as_index=False)["population"].sum()
    else:
        pop_map = (pp.sort_values(["_GEOID11"])
                     .drop_duplicates(subset=["_GEOID11"], keep="last")
                     .loc[:, ["_GEOID11","population"]])

# ============================== Prepare CRIME + Merge ==============================
cc = crime.copy()
cc["_GEOID11"] = _key(cc[crime_geoid_col], JOIN_LEN)

out = cc.merge(pop_map, how="left", on="_GEOID11")

# Resmi GEOID kolonunu öne al (mevcut geoid kolonlarını bozma)
# Eğer zaten "GEOID" varsa dokunma; yoksa üret.
if "GEOID" not in out.columns:
    out.insert(0, "GEOID", out["_GEOID11"].astype("string"))

out.drop(columns=["_GEOID11"], inplace=True, errors="ignore")
out["GEOID"] = out["GEOID"].astype("string")

# ============================== NaN Teşhis & Opsiyonel Doldurma ==============================
unmatched_mask = out["population"].isna()
n_unmatched = int(unmatched_mask.sum())
if n_unmatched:
    rate = n_unmatched / len(out) if len(out) else 0
    log(f"⚠️ Uyarı: population eşleşmeyen satır = {n_unmatched:,} (%{rate:.2%})")
    try:
        sample = out.loc[unmatched_mask, ["GEOID"]].head(50)
        sample_path = BASE_DIR / "unmatched_sample.csv"
        sample.to_csv(sample_path, index=False)
        log(f"🧪 unmatched örnek → {sample_path}")
    except Exception:
        pass
    if POP_FILL_ZERO:
        out["population"] = out["population"].fillna(0)

# ============================== Save ==============================
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False, na_rep="")

log(f"✅ Kaydedildi → {OUT_PATH}")
try:
    null_rate = out["population"].isna().mean()
    log(f"📊 out satır={len(out):,} | population NaN oranı={null_rate:.2%} | örnek:")
    with pd.option_context("display.max_columns", 60, "display.width", 1600):
        log(out[["GEOID","population"]].head(10).to_string(index=False))
except Exception as e:
    log(f"ℹ️ Önizleme atlandı: {e}")
