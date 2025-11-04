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

# --- Temel temizleyici ---
def _clean_geoid_scalar(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # Bilimsel/float gösterimler (ör: 6.0755980501E10, 60755980501.0)
    try:
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?([eE][+\-]?[0-9]+)?", s):
            return str(int(float(s)))
    except Exception:
        pass
    # Genel durum: rakam dışını at
    return re.sub(r"\D+", "", s)

def _digits_only(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_clean_geoid_scalar).fillna("")

def _mode_len(series: pd.Series) -> int:
    if series.empty: return 11
    L = series.astype(str).str.len()
    m = L.mode(dropna=True)
    return int(m.iloc[0]) if not m.empty else int(L.dropna().median())

def _key(series: pd.Series, L: int) -> pd.Series:
    """Önce ACS son-11 desenini yakala; yoksa klasik rakam temizliği."""
    s = series.astype(str)
    tail11 = s.str.extract(r'(\d{11})$', expand=False)  # ...US###########
    out = pd.Series(index=s.index, dtype="string")
    has_tail = tail11.notna()
    out[has_tail] = tail11[has_tail].astype("string")
    # kalanlar: rakam temizle + zfill + kırp
    rest = ~has_tail
    if rest.any():
        cleaned = _digits_only(s[rest]).str.zfill(L).str[:L]
        out[rest] = cleaned.astype("string")
    return out.astype("string")

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

def _find_population_col(df: pd.DataFrame, forced: str | None = None) -> str | None:
    if forced and forced in df.columns:
        return forced
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

# ============================== Opsiyonel: Crosswalk ==============================
def _detect_crosswalk_cols(df: pd.DataFrame):
    candidates = [
        ("geoid_from","geoid_to"), ("tract2010","tract2020"),
        ("geoid10","geoid20"), ("GEOID10","GEOID20"), ("from","to")
    ]
    cols = set(df.columns)
    for a,b in candidates:
        if a in cols and b in cols:
            return a,b
    c2010 = [c for c in df.columns if "2010" in c]
    c2020 = [c for c in df.columns if "2020" in c]
    if c2010 and c2020:
        return c2010[0], c2020[0]
    return None, None

def _apply_crosswalk_if_provided(pp: pd.DataFrame, join_len: int) -> pd.DataFrame:
    path = os.getenv("CROSSWALK_CSV", "").strip()
    if not path:
        return pp
    p = Path(path)
    if not p.exists():
        log(f"ℹ️ CROSSWALK_CSV bulunamadı: {p}")
        return pp
    try:
        cw = pd.read_csv(p, dtype=str, low_memory=False)
    except Exception as e:
        log(f"⚠️ Crosswalk okunamadı: {e}")
        return pp
    src_col, dst_col = _detect_crosswalk_cols(cw)
    if not src_col or not dst_col:
        log("⚠️ Crosswalk kolonları tespit edilemedi (örn. geoid10/geoid20).")
        return pp
    cw["_src"] = _key(cw[src_col], join_len)
    cw["_dst"] = _key(cw[dst_col], join_len)
    cw_map = dict(zip(cw["_src"], cw["_dst"]))
    before_unique = pp["_key"].nunique()
    pp = pp.copy()
    pp["_key"] = pp["_key"].map(lambda x: cw_map.get(x, x))
    after_unique = pp["_key"].nunique()
    log(f"[info] Crosswalk uygulandı: uniq_keys {before_unique:,} → {after_unique:,}")
    return pp

# ============================== Config ==============================
BASE_DIR      = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_ZIP  = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR  = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))
CRIME_OUTPUT  = str(BASE_DIR / "sf_crime_03.csv")

JOIN_LEN      = int(os.getenv("JOIN_LEN", "11"))    # tract
SF_PREFIX     = os.getenv("SF_PREFIX_FILTER", "06075").strip()
FORCE_POP_COL = os.getenv("POPULATION_COL", "").strip() or None

# Tract yeniden inşası için FIPS
STATE_FIPS  = os.getenv("STATE_FIPS",  "06").strip().zfill(2)
COUNTY_FIPS = os.getenv("COUNTY_FIPS", "075").strip().zfill(3)

# ZIP varsa aç
safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

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
crime = pd.read_csv(crime_path, low_memory=False, dtype=str)
pop   = pd.read_csv(pop_path,   low_memory=False, dtype=str)

crime_geoid_col = _find_geoid_col(crime)
if not crime_geoid_col: raise RuntimeError("Suç veri setinde GEOID kolonu yok.")
pop_geoid_col = _find_geoid_col(pop)
if not pop_geoid_col:  raise RuntimeError("Nüfus CSV’de GEOID kolonu yok.")
pop_val_col   = _find_population_col(pop, FORCE_POP_COL)
if not pop_val_col:    raise RuntimeError("Nüfus CSV’de nüfus değer kolonu yok (population/B01003_001E/estimate/...).")

crime_len = _mode_len(_key(crime[crime_geoid_col], JOIN_LEN))
pop_len   = _mode_len(_key(pop[pop_geoid_col], JOIN_LEN))
log(f"[info] crime GEO len≈{crime_len} | pop GEO len≈{pop_len} | join_len={JOIN_LEN} (tract)")
log(f"[info] pop_val_col={pop_val_col!r}")

# ============================== Prepare POP ==============================
pp0 = pop[[pop_geoid_col, pop_val_col]].copy()
pp0["_key_raw"] = _digits_only(pp0[pop_geoid_col])
pp0["_key"]     = _key(pp0[pop_geoid_col], JOIN_LEN)
pp0["population"] = _num(pp0[pop_val_col]).fillna(0)

# Teşhis: ham uzunluk ve prefiks dağılımı
try:
    len_mode = int(pp0["_key"].str.len().mode().iat[0])
    log(f"[diag] POP _key length mode: {len_mode}")
except Exception:
    pass
try:
    pref_counts = (pp0["_key"].str[:5].value_counts().head(12))
    log("[diag] POP _key ilk5 prefiks (top-12):")
    for k, v in pref_counts.items():
        log(f"   • {k}: {v}")
except Exception as _e:
    log(f"[diag] Prefiks sayımı atlandı: {_e}")

# ---- Prefiks olmayan tract kodlarını (ör. '980501') 06+075 ile yeniden inşa ----
# Kriter: _key '06' ile başlamıyorsa VE uzunluk <= 6 ise büyük çoğunluğu prefikssizdir.
share_not_06 = (pp0["_key"].str.startswith("06") == False).mean()
median_len   = int(pp0["_key"].str.len().median()) if len(pp0) else 0
if share_not_06 > 0.8 and median_len <= 6:
    log(f"⚙️ Prefikssiz tract tespit edildi (share_not_06={share_not_06:.2f}, median_len={median_len}). "
        f"{STATE_FIPS}+{COUNTY_FIPS}+tract6 ile yeniden inşa ediliyor…")
    tract6 = pp0["_key"].str.zfill(6).str[-6:]
    pp0["_key"] = (STATE_FIPS + COUNTY_FIPS + tract6).astype("string")

# SF prefix filtre (otomatik fallback'lı)
pp = pp0.copy()
if SF_PREFIX:
    before = len(pp)
    pp = pp[pp["_key"].str.startswith(SF_PREFIX)]
    log(f"[info] POP SF filtresi: prefix={SF_PREFIX} | {before:,} → {len(pp):,}")
    if len(pp) == 0 and before > 0:
        log("⚠️ SF_PREFIX filtresi tüm nüfusu eledi. Otomatik olarak filtresiz moda dönüyorum.")
        pp = pp0.copy()

# Crosswalk (opsiyonel)
pp = _apply_crosswalk_if_provided(pp, join_len=JOIN_LEN)

# Aggregation / duplicates
if pp["_key"].str.len().median() > JOIN_LEN:
    pp = pp.groupby("_key", as_index=False)["population"].sum()
else:
    pp = pp[["_key","population"]].drop_duplicates("_key", keep="last")

# ============================== Prepare CRIME ==============================
cc = crime.copy()
cc["_key"] = _key(cc[crime_geoid_col], JOIN_LEN)

# --- Kesişim teşhisi ---
left_keys  = set(cc["_key"].unique())
right_keys = set(pp["_key"].unique())
inter      = left_keys & right_keys
log(f"[debug] left_keys={len(left_keys):,} | right_keys={len(right_keys):,} | intersection={len(inter):,}")
if len(inter) == 0:
    samp_left  = list(sorted(left_keys))[:5]
    samp_right = list(sorted(right_keys))[:5]
    log(f"[debug] örnek left keys:  {samp_left}")
    log(f"[debug] örnek right keys: {samp_right}")
    log("⚠️ Hiç eşleşme yok: Kaynak dosyada prefiks/vintaj farkı olabilir. "
        "POPULATION_PATH doğru SF tract verisine işaret ediyor mu?")

# ============================== Merge & GEOID Temizliği ==============================
out = cc.merge(pp, how="left", on="_key")

geoid_like_cols = [c for c in out.columns if c.lower().startswith("geoid")]
out.drop(columns=[c for c in geoid_like_cols if c != "_key"], errors="ignore", inplace=True)

out.insert(0, "GEOID", out["_key"].astype("string"))
out.drop(columns=["_key"], inplace=True)

out["GEOID"] = out["GEOID"].astype("string")

bad = out["GEOID"].fillna("").str.fullmatch(r"\d{11}") == False
if bad.any():
    n_bad = int(bad.sum())
    log(f"⚠️ Uyarı: {n_bad} satırda GEOID 11 hane değil (örn: {out.loc[bad, 'GEOID'].head(3).tolist()})")

# ============================== Save ==============================
Path(CRIME_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(CRIME_OUTPUT, index=False, na_rep="")

# ============================== Summary Logs ==============================
log(f"✅ Kaydedildi → {CRIME_OUTPUT}")
try:
    null_rate = out["population"].isna().mean()
    match_rate = 1.0 - null_rate
    log(f"📊 satır: crime={len(crime):,} | pop={len(pp):,} | out={len(out):,}")
    log(f"🔗 match_rate={match_rate:.2%} | population NaN oranı={null_rate:.2%}")
    with pd.option_context("display.max_columns", 60, "display.width", 1600):
        log(out[["GEOID","population"]].head(10).to_string(index=False))
except Exception as e:
    log(f"ℹ️ Önizleme atlandı: {e}")
