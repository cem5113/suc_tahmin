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

# --- GEOID temizleyici: scientific notation & float görünümünü düzelt ---
def _clean_geoid_scalar(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # Bilimsel gösterim / float (ör: 6.0755980501E10, 60755980501.0)
    try:
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?([eE][+\-]?[0-9]+)?", s):
            as_int = int(float(s))
            return str(as_int)
    except Exception:
        pass
    # Genel durum: rakam dışını at
    return re.sub(r"\D+", "", s)

def _digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).map(_clean_geoid_scalar).fillna("")

def _mode_len(series: pd.Series) -> int:
    if series.empty: return 11
    L = series.astype(str).str.len()
    m = L.mode(dropna=True)
    return int(m.iloc[0]) if not m.empty else int(L.dropna().median())

def _key(series: pd.Series, L: int) -> pd.Series:
    # Yalnızca rakamları al; kısa ise zfill, uzun ise kırp
    s = _digits_only(series)
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
    """
    Crosswalk CSV için muhtemel kolon setleri:
      - 'geoid_from','geoid_to'
      - 'tract2010','tract2020'
      - 'geoid10','geoid20'
    Dönüş: (src_col, dst_col) veya (None,None)
    """
    candidates = [
        ("geoid_from","geoid_to"), ("tract2010","tract2020"),
        ("geoid10","geoid20"), ("GEOID10","GEOID20"), ("from","to")
    ]
    cols = set(df.columns)
    for a,b in candidates:
        if a in cols and b in cols:
            return a,b
    # zayıf sezgi: içinde 2010/2020 geçen ilk iki kolon
    c2010 = [c for c in df.columns if "2010" in c]
    c2020 = [c for c in df.columns if "2020" in c]
    if c2010 and c2020:
        return c2010[0], c2020[0]
    return None, None

def _apply_crosswalk_if_provided(pp: pd.DataFrame, key_col: str, join_len: int) -> pd.DataFrame:
    """
    Env: CROSSWALK_CSV varsa pp['_key'] değerlerini dönüştürür.
    Crosswalk haritası tract→tract düzeyinde olmalı (11 hane).
    """
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

# Birleşim anahtar uzunluğu (tract için 11)
JOIN_LEN      = int(os.getenv("JOIN_LEN", "11"))

# SF'ye filtre için prefiks; boş ise filtre uygulanmaz
SF_PREFIX     = os.getenv("SF_PREFIX_FILTER", "06075").strip()

# İsteğe bağlı: nüfus kolonunu zorla
FORCE_POP_COL = os.getenv("POPULATION_COL", "").strip() or None

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
crime = pd.read_csv(crime_path, low_memory=False, dtype=str)
pop   = pd.read_csv(pop_path,   low_memory=False, dtype=str)

crime_geoid_col = _find_geoid_col(crime)
if not crime_geoid_col: raise RuntimeError("Suç veri setinde GEOID kolonu yok.")
pop_geoid_col = _find_geoid_col(pop)
if not pop_geoid_col:  raise RuntimeError("Nüfus CSV’de GEOID kolonu yok.")
pop_val_col   = _find_population_col(pop, FORCE_POP_COL)
if not pop_val_col:    raise RuntimeError("Nüfus CSV’de nüfus değer kolonu yok (population/B01003_001E/estimate/...).")

crime_len = _mode_len(_digits_only(crime[crime_geoid_col]))
pop_len   = _mode_len(_digits_only(pop[pop_geoid_col]))
log(f"[info] crime GEO len≈{crime_len} | pop GEO len≈{pop_len} | join_len={JOIN_LEN} (tract)")
log(f"[info] pop_val_col={pop_val_col!r}")

# ============================== Prepare POP ==============================
pp0 = pop[[pop_geoid_col, pop_val_col]].copy()
pp0["_key"] = _key(pp0[pop_geoid_col], JOIN_LEN)
pp0["population"] = _num(pp0[pop_val_col]).fillna(0)

# Teşhis: Prefiks dağılımı (ilk 5 hane)
try:
    pref_counts = (pp0["_key"].str[:5].value_counts().head(12))
    log("[diag] POP _key ilk5 prefiks (top-12):")
    for k, v in pref_counts.items():
        log(f"   • {k}: {v}")
except Exception as _e:
    log(f"[diag] Prefiks sayımı atlandı: {_e}")

pp = pp0

# Opsiyonel: SF prefiksi ile filtre (gürültüyü azaltır)
if SF_PREFIX:
    before = len(pp)
    pp = pp[pp["_key"].str.startswith(SF_PREFIX)]
    log(f"[info] POP SF filtresi: prefix={SF_PREFIX} | {before:,} → {len(pp):,}")
    # Otomatik geri dönüş: eğer filtre hepsini sildiyse, filtreyi kaldır (pp0'a dön)
    if len(pp) == 0 and before > 0:
        log("⚠️ SF_PREFIX filtresi tüm nüfusu eledi. Otomatik olarak filtresiz moda dönüyorum.")
        pp = pp0.copy()

# Crosswalk (opsiyonel): varsa uygula (2010↔2020 dönüşümü)
pp = _apply_crosswalk_if_provided(pp, key_col="_key", join_len=JOIN_LEN)

# Pop seviyesi tract üstü ise 11'e aggregate (sum)
pop_len_checked = _mode_len(_digits_only(pop[pop_geoid_col]))
if pop_len_checked > JOIN_LEN:
    pp = pp.groupby("_key", as_index=False)["population"].sum()
else:
    pp = pp[["_key","population"]].drop_duplicates("_key", keep="last")

# ============================== Prepare CRIME ==============================
cc = crime.copy()
cc["_key"] = _key(cc[crime_geoid_col], JOIN_LEN)

# --- Teşhis: Anahtar kümeleri ve kesişim ---
left_keys  = set(cc["_key"].unique())
right_keys = set(pp["_key"].unique())
inter      = left_keys & right_keys
log(f"[debug] left_keys={len(left_keys):,} | right_keys={len(right_keys):,} | intersection={len(inter):,}")

if len(inter) == 0:
    samp_left  = list(sorted(left_keys))[:5]
    samp_right = list(sorted(right_keys))[:5]
    log(f"[debug] örnek left keys:  {samp_left}")
    log(f"[debug] örnek right keys: {samp_right}")
    log("⚠️ Hiç eşleşme yok: Muhtemelen GEOID scientific-notation / vintaj (2010↔2020) farkı veya SF dışı pop kaynağı.")

# ============================== Merge & GEOID Temizliği ==============================
out = cc.merge(pp, how="left", on="_key")

# Tüm GEOID türevlerini at ve en başa 11 hanelik string GEOID koy
geoid_like_cols = [c for c in out.columns if c.lower().startswith("geoid")]
out.drop(columns=[c for c in geoid_like_cols if c != "_key"], errors="ignore", inplace=True)

# Tek resmi GEOID sütunu:
out.insert(0, "GEOID", out["_key"].astype("string"))
out.drop(columns=["_key"], inplace=True)

# Tip güvenliği: GEOID -> string (11 hane)
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
