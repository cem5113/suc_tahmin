# update_population.py — GEOID (11 hane, tract) zenginleştirme → sf_crime_03.csv
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ----------------------------- Helpers -----------------------------
def _digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")

def _mode_len(series: pd.Series) -> int:
    if series.empty:
        return 11
    L = series.astype(str).str.len()
    m = L.mode(dropna=True)
    return int(m.iloc[0]) if not m.empty else int(L.dropna().median())

def _key(series: pd.Series, L: int) -> pd.Series:
    s = _digits_only(series).str.replace(" ", "", regex=False)
    return s.str.zfill(L).str[:L]

def _find_geoid_col(df: pd.DataFrame) -> str | None:
    cands = [
        "GEOID","geoid","geo_id","GEOID10","geoid10","GeoID",
        "tract","TRACT","tract_geoid","TRACT_GEOID",
        "geography_id","GEOID2",
    ]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low:
            return low[n.lower()]
    for c in df.columns:
        if "geoid" in c.lower():
            return c
    return None

def _find_population_col(df: pd.DataFrame) -> str | None:
    cands = ["population","pop","total_population","B01003_001E","estimate","total","value"]
    low = {c.lower(): c for c in df.columns}
    for n in cands:
        if n.lower() in low:
            return low[n.lower()]
    for c in df.columns:
        if re.fullmatch(r"(pop.*|.*population.*|value)", c, flags=re.I):
            return c
    return None

def _len_ok(s: pd.Series, L: int) -> float:
    s = s.fillna("").astype(str)
    return float((s.str.len() == L).mean())

def _level_name(L: int) -> str:
    return {5: "county", 11: "tract", 12: "blockgroup", 15: "block"}.get(L, f"L={L}")

def _clean_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

# ----------------------------- Paths/ENV -----------------------------
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Crime input otomatik bul
CRIME_INPUT = os.getenv("CRIME_INPUT", "") or None
if not CRIME_INPUT:
    for p in [BASE_DIR / "sf_crime_02.csv", Path("sf_crime_02.csv"),
              BASE_DIR / "sf_crime.csv",     Path("sf_crime.csv")]:
        if p.exists():
            CRIME_INPUT = str(p); break
if not CRIME_INPUT or not Path(CRIME_INPUT).exists():
    raise FileNotFoundError("CRIME_INPUT bulunamadı. 'sf_crime_02.csv' veya 'sf_crime.csv' gereklidir.")

CRIME_OUTPUT = str(BASE_DIR / "sf_crime_03.csv")

# Population input (yerel CSV veya artifact’tan kopyalanmış CSV)
POPULATION_PATH = (os.getenv("POPULATION_PATH", "") or "").strip()
if not POPULATION_PATH:
    cand = BASE_DIR / "sf_population.csv"
    if cand.exists():
        POPULATION_PATH = str(cand)
    elif Path("sf_population.csv").exists():
        POPULATION_PATH = "sf_population.csv"
    else:
        raise FileNotFoundError("Nüfus CSV bulunamadı (sf_population.csv). POPULATION_PATH ile belirtin.")
else:
    if POPULATION_PATH.startswith(("http://","https://")):
        raise ValueError("CSV-ONLY: POPULATION_PATH yerel bir CSV olmalı (URL kabul edilmez).")
    if not Path(POPULATION_PATH).exists():
        raise FileNotFoundError(f"POPULATION_PATH yok: {POPULATION_PATH}")

# ----------------------------- Read (string!) -----------------------------
# Tüm kolonları string okumak GEOID'nin float'a dönüşmesini engeller
crime = pd.read_csv(CRIME_INPUT, low_memory=False, dtype=str)
crime_geoid_col = _find_geoid_col(crime)
if not crime_geoid_col:
    raise RuntimeError("Suç veri setinde GEOID kolonu bulunamadı.")

pop = pd.read_csv(POPULATION_PATH, low_memory=False, dtype=str)
pop_geoid_col = _find_geoid_col(pop)
if not pop_geoid_col:
    raise RuntimeError("Nüfus CSV’de GEOID kolonu bulunamadı (örn. GEOID/geography_id).")
pop_val_col = _find_population_col(pop)
if not pop_val_col:
    raise RuntimeError("Nüfus CSV’de nüfus değeri kolonu bulunamadı (örn. population/B01003_001E/estimate).")

# ----------------------------- Force join_len=11 -----------------------------
crime_len = _mode_len(_digits_only(crime[crime_geoid_col]))
pop_len   = _mode_len(_digits_only(pop[pop_geoid_col]))
join_len  = 11   # <— TRACT seviyesine zorunlu
print(f"[info] crime GEO len≈{crime_len} | pop GEO len≈{pop_len} | join_len={join_len} ({_level_name(join_len)})")

# ----------------------------- Prep Population -----------------------------
pp = pop[[pop_geoid_col, pop_val_col]].copy()
pp["_key"] = _key(pp[pop_geoid_col], join_len)

pp["population"] = _clean_to_numeric(pp[pop_val_col]).fillna(0)

# pop GEOID seviyesi > 11 ise (ör. 12, 15) → 11'e aggregate (sum)
if pop_len > join_len:
    pp = pp.groupby("_key", as_index=False)["population"].sum()
else:
    # Aynı anahtarlar varsa sonuncuyu al; 11’e upsample etmiyoruz (county vb. ise bire bir eşleşme az olabilir)
    pp = pp[["_key", "population"]].drop_duplicates("_key", keep="last")

# ----------------------------- Prep Crime & Merge -----------------------------
cc = crime.copy()
cc["_key"] = _key(cc[crime_geoid_col], join_len)

# Kalite göstergeleri
ok_pop   = _len_ok(pp["_key"], join_len)
ok_crime = _len_ok(cc["_key"], join_len)
inter = len(set(cc["_key"]).intersection(set(pp["_key"])))
print(f"🔎 GEO normalize: level={_level_name(join_len)} (L={join_len}) | pop_ok={ok_pop:.2%} | crime_ok={ok_crime:.2%}")
print(f"🔎 Kesişim anahtar sayısı: {inter:,}")

out = cc.merge(pp, how="left", on="_key")
out.drop(columns=["_key"], errors="ignore", inplace=True)

# ----------------------------- Save & Logs -----------------------------
Path(CRIME_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(CRIME_OUTPUT, index=False)
print(f"✅ Kaydedildi → {CRIME_OUTPUT}")

try:
    null_rate = out["population"].isna().mean()
    print(f"📊 satır: crime={len(crime):,} | pop={len(pp):,} | out={len(out):,} | population NaN oranı={null_rate:.2%}")
    with pd.option_context("display.max_columns", 60, "display.width", 2000):
        print(out[[crime_geoid_col, "population"]].head(10).to_string(index=False))
except Exception as e:
    print(f"ℹ️ Önizleme atlandı: {e}")

if __name__ == "__main__":
    pass
