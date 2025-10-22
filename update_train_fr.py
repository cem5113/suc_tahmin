# update_train_fr.py

from __future__ import annotations
import os, re, zipfile
from pathlib import Path
from typing import Optional, List

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

def _key_11(series: pd.Series) -> pd.Series:
    # GEOID → sadece rakamlar, boşluk sil, 11 haneye zfill, fazlaysa kes.
    s = _digits_only(series).str.replace(" ", "", regex=False)
    return s.str.zfill(11).str[:11]

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

def pick(paths: List[Path | str]) -> Optional[str]:
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ============================== Config ==============================
BASE_DIR      = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")); BASE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_ZIP  = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR  = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))

# I/O
CRIME_INPUT_NAME   = os.getenv("CRIME_INPUT_NAME", "fr_crime_04.csv")
CRIME_OUTPUT_NAME  = os.getenv("CRIME_OUTPUT_NAME", "fr_crime_05.csv")
TRAIN_PATH_ENV     = os.getenv("TRAIN_STOPS_NAME", "sf_train_stops_with_geoid.csv")

CRIME_IN_PATHS = [
    ARTIFACT_DIR / CRIME_INPUT_NAME,
    BASE_DIR / CRIME_INPUT_NAME,
    Path(CRIME_INPUT_NAME),
]
TRAIN_IN_PATHS = [
    ARTIFACT_DIR / TRAIN_PATH_ENV,
    BASE_DIR / TRAIN_PATH_ENV,
    Path(TRAIN_PATH_ENV),
]

# Hangi tren özelliklerini arayalım? (dosyada ne varsa alır)
TRAIN_FEATS_CANDIDATES = [
    "train_stop_count", "distance_to_train_m",
    "train_within_300m", "train_within_600m", "train_within_900m",
    "train_within_250m", "train_within_500m", "train_within_750m", "train_within_1000m",
    "train_0_300m", "train_300_600m", "train_600_900m",
]

# ============================== Run ==============================
def main():
    # 0) unzip (varsa)
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) giriş dosyaları
    crime_path = pick(CRIME_IN_PATHS)
    if not crime_path:
        raise FileNotFoundError(f"❌ CRIME bulunamadı: {CRIME_INPUT_NAME}")
    train_path = pick(TRAIN_IN_PATHS)
    if not train_path:
        raise FileNotFoundError(f"❌ TRAIN dosyası bulunamadı: {TRAIN_PATH_ENV} (artifact/yerel)")

    log(f"📥 crime: {crime_path}")
    log(f"📥 train: {train_path}")

    # 2) oku (tamamen string) ve GEOID’leri 11 haneye normalize et
    crime = pd.read_csv(crime_path, low_memory=False, dtype=str)
    train = pd.read_csv(train_path, low_memory=False, dtype=str)

    c_geoid = _find_geoid_col(crime)
    t_geoid = _find_geoid_col(train)
    if not c_geoid: raise RuntimeError("Suç veri setinde GEOID kolonu yok.")
    if not t_geoid: raise RuntimeError("Tren verisinde GEOID kolonu yok.")

    # Crime: tek, string ve 11 hanelik GEOID’i garanti et
    crime["_key"] = _key_11(crime[c_geoid])
    crime.drop(columns=[c for c in crime.columns if c.lower().startswith("geoid") and c != "_key"],
               errors="ignore", inplace=True)
    crime.insert(0, "GEOID", crime["_key"].astype("string"))
    crime.drop(columns=["_key"], inplace=True)

    # Train: GEOID+özet
    train["_key"] = _key_11(train[t_geoid])

    present_cols = [c for c in TRAIN_FEATS_CANDIDATES if c in train.columns]
    if not present_cols:
        # sadece satır bazlı durak listesi ise: GEOID başına tren durağı say
        train_agg = (train
                     .groupby("_key", as_index=False)
                     .size()
                     .rename(columns={"size":"train_stop_count"}))
    else:
        # sayısal yap, GEOID bazında tekilleştir
        agg_dict = {}
        for c in present_cols:
            # isimden çıkarım: count/within → sum; distance → min
            if re.search(r"(count|within)", c, flags=re.I):
                agg_dict[c] = "sum"
            elif re.search(r"(dist|distance)", c, flags=re.I):
                agg_dict[c] = "min"
            else:
                agg_dict[c] = "sum"
        tr_num = train[["_key"] + present_cols].copy()
        for c in present_cols:
            tr_num[c] = pd.to_numeric(tr_num[c], errors="coerce")
        train_agg = tr_num.groupby("_key", as_index=False).agg(agg_dict)

        # güvenlik: stop sayısını da ekle
        if "train_stop_count" not in train_agg.columns:
            add_cnt = (train.groupby("_key", as_index=False)
                              .size()
                              .rename(columns={"size":"train_stop_count"}))
            train_agg = train_agg.merge(add_cnt, on="_key", how="outer")

    train_agg.rename(columns={"_key":"GEOID"}, inplace=True)

    # 3) Merge (GEOID-temelli, many_to_one)
    out = crime.merge(train_agg, on="GEOID", how="left", validate="many_to_one")

    # 4) Tip/NaN temizlik
    # Int uygun olanları doldur
    for col in out.columns:
        if re.search(r"(count|within)_?\d*m$", col) or col.endswith("train_stop_count"):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("Int64")
    if "distance_to_train_m" in out.columns:
        out["distance_to_train_m"] = pd.to_numeric(out["distance_to_train_m"], errors="coerce")

    # GEOID görünümü: kesin string 11 hane (float .0 mümkün değil)
    out["GEOID"] = out["GEOID"].astype("string")

    # 5) Kaydet & özet
    out_path = str(BASE_DIR / CRIME_OUTPUT_NAME)
    ensure_parent(out_path)
    out.to_csv(out_path, index=False)
    log(f"✅ Kaydedildi → {out_path}")

    try:
        r, c = out.shape
        have_cols = [c for c in ["train_stop_count","distance_to_train_m",
                                 "train_within_300m","train_within_600m","train_within_900m",
                                 "train_0_300m","train_300_600m","train_600_900m"] if c in out.columns]
        log(f"📊 satır={r:,} sütun={c} | eklenen={have_cols}")
        with pd.option_context("display.max_columns", 80, "display.width", 1600):
            prev_cols = ["GEOID"] + have_cols[:4]
            prev_cols = [col for col in prev_cols if col in out.columns]
            log(pd.DataFrame(out[prev_cols].head(10)).to_string(index=False))
    except Exception as e:
        log(f"ℹ️ Önizleme atlandı: {e}")

if __name__ == "__main__":
    main()
