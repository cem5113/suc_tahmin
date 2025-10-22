# update_bus_fr.py 

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
CRIME_INPUT_NAME   = os.getenv("CRIME_INPUT_NAME", "fr_crime_03.csv")
CRIME_OUTPUT_NAME  = os.getenv("CRIME_OUTPUT_NAME", "fr_crime_04.csv")
CRIME_IN_PATHS = [
    ARTIFACT_DIR / CRIME_INPUT_NAME,
    BASE_DIR / CRIME_INPUT_NAME,
    Path(CRIME_INPUT_NAME),
]
BUS_PATH_ENV = os.getenv("BUS_CANON_RAW", "sf_bus_stops_with_geoid.csv")
BUS_IN_PATHS = [
    ARTIFACT_DIR / BUS_PATH_ENV,
    BASE_DIR / BUS_PATH_ENV,
    Path(BUS_PATH_ENV),
]

# ============================== Run ==============================
def main():
    # 0) unzip (varsa)
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) giriş dosyaları
    crime_path = pick(CRIME_IN_PATHS)
    if not crime_path:
        raise FileNotFoundError(f"❌ CRIME bulunamadı: {CRIME_INPUT_NAME}")
    bus_path = pick(BUS_IN_PATHS)
    if not bus_path:
        raise FileNotFoundError(f"❌ BUS dosyası bulunamadı: {BUS_PATH_ENV} (artifact/yerel)")

    log(f"📥 crime: {crime_path}")
    log(f"📥 bus  : {bus_path}")

    # 2) oku (tamamen string) ve GEOID’leri 11 haneye normalize et
    crime = pd.read_csv(crime_path, low_memory=False, dtype=str)
    bus   = pd.read_csv(bus_path,   low_memory=False, dtype=str)

    c_geoid = _find_geoid_col(crime)
    b_geoid = _find_geoid_col(bus)
    if not c_geoid: raise RuntimeError("Suç veri setinde GEOID kolonu yok.")
    if not b_geoid: raise RuntimeError("Otobüs verisinde GEOID kolonu yok.")

    crime["_key"] = _key_11(crime[c_geoid])
    bus["_key"]   = _key_11(bus[b_geoid])

    # Final çıktıda GEOID sütununu tek, string ve 11 hane tut
    # (mevcut GEOID’leri kaldırıp _key’i resmi GEOID yapacağız)
    crime.drop(columns=[c for c in crime.columns if c.lower().startswith("geoid")], errors="ignore", inplace=True)
    crime.insert(0, "GEOID", crime["_key"].astype("string"))
    crime.drop(columns=["_key"], inplace=True)

    # 3) BUS özellikleri (GEOID-temelli)
    # Eğer dosyada zaten GEOID-özet sütunları varsa doğrudan kullan; yoksa en azından stop sayısını üret.
    # Aşağıdaki potansiyel sütunları toparla:
    # - bus_stop_count
    # - distance_to_bus_m
    # - bus_within_300m, bus_within_600m, bus_within_900m, ...
    bus_feats_candidates = [
        "bus_stop_count", "distance_to_bus_m",
        "bus_within_300m", "bus_within_600m", "bus_within_900m",
        "bus_within_250m", "bus_within_500m", "bus_within_750m", "bus_within_1000m",
    ]
    present_cols = [c for c in bus_feats_candidates if c in bus.columns]

    # Eğer bu sütunlar yoksa, en azından GEOID başına stop say
    if not present_cols:
        # satır satır durak tablosuysa: her satır = bir durak ⇒ count
        bus_grp = (bus
                   .groupby("_key", as_index=False)
                   .size()
                   .rename(columns={"size":"bus_stop_count"}))
        bus_agg = bus_grp
    else:
        # bu sütunlar varsa, GEOID bazında tekilleştir (mean/sum mantığı):
        agg_dict = {}
        for c in present_cols:
            # isimden tümevarım: içinde 'count' veya 'within' geçiyorsa sum; 'distance' ise min
            if re.search(r"(count|within)", c, flags=re.I):
                agg_dict[c] = "sum"
            elif re.search(r"(dist|distance)", c, flags=re.I):
                agg_dict[c] = "min"
            else:
                # emin olamıyorsak sum ile ilerleyelim (0/1 veya sayısal metriklerin çoğu için güvenli)
                agg_dict[c] = "sum"

        bus["_key"] = _key_11(bus[b_geoid])  # garanti
        bus_num = bus.copy()
        # sayısal yap (hata → NaN → 0)
        for c in present_cols:
            bus_num[c] = pd.to_numeric(bus_num[c], errors="coerce")
        bus_agg = (bus_num
                   .groupby("_key", as_index=False)
                   .agg(agg_dict))

        # Yoksa stop sayısını da ekle (güvenlik için)
        if "bus_stop_count" not in bus_agg.columns:
            add_cnt = (bus
                       .groupby("_key", as_index=False)
                       .size()
                       .rename(columns={"size":"bus_stop_count"}))
            bus_agg = bus_agg.merge(add_cnt, on="_key", how="outer")

    # 4) Merge (GEOID-temelli)
    # Crime tekrar _key hazırlayıp birleştirelim (üstte GEOID’i set ettik)
    crime["_key"] = crime["GEOID"].astype(str)
    out = crime.merge(bus_agg.rename(columns={"_key":"GEOID"}), on="GEOID", how="left")

    # 5) Tip ve NaN doldurma
    # Varsayılan: sayılabilir kolonları 0 ile doldur
    for c in ["bus_stop_count"] + [col for col in out.columns if re.search(r"bus_within_\\d+m", col or "", flags=re.I)]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int64")

    if "distance_to_bus_m" in out.columns:
        out["distance_to_bus_m"] = pd.to_numeric(out["distance_to_bus_m"], errors="coerce")

    # Son görünüm: GEOID string, 11 hane (asla float değil)
    out["GEOID"] = out["GEOID"].astype("string")

    # 6) Kaydet
    out_path = str(BASE_DIR / CRIME_OUTPUT_NAME)
    ensure_parent(out_path)
    out.to_csv(out_path, index=False)
    log(f"✅ Kaydedildi → {out_path}")

    # 7) Hızlı özet
    try:
        r, c = out.shape
        have_cols = [c for c in ["bus_stop_count","distance_to_bus_m","bus_within_300m","bus_within_600m","bus_within_900m"] if c in out.columns]
        log(f"📊 satır={r:,} sütun={c} | eklenen={have_cols}")
        with pd.option_context("display.max_columns", 80, "display.width", 1600):
            prev_cols = ["GEOID"] + have_cols[:3]
            prev_cols = [col for col in prev_cols if col in out.columns]
            log(out[prev_cols].head(10).to_string(index=False))
    except Exception as e:
        log(f"ℹ️ Önizleme atlandı: {e}")

if __name__ == "__main__":
    main()
