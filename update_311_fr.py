from __future__ import annotations

import os, re, zipfile
from pathlib import Path
from typing import Optional, List

import pandas as pd
import geopandas as gpd

# =========================
# BASIC UTILS
# =========================
def log(msg: str):
    print(msg, flush=True)

def ensure_parent(path: Path | str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: Path | str):
    try:
        ensure_parent(path)
        path = str(path)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic replace
        log(f"💾 Kaydedildi: {path} (satır={len(df):,})")
    except Exception as e:
        b = str(path) + ".bak"
        try:
            df.to_csv(b, index=False)
        except Exception:
            pass
        log(f"❌ Kaydetme hatası: {path} — Yedek: {b}\n{e}")

def to_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

_DEF_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

def normalize_geoid(s: pd.Series, target_len: int = _DEF_GEOID_LEN) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:int(target_len)].str.zfill(int(target_len))

def first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

# =========================
# CONFIG (Artifact ZIP → extract → read)
# =========================
ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped"))  # zip buraya açılır

# Giriş/Çıkış (fr_crime_01 varsa onu kullan; yoksa fr_crime)
FR_BASE_IN_ENV = os.getenv("FR_BASE_IN", "")  # opsiyonel override (tam yol veya dosya adı)
INPUT_FR_01_FILENAME = os.getenv("FR_CRIME_01_FILE", "fr_crime_01.csv")
INPUT_FR_RAW_FILENAME = os.getenv("FR_CRIME_FILE", "fr_crime.csv")

OUTPUT_DIR = Path(os.getenv("FR_OUTPUT_DIR", str(ARTIFACT_DIR)))
OUTPUT_FR_02_FILENAME = os.getenv("FR_CRIME_02_FILE", "fr_crime_02.csv")

# ZIP içinden/klasörden aranan dosyalar (öncelik: unzip edilen klasör)
# Not: artifact içinde farklı isimler kullanıyorsan buraya ekle.
def build_candidates():
    return {
        "FR_311": [
            ARTIFACT_DIR / "sf_311_last_5_years.csv",
            Path("crime_prediction_data") / "sf_311_last_5_years.csv",
            Path("sf_311_last_5_years.csv"),
        ],
        "CENSUS": [
            ARTIFACT_DIR / "sf_census_blocks_with_population.geojson",
            Path("crime_prediction_data") / "sf_census_blocks_with_population.geojson",
            Path("./sf_census_blocks_with_population.geojson"),
        ],
        "FR_BASE": (
            [Path(FR_BASE_IN_ENV)] if FR_BASE_IN_ENV else []
        ) + [
            ARTIFACT_DIR / INPUT_FR_01_FILENAME,
            ARTIFACT_DIR / INPUT_FR_RAW_FILENAME,
            Path("crime_prediction_data") / INPUT_FR_01_FILENAME,
            Path("crime_prediction_data") / INPUT_FR_RAW_FILENAME,
            Path(INPUT_FR_01_FILENAME),
            Path(INPUT_FR_RAW_FILENAME),
        ],
    }

# =========================
# ZIP HELPERS (zip-slip güvenliği)
# =========================
def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))
    except Exception:
        return False

def safe_unzip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        log(f"ℹ️ Artifact ZIP bulunamadı: {zip_path} — klasörlerden denenecek.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"📦 ZIP açılıyor: {zip_path} → {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in zf.infolist():
            out_path = dest_dir / m.filename
            if not _is_within_directory(dest_dir, out_path.parent):
                raise RuntimeError(f"Zip path outside target dir engellendi: {m.filename}")
            if m.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, 'r') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    log("✅ ZIP çıkarma tamam.")

# =========================
# GEO HELPERS (lat/lon → GEOID)
# =========================

def _load_blocks(CENSUS_GEOJSON_CANDIDATES: List[Path]) -> gpd.GeoDataFrame:
    census_path = first_existing(CENSUS_GEOJSON_CANDIDATES)
    if census_path is None:
        raise FileNotFoundError("❌ GEOID poligon dosyası yok: sf_census_blocks_with_population.geojson")

    gdf = gpd.read_file(census_path)
    if "GEOID" not in gdf.columns:
        cand = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID benzeri sütun yok.")
        gdf = gdf.rename(columns={cand[0]: "GEOID"})

    # GEOID uzunluğunu veri içinden türet
    tlen = int(gdf["GEOID"].astype(str).str.len().mode().iat[0])
    gdf["GEOID"] = normalize_geoid(gdf["GEOID"], tlen)

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf[["GEOID", "geometry"]].copy()


def ensure_fr_geoid(fr: pd.DataFrame, CENSUS_GEOJSON_CANDIDATES: List[Path]) -> pd.DataFrame:
    fr = fr.copy()
    if "GEOID" in fr.columns and fr["GEOID"].notna().any():
        fr["GEOID"] = normalize_geoid(fr["GEOID"], _DEF_GEOID_LEN)
        return fr

    # lat/lon alternatif isimler → standardize
    alt = {"lat": "latitude", "y": "latitude", "lon": "longitude", "long": "longitude", "x": "longitude"}
    for k, v in alt.items():
        if k in fr.columns and v not in fr.columns:
            fr[v] = fr[k]

    if not {"latitude", "longitude"}.issubset(fr.columns):
        raise ValueError("❌ fr_crime içinde GEOID yok ve lat/lon bulunamadı.")

    # BBOX kaba filtre (SF)
    min_lon, min_lat, max_lon, max_lat = (-123.2, 37.6, -122.3, 37.9)
    fr = fr[(pd.to_numeric(fr["latitude"], errors="coerce").between(min_lat, max_lat)) &
            (pd.to_numeric(fr["longitude"], errors="coerce").between(min_lon, max_lon))].copy()

    blocks = _load_blocks(CENSUS_GEOJSON_CANDIDATES)
    gdf = gpd.GeoDataFrame(
        fr,
        geometry=gpd.points_from_xy(pd.to_numeric(fr["longitude"]), pd.to_numeric(fr["latitude"])),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf, blocks, how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], _DEF_GEOID_LEN)
    out = out.dropna(subset=["GEOID"]).copy()
    return out

# =========================
# TIME HELPERS (datetime → date, event_hour, hr_key)
# =========================
hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
_TS_CANDS = ["datetime", "timestamp", "occurred_at", "reported_at", "requested_datetime", "date_time"]
_HOUR_CANDS = ["hour", "event_hour", "hr", "Hour"]

def hour_from_range(s: str) -> Optional[int]:
    m = hr_pat.match(str(s))
    return int(m.group(1)) % 24 if m else None


def hour_to_range(h: int) -> str:
    h = int(h) % 24
    a = (h // 3) * 3
    b = min(a + 3, 24)
    return f"{a:02d}-{b:02d}"


def ensure_fr_time_keys(fr: pd.DataFrame) -> pd.DataFrame:
    fr = fr.copy()
    ts_col = next((c for c in _TS_CANDS if c in fr.columns), None)
    if ts_col:
        fr[ts_col] = pd.to_datetime(fr[ts_col], errors="coerce")
        fr["date"] = fr[ts_col].dt.date
        if any(c in fr.columns for c in _HOUR_CANDS):
            hcol = next(c for c in _HOUR_CANDS if c in fr.columns)
            hour = pd.to_numeric(fr[hcol], errors="coerce").fillna(fr[ts_col].dt.hour).astype(int)
        else:
            hour = fr[ts_col].dt.hour.fillna(0).astype(int)
    else:
        if "date" in fr.columns:
            fr["date"] = to_date(fr["date"])
        else:
            log("⚠️ Zaman kolonu yok → 'date' boş kalacak (gevşek join).")
            fr["date"] = pd.NaT
        if any(c in fr.columns for c in _HOUR_CANDS):
            hcol = next(c for c in _HOUR_CANDS if c in fr.columns)
            hour = pd.to_numeric(fr[hcol], errors="coerce").fillna(0).astype(int)
        else:
            hour = pd.Series([0] * len(fr), index=fr.index)

    fr["event_hour"] = (hour % 24).astype("int16")
    fr["hr_key"] = ((fr["event_hour"].astype(int) // 3) * 3).astype("int16")
    return fr

# =========================
# 311 SUMMARY LOAD & STANDARDIZE
# =========================

def load_311_summary_from_candidates(cands_311: List[Path]) -> pd.DataFrame:
    p = first_existing(cands_311)
    if p is None:
        raise FileNotFoundError("❌ 311 özet dosyası bulunamadı (artifact veya klasörler).")
    log(f"📥 311 özeti yükleniyor: {p}")
    df = pd.read_csv(p, low_memory=False, dtype={"GEOID": "string"})

    # kolon adları normalizasyonu
    if "311_request_count" not in df.columns:
        for c in ["count", "requests", "n"]:
            if c in df.columns:
                df = df.rename(columns={c: "311_request_count"})
                break

    # hour_range / event_hour / hr_key kesinleştir
    if "hour_range" not in df.columns and "event_hour" in df.columns:
        df["hour_range"] = df["event_hour"].apply(hour_to_range)

    if "event_hour" not in df.columns and "hour_range" in df.columns:
        df["event_hour"] = df["hour_range"].apply(hour_from_range).astype("Int64")

    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], _DEF_GEOID_LEN)
    if "date" in df.columns:
        df["date"] = to_date(df["date"])

    # hr_key
    if "event_hour" in df.columns:
        df["hr_key"] = ((pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3
    elif "hour_range" in df.columns:
        df["hr_key"] = df["hour_range"].apply(hour_from_range).fillna(0).astype(int)
    else:
        df["hr_key"] = 0

    keep = ["GEOID", "date", "hour_range", "hr_key", "311_request_count"]
    present = [c for c in keep if c in df.columns]
    df = df[present].dropna(subset=["GEOID"]).copy()

    if {"GEOID", "date", "hr_key"}.issubset(df.columns):
        df = (
            df.sort_values(["GEOID", "date", "hr_key"]).drop_duplicates(
                subset=["GEOID", "date", "hr_key"], keep="last"
            )
        )
    log(f"📊 311 özet: {len(df):,} satır")
    return df

# =========================
# MAIN
# =========================

def main():
    # 0) ZIP varsa güvenli şekilde aç
    safe_unzip(ARTIFACT_ZIP, ARTIFACT_DIR)

    # 1) Aday yolları hazırla
    CANDS = build_candidates()

    # 2) 311 özetini yükle
    df311 = load_311_summary_from_candidates(CANDS["FR_311"])

    # 3) fr_crime tabanını yükle
    base_path = first_existing(CANDS["FR_BASE"])
    if base_path is None:
        raise FileNotFoundError("❌ fr_crime tabanı bulunamadı (artifact veya klasörler).")
    base = pd.read_csv(base_path, low_memory=False)
    log(f"📊 fr_base: {len(base):,} satır, {len(base.columns)} sütun — {base_path}")

    # 4) GEOID & zaman anahtarları
    base = ensure_fr_geoid(base, CANDS["CENSUS"])
    base = ensure_fr_time_keys(base)

    # 5) 311 tarafı: join için minimal kolonlar
    cols_311 = ["GEOID", "date", "hour_range", "hr_key", "311_request_count"]
    df311 = df311[[c for c in cols_311 if c in df311.columns]].copy()

    # 6) Birleştirme mantığı (en spesifikten genele)
    if base["date"].notna().any() and {"date", "hr_key"}.issubset(base.columns) and {"date", "hr_key"}.issubset(df311.columns):
        merged = base.merge(df311, on=["GEOID", "date", "hr_key"], how="left")
        join_mode = "GEOID + date + hr_key"
    elif "hr_key" in base.columns and "hr_key" in df311.columns:
        merged = base.merge(df311.drop(columns=["date"], errors="ignore"), on=["GEOID", "hr_key"], how="left")
        join_mode = "GEOID + hr_key"
    else:
        merged = base.merge(
            df311.drop(columns=["date", "hr_key"], errors="ignore").drop_duplicates("GEOID"),
            on=["GEOID"],
            how="left",
        )
        join_mode = "GEOID"

    # 7) NA → 0
    if "311_request_count" in merged.columns:
        merged["311_request_count"] = (
            pd.to_numeric(merged["311_request_count"], errors="coerce").fillna(0).astype("int32")
        )
    else:
        merged["311_request_count"] = 0

    # 8) Yaz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FR_02_FILENAME
    safe_save_csv(merged, out_path)

    log(f"🔗 Join modu: {join_mode}")
    log(f"✅ fr_crime + 311 birleşti → {out_path} (satır={len(merged):,})")

    try:
        preview_cols = [c for c in ["GEOID", "date", "hr_key", "hour_range", "311_request_count"] if c in merged.columns]
        log(merged[preview_cols].head(10).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
