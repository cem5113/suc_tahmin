#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_311_fr.py — 311 özetini fr_crime ile birleştirip fr_crime_02.csv üretir.

Özet:
  - Girdi:  (öncelik) fr_crime_01.csv  -> yoksa fr_crime.csv
  - 311:    BASE_DIR, ARTIFACT_DIR, FALLBACK_DIRS altında CSV/Parquet ve isim varyantları
  - Çıkış:  fr_crime_02.csv  (aynı BASE_DIR altında)
"""

from __future__ import annotations

import os, re
from pathlib import Path
from typing import Optional, List

import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Giriş/Çıkış (fr_crime_01 varsa onu kullan; yoksa fr_crime)
FR_BASE_IN_ENV = os.getenv("FR_BASE_IN", "")  # opsiyonel override
FR_BASE_01     = Path(BASE_DIR) / "fr_crime_01.csv"
FR_BASE_RAW    = Path(BASE_DIR) / "fr_crime.csv"
FR_BASE_IN     = Path(FR_BASE_IN_ENV) if FR_BASE_IN_ENV else (FR_BASE_01 if FR_BASE_01.exists() else FR_BASE_RAW)

FR_OUT_PATH    = Path(BASE_DIR) / "fr_crime_02.csv"

# 311 özet arama kökleri ve ad adayları
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifact")
FALLBACK_DIRS = [p.strip() for p in os.getenv("FALLBACK_DIRS", "").split(",") if p.strip()]

_311_NAME_CANDS = [
    "sf_311_last_5_years.csv",
    "sf_311_last_5_year.csv",
    "sf_311_last_5_years_y.csv",
    "sf_311_last_5_years.parquet",
    "sf_311_last_5_year.parquet",
    "sf_311_last_5_years_y.parquet",
]

# GEOID uzunluğu
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ---------------- time utils ----------------
hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
_TS_CANDS = ["datetime","timestamp","occurred_at","reported_at","requested_datetime","date_time"]
_HOUR_CANDS = ["hour","event_hour","hr","Hour"]

def hour_from_range(s: str) -> Optional[int]:
    m = hr_pat.match(str(s))
    return int(m.group(1)) % 24 if m else None

def hour_to_range(h: int) -> str:
    h = int(h) % 24
    a = (h // 3) * 3
    b = min(a + 3, 24)
    return f"{a:02d}-{b:02d}"

# =========================
# HELPERS
# =========================
def log(msg: str):
    print(msg, flush=True)

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: Path):
    ensure_parent(path)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def normalize_geoid(s: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:int(target_len)].str.zfill(int(target_len))

def to_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def pick_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

# =========================
# LOAD 311 SUMMARY (BASE/ARTIFACT/FALLBACK; CSV/PARQUET)
# =========================
def load_311_summary() -> pd.DataFrame:
    roots = [Path(BASE_DIR), Path(ARTIFACT_DIR)] + [Path(p) for p in FALLBACK_DIRS]
    roots = [r.expanduser().resolve() for r in roots if str(r).strip()]

    for r in roots:
        if not r.exists():
            continue
        # İsim adaylarını rglob ile tara
        for name in _311_NAME_CANDS:
            for p in r.rglob(name):
                try:
                    log(f"📥 311 özeti yükleniyor: {p}")
                    if p.suffix.lower() == ".parquet":
                        df = pd.read_parquet(p)
                    else:
                        df = pd.read_csv(p, low_memory=False)
                    # dtype düzeltmeleri
                    if "GEOID" in df.columns:
                        df["GEOID"] = df["GEOID"].astype("string")

                    # kolon adları normalizasyonu
                    if "311_request_count" not in df.columns:
                        for c in ["count","requests","n"]:
                            if c in df.columns:
                                df = df.rename(columns={c:"311_request_count"})
                                break

                    # hour_range / event_hour / hr_key kesinleştir
                    if "hour_range" not in df.columns and "event_hour" in df.columns:
                        df["hour_range"] = df["event_hour"].apply(hour_to_range)

                    if "event_hour" not in df.columns and "hour_range" in df.columns:
                        df["event_hour"] = df["hour_range"].apply(hour_from_range).astype("Int64")

                    if "GEOID" in df.columns:
                        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
                    if "date" in df.columns:
                        df["date"] = to_date(df["date"])

                    # hr_key
                    if "event_hour" in df.columns:
                        df["hr_key"] = ((pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3
                    elif "hour_range" in df.columns:
                        df["hr_key"] = df["hour_range"].apply(hour_from_range).fillna(0).astype(int)
                    else:
                        df["hr_key"] = 0

                    # sade kolon kümesi
                    keep = ["GEOID","date","hour_range","hr_key","311_request_count"]
                    present = [c for c in keep if c in df.columns]
                    df = df[present].dropna(subset=["GEOID"]).copy()

                    # anahtar eşsizliği
                    if {"GEOID","date","hr_key"}.issubset(df.columns):
                        df = (df.sort_values(["GEOID","date","hr_key"])
                                .drop_duplicates(subset=["GEOID","date","hr_key"], keep="last"))

                    log(f"📊 311 özet: {len(df)} satır (kaynak: {p})")
                    return df
                except Exception as e:
                    log(f"⚠️ 311 dosyası okunamadı ({p}): {e}")
                    continue

    raise FileNotFoundError(
        "311 özeti bulunamadı: BASE_DIR/ARTIFACT/FALLBACK içinde "
        "sf_311_last_5_year(s)[_y].(csv|parquet) yok."
    )

# =========================
# GEOID for fr_crime (from lat/lon if needed) — opsiyonel
# =========================
def load_blocks() -> "gpd.GeoDataFrame":
    import geopandas as gpd  # yalnızca gerektiğinde import
    bp_cands = [
        Path(BASE_DIR) / "sf_census_blocks_with_population.geojson",
        Path("./sf_census_blocks_with_population.geojson"),
    ]
    bp = pick_existing(bp_cands)
    if bp is None:
        raise FileNotFoundError("Nüfus blokları GeoJSON yok: sf_census_blocks_with_population.geojson")
    gdf = gpd.read_file(bp)
    if "GEOID" not in gdf.columns:
        cand = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID yok.")
        gdf = gdf.rename(columns={cand[0]:"GEOID"})
    gdf["GEOID"] = normalize_geoid(gdf["GEOID"], DEFAULT_GEOID_LEN)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif getattr(gdf.crs, "to_epsg", lambda: None)() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf[["GEOID","geometry"]].copy()

def ensure_fr_geoid(fr: pd.DataFrame) -> pd.DataFrame:
    # Zaten GEOID varsa normalleştir ve dön
    if "GEOID" in fr.columns and fr["GEOID"].notna().any():
        fr = fr.copy()
        fr["GEOID"] = normalize_geoid(fr["GEOID"], DEFAULT_GEOID_LEN)
        return fr

    # lat/lon alternatif isimleri
    fr = fr.copy()
    alt = {"lat":"latitude","y":"latitude","lon":"longitude","long":"longitude","x":"longitude"}
    for k,v in alt.items():
        if k in fr.columns and v not in fr.columns:
            fr[v] = fr[k]
    if not {"latitude","longitude"}.issubset(fr.columns):
        # GEOID yok ve koordinatlar da yok → olduğu gibi dön (join GEOID’siz kalır)
        log("⚠️ fr_crime tablosunda GEOID yok ve lat/lon bulunamadı. GEOID eşlemesi yapılmayacak.")
        return fr

    # BBOX kaba filtre (SF)
    min_lon, min_lat, max_lon, max_lat = (-123.2, 37.6, -122.3, 37.9)
    fr = fr[
        pd.to_numeric(fr["latitude"],  errors="coerce").between(min_lat, max_lat) &
        pd.to_numeric(fr["longitude"], errors="coerce").between(min_lon, max_lon)
    ].copy()

    # GEOID hesapla (point-in-polygon)
    try:
        import geopandas as gpd
        blocks = load_blocks()  # type: ignore
        gdf = gpd.GeoDataFrame(
            fr,
            geometry=gpd.points_from_xy(pd.to_numeric(fr["longitude"]), pd.to_numeric(fr["latitude"])),
            crs="EPSG:4326"
        )
        joined = gpd.sjoin(gdf, blocks, how="left", predicate="within")
        out = pd.DataFrame(joined.drop(columns=["geometry","index_right"], errors="ignore"))
        out["GEOID"] = normalize_geoid(out["GEOID"], DEFAULT_GEOID_LEN)
        out = out.dropna(subset=["GEOID"]).copy()
        return out
    except Exception as e:
        log(f"⚠️ GEOID çıkarımı atlandı: {e}")
        return fr

# =========================
# TIME KEYS for fr_crime
# =========================
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
            hour = pd.Series([0]*len(fr), index=fr.index)

    fr["event_hour"] = (hour % 24).astype("int16")
    fr["hr_key"] = ((fr["event_hour"].astype(int) // 3) * 3).astype("int16")
    return fr

# =========================
# MAIN
# =========================
def main():
    log(f"📁 Giriş: {FR_BASE_IN}")
    if not FR_BASE_IN.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {FR_BASE_IN}")
    base = pd.read_csv(FR_BASE_IN, low_memory=False)
    log(f"📊 fr_base: {len(base)} satır, {len(base.columns)} sütun")

    # 311 özetini yükle (GEOID bazlı, hazır)
    df311 = load_311_summary()

    # fr_crime tarafında GEOID & zaman anahtarları
    base = ensure_fr_geoid(base)
    base = ensure_fr_time_keys(base)

    # 311 tarafı: join için minimal kolonlar
    cols_311 = ["GEOID","date","hour_range","hr_key","311_request_count"]
    df311 = df311[[c for c in cols_311 if c in df311.columns]].copy()

    # birleştirme anahtarları
    keys_full    = ["GEOID","date","hr_key"]
    keys_geoidhr = ["GEOID","hr_key"]
    keys_geoid   = ["GEOID"]

    # öncelik sırasına göre join
    if base["date"].notna().any() and {"date","hr_key"}.issubset(base.columns) and {"date","hr_key"}.issubset(df311.columns):
        merged = base.merge(df311, on=keys_full, how="left")
        join_mode = "GEOID + date + hr_key"
    elif "hr_key" in base.columns and "hr_key" in df311.columns:
        merged = base.merge(df311.drop(columns=["date"], errors="ignore"), on=keys_geoidhr, how="left")
        join_mode = "GEOID + hr_key"
    else:
        merged = base.merge(
            df311.drop(columns=["date","hr_key"], errors="ignore").drop_duplicates("GEOID"),
            on=keys_geoid,
            how="left",
        )
        join_mode = "GEOID"

    # NA → 0
    if "311_request_count" in merged.columns:
        merged["311_request_count"] = pd.to_numeric(merged["311_request_count"], errors="coerce").fillna(0).astype("int32")
    else:
        merged["311_request_count"] = 0

    # Kaydet
    safe_save_csv(merged, FR_OUT_PATH)
    log(f"🔗 Join modu: {join_mode}")
    log(f"✅ fr_crime + 311 birleşti → {FR_OUT_PATH} ({len(merged)} satır)")

    try:
        with pd.option_context("display.max_columns", 80, "display.width", 1600):
            print(merged.head(5).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
