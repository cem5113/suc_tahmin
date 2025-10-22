# update_911_fr.py
from __future__ import annotations
import os, re
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import geopandas as gpd

# =========================
# BASIC UTILS
# =========================
def log(msg: str): print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
        log(f"💾 Kaydedildi: {path} (satır={len(df):,})")
    except Exception as e:
        b = path + ".bak"
        df.to_csv(b, index=False)
        log(f"❌ Kaydetme hatası: {path} — Yedek: {b}\n{e}")

def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

# =========================
# CONFIG
# =========================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
INPUT_CRIME = os.getenv("FR_CRIME_FILE", "fr_crime.csv")
OUTPUT_MERGED = os.getenv("FR_CRIME_01_FILE", "fr_crime_01.csv")

# 911 özet adayları (önce _y)
FR_911_CANDIDATES = [
    Path(BASE_DIR) / "sf_911_last_5_year_y.csv",
    Path(BASE_DIR) / "sf_911_last_5_year.csv",
]

# GEOID poligonları (Fransa için senin blok/veri kümen)
CENSUS_GEOJSON_CANDIDATES = [
    Path(BASE_DIR) / "sf_census_blocks_with_population.geojson",
    Path("./fr_census_blocks_with_population.geojson"),
]

# Rolling pencere (opsiyonel)
ROLL_WINDOWS = (3, 7)

# =========================
# GEO HELPERS
# =========================
def _load_blocks() -> tuple[gpd.GeoDataFrame, int]:
    census_path = next((p for p in CENSUS_GEOJSON_CANDIDATES if p.exists()), None)
    if census_path is None:
        raise FileNotFoundError("❌ GEOID poligon dosyası yok: fr_census_blocks_with_population.geojson")

    gdf_blocks = gpd.read_file(census_path)
    if "GEOID" not in gdf_blocks.columns:
        cand = [c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID benzeri sütun yok.")
        gdf_blocks = gdf_blocks.rename(columns={cand[0]: "GEOID"})

    # GEOID hedef uzunluğunu veri moduna göre çıkar
    tlen = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
    gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], int(tlen))

    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True)
    elif gdf_blocks.crs.to_epsg() != 4326:
        gdf_blocks = gdf_blocks.to_crs(4326)
    return gdf_blocks, int(tlen)

def ensure_geoid_from_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """fr_crime.csv içinde GEOID yoksa lat/lon → GEOID atar."""
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df

    lat_col = next((c for c in ["latitude", "lat", "y"] if c in df.columns), None)
    lon_col = next((c for c in ["longitude", "lon", "x"] if c in df.columns), None)
    if not lat_col or not lon_col:
        raise ValueError("❌ fr_crime.csv içinde GEOID yok ve lat/lon bulunamadı.")

    gdf_blocks, tlen = _load_blocks()

    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col]).copy()

    pts = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    return out

# =========================
# 911 SUMMARY HELPERS
# =========================
HR_PAT = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")

def _fmt_hr_range(hr):
    m = HR_PAT.match(str(hr))
    if not m:
        return None
    a = int(m.group(1)) % 24
    b = int(m.group(2))
    b = b if b > a else min(a + 3, 24)
    return f"{a:02d}-{b:02d}"

def _hr_key_from_range(hr):
    m = HR_PAT.match(str(hr))
    return int(m.group(1)) % 24 if m else None

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham 911 → GEOID, date, hour_range seviyesinde özetler."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["GEOID", "date", "hour_range",
                                     "911_request_count_hour_range",
                                     "911_request_count_daily(before_24_hours)"])
    df = raw.copy()
    # olası timestamp kolonları
    ts_col = next((c for c in ["received_time","received_datetime","date","datetime",
                               "timestamp","call_received_datetime"] if c in df.columns), None)
    if ts_col is None:
        raise ValueError("911 ham veride zaman kolonu bulunamadı.")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour.fillna(0).astype(int) % 24
    start = (df["event_hour"] // 3) * 3
    df["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")

    grp_hr = (["GEOID"] if "GEOID" in df.columns else []) + ["date", "hour_range"]
    grp_day = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]

    hr_agg = df.groupby(grp_hr, dropna=False).size().reset_index(name="911_request_count_hour_range")
    day_agg = df.groupby(grp_day, dropna=False).size().reset_index(name="911_request_count_daily(before_24_hours)")
    out = hr_agg.merge(day_agg, on=grp_day, how="left")

    cols_tail = [c for c in ["date","hour_range","GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]

def read_911_summary() -> pd.DataFrame:
    src = next((p for p in FR_911_CANDIDATES if p.exists()), None)
    if src is None:
        raise FileNotFoundError("❌ 911 özeti bulunamadı: fr_911_last_5_year_y.csv / fr_911_last_5_year.csv")

    log(f"📥 911 özeti yükleniyor: {src}")
    df = pd.read_csv(src, low_memory=False, dtype={"GEOID": "string"})

    # zaten özet mi?
    is_summary = (
        {"date", "hour_range"}.issubset(df.columns) and
        any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    )

    if is_summary:
        cnt_col = next(c for c in ["911_request_count_hour_range","call_count","count","requests","n"] if c in df.columns)
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})
        df["date"] = to_date(df["date"])
        # hour_range normalize
        df["hour_range"] = df["hour_range"].apply(_fmt_hr_range)
    else:
        df = make_standard_summary(df)

    # GEOID normalize (uzunluğu bloklardan öğren)
    gdf_blocks, tlen = _load_blocks()
    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], tlen)

    # günlük toplam yoksa ekle
    if "911_request_count_daily(before_24_hours)" not in df.columns:
        keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
        day = (df.groupby(keys, dropna=False)["911_request_count_hour_range"]
                 .sum().reset_index(name="911_request_count_daily(before_24_hours)"))
        df = df.merge(day, on=keys, how="left")

    # türev anahtarlar
    df = df.dropna(subset=["date","hour_range"]).copy()
    df["hr_key"] = df["hour_range"].apply(_hr_key_from_range).astype("int16")
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.weekday.astype("int8")
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype("int8")
    _season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                   6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    df["season"] = df["month"].map(_season_map).astype("category")

    # Rolling (opsiyonel ama faydalı)
    day_unique = (df[["GEOID","date","911_request_count_daily(before_24_hours)"]]
                    .drop_duplicates(subset=["GEOID","date"])
                    .rename(columns={"911_request_count_daily(before_24_hours)":"daily_cnt"})
                    .sort_values(["GEOID","date"]).reset_index(drop=True))
    hr_unique = (df.groupby(["GEOID","hr_key","date"], as_index=False)["911_request_count_hour_range"]
                   .sum()
                   .rename(columns={"911_request_count_hour_range":"hr_cnt"})
                   .sort_values(["GEOID","hr_key","date"]).reset_index(drop=True))

    for W in ROLL_WINDOWS:
        day_unique[f"911_geo_last{W}d"] = (
            day_unique.groupby("GEOID")["daily_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")
        hr_unique[f"911_geo_hr_last{W}d"] = (
            hr_unique.groupby(["GEOID","hr_key"])["hr_cnt"].transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

    enriched = df.merge(hr_unique, on=["GEOID","hr_key","date"], how="left")
    enriched = enriched.merge(day_unique, on=["GEOID","date"], how="left")

    return enriched

# =========================
# CRIME + 911 MERGE
# =========================
def main():
    base_dir = Path(BASE_DIR)
    crime_path = base_dir / INPUT_CRIME
    out_path = base_dir / OUTPUT_MERGED

    if not crime_path.exists():
        raise FileNotFoundError(f"❌ fr_crime.csv bulunamadı: {crime_path}")

    # 1) 911 özetini yükle (GEOID bazlı, hazır)
    fr911 = read_911_summary()
    log(f"📊 911 özet boyutu: {fr911.shape[0]:,} × {fr911.shape[1]}")

    # 2) Suç verisini yükle ve GEOID’yi garanti et
    crime = pd.read_csv(crime_path, low_memory=False, dtype={"GEOID":"string"})
    log(f"📥 fr_crime.csv satır: {len(crime):,}")

    try:
        gdf_blocks, tlen = _load_blocks()
    except Exception:
        # GEOID normalize ederken fallback — ama normalde var olmalı
        tlen = 11

    if "GEOID" in crime.columns and crime["GEOID"].notna().any():
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)
    else:
        crime = ensure_geoid_from_latlon(crime)
        crime["GEOID"] = normalize_geoid(crime["GEOID"], tlen)

    # 3) hr_key ve tarih bilgisi
    if "event_hour" not in crime.columns:
        raise ValueError("❌ fr_crime.csv içinde 'event_hour' kolonu yok (0-23 saat).")
    crime["hr_key"] = ((pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3

    has_date_col = ("date" in crime.columns) or ("datetime" in crime.columns)
    if has_date_col:
        if "date" not in crime.columns:
            crime["date"] = to_date(crime["datetime"])
        else:
            crime["date"] = to_date(crime["date"])

    # 4) Birleştirme stratejisi
    if has_date_col:
        keys = ["GEOID","date","hr_key"]
        merged = crime.merge(fr911, on=keys, how="left")
        log("🔗 Join modu: DATE-BASED (GEOID, date, hr_key)")
    else:
        # takvim-bazlı (median)
        cal_keys = ["GEOID","hr_key","day_of_week","season"]
        agg_cols = [
            "911_request_count_hour_range",
            "911_request_count_daily(before_24_hours)",
            "daily_cnt","hr_cnt",
            "911_geo_last3d","911_geo_last7d",
            "911_geo_hr_last3d","911_geo_hr_last7d",
        ]
        cal = (fr911.groupby(cal_keys, as_index=False)[agg_cols]
                     .median(numeric_only=True))
        if "day_of_week" not in crime.columns:
            # yoksa 0 ver (düşük etkili ama boş bırakmaktan iyi)
            crime["day_of_week"] = 0
        if "season" not in crime.columns:
            if "month" in crime.columns:
                _smap = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",
                         5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
                crime["season"] = crime["month"].map(_smap).fillna("Summer")
            else:
                crime["season"] = "Summer"
        merged = crime.merge(cal, on=cal_keys, how="left")
        log("🔗 Join modu: CALENDAR-BASED (GEOID, hr_key, day_of_week, season)")

    # 5) Sayısal kolonları doldur
    fill_cols = [
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "daily_cnt","hr_cnt",
        "911_geo_last3d","911_geo_last7d",
        "911_geo_hr_last3d","911_geo_hr_last7d",
    ]
    for c in fill_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    # 6) Yaz
    safe_save_csv(merged, str(out_path))

    # küçük bir özet
    try:
        log(merged[["GEOID","date","hr_key","911_request_count_hour_range",
                    "911_request_count_daily(before_24_hours)"]].head(8).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
