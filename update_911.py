# update_911.py
from __future__ import annotations
import os, re, io, time, requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

# zoneinfo güvenli import (SF saat dilimi), olmazsa zarifçe devre dışı
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None
    
# =========================
# CONFIG & PATHS
# =========================

# ✅ Panel ile tutarlılık: varsayılan GEOID uzunluğu 11
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# Çalışma dizini ENV ile yönetilebilir
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# 911 summary dosya adları
LOCAL_NAME = "sf_911_last_5_year.csv"
local_summary_path = Path(BASE_DIR) / LOCAL_NAME
Y_NAME = "sf_911_last_5_year_y.csv"
y_summary_path = Path(BASE_DIR) / Y_NAME

# Crime grid
CRIME_GRID_CANDIDATES = [
    Path(BASE_DIR) / "sf_crime_grid_full_labeled.csv",
    Path("./sf_crime_grid_full_labeled.csv"),
]
merged_output_path = Path(BASE_DIR) / "sf_crime_01.csv"

# Census blocks (komşu için)
CENSUS_CANDIDATES = [
    Path(BASE_DIR) / "sf_census_blocks_with_population.geojson",
    Path("./sf_census_blocks_with_population.geojson"),
]

# API / kaynak
SF911_API_URL   = os.getenv("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF_APP_TOKEN    = os.getenv("SF911_API_TOKEN", "")
AGENCY_FILTER   = os.getenv("SF911_AGENCY_FILTER", "agency like '%Police%'")
REQUEST_TIMEOUT = int(os.getenv("SF911_REQUEST_TIMEOUT", "60"))
CHUNK_LIMIT     = int(os.getenv("SF911_CHUNK_LIMIT", "50000"))
MAX_RETRIES     = int(os.getenv("SF911_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS = float(os.getenv("SF911_SLEEP", "0.2"))
BULK_RANGE      = os.getenv("SF911_BULK_RANGE", "1").lower() in ("1","true","yes","on")
IS_V3           = "/api/v3/views/" in SF911_API_URL
V3_PAGE_LIMIT   = int(os.getenv("SF_V3_PAGE_LIMIT", "1000"))
SF911_RECENT_HOURS = int(os.getenv("SF911_RECENT_HOURS", "6"))

# Release taban URL — `_y` ÖNCELİKLİ, sonra eski ada düş
RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()

RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",  
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]

# =========================
# LOG / HELPERS
# =========================

def safe_dropna(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    cols = [c for c in subset if c in df.columns]
    return df if not cols else df.dropna(subset=cols)

def log(msg: str):
    print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        log(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        log(f"📁 Yedek oluşturuldu: {path}.bak")

def _to_date_series(x):
    """UTC -> SF yerel tarihe dönüştürmeyi dener; olmazsa naive tarihe düşer."""
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce")
        if SF_TZ is not None:
            s = s.dt.tz_convert(SF_TZ)
        return s.dt.date.dropna()
    except Exception:
        return pd.to_datetime(x, errors="coerce").dt.date.dropna()

def log_shape(df, label):
    r, c = df.shape
    log(f"📊 {label}: {r} satır × {c} sütun")

def log_date_range(df, date_col="date", label="911"):
    if date_col not in df.columns:
        log(f"⚠️ {label}: '{date_col}' kolonu yok.")
        return
    s = _to_date_series(df[date_col])
    if s.empty:
        log(f"⚠️ {label}: tarih parse edilemedi.")
        return
    log(f"🧭 {label} tarihi aralığı: {s.min()} → {s.max()} (gün={s.nunique()})")

def missing_report(df: pd.DataFrame, label: str, out_dir: str = BASE_DIR) -> pd.DataFrame:
    """
    Sütun bazında NaN sayısı ve oranını raporlar; CSV'ye yazar ve log'lar.
    """
    if df is None or df.empty:
        log(f"🕳️ NaN raporu ({label}): DF boş.")
        return pd.DataFrame(columns=["column", "missing", "total", "ratio"])

    miss = df.isna().sum().sort_values(ascending=False)
    rep = pd.DataFrame({
        "missing": miss,
        "total": len(df),
        "ratio": (miss / max(len(df), 1)).round(6)
    })
    rep.index.name = "column"
    rep = rep.reset_index()

    # Dosyaya yaz
    safe_save_csv(rep, str(Path(out_dir) / f"missing_{re.sub(r'[^A-Za-z0-9_]+','_', label)}.csv"))

    # Log: sadece NaN > 0 olan ilk 10 sütunu kısaca göster
    top = rep[rep["missing"] > 0].head(10)
    if not top.empty:
        summary_str = ", ".join(f"{r['column']}:{int(r['missing'])}({r['ratio']:.1%})" for _, r in top.iterrows())
        log(f"🕳️ NaN raporu ({label}) → {summary_str}")
    else:
        log(f"🕳️ NaN raporu ({label}) → NaN yok.")

    # Tamamen NaN olan sütunları ayrıca not düş
    all_nan_cols = rep.loc[rep["ratio"] == 1.0, "column"].tolist()
    if all_nan_cols:
        log(f"⚠️ Tamamen NaN sütunlar ({label}): {all_nan_cols}")

    return rep

def dump_nan_samples(
    df: pd.DataFrame,
    label: str,
    key_cols: tuple = ("GEOID", "date", "hr_key", "day_of_week", "season"),
    n: int = 5,
    out_dir: str | None = None,
) -> Optional[pd.DataFrame]:
    """
    NaN içeren HER sütun için en fazla n adet örnek satır döker.
    Çıktı: nan_samples_<label>.csv  (BASE_DIR altında)
    Kolonlar: __nan_in, __row_index + (key_cols ∩ df) + [ilgili kolon]
    """
    if df is None or df.empty:
        log(f"🧪 NaN örnekleri ({label}): DF boş; örnek yok.")
        return None

    if out_dir is None:
        out_dir = BASE_DIR

    cols_with_nan = [c for c in df.columns if df[c].isna().any()]
    if not cols_with_nan:
        log(f"🧪 NaN örnekleri ({label}): NaN yok.")
        return None

    keep_keys = [k for k in key_cols if k in df.columns]
    samples = []
    total_rows = 0

    for c in cols_with_nan:
        cols = keep_keys + [c]
        ex = df.loc[df[c].isna(), cols].head(n).copy()
        if ex.empty:
            continue
        ex.insert(0, "__nan_in", c)
        ex.insert(1, "__row_index", ex.index)  # orijinal satır index’i
        samples.append(ex)
        total_rows += len(ex)

    if not samples:
        log(f"🧪 NaN örnekleri ({label}): NaN var ama örnek çekilemedi.")
        return None

    out = pd.concat(samples, ignore_index=True)
    out_path = Path(out_dir) / f"nan_samples_{re.sub(r'[^A-Za-z0-9_]+','_', label)}.csv"
    safe_save_csv(out, str(out_path))
    # Log’ta ilk birkaç kolon/satır
    log(f"🧪 NaN örnekleri ({label}) → {len(cols_with_nan)} sütunda {total_rows} örnek satır yazıldı → {out_path.name}")
    return out

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    """Sadece rakamları al, soldan L karaktere kes ve zfill(L) yap (panel ile uyumlu)."""
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def is_lfs_pointer_file(p: Path) -> bool:
    """Git LFS pointer dosyası olup olmadığını hızlıca kontrol et."""
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False

def _pick_working_release_url(candidates: list[str]) -> str:
    for u in candidates:
        if not u:
            continue
        try:
            r = requests.get(u, timeout=20)
            # LFS pointer veya boş içeriği ele
            if r.ok and r.content and len(r.content) > 200 and b"git-lfs" not in r.content[:200].lower():
                log(f"⬇️ Release kaynağı seçildi: {u}")
                return u
            else:
                log(f"⚠️ Uygun değil (boş/küçük/LFS pointer olabilir): {u}")
        except Exception as e:
            log(f"⚠️ Ulaşılamadı: {u} ({e})")
    raise RuntimeError("❌ Hiçbir release 911 URL’i erişilebilir değil.")

# Komşu ayarları
ENABLE_NEIGHBORS  = os.getenv("ENABLE_NEIGHBORS", "1").lower() in ("1","true","yes","on")
NEIGHBOR_METHOD   = os.getenv("NEIGHBOR_METHOD", "touches")  # touches | radius
NEIGHBOR_RADIUS_M = float(os.getenv("NEIGHBOR_RADIUS_M", "500"))

# BBOX temizliği
SF_BBOX = (-123.2, 37.6, -122.3, 37.9)

# Zaman pencereleri (rolling)
ROLL_WINDOWS = (3, 7)

# --- KOMBİNE KAYIT HELPER ---
def save_911_both(df: pd.DataFrame):
    """911 özetini hem normal ada hem _y ada kaydet."""
    safe_save_csv(df, str(local_summary_path))
    safe_save_csv(df, str(y_summary_path))
    log(f"💾 911 özet yazıldı → {local_summary_path.name} & {y_summary_path.name} (satır={len(df)})")

# =========================
# IO HELPERS
# =========================

def read_large_csv_in_chunks(path, usecols=None, chunksize=200_000):
    try:
        it = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"}, usecols=usecols, chunksize=chunksize)
        return pd.concat(it, ignore_index=True)
    except ValueError:
        it = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"}, chunksize=chunksize)
        return pd.concat(it, ignore_index=True)

# =========================
# 911 FETCH & SUMMARY
# =========================

def try_small_request(params, headers):
    p = dict(params)
    p["$limit"], p["$offset"] = 1, 0
    r = requests.get(SF911_API_URL, headers=headers, params=p, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r

def _load_blocks() -> tuple[gpd.GeoDataFrame, int]:
    census_path = next((p for p in CENSUS_CANDIDATES if p.exists()), None)
    if census_path is None:
        raise FileNotFoundError("❌ Nüfus blokları GeoJSON yok (crime_prediction_data/ veya kök).")
    gdf_blocks = gpd.read_file(census_path)
    if "GEOID" not in gdf_blocks.columns:
        cand = [c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID benzeri bir sütun yok.")
        gdf_blocks = gdf_blocks.rename(columns={cand[0]: "GEOID"})
    tlen = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
    gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], tlen)
    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True)
    elif gdf_blocks.crs.to_epsg() != 4326:
        gdf_blocks = gdf_blocks.to_crs(4326)
    return gdf_blocks, tlen

def ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df
    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "intersection_point" in df.columns:
            def _lon(x):
                if isinstance(x, dict) and "coordinates" in x: return x["coordinates"][0]
                if isinstance(x, str):
                    m = re.search(r"[-\d\.]+,\s*[-\d\.]+", x)
                    if m:
                        lo, la = m.group(0).split(","); return float(lo)
                return None
            def _lat(x):
                if isinstance(x, dict) and "coordinates" in x: return x["coordinates"][1]
                if isinstance(x, str):
                    m = re.search(r"[-\d\.]+,\s*[-\d\.]+", x)
                    if m:
                        lo, la = m.group(0).split(","); return float(la)
                return None
            df["longitude"], df["latitude"] = df["intersection_point"].apply(_lon), df["intersection_point"].apply(_lat)
        for a,b in (("y","x"),("lat","long")):
            if a in df.columns and b in df.columns and "latitude" not in df.columns:
                df["latitude"], df["longitude"] = pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")
                break
    if "latitude" in df.columns and "longitude" in df.columns:
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df = df[(df["latitude"].between(min_lat, max_lat)) & (df["longitude"].between(min_lon, max_lon))]
    df = df.dropna(subset=["latitude","longitude"]).copy()

    gdf_blocks, tlen = _load_blocks()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID","geometry"]], how="left", predicate="within")
    out = pd.DataFrame(gdf.drop(columns=["geometry","index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    out = out.dropna(subset=["GEOID"]).copy()
    return out

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["GEOID","date","hour_range","911_request_count_hour_range","911_request_count_daily(before_24_hours)"])
    df = raw.copy()
    ts_col = None
    for cand in ["received_time","received_datetime","date","datetime","timestamp","call_received_datetime"]:
        if cand in df.columns:
            ts_col = cand; break
    if ts_col is None:
        raise ValueError("Zaman kolonu bulunamadı (received_time/received_datetime/date).")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour
    eh = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int) % 24
    start = (eh // 3) * 3
    df["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")

    has_geoid = "GEOID" in df.columns
    grp_hr  = (["GEOID"] if has_geoid else []) + ["date","hour_range"]
    grp_day = (["GEOID"] if has_geoid else []) + ["date"]

    hr_agg = df.groupby(grp_hr, dropna=False, observed=True).size().reset_index(name="911_request_count_hour_range")
    day_agg = df.groupby(grp_day, dropna=False, observed=True).size().reset_index(name="911_request_count_daily(before_24_hours)")
    out = hr_agg.merge(day_agg, on=grp_day, how="left")

    cols_tail = [c for c in ["date","hour_range","GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]

# -------------------- LOCAL BASE (artifact/regular) & RELEASE ------------------

def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 tabanı okunuyor: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID":"string"})
    is_already_summary = (
        {"date","hour_range"}.issubset(df.columns) and
        any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    )
    if is_already_summary:
        cnt_col = next(c for c in ["911_request_count_hour_range","call_count","count","requests","n"] if c in df.columns)
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})
        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m: return None
            a = int(m.group(1)) % 24; b = int(m.group(2)); b = b if b > a else min(a+3, 24)
            return f"{a:02d}-{b:02d}"
        df["hour_range"] = df["hour_range"].apply(_fmt_hr)
        if "911_request_count_daily(before_24_hours)" not in df.columns:
            keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
            day = df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"] \
                    .sum().reset_index(name="911_request_count_daily(before_24_hours)")
            df = df.merge(day, on=keys, how="left")
        if min_date is not None:
            df = df[df["date"] >= min_date]
        cols_tail = [c for c in ["date","hour_range","GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in cols_tail] + cols_tail
        return df[cols]
    # değilse ham → özet
    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date]
    return std

def ensure_local_911_base() -> Optional[Path]:
    """
    Tercih sırası:
      1) artifact/çalışma alanından sf_911_last_5_year_y.csv (önceki full pipeline’dan)
      2) yerel sf_911_last_5_year.csv
      3) yoksa None (release fallback kullanılacak)
    """
    y_candidates = [
        Path(BASE_DIR) / "sf_911_last_5_year_y.csv",
        Path("./sf_911_last_5_year_y.csv"),
        Path("outputs/sf_911_last_5_year_y.csv"),
    ]
    for p in y_candidates:
        if p.exists():
            if p.suffix == ".csv" and is_lfs_pointer_file(p):
                continue
            log(f"📦 911 base (preferred Y) bulundu: {p}")
            return p

    regular_candidates = [
        Path(BASE_DIR) / "sf_911_last_5_year.csv",
        Path("./sf_911_last_5_year.csv"),
    ]
    for p in regular_candidates:
        if p.exists():
            if p.suffix == ".csv" and is_lfs_pointer_file(p):
                continue
            log(f"📦 911 base (regular) bulundu: {p}")
            return p

    return None

# ------------------------- RELEASE → BASE SUMMARY ----------------------------

def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release 911 özeti indiriliyor: {url}")
    r = requests.get(url, timeout=120); r.raise_for_status()
    tmp = Path(BASE_DIR) / "_tmp_911.csv"; ensure_parent(str(tmp))
    tmp.write_bytes(r.content)
    df = pd.read_csv(tmp, low_memory=False, dtype={"GEOID":"string"})
    is_already_summary = ( {"date","hour_range"}.issubset(df.columns)
                           and any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"]) )
    if is_already_summary:
        cnt_col = next(c for c in ["911_request_count_hour_range","call_count","count","requests","n"] if c in df.columns)
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})
        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        # hour_range normalize
        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m: return None
            a = int(m.group(1)) % 24; b = int(m.group(2)); b = b if b > a else min(a+3, 24)
            return f"{a:02d}-{b:02d}"
        df["hour_range"] = df["hour_range"].apply(_fmt_hr)
        if "911_request_count_daily(before_24_hours)" not in df.columns:
            keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
            day = df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"].sum().reset_index(name="911_request_count_daily(before_24_hours)")
            df = df.merge(day, on=keys, how="left")
        if min_date is not None:
            df = df[df["date"] >= min_date]
        cols_tail = [c for c in ["date","hour_range","GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in cols_tail] + cols_tail
        return df[cols]
    # değilse ham → özet
    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date]
    return std

# ----------------------- API INCREMENT (RANGE) --------------------------------

def fetch_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    dt_candidates = ["received_time","received_datetime","date","datetime","call_datetime","received_dttm","call_date"]
    headers = {"X-App-Token": SF_APP_TOKEN} if SF_APP_TOKEN else {}
    rng_start = f"{start_day}T00:00:00"; rng_end = f"{end_day}T23:59:59"
    chosen_dt, last_err = None, None
    for dt_col in dt_candidates:
        base_where = f"{dt_col} between '{rng_start}' and '{rng_end}'"
        for wc in [base_where + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""), base_where]:
            try:
                try_small_request({"$where": wc}, headers)
                chosen_dt = dt_col; break
            except Exception as e:
                last_err = e; continue
        if chosen_dt: break
    if chosen_dt is None:
        log(f"    ❌ Aralık için uygun datetime kolonu bulunamadı. Son hata: {last_err}")
        return None
    pieces, offset, page = [], 0, 1
    where_list = [f"{chosen_dt} between '{rng_start}' and '{rng_end}'" + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""),
                  f"{chosen_dt} between '{rng_start}' and '{rng_end}'"]
    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(SF911_API_URL, headers=headers, params={"$where": where_list[0], "$limit": CHUNK_LIMIT, "$offset": offset}, timeout=REQUEST_TIMEOUT)
                if r.status_code == 400:
                    r = requests.get(SF911_API_URL, headers=headers, params={"$where": where_list[1], "$limit": CHUNK_LIMIT, "$offset": offset}, timeout=REQUEST_TIMEOUT)
                r.raise_for_status(); df = pd.read_json(io.BytesIO(r.content)); break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"    ❌ range page {page} (offset={offset}) hata: {e}")
                    df = None; break
                time.sleep(1.0 + attempt*0.5)
        if df is None or df.empty:
            if page == 1: log("    (bu aralıkta veri yok)")
            break
        log(f"    + {len(df)} satır (range-page={page}, offset={offset})")
        pieces.append(df)
        if len(df) < CHUNK_LIMIT: break
        offset += CHUNK_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)
    if not pieces: return None
    return pd.concat(pieces, ignore_index=True)

def fetch_v3_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    from requests.adapters import HTTPAdapter, Retry
    sess = requests.Session()
    retries = Retry(total=5, connect=5, read=5, backoff_factor=1.2,
                    status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://",  HTTPAdapter(max_retries=retries))

    headers = {"Accept":"application/json"}
    if SF_APP_TOKEN:
        headers["X-App-Token"] = SF_APP_TOKEN

    dt_candidates = ["received_time","received_datetime","date","datetime","call_datetime","received_dttm","call_date"]
    rng_start = f"{start_day}T00:00:00"; rng_end = f"{end_day}T23:59:59"

    chosen_dt, cols = None, None
    for dtc in dt_candidates:
        where = f"{dtc} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER:
            where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT 1 OFFSET 0"
        try:
            r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if obj.get("data"):
                chosen_dt = dtc
                cols = [c.get("fieldName") or c.get("name") or f"c{i}"
                        for i,c in enumerate(obj.get("meta",{}).get("view",{}).get("columns",[]))]
                break
        except Exception:
            continue
    if not chosen_dt:
        log("    ❌ v3: uygun datetime kolonu bulunamadı.")
        return None

    all_rows, offset, page = [], 0, 1
    while True:
        where = f"{chosen_dt} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER:
            where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT {V3_PAGE_LIMIT} OFFSET {offset}"

        got = 0
        for attempt in range(MAX_RETRIES):
            try:
                r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                obj = r.json()
                data = obj.get("data", [])
                if data:
                    for row in data:
                        if isinstance(row, list):
                            all_rows.append({cols[i]: (row[i] if i < len(cols) else None) for i in range(len(cols))})
                        elif isinstance(row, dict):
                            all_rows.append(row)
                    got = len(data)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"    ❌ v3 range page {page} (offset={offset}) hata: {e}")
                time.sleep(1.0 + attempt*0.5)
        if got < V3_PAGE_LIMIT:
            break
        offset += V3_PAGE_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)

    return pd.DataFrame(all_rows)

def write_recent_csv(raw: pd.DataFrame, hours: int = SF911_RECENT_HOURS):
    ts_col = next((c for c in ["received_time","received_datetime","date","datetime",
                               "timestamp","call_received_datetime","ts"] if c in raw.columns), None)
    if not ts_col:
        return
    tmp = raw.copy()
    tmp["ts"] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp[tmp["ts"].notna()]
    if tmp.empty:
        return

    lat_col = next((c for c in ["latitude","lat","y"] if c in raw.columns), None)
    lon_col = next((c for c in ["longitude","lon","x"] if c in raw.columns), None)

    tmax = tmp["ts"].max()
    cutoff = tmax - pd.Timedelta(hours=hours)

    out = pd.DataFrame({"ts": tmp["ts"]})
    if lat_col: out["lat"] = pd.to_numeric(tmp[lat_col], errors="coerce")
    if lon_col: out["lon"] = pd.to_numeric(tmp[lon_col], errors="coerce")
    out = out[out["ts"] >= cutoff].copy()

    path = Path(BASE_DIR) / "sf_911_recent.csv"
    safe_save_csv(out, str(path))
    log(f"ℹ️ sf_911_recent.csv yazıldı (son {hours} saat): {len(out)} satır")

def incremental_summary(start_day: datetime.date, end_day: datetime.date) -> pd.DataFrame:
    if start_day is None or end_day is None or end_day < start_day:
        return pd.DataFrame()
    log(f"🌐 API artımlı: {start_day} → {end_day} ({(end_day - start_day).days + 1} gün)")
    raw = None
    if BULK_RANGE:
        raw = fetch_v3_range_all_chunks(start_day, end_day) if IS_V3 else fetch_range_all_chunks(start_day, end_day)
    try:
        if raw is not None and not raw.empty:
            write_recent_csv(raw, hours=SF911_RECENT_HOURS)
    except Exception as e:
        log(f"⚠️ recent yazımı atlandı: {e}")
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        raw = ensure_geoid(raw)
    except Exception as e:
        log(f"⚠️ ensure_geoid sırasında hata: {e}; GEOID’siz özet üretilecek")
    return make_standard_summary(raw)

# =========================
# MAIN: LOCAL (Y/regular) → RELEASE (Y → regular) → API FALLBACK → ENRICH PREP
# =========================
import traceback, sys  # ayrıntılı hata çıktısı için

five_years_ago = datetime.now(timezone.utc).date() - timedelta(days=5*365)
log(f"📁 911 yerel özet yolu: {local_summary_path}")

final_911 = None

# 1) YEREL TABAN (önce Y, yoksa regular)
try:
    base_csv_path = ensure_local_911_base()
    if base_csv_path is not None:
        final_911 = summary_from_local(base_csv_path, min_date=five_years_ago)
        save_911_both(final_911)
        log(f"✅ Yerel 911 özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")
    else:
        log("ℹ️ Yerel 911 özeti bulunamadı; release denenecek.")
except Exception as e:
    log("⚠️ Yerel 911 özet okunurken hata:")
    log("".join(traceback.format_exception(e)))

# 2) RELEASE (önce _y.csv, olmazsa orijinal csv)
if final_911 is None or final_911.empty:
    try:
        release_url = _pick_working_release_url(RAW_911_URL_CANDIDATES)
        final_911 = summary_from_release(release_url, min_date=five_years_ago)
        save_911_both(final_911)
        log(f"✅ Release özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")
    except Exception as e:
        log("⚠️ Release fallback başarısız; API fallback denenecek:")
        log("".join(traceback.format_exception(e)))

# 3) API FALLBACK (release da başarısız/boşsa)
if final_911 is None or final_911.empty:
    try:
        today_sf = (datetime.now(SF_TZ) if SF_TZ is not None else datetime.now()).date()
        final_911 = incremental_summary(five_years_ago, today_sf)
        if final_911 is None or final_911.empty:
            log("❌ 911 tabanı üretilemedi: Yerel yok, release erişilemedi/boş ve API da boş döndü.")
            sys.exit(1)
        save_911_both(final_911)
        log(f"✅ API tabanlı 911 özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")
    except Exception as e:
        log("❌ API fallback sırasında hata:")
        log("".join(traceback.format_exception(e)))
        sys.exit(1)

# 4) NORMALIZE ( güvenli / kolonlar yoksa kırmaz )
if final_911 is None or final_911.empty:
    log("⚠️ 911 özeti üretilemedi (boş). Çıkılıyor.")
    sys.exit(1)

# Bu iki değişkeni try'dan ÖNCE başlat → NameError riski yok
_day_unique = pd.DataFrame()
_hr_unique  = pd.DataFrame()

hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")

try:
    final_911 = final_911.dropna(subset=["GEOID","date","hour_range"]).copy()
    if "GEOID" in final_911.columns:
        final_911["GEOID"] = normalize_geoid(final_911["GEOID"], DEFAULT_GEOID_LEN)
    if "date" in final_911.columns:
        final_911["date"] = to_date(final_911["date"])

    # hr_pat zaten yukarıda tanımlı → tekrar etmiyoruz
    def _hr_key_from_range(hr):
        m = hr_pat.match(str(hr))
        return int(m.group(1)) % 24 if m else None

    if "hour_range" in final_911.columns:
        final_911["hr_key"] = final_911["hour_range"].apply(_hr_key_from_range).astype("int16")

    if "date" in final_911.columns:
        final_911["day_of_week"] = pd.to_datetime(final_911["date"]).dt.weekday.astype("int8")
        final_911["month"] = pd.to_datetime(final_911["date"]).dt.month.astype("int8")
        _season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
        final_911["season"] = final_911["month"].map(_season_map).astype("category")

    log_shape(final_911, "911 summary (normalize)")
    log_date_range(final_911, "date", "911")

    # ⬇️ final_911 için raporu BURADA tek kez al
    if 'missing_report' in globals():
        missing_report(final_911, "911_summary_normalize")
    if 'dump_nan_samples' in globals():
        dump_nan_samples(final_911, "911_summary_normalize")

    # Günlük toplam yoksa üret
    if "911_request_count_daily(before_24_hours)" not in final_911.columns:
        keys_day = [k for k in ["GEOID", "date"] if k in final_911.columns]
        if {"GEOID","date","hour_range","911_request_count_hour_range"}.issubset(final_911.columns):
            _tmp_day = (final_911.groupby(keys_day, observed=True)["911_request_count_hour_range"]
                        .sum().reset_index(name="911_request_count_daily(before_24_hours)"))
            final_911 = final_911.merge(_tmp_day, on=keys_day, how="left")
        else:
            # hiçbiri yoksa yine de kolonu yarat (0) → downstream garanti
            final_911["911_request_count_daily(before_24_hours)"] = 0

    # 1) Günlük baz (GEOID × date) — dinamik kolon seçimi (daily_cnt'yi GARANTİ yarat)
    if "911_request_count_daily(before_24_hours)" not in final_911.columns:
        final_911["911_request_count_daily(before_24_hours)"] = 0
    # emniyet: sayısal yap
    final_911["911_request_count_daily(before_24_hours)"] = pd.to_numeric(
        final_911["911_request_count_daily(before_24_hours)"], errors="coerce"
    ).fillna(0).astype("int32")

    cols_day = [c for c in ["GEOID", "date", "911_request_count_daily(before_24_hours)"] if c in final_911.columns]
    _day_unique = (
        final_911[cols_day]
        .dropna(subset=[c for c in ["date"] if c in cols_day])
        .drop_duplicates(subset=[c for c in ["GEOID","date"] if c in cols_day])
        .sort_values([c for c in ["GEOID","date"] if c in cols_day])
        .rename(columns={"911_request_count_daily(before_24_hours)": "daily_cnt"})
        .reset_index(drop=True)
    )

    # emniyet: eğer bir şekilde daily_cnt gelmemişse oluştur
    if "daily_cnt" not in _day_unique.columns:
        if "911_request_count_hour_range" in final_911.columns and {"GEOID","date"}.issubset(final_911.columns):
            _tmp_day2 = (final_911.groupby(["GEOID","date"], observed=True)
                         ["911_request_count_hour_range"].sum().reset_index()
                         .rename(columns={"911_request_count_hour_range": "daily_cnt"}))
            _day_unique = _day_unique.merge(
                _tmp_day2, on=[c for c in ["GEOID","date"] if c in _day_unique.columns], how="left"
            )
        else:
            _day_unique["daily_cnt"] = 0
    _day_unique["daily_cnt"] = pd.to_numeric(_day_unique["daily_cnt"], errors="coerce").fillna(0).astype("int32")

    # === GEOID bazında son 3/7 gün 911 toplamı (bugün hariç) ===
    try:
        if not _day_unique.empty and {"GEOID", "date", "daily_cnt"}.issubset(_day_unique.columns):
            _day_unique["date"] = pd.to_datetime(_day_unique["date"], errors="coerce").dt.date
            _day_unique = _day_unique.dropna(subset=["date"]).copy()

            def _expand_daily(g):
                s = pd.Series(g["daily_cnt"].values,
                              index=pd.to_datetime(g["date"], errors="coerce")).sort_index()
                full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
                s = s.reindex(full_idx, fill_value=0)
                out = pd.DataFrame({
                    "date_ts": s.index,
                    "daily_cnt": s.values
                })
                out["911_geo_last3d"] = (
                    out["daily_cnt"].rolling(3, min_periods=1).sum().shift(1)
                )
                out["911_geo_last7d"] = (
                    out["daily_cnt"].rolling(7, min_periods=1).sum().shift(1)
                )
                return out

            _rolled_list = []
            for geoid, grp in _day_unique.groupby("GEOID", observed=True, sort=False):
                r = _expand_daily(grp)
                r.insert(0, "GEOID", geoid)
                _rolled_list.append(r)

            _geo_roll = pd.concat(_rolled_list, ignore_index=True)
            _geo_roll["date"] = _geo_roll["date_ts"].dt.date
            _geo_roll = _geo_roll.drop(columns=["date_ts"])

            _day_unique = _day_unique.merge(
                _geo_roll,
                on=["GEOID", "date"],
                how="left"
            )

            for c in ["911_geo_last3d", "911_geo_last7d"]:
                if c in _day_unique.columns:
                    _day_unique[c] = pd.to_numeric(_day_unique[c], errors="coerce").fillna(0).astype("int32")

            # 2) Saat dilimi baz
            if {"GEOID","hr_key","date","911_request_count_hour_range"}.issubset(final_911.columns):
                _hr_unique = (
                    final_911.groupby(["GEOID","hr_key","date"], as_index=False, observed=True)["911_request_count_hour_range"]
                    .sum()
                    .rename(columns={"911_request_count_hour_range": "hr_cnt"})
                    .sort_values(["GEOID","hr_key","date"])
                    .reset_index(drop=True)
                )
            else:
                _hr_unique = pd.DataFrame(columns=["GEOID","hr_key","date","hr_cnt"])

            # =========================
            # ROLLING (3g/7g)
            # =========================
            for W in ROLL_WINDOWS:
                if not _day_unique.empty:
                    # günlük rolling için güvenli kolon seçimi
                    val_col = "daily_cnt" if "daily_cnt" in _day_unique.columns else (
                        "911_request_count_daily(before_24_hours)" if "911_request_count_daily(before_24_hours)" in _day_unique.columns else None
                    )
                    if val_col is None:
                        _day_unique["daily_cnt"] = 0
                        val_col = "daily_cnt"

                    if "GEOID" in _day_unique.columns:
                        _day_unique[f"911_geo_last{W}d"] = (
                            _day_unique.groupby("GEOID", observed=True)[val_col]
                            .transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
                        ).astype("float32")
                    else:
                        # GEOID yoksa şehir geneli tek seri rolling
                        _day_unique[f"911_geo_last{W}d"] = (
                            _day_unique.sort_values("date")[val_col]
                            .rolling(W, min_periods=1).sum().shift(1)
                        ).astype("float32")

            # ⬇️ Rolling sonrası raporlar
            missing_report(_day_unique, "911_day_unique_after_roll")
            dump_nan_samples(_day_unique, "911_day_unique_after_roll")
            missing_report(_hr_unique, "911_hr_unique_after_roll")
            dump_nan_samples(_hr_unique, "911_hr_unique_after_roll")

    except Exception as e:
        log("⚠️ Normalize sırasında uyarı; devam ediliyor:")
        log("".join(traceback.format_exception(e)))

    # 5) ARTIMLI GÜNCELLEME (mevcut taban üzerine yeni günleri ekle)
    base_max_date = to_date(final_911["date"]).max() if "date" in final_911.columns and not final_911.empty else None
    today_sf = (datetime.now(SF_TZ) if SF_TZ is not None else datetime.now()).date()

    if base_max_date is None:
        fetch_start, fetch_end = today_sf, today_sf
    else:
        fetch_start, fetch_end = base_max_date + timedelta(days=1), today_sf
        if fetch_start > fetch_end:
            fetch_start = fetch_end

    log(f"🗓️ İndirme aralığı: {fetch_start} → {fetch_end} ({(fetch_end - fetch_start).days + 1} gün)")

    try:
        inc = incremental_summary(fetch_start, fetch_end)
        if inc is not None and not inc.empty:
            if "GEOID" in inc.columns:
                inc["GEOID"] = normalize_geoid(inc["GEOID"], DEFAULT_GEOID_LEN)
            if "date" in inc.columns:
                inc["date"] = to_date(inc["date"])

            before = len(final_911)
            final_911 = pd.concat([final_911, inc], ignore_index=True)

            subset_cols = [c for c in ["GEOID","date","hour_range"] if c in final_911.columns]
            final_911 = (
                final_911
                .dropna(subset=["date"])
                .sort_values(subset_cols if subset_cols else (["date"] if "date" in final_911.columns else None))
                .drop_duplicates(subset=subset_cols if subset_cols else (["date"] if "date" in final_911.columns else None),
                                 keep="last")
            )
            if "date" in final_911.columns:
                final_911 = final_911[final_911["date"] >= five_years_ago]

            save_911_both(final_911)
            log(f"💾 911 özet GÜNCELLENDİ (base+API) → {local_summary_path} & {y_summary_path} (+{len(final_911)-before:,} satır)")
        else:
            log("ℹ️ API tarafında yeni gün yok veya boş döndü; mevcut taban veri kullanılacak.")
    except Exception as e:
        log("⚠️ Artımlı güncelleme sırasında hata (mevcut taban veri kullanılacak):")
        log("".join(traceback.format_exception(e)))

    # 6) SON KONTROL
    if final_911 is None or final_911.empty:
        log("⚠️ 911 özeti üretilemedi (boş). Çıkılıyor.")
        sys.exit(1)

except Exception as e:
    log("❌ Dış normalize+incremental try bloğunda hata:")
    import traceback
    log("".join(traceback.format_exception(e)))

# =========================
# STANDARDIZE + DERIVED KEYS (hr_key, dow, season)
# =========================
def _hr_key_from_range(hr):
    m = hr_pat.match(str(hr))
    return int(m.group(1)) % 24 if m else None

final_911 = final_911.dropna(subset=["GEOID","date","hour_range"]).copy()
final_911["GEOID"] = normalize_geoid(final_911["GEOID"], DEFAULT_GEOID_LEN)
final_911["date"] = to_date(final_911["date"])
final_911["hr_key"] = final_911["hour_range"].apply(_hr_key_from_range).astype("int16")
final_911["day_of_week"] = pd.to_datetime(final_911["date"]).dt.weekday.astype("int8")
final_911["month"] = pd.to_datetime(final_911["date"]).dt.month.astype("int8")
_season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
final_911["season"] = final_911["month"].map(_season_map).astype("category")

log_shape(final_911, "911 summary (normalize)")
log_date_range(final_911, "date", "911")
missing_report(final_911, "911_summary_normalize")

# =========================
# ROLLING (3g/7g) — GEOID ve GEOID×hr_key
# =========================
missing_report(_day_unique, "911_day_unique_after_roll")
dump_nan_samples(_day_unique, "911_day_unique_after_roll")

missing_report(_hr_unique, "911_hr_unique_after_roll")
dump_nan_samples(_hr_unique, "911_hr_unique_after_roll")


# =========================
# KOMŞU GEOID ÖZELLİKLERİ (günlük baz)
# =========================
def build_neighbors(method: str = "touches", radius_m: float = 500.0) -> pd.DataFrame:
    ...
    # Burayı gerçek komşuluk hesap kodunla doldurabilirsin.

neighbors_df = None
if ENABLE_NEIGHBORS:
    try:
        neighbors_df = build_neighbors(NEIGHBOR_METHOD, NEIGHBOR_RADIUS_M)
        log_shape(neighbors_df, f"Komşu haritası ({NEIGHBOR_METHOD})")
    except Exception as e:
        log(f"⚠️ Komşu haritası üretilemedi: {e}")
        neighbors_df = None

# ✅ _neighbor_roll'u bu aşamada tanımla
_neighbor_roll = None
if neighbors_df is not None and not neighbors_df.empty:
    day_nbr = neighbors_df.merge(
        _day_unique.rename(columns={"GEOID":"nbr"}), on="nbr", how="left"
    )
    day_nbr = (day_nbr.groupby(["GEOID","date"], as_index=False, observed=True)["daily_cnt"]
               .sum().rename(columns={"daily_cnt":"nbr_daily_cnt"}))
    _neighbor_roll = day_nbr.sort_values(["GEOID","date"]).reset_index(drop=True)
    for W in ROLL_WINDOWS:
        _neighbor_roll[f"911_neighbors_last{W}d"] = (
            _neighbor_roll.groupby("GEOID")["nbr_daily_cnt"]
            .transform(lambda s: s.rolling(W, min_periods=1).sum().shift(1))
        ).astype("float32")

if _neighbor_roll is not None:
    missing_report(_neighbor_roll, "911_neighbor_roll")
    dump_nan_samples(_neighbor_roll, "911_neighbor_roll")


# =========================
# MERGE STRATEJİSİ
# =========================
if _hr_unique is None or _hr_unique.empty:
    _hr_unique = pd.DataFrame(columns=["GEOID","hr_key","date","hr_cnt"])

if _day_unique is None or _day_unique.empty:
    _day_unique = pd.DataFrame(columns=["GEOID","date","daily_cnt","911_geo_last3d","911_geo_last7d"])

_enriched = final_911.merge(_hr_unique, on=["GEOID","hr_key","date"], how="left")
_enriched = _enriched.merge(_day_unique, on=["GEOID","date"], how="left")
if _neighbor_roll is not None:
    _enriched = _enriched.merge(
        _neighbor_roll[["GEOID","date"] + [f"911_neighbors_last{W}d" for W in ROLL_WINDOWS if f"911_neighbors_last{W}d" in _neighbor_roll.columns]],
        on=["GEOID","date"], how="left"
    )

missing_report(_enriched, "911_enriched_before_grid_merge")
dump_nan_samples(_enriched, "911_enriched_before_grid_merge")

crime_grid_path = next((p for p in CRIME_GRID_CANDIDATES if p.exists()), None)
if crime_grid_path is None:
    raise FileNotFoundError("❌ Suç grid yok: crime_prediction_data/sf_crime_grid_full_labeled.csv (veya kökte).")
crime = pd.read_csv(crime_grid_path, dtype={"GEOID": str}, low_memory=False)
log(f"📥 Suç grid yüklendi: {len(crime)} satır ({crime_grid_path})")
log_shape(crime, "CRIME grid — ham")

crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)

if "event_hour" not in crime.columns:
    raise ValueError("❌ Suç grid dosyasında 'event_hour' yok.")
crime["hr_key"] = ((pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3

has_date_col = ("date" in crime.columns) or ("datetime" in crime.columns)
if has_date_col:
    if "date" not in crime.columns:
        crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
    else:
        crime["date"] = to_date(crime["date"])
    keys = ["GEOID","date","hr_key"]
    merged = crime.merge(_enriched, on=keys, how="left")
    log("🔗 Join modu: DATE-BASED (GEOID, date, hr_key)")
else:
    cal_keys = ["GEOID","hr_key","day_of_week","season"]

    base_agg_candidates = [
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "911_geo_last3d", "911_geo_last7d",
        # "911_geo_hr_last3d", "911_geo_hr_last7d",  # üretilmiyor → listeye ekleme
    ]
    neighbor_agg_candidates = [f"911_neighbors_last{W}d" for W in ROLL_WINDOWS]

    agg_cols = [c for c in (base_agg_candidates + neighbor_agg_candidates) if c in _enriched.columns]

    if agg_cols:
        cal_agg = (_enriched.groupby(cal_keys, as_index=False, observed=True)[agg_cols]
                            .median(numeric_only=True))
    else:
        # hiç metrik yoksa, yine de groupby sonucu boş kalmasın
        cal_agg = (_enriched.groupby(cal_keys, as_index=False, observed=True)
                            .size().rename(columns={"size":"rows"}))

    if "day_of_week" not in crime.columns:
        log("ℹ️ crime grid’de day_of_week yok → 0 atanıyor (düşük etkili).")
        crime["day_of_week"] = 0
    if "season" not in crime.columns:
        if "month" in crime.columns:
            _smap = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
            crime["season"] = crime["month"].map(_smap).fillna("Summer")
        else:
            crime["season"] = "Summer"
    merged = crime.merge(cal_agg, on=cal_keys, how="left")
    log("🔗 Join modu: CALENDAR-BASED (GEOID, hr_key, day_of_week, season)")

fill_candidates = [
    "911_request_count_hour_range",
    "911_request_count_daily(before_24_hours)",
    "911_geo_last3d","911_geo_last7d",
] + ([f"911_neighbors_last{W}d" for W in ROLL_WINDOWS] if _neighbor_roll is not None else [])
fill_cols = [c for c in fill_candidates if c in merged.columns]

for c in fill_cols:
    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("int32")

missing_report(merged, "crime_x_911_after_fill")
dump_nan_samples(merged, "crime_x_911_after_fill")

all_nan_cols = [c for c in merged.columns if merged[c].isna().all()]
if all_nan_cols:
    log(f"🧹 CRIME×911: tamamen NaN sütunlar atılıyor → {all_nan_cols}")
    merged = merged.drop(columns=all_nan_cols)

safe_save_csv(merged, str(merged_output_path))
log_shape(merged, "CRIME⨯911 (kayıt öncesi)")
log(f"✅ Suç + 911 birleştirmesi tamamlandı → {merged_output_path}")

try:
    print(merged.head(5).to_string(index=False))
except Exception:
    pass
