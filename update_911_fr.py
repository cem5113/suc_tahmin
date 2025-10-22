# update_911_fr.py
from __future__ import annotations

import io
import os
import re
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

# =========================
# CONFIG
# =========================
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Tarih aralığı (raporlama için)
TODAY_UTC = datetime.now(timezone.utc).date()
FIVE_YEARS_AGO = TODAY_UTC - timedelta(days=5 * 365)

# Çıktı (hedef: fr_crime_01.csv)
OUT_FR_CRIME_01 = str(Path(BASE_DIR) / "fr_crime_01.csv")

# Yerel 911 özetleri
LOCAL_911   = Path(BASE_DIR) / "sf_911_last_5_year.csv"
LOCAL_911_Y = Path(BASE_DIR) / "sf_911_last_5_year_y.csv"

# Socrata API (fallback)
SF911_API_URL = os.getenv("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF_APP_TOKEN = os.getenv("SF911_API_TOKEN", "")
AGENCY_FILTER = os.getenv("SF911_AGENCY_FILTER", "agency like '%Police%'")
REQUEST_TIMEOUT = int(os.getenv("SF911_REQUEST_TIMEOUT", "60"))
CHUNK_LIMIT = int(os.getenv("SF911_CHUNK_LIMIT", "50000"))
MAX_RETRIES = int(os.getenv("SF911_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS = float(os.getenv("SF911_SLEEP", "0.2"))
IS_V3 = "/api/v3/views/" in SF911_API_URL
V3_PAGE_LIMIT = int(os.getenv("SF_V3_PAGE_LIMIT", "1000"))

# Release (önce _y, sonra regular)
RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()
RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]

# =========================
# UTILS
# =========================
def log(msg: str): 
    print(msg, flush=True)

def ensure_parent(path: str): 
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        log(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        log(f"📁 Yedek oluşturuldu: {path}.bak")

def to_date(s): 
    return pd.to_datetime(s, errors="coerce").dt.date

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:int(target_len)].str.zfill(int(target_len))

def log_shape(df, label):
    try:
        r, c = df.shape
        log(f"📊 {label}: {r} satır × {c} sütun")
    except Exception:
        pass

def _mtime(p: Path) -> float:
    try: 
        return p.stat().st_mtime
    except Exception: 
        return -1.0

# =========================
# 911 SUMMARY HELPERS
# =========================
ALLOWED_911_FEATURES = {
    "event_hour",
    "911_request_count_hour_range",
    "received_time",
    "911_request_count_daily(before_24_hours)",
    "date",
    "hour_range",
    "GEOID",
}

def _fmt_hour_range_from_hour(h: int) -> str:
    h = int(h) % 24
    a = (h // 3) * 3
    b = min(a + 3, 24)
    return f"{a:02d}-{b:02d}"

def _hour_from_range(s: str) -> int:
    m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(s))
    return int(m.group(1)) % 24 if m else 0

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham 911 satırlarından saatlik ve günlük sayılar üret."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[
            "GEOID","date","hour_range","event_hour",
            "911_request_count_hour_range","911_request_count_daily(before_24_hours)"
        ])

    df = raw.copy()
    ts_col = None
    for cand in ["received_time","received_datetime","date","datetime","timestamp","call_received_datetime","call_datetime","received_dttm","call_date"]:
        if cand in df.columns:
            ts_col = cand; break
    if ts_col is None:
        raise ValueError("911 kaynağında zaman kolonu bulunamadı.")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[df[ts_col].notna()].copy()
    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour.astype("int16")
    df["hour_range"] = df["event_hour"].apply(_fmt_hour_range_from_hour)

    grp_hr  = (["GEOID"] if "GEOID" in df.columns else []) + ["date","hour_range"]
    grp_day = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]

    hr  = df.groupby(grp_hr,  dropna=False, observed=True).size().reset_index(name="911_request_count_hour_range")
    day = df.groupby(grp_day, dropna=False, observed=True).size().reset_index(name="911_request_count_daily(before_24_hours)")
    out = hr.merge(day, on=grp_day, how="left")
    out["event_hour"] = out["hour_range"].apply(_hour_from_range).astype("int16")

    if "received_time" not in out.columns:
        out["received_time"] = pd.NaT
    if "GEOID" in out.columns:
        out["GEOID"] = normalize_geoid(out["GEOID"], DEFAULT_GEOID_LEN)
    out["date"] = to_date(out["date"])

    keep = ["event_hour","911_request_count_hour_range","received_time",
            "911_request_count_daily(before_24_hours)","date","hour_range","GEOID"]
    keep_present = [c for c in keep if c in out.columns]
    tail = [c for c in ["date","hour_range","GEOID"] if c in out.columns]
    head = [c for c in keep_present if c not in tail]
    return out[head + tail]

def coerce_to_summary_like(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cnt in ["911_request_count_hour_range","call_count","count","requests","n"]:
        if cnt in df.columns:
            if cnt != "911_request_count_hour_range":
                df = df.rename(columns={cnt: "911_request_count_hour_range"})
            break
    if "hour_range" in df.columns:
        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m: return None
            a = int(m.group(1)) % 24
            b = int(m.group(2))
            b = b if b > a else min(a+3, 24)
            return f"{a:02d}-{b:02d}"
        df["hour_range"] = df["hour_range"].apply(_fmt_hr)
    if "date" in df.columns:
        df["date"] = to_date(df["date"])
    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    if "911_request_count_daily(before_24_hours)" not in df.columns and \
       {"date","hour_range","911_request_count_hour_range"}.issubset(df.columns):
        keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
        day = df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"].sum() \
                .reset_index(name="911_request_count_daily(before_24_hours)")
        df = df.merge(day, on=keys, how="left")

    if "event_hour" not in df.columns and "hour_range" in df.columns:
        df["event_hour"] = df["hour_range"].apply(_hour_from_range).astype("int16")
    if "received_time" not in df.columns:
        df["received_time"] = pd.NaT

    keep = [c for c in df.columns if c in ALLOWED_911_FEATURES or c in {"event_hour"}]
    return df[keep]

def ensure_local_911_base() -> Optional[Path]:
    for p in [LOCAL_911_Y, Path("./sf_911_last_5_year_y.csv"), Path("outputs/sf_911_last_5_year_y.csv")]:
        if p.exists(): log(f"📦 Yerel 911 (Y) bulundu: {p}"); return p
    for p in [LOCAL_911, Path("./sf_911_last_5_year.csv")]:
        if p.exists(): log(f"📦 Yerel 911 (regular) bulundu: {p}"); return p
    return None

def _pick_working_release_url(candidates: List[str]) -> str:
    for u in candidates:
        if not u: continue
        try:
            r = requests.get(u, timeout=20)
            if r.ok and r.content and len(r.content) > 200 and b"git-lfs" not in r.content[:200].lower():
                log(f"⬇️ Release seçildi: {u}")
                return u
            else:
                log(f"⚠️ Uygun değil (boş/LFS olabilir): {u}")
        except Exception as e:
            log(f"⚠️ Erişilemedi: {u} ({e})")
    raise RuntimeError("❌ Release URL’lerine ulaşılamadı.")

def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 okunuyor: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID":"string"})
    is_summary = {"date","hour_range"}.issubset(df.columns) and \
                 any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    df = coerce_to_summary_like(df) if is_summary else make_standard_summary(df)
    if min_date is not None and "date" in df.columns:
        df = df[df["date"] >= min_date]
    return df

def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release indiriliyor: {url}")
    r = requests.get(url, timeout=120); r.raise_for_status()
    tmp = Path(BASE_DIR) / "_tmp_911.csv"
    ensure_parent(str(tmp)); tmp.write_bytes(r.content)
    df = pd.read_csv(tmp, low_memory=False, dtype={"GEOID":"string"})
    is_summary = {"date","hour_range"}.issubset(df.columns) and \
                 any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    df = coerce_to_summary_like(df) if is_summary else make_standard_summary(df)
    if min_date is not None and "date" in df.columns:
        df = df[df["date"] >= min_date]
    return df

def try_small_request(params, headers):
    p = dict(params); p["$limit"], p["$offset"] = 1, 0
    r = requests.get(SF911_API_URL, headers=headers, params=p, timeout=REQUEST_TIMEOUT)
    r.raise_for_status(); return r

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
        log(f" ❌ Uygun datetime kolonu bulunamadı. Son hata: {last_err}")
        return None

    pieces, offset, page = [], 0, 1
    where_list = [f"{chosen_dt} between '{rng_start}' and '{rng_end}'" + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""),
                  f"{chosen_dt} between '{rng_start}' and '{rng_end}'"]
    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(SF911_API_URL, headers=headers,
                                 params={"$where": where_list[0], "$limit": CHUNK_LIMIT, "$offset": offset},
                                 timeout=REQUEST_TIMEOUT)
                if r.status_code == 400:
                    r = requests.get(SF911_API_URL, headers=headers,
                                     params={"$where": where_list[1], "$limit": CHUNK_LIMIT, "$offset": offset},
                                     timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                df = pd.read_json(io.BytesIO(r.content)); break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f" ❌ range page {page} (offset={offset}) hata: {e}")
                    df = None; break
                time.sleep(1.0 + attempt*0.5)
        if df is None or df.empty:
            if page == 1: log(" (bu aralıkta veri yok)")
            break
        pieces.append(df)
        if len(df) < CHUNK_LIMIT: break
        offset += CHUNK_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)
    if not pieces: return None
    return pd.concat(pieces, ignore_index=True)

def fetch_v3_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    sess = requests.Session()
    retries = Retry(total=5, connect=5, read=5, backoff_factor=1.2,
                    status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    headers = {"Accept":"application/json"}
    if SF_APP_TOKEN: headers["X-App-Token"] = SF_APP_TOKEN

    dt_candidates = ["received_time","received_datetime","date","datetime","call_datetime","received_dttm","call_date"]
    rng_start = f"{start_day}T00:00:00"; rng_end = f"{end_day}T23:59:59"

    chosen_dt, cols = None, None
    for dtc in dt_candidates:
        where = f"{dtc} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER: where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT 1 OFFSET 0"
        try:
            r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if obj.get("data"):
                chosen_dt = dtc
                cols = [c.get("fieldName") or c.get("name") or f"c{i}" for i,c in enumerate(obj.get("meta",{}).get("view",{}).get("columns",[]))]
                break
        except Exception:
            continue
    if not chosen_dt:
        log(" ❌ v3: uygun datetime kolonu yok.")
        return None

    all_rows, offset, page = [], 0, 1
    while True:
        where = f"{chosen_dt} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER: where += f" AND {AGENCY_FILTER}"
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
                    log(f" ❌ v3 range page {page} (offset={offset}) hata: {e}")
                time.sleep(1.0 + attempt*0.5)
        if got < V3_PAGE_LIMIT: break
        offset += V3_PAGE_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)
    return pd.DataFrame(all_rows)

def minimal_911_summary_ensure() -> pd.DataFrame:
    """Yerelde/yayın’da yoksa 5 yıllık 911 özetini üret/indir."""
    five_years_ago = FIVE_YEARS_AGO
    final_911 = None
    try:
        p = ensure_local_911_base()
        if p is not None:
            final_911 = summary_from_local(p, min_date=five_years_ago)
            safe_save_csv(final_911, str(LOCAL_911))
            safe_save_csv(final_911, str(LOCAL_911_Y))
            log(f"✅ Yerel 911 özet (satır: {len(final_911)})")
    except Exception as e:
        log("⚠️ Yerel 911 okunurken hata:")
        log("".join(traceback.format_exception(e)))

    if final_911 is None or final_911.empty:
        try:
            url = _pick_working_release_url(RAW_911_URL_CANDIDATES)
            final_911 = summary_from_release(url, min_date=five_years_ago)
            safe_save_csv(final_911, str(LOCAL_911))
            safe_save_csv(final_911, str(LOCAL_911_Y))
            log(f"✅ Release 911 özet (satır: {len(final_911)})")
        except Exception as e:
            log("⚠️ Release başarısız; API fallback denenecek:")
            log("".join(traceback.format_exception(e)))

    if final_911 is None or final_911.empty:
        try:
            today = TODAY_UTC
            raw = fetch_v3_range_all_chunks(five_years_ago, today) if IS_V3 else fetch_range_all_chunks(five_years_ago, today)
            if raw is None or raw.empty:
                raise RuntimeError("API boş döndü.")
            final_911 = make_standard_summary(raw)
            safe_save_csv(final_911, str(LOCAL_911))
            safe_save_csv(final_911, str(LOCAL_911_Y))
            log(f"✅ API 911 özet (satır: {len(final_911)})")
        except Exception as e:
            log("❌ API fallback hatası:")
            log("".join(traceback.format_exception(e)))
            raise
    return final_911

# =========================
# CRIME (sf_crime_y.csv / sf_crime.csv) OKUYUCU
# =========================
def pick_crime_source() -> Path:
    """Öncelik: sf_crime_y.csv → sf_crime.csv; varsa en güncelini seç."""
    cands: List[Path] = []
    for name in ["sf_crime_y.csv", "sf_crime.csv"]:
        for base in [Path(BASE_DIR), Path(".")]:
            p = base / name
            if p.exists():
                cands.append(p)
    if not cands:
        raise FileNotFoundError("Suç tabanı bulunamadı (sf_crime_y.csv veya sf_crime.csv).")
    # sf_crime_y.csv'e öncelik; yoksa mtime'a göre en güncel
    y = [p for p in cands if p.name == "sf_crime_y.csv"]
    if y: 
        return max(y, key=_mtime)
    return max([p for p in cands if p.name == "sf_crime.csv"], key=_mtime)

def read_crime_df(path: Path) -> pd.DataFrame:
    log(f"📥 Crime kaynağı: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID":"string"})
    if df.empty:
        raise ValueError("Suç dosyası boş.")

    # GEOID
    if "GEOID" not in df.columns:
        raise ValueError("Suç verisinde GEOID kolonu yok.")
    df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    # Zamanı yakala: date+time veya tek datetime
    ts_col = None
    if "datetime" in df.columns:
        ts_col = "datetime"
    else:
        # date + time varsa birleştir
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            t = df["time"] if "time" in df.columns else "00:00:00"
            dt = pd.to_datetime(d.dt.strftime("%Y-%m-%d") + " " + t.astype(str), errors="coerce")
            df["datetime"] = dt
            ts_col = "datetime"
        else:
            # başka zaman kolonlarını dene
            for c in ["occurred_at","report_datetime","incident_datetime","timestamp","received_time"]:
                if c in df.columns:
                    ts_col = c; break
            if ts_col is None:
                # her kolonu dene (name contains time/date)
                for c in df.columns:
                    if re.search(r"(time|date)", str(c), flags=re.I):
                        s = pd.to_datetime(df[c], errors="coerce")
                        if s.notna().mean() > 0.5:
                            df["datetime"] = s; ts_col = "datetime"; break

    if ts_col is None:
        raise ValueError("Suç verisinde zaman bilgisi bulunamadı (date/time veya datetime).")

    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[df["_ts"].notna()].copy()
    df["date"] = df["_ts"].dt.date
    df["event_hour"] = df["_ts"].dt.hour.astype("int16")
    df["hr_key"] = ((df["event_hour"] // 3) * 3).astype(int)
    df["hour_range"] = df["hr_key"].apply(lambda h: f"{h:02d}-{min(h+3,24):02d}")
    df["hour_ts"] = pd.to_datetime(df["date"].astype(str)) + pd.to_timedelta(df["hr_key"], unit="h")
    return df

# =========================
# 911 → ÖZELLİK ÜRET (LEAKAGE-SAFE) ve MERGE
# =========================
def prepare_calls(calls: pd.DataFrame) -> pd.DataFrame:
    df = calls.copy()
    if "911_request_count_hour_range" in df.columns:
        df = df.rename(columns={"911_request_count_hour_range": "hr_cnt"})
    if "911_request_count_daily(before_24_hours)" in df.columns:
        df = df.rename(columns={"911_request_count_daily(before_24_hours)": "daily_cnt"})

    if "date" not in df.columns or "hour_range" not in df.columns:
        raise ValueError("911 özetinde 'date' ve 'hour_range' kolonları gerekli.")

    df["date"] = to_date(df["date"])
    df["hr_key"] = df["hour_range"].str.slice(0,2).astype(int)
    df["hour_ts"] = pd.to_datetime(df["date"].astype(str)) + pd.to_timedelta(df["hr_key"], unit="h")

    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    df = df.sort_values(["GEOID","hour_ts"]).reset_index(drop=True)

    if "hr_cnt" not in df.columns:
        df["hr_cnt"] = 0
    if "daily_cnt" not in df.columns:
        tmp = df.groupby(["GEOID","date"], observed=True)["hr_cnt"].sum().reset_index(name="daily_cnt")
        df = df.merge(tmp, on=["GEOID","date"], how="left")

    # sızıntı güvenli laglar (olaylarla birleştirmeden ÖNCE shift)
    df["hr_cnt_t1"] = df.groupby("GEOID", observed=True)["hr_cnt"].shift(1).fillna(0).astype(int)
    df["hr_cnt_last3h"] = (df.groupby("GEOID", observed=True)["hr_cnt"]
                             .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
                             .fillna(0).astype(int))
    df["daily_cnt_prev"] = df.groupby("GEOID", observed=True)["daily_cnt"].shift(1).fillna(0).astype(int)

    # varsa dışarıdan gelen pencereler (3g/7g)
    for col in ["911_geo_last3d","911_geo_last7d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    keep_cols = ["GEOID","hour_ts","hr_cnt_t1","hr_cnt_last3h","daily_cnt_prev","date"]
    for k in ["911_geo_last3d","911_geo_last7d"]:
        if k in df.columns: keep_cols.append(k)
    return df[keep_cols]

def merge_crime_with_911(crime_df: pd.DataFrame, calls_feat: pd.DataFrame) -> pd.DataFrame:
    out = crime_df.merge(calls_feat, on=["GEOID","hour_ts"], how="left")
    for c in ["hr_cnt_t1","hr_cnt_last3h","daily_cnt_prev","911_geo_last3d","911_geo_last7d"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out

# =========================
# MAIN
# =========================
def main():
    log(f"📁 BASE_DIR: {BASE_DIR}")
    log(f"🗓️ Tarih aralığı: {FIVE_YEARS_AGO} → {TODAY_UTC}")

    # 1) 911 özetini hazırla/garanti et (öncelik: *_y.csv > regular)
    calls = minimal_911_summary_ensure()
    # Son 5 yıl filtresi ve bugüne kadar doldurma zaten yapıldı
    log_shape(calls, "911 summary (loaded)")

    # 2) Crime kaynağını seç + oku
    crime_path = pick_crime_source()
    crime_df = read_crime_df(crime_path)
    log_shape(crime_df, "crime (loaded)")

    # 3) 911 lag özelliklerini hazırla (leakage-safe)
    calls_feat = prepare_calls(calls)
    log_shape(calls_feat, "911 features (prepared)")

    # 4) Birleştir → fr_crime_01.csv
    merged = merge_crime_with_911(crime_df, calls_feat)
    log_shape(merged, "crime × 911 (merged)")

    # 5) Kaydet
    safe_save_csv(merged, OUT_FR_CRIME_01)
    log(f"✅ Yazıldı: {OUT_FR_CRIME_01}")

    # 6) Kısa önizleme (opsiyonel)
    try:
        preview_cols = ["GEOID","date","hour_range","hr_cnt_t1","hr_cnt_last3h","daily_cnt_prev"]
        show = [c for c in preview_cols if c in merged.columns]
        if show:
            print(merged[show].head(10).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
