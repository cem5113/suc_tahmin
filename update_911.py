# update_911.py

from __future__ import annotations

import io
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


# =========================
# CONFIG
# =========================

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

LOCAL_NAME = "sf_911_last_5_year.csv"
Y_NAME = "sf_911_last_5_year_y.csv"
local_summary_path = Path(BASE_DIR) / LOCAL_NAME
y_summary_path = Path(BASE_DIR) / Y_NAME

# Socrata API
SF911_API_URL = os.getenv("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF_APP_TOKEN = os.getenv("SF911_API_TOKEN", "")
AGENCY_FILTER = os.getenv("SF911_AGENCY_FILTER", "agency like '%Police%'")
REQUEST_TIMEOUT = int(os.getenv("SF911_REQUEST_TIMEOUT", "60"))
CHUNK_LIMIT = int(os.getenv("SF911_CHUNK_LIMIT", "50000"))
MAX_RETRIES = int(os.getenv("SF911_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS = float(os.getenv("SF911_SLEEP", "0.2"))
IS_V3 = "/api/v3/views/" in SF911_API_URL
V3_PAGE_LIMIT = int(os.getenv("SF_V3_PAGE_LIMIT", "1000"))

# Releases (prefer _y, then regular)
RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()
RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]


# =========================
# HELPERS
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

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:int(target_len)].str.zfill(int(target_len))

def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def log_shape(df, label):
    try:
        r, c = df.shape
        log(f"📊 {label}: {r} satır × {c} sütun")
    except Exception:
        pass

def log_date_range(df, date_col="date", label="911"):
    if date_col not in df.columns:
        log(f"⚠️ {label}: '{date_col}' yok.")
        return
    s = pd.to_datetime(df[date_col], errors="coerce")
    s = s.dt.date.dropna()
    if s.empty:
        log(f"⚠️ {label}: tarih parse edilemedi.")
        return
    log(f"🧭 {label} tarihi aralığı: {s.min()} → {s.max()} (gün={s.nunique()})")


# =========================
# CORE SUMMARIZATION
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

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """From raw 911 rows -> hourly + daily counts. No lat/lon, no spatial joins."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["GEOID","date","hour_range","event_hour",
                                     "911_request_count_hour_range","911_request_count_daily(before_24_hours)"])

    df = raw.copy()

    # pick timestamp column
    ts_col = None
    for cand in ["received_time","received_datetime","date","datetime","timestamp","call_received_datetime"]:
        if cand in df.columns:
            ts_col = cand; break
    if ts_col is None:
        raise ValueError("Zaman kolonu yok (received_time/received_datetime/date/...)")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[df[ts_col].notna()].copy()

    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour.astype("int16")
    df["hour_range"] = df["event_hour"].apply(_fmt_hour_range_from_hour)

    # hourly counts
    grp_hr = (["GEOID"] if "GEOID" in df.columns else []) + ["date","hour_range"]
    hr = df.groupby(grp_hr, dropna=False, observed=True).size().reset_index(name="911_request_count_hour_range")

    # daily counts (previous 24h ≈ same-day daily total for modeling; if stricter is needed, shift can be added)
    grp_day = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
    day = df.groupby(grp_day, dropna=False, observed=True).size().reset_index(name="911_request_count_daily(before_24_hours)")

    out = hr.merge(day, on=grp_day, how="left")

    # If we have event_hour in original rows, we can approximate by the start of hour_range; else derive from label
    def _hour_from_range(s: str) -> int:
        m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(s))
        return int(m.group(1)) % 24 if m else 0
    out["event_hour"] = out["hour_range"].apply(_hour_from_range).astype("int16")

    # keep requested order
    keep = ["event_hour","911_request_count_hour_range","received_time",
            "911_request_count_daily(before_24_hours)","date","hour_range","GEOID"]
    # received_time doesn't exist after aggregation; include placeholder if necessary to satisfy schema
    if "received_time" not in out.columns:
        out["received_time"] = pd.NaT
    # normalize
    if "GEOID" in out.columns:
        out["GEOID"] = normalize_geoid(out["GEOID"], DEFAULT_GEOID_LEN)
    out["date"] = to_date(out["date"])

    # final column order filtered by presence
    keep_present = [c for c in keep if c in out.columns]
    # put date/hour_range/GEOID last as in your spec
    tail = [c for c in ["date","hour_range","GEOID"] if c in out.columns]
    head = [c for c in keep_present if c not in tail]
    out = out[head + tail]
    return out


def coerce_to_summary_like(df: pd.DataFrame) -> pd.DataFrame:
    """If incoming CSV is already an hourly summary, normalize names and formats."""
    df = df.copy()
    # unify count column name
    for cnt in ["911_request_count_hour_range","call_count","count","requests","n"]:
        if cnt in df.columns:
            if cnt != "911_request_count_hour_range":
                df = df.rename(columns={cnt: "911_request_count_hour_range"})
            break

    # hour_range normalize
    if "hour_range" in df.columns:
        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m: return None
            a = int(m.group(1)) % 24; b = int(m.group(2)); b = b if b > a else min(a+3, 24)
            return f"{a:02d}-{b:02d}"
        df["hour_range"] = df["hour_range"].apply(_fmt_hr)

    # date normalize
    if "date" in df.columns:
        df["date"] = to_date(df["date"])

    # geoid normalize
    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    # ensure daily column
    if "911_request_count_daily(before_24_hours)" not in df.columns and \
       {"date","hour_range","911_request_count_hour_range"}.issubset(df.columns):
        keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
        day = df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"] \
                .sum().reset_index(name="911_request_count_daily(before_24_hours)")
        df = df.merge(day, on=keys, how="left")

    # ensure event_hour
    if "event_hour" not in df.columns and "hour_range" in df.columns:
        def _hour_from_range(s: str) -> int:
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(s))
            return int(m.group(1)) % 24 if m else 0
        df["event_hour"] = df["hour_range"].apply(_hour_from_range).astype("int16")

    # received_time is not defined in summaries; add placeholder
    if "received_time" not in df.columns:
        df["received_time"] = pd.NaT

    keep = [c for c in df.columns if c in ALLOWED_911_FEATURES or c in {"event_hour"}]
    return df[keep]


# =========================
# LOADERS
# =========================

def ensure_local_911_base() -> Optional[Path]:
    y_candidates = [
        Path(BASE_DIR) / "sf_911_last_5_year_y.csv",
        Path("./sf_911_last_5_year_y.csv"),
        Path("outputs/sf_911_last_5_year_y.csv"),
    ]
    for p in y_candidates:
        if p.exists() and p.suffix == ".csv":
            log(f"📦 Yerel 911 (Y) bulundu: {p}")
            return p

    regular_candidates = [
        Path(BASE_DIR) / "sf_911_last_5_year.csv",
        Path("./sf_911_last_5_year.csv"),
    ]
    for p in regular_candidates:
        if p.exists() and p.suffix == ".csv":
            log(f"📦 Yerel 911 (regular) bulundu: {p}")
            return p
    return None

def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 okunuyor: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID":"string"})
    is_already_summary = (
        {"date","hour_range"}.issubset(df.columns)
        and any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    )
    df = coerce_to_summary_like(df) if is_already_summary else make_standard_summary(df)
    if min_date is not None and "date" in df.columns:
        df = df[df["date"] >= min_date]
    return df

def _pick_working_release_url(candidates: list[str]) -> str:
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

def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release indiriliyor: {url}")
    r = requests.get(url, timeout=120); r.raise_for_status()
    tmp = Path(BASE_DIR) / "_tmp_911.csv"
    ensure_parent(str(tmp))
    tmp.write_bytes(r.content)
    df = pd.read_csv(tmp, low_memory=False, dtype={"GEOID":"string"})
    is_already_summary = (
        {"date","hour_range"}.issubset(df.columns)
        and any(c in df.columns for c in ["911_request_count_hour_range","call_count","count","requests","n"])
    )
    df = coerce_to_summary_like(df) if is_already_summary else make_standard_summary(df)
    if min_date is not None and "date" in df.columns:
        df = df[df["date"] >= min_date]
    return df


# =========================
# API (INCREMENTAL RANGE)
# =========================

def try_small_request(params, headers):
    p = dict(params)
    p["$limit"], p["$offset"] = 1, 0
    r = requests.get(SF911_API_URL, headers=headers, params=p, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r

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
            if page == 1:
                log(" (bu aralıkta veri yok)")
            break

        pieces.append(df)
        if len(df) < CHUNK_LIMIT:
            break
        offset += CHUNK_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)

    if not pieces:
        return None
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

        if got < V3_PAGE_LIMIT:
            break
        offset += V3_PAGE_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)

    return pd.DataFrame(all_rows)

def incremental_summary(start_day, end_day) -> pd.DataFrame:
    if start_day is None or end_day is None or end_day < start_day:
        return pd.DataFrame()
    log(f"🌐 API artımlı: {start_day} → {end_day} ({(end_day - start_day).days + 1} gün)")
    raw = fetch_v3_range_all_chunks(start_day, end_day) if IS_V3 else fetch_range_all_chunks(start_day, end_day)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return make_standard_summary(raw)


# =========================
# MAIN
# =========================

def save_911_both(df: pd.DataFrame):
    safe_save_csv(df, str(local_summary_path))
    safe_save_csv(df, str(y_summary_path))
    log(f"💾 Kaydedildi → {local_summary_path.name} & {y_summary_path.name} (satır={len(df)})")

def main():
    five_years_ago = datetime.now(timezone.utc).date() - timedelta(days=5*365)
    log(f"📁 Hedef: {local_summary_path}")
    final_911 = None

    # 1) Local
    try:
        base_csv_path = ensure_local_911_base()
        if base_csv_path is not None:
            final_911 = summary_from_local(base_csv_path, min_date=five_years_ago)
            save_911_both(final_911)
            log(f"✅ Yerel özet okundu (satır: {len(final_911)})")
        else:
            log("ℹ️ Yerel bulunamadı; release denenecek.")
    except Exception as e:
        log("⚠️ Yerel okunurken hata:")
        log("".join(traceback.format_exception(e)))

    # 2) Release
    if final_911 is None or final_911.empty:
        try:
            release_url = _pick_working_release_url(RAW_911_URL_CANDIDATES)
            final_911 = summary_from_release(release_url, min_date=five_years_ago)
            save_911_both(final_911)
            log(f"✅ Release özet indirildi (satır: {len(final_911)})")
        except Exception as e:
            log("⚠️ Release başarısız; API fallback denenecek:")
            log("".join(traceback.format_exception(e)))

    # 3) API fallback
    if final_911 is None or final_911.empty:
        try:
            today = datetime.utcnow().date()
            final_911 = incremental_summary(five_years_ago, today)
            if final_911 is None or final_911.empty:
                log("❌ 911 özet üretilemedi: lokal yok, release yok, API boş.")
                sys.exit(1)
            save_911_both(final_911)
            log(f"✅ API özet üretildi (satır: {len(final_911)})")
        except Exception as e:
            log("❌ API fallback hatası:")
            log("".join(traceback.format_exception(e)))
            sys.exit(1)

    # 4) Final normalize & keep only the requested columns
    keep_cols = ["event_hour","911_request_count_hour_range","received_time",
                 "911_request_count_daily(before_24_hours)","date","hour_range","GEOID"]
    final_911 = final_911[[c for c in keep_cols if c in final_911.columns]].copy()

    # Enforce ordering
    tail = [c for c in ["date","hour_range","GEOID"] if c in final_911.columns]
    head = [c for c in keep_cols if c not in tail and c in final_911.columns]
    final_911 = final_911[head + tail]

    save_911_both(final_911)
    log_shape(final_911, "911 summary (final)")
    log_date_range(final_911, "date", "911")

    # Preview
    try:
        print(final_911.head(10).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
