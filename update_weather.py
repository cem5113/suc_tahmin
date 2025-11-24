# update_weather.py 

from datetime import datetime, timedelta, date, timezone
import os
import pandas as pd
import numpy as np

# Opsiyonel: PyGithub ve Meteostat
try:
    from github import Github  # only if upload wanted
except Exception:
    Github = None

try:
    from meteostat import Daily, Point
except Exception as e:
    print("⚠️ meteostat kurulu değilse yalnızca mevcut weather CSV'si ile devam ederiz:", e)
    Daily = None
    Point = None

pd.options.mode.copy_on_write = True

# =====================================================================================
# AYARLAR
# =====================================================================================
DATA_DIR      = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").rstrip("/")
WEATHER_CSV   = os.getenv("WEATHER_CSV", os.path.join(DATA_DIR, "sf_weather_5years.csv"))

UPLOAD_WEATHER_TO_GH = os.getenv("UPLOAD_WEATHER_TO_GH", "0") in ("1", "true", "True")
PROBE_GH_STATUS      = os.getenv("PROBE_GH_STATUS", "1") in ("1", "true", "True")
GITHUB_TOKEN         = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
REPO_NAME            = os.getenv("REPO_NAME", "cem5113/crime_prediction_data")
WEATHER_TARGET_PATH  = os.getenv("WEATHER_TARGET_PATH", f"{DATA_DIR}/sf_weather_5years.csv")

# Upload modu: force_update | skip_if_same
GH_UPLOAD_MODE       = os.getenv("GH_UPLOAD_MODE", "skip_if_same").strip()

# Meteostat ayarları
LAT, LON = float(os.getenv("WX_LAT", "37.7749")), float(os.getenv("WX_LON", "-122.4194"))
HOT_DAY_THRESHOLD_C = float(os.getenv("HOT_DAY_THRESHOLD_C", "25.0"))

# =====================================================================================
# TARİH PENCERESİ
# =====================================================================================
def five_year_window(today: date):
    try:
        start = today.replace(year=today.year - 5)
    except ValueError:
        start = today - timedelta(days=365*5 + 2)
    return (start + timedelta(days=1), today)

today = date.today()
win_start, win_end = five_year_window(today)
print(f"📅 5Y Pencere: {win_start} → {win_end}")

# =====================================================================================
# YARDIMCILAR
# =====================================================================================
def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lmap = {c.lower(): c for c in d.columns}
    def has(c): return c in lmap
    def col(c): return lmap[c]

    if has("date"):
        d[col("date")] = to_date(d[col("date")])
    elif has("time"):
        d["date"] = to_date(d[col("time")])
    elif has("datetime"):
        d["date"] = to_date(d[col("datetime")])

    ren = {}
    if has("temp_min") and not has("tmin"): ren[col("temp_min")] = "tmin"
    if has("temp_max") and not has("tmax"): ren[col("temp_max")] = "tmax"
    if has("precipitation_mm") and not has("prcp"): ren[col("precipitation_mm")] = "prcp"
    if has("prcp_mm") and not has("prcp"): ren[col("prcp_mm")] = "prcp"
    if has("taverage") and not has("tavg"): ren[col("taverage")] = "tavg"
    d.rename(columns=ren, inplace=True)

    for c in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["tavg", "tmin", "tmax", "prcp"]:
        if c not in d.columns:
            d[c] = np.nan

    d["temp_range"] = (d["tmax"] - d["tmin"]).astype(float)
    d["is_rainy"] = (pd.to_numeric(d.get("prcp", np.nan), errors="coerce").fillna(0) > 0).astype("Int64")
    d["is_hot_day"] = (pd.to_numeric(d.get("tmax", np.nan), errors="coerce") > HOT_DAY_THRESHOLD_C).astype("Int64")

    if "date" not in d.columns:
        d["date"] = pd.NaT
    d["date"] = to_date(d["date"])
    d.dropna(subset=["date"], inplace=True)
    d = d.drop_duplicates(subset=["date"]).sort_values("date")

    d = d[(d["date"] >= win_start) & (d["date"] <= win_end)].copy()

    final_cols = ["date", "tavg", "tmin", "tmax", "prcp", "temp_range", "is_rainy", "is_hot_day"]
    for c in final_cols:
        if c not in d.columns:
            d[c] = np.nan
    return d[final_cols]

def fetch_weather(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    if Daily is None or Point is None:
        print("ℹ️ meteostat yok → boş DataFrame dönüyorum.")
        return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])
    start_dt = datetime(start_d.year, start_d.month, start_d.day)
    end_dt   = datetime(end_d.year, end_d.month, end_d.day)
    df = Daily(Point(lat, lon), start_dt, end_dt).fetch().reset_index()
    df.rename(columns={"time": "date"}, inplace=True)
    return normalize_weather_columns(df)

def read_existing_weather(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])
    try:
        ex = pd.read_csv(path, low_memory=False)
        return normalize_weather_columns(ex)
    except Exception as e:
        print("⚠️ Mevcut weather dosyası okunamadı, baştan çekilecek:", e)
        return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])

# =====================================================================================
# (YENİ) EKSİK GÜNLERİ PREV-YEAR SAME-WEEK MEAN İLE DOLDUR
# =====================================================================================
def fill_missing_prev_year_same_week(allw: pd.DataFrame) -> pd.DataFrame:
    """
    5Y pencere içinde tam tarih evreni kurar.
    Eksik günleri bir önceki yıl aynı ISO haftasının ortalamasıyla doldurur.
    """
    if allw.empty:
        return allw

    num_cols = ["tavg","tmin","tmax","prcp"]

    # tam tarih evreni
    full_dates = pd.date_range(pd.to_datetime(win_start), pd.to_datetime(win_end), freq="D")
    full_df = pd.DataFrame({"date": full_dates.date})

    out = full_df.merge(allw, on="date", how="left")

    # ISO hafta bilgisi (mevcut veriden)
    base = allw.copy()
    base["date_ts"] = pd.to_datetime(base["date"])
    iso = base["date_ts"].dt.isocalendar()
    base["iso_year"] = iso.year.astype(int)
    base["iso_week"] = iso.week.astype(int)

    week_means = (
        base.groupby(["iso_year","iso_week"], as_index=False)[num_cols]
        .mean()
        .rename(columns={c: f"mean_{c}" for c in num_cols})
    )

    # out ISO info
    out["date_ts"] = pd.to_datetime(out["date"])
    iso_out = out["date_ts"].dt.isocalendar()
    out["iso_year"] = iso_out.year.astype(int)
    out["iso_week"] = iso_out.week.astype(int)

    missing_mask = out[num_cols].isna().any(axis=1)

    out.loc[missing_mask, "prev_iso_year"] = out.loc[missing_mask, "iso_year"] - 1
    out.loc[missing_mask, "prev_iso_week"] = out.loc[missing_mask, "iso_week"]

    out = out.merge(
        week_means,
        left_on=["prev_iso_year","prev_iso_week"],
        right_on=["iso_year","iso_week"],
        how="left",
        suffixes=("","_prev")
    )

    # doldur
    for c in num_cols:
        out[c] = out[c].fillna(out[f"mean_{c}"])

    # türevleri yeniden hesapla
    out["temp_range"] = (out["tmax"] - out["tmin"]).astype(float)
    out["is_rainy"]   = (pd.to_numeric(out["prcp"], errors="coerce").fillna(0) > 0).astype("Int64")
    out["is_hot_day"] = (pd.to_numeric(out["tmax"], errors="coerce") > HOT_DAY_THRESHOLD_C).astype("Int64")

    keep = ["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"]
    out = out[keep].drop_duplicates("date").sort_values("date")

    # rapor
    remain = out[num_cols].isna().sum().sum()
    if remain > 0:
        print(f"⚠️ Prev-year same-week ile doldurulamayan NaN sayısı: {remain}")
    else:
        print("✅ Tüm eksikler prev-year same-week ile dolduruldu.")

    return out

# ---------- GitHub yardımcıları ----------
def _get_repo():
    if Github is None or not GITHUB_TOKEN:
        return None
    try:
        return Github(GITHUB_TOKEN).get_repo(REPO_NAME)
    except Exception as e:
        print("⚠️ Repo erişimi başarısız:", e)
        return None

def github_file_status(path: str):
    """
    Döndürür: dict(exists: bool, size:int|None, sha:str|None, html_url:str|None, last_commit_iso:str|None, content:str|None)
    """
    repo = _get_repo()
    if repo is None:
        return {"exists": False, "size": None, "sha": None, "html_url": None, "last_commit_iso": None, "content": None}
    try:
        contents = repo.get_contents(path)
        commits = repo.get_commits(path=path)
        last_iso = None
        try:
            c = next(iter(commits))
            dt = getattr(c.commit.author, "date", None)
            if isinstance(dt, datetime):
                last_iso = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        return {
            "exists": True,
            "size": getattr(contents, "size", None),
            "sha": getattr(contents, "sha", None),
            "html_url": getattr(contents, "html_url", None),
            "last_commit_iso": last_iso,
            "content": contents.decoded_content.decode("utf-8", errors="ignore"),
        }
    except Exception:
        return {"exists": False, "size": None, "sha": None, "html_url": None, "last_commit_iso": None, "content": None}

def upsert_github_csv_smart(df: pd.DataFrame, target_path: str):
    repo = _get_repo()
    if repo is None:
        print("ℹ️ GitHub upload atlandı (token veya PyGithub yok).")
        return

    csv_str = df.to_csv(index=False)
    status = github_file_status(target_path)

    if PROBE_GH_STATUS:
        if status["exists"]:
            print(f"🔎 GH Durum: VAR — {target_path} (boyut={status['size']}, son_commit={status['last_commit_iso']})")
        else:
            print(f"🔎 GH Durum: YOK — {target_path}")

    if not UPLOAD_WEATHER_TO_GH:
        return

    same = (status["content"] == csv_str) if status["exists"] and status["content"] is not None else False

    if status["exists"]:
        if GH_UPLOAD_MODE == "skip_if_same" and same:
            print("✅ GH güncel: içerik aynı, update atlandı.")
            return
        try:
            repo.update_file(
                status["html_url"].split("blob/")[-1].split("/", 1)[-1] if status["html_url"] else target_path,
                f"update {os.path.basename(target_path)}",
                csv_str,
                status["sha"],
                branch="main"
            )
            print(f"✅ GitHub güncellendi: {target_path}")
        except Exception:
            contents = repo.get_contents(target_path)
            repo.update_file(contents.path, f"update {os.path.basename(target_path)}", csv_str, contents.sha, branch="main")
            print(f"✅ GitHub güncellendi: {target_path}")
    else:
        repo.create_file(target_path, f"add {os.path.basename(target_path)}", csv_str, branch="main")
        print(f"🆕 GitHub oluşturuldu: {target_path}")

# =====================================================================================
# (YENİ) WEATHER DF CACHE
# =====================================================================================
_WEATHER_LATEST: pd.DataFrame | None = None
def get_weather_df() -> pd.DataFrame:
    global _WEATHER_LATEST
    if _WEATHER_LATEST is not None:
        return _WEATHER_LATEST
    if os.path.exists(WEATHER_CSV):
        try:
            df = pd.read_csv(WEATHER_CSV, low_memory=False)
            return normalize_weather_columns(df)
        except Exception:
            pass
    return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])

# =====================================================================================
# WEATHER GÜNCELLE (MERGE/08 YOK)
# =====================================================================================
existing = read_existing_weather(WEATHER_CSV)
last_date = existing["date"].max() if not existing.empty else None
fetch_start = (last_date + timedelta(days=1)) if pd.notna(last_date) else win_start
fetch_end   = win_end

if fetch_start <= fetch_end:
    print(f"📥 Meteostat Daily: {fetch_start} → {fetch_end}")
    neww = fetch_weather(LAT, LON, fetch_start, fetch_end)
    print(f"✅ Yeni gün sayısı: {len(neww)}")
    allw = pd.concat([existing, neww], ignore_index=True) if not existing.empty else neww.copy()
else:
    print("ℹ️ Weather güncel; indirilecek yeni gün yok.")
    allw = existing.copy()

# normalize + pencere kırp + tekilleştir
allw = normalize_weather_columns(allw)
allw = allw.drop_duplicates(subset=["date"]).sort_values("date")
allw = allw[(allw["date"] >= win_start) & (allw["date"] <= win_end)].copy()

# ✅ NEW: eksik günleri doldur
allw = fill_missing_prev_year_same_week(allw)

# Kaydet (local)
os.makedirs(os.path.dirname(WEATHER_CSV), exist_ok=True)
allw.to_csv(WEATHER_CSV, index=False)
print(f"💾 Weather kaydedildi: {WEATHER_CSV} — {len(allw)} satır, {allw['date'].min()} → {allw['date'].max()}")

# Bellek içi cache
_WEATHER_LATEST = allw.copy()

# GitHub durumu raporla + gerekirse yükle
if Github is not None and (PROBE_GH_STATUS or UPLOAD_WEATHER_TO_GH):
    upsert_github_csv_smart(allw, WEATHER_TARGET_PATH)
