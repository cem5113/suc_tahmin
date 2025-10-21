# update_weather.py — weather'ı güncelle + sf_crime_07 ile birleştir → sf_crime_08.csv
# Robust, header-safe, çok-kent destekli (prefix: sf, fr, ...)

from datetime import datetime, timedelta, date
import os
import sys
from typing import Optional, List
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

# =====================================================================================
# AYARLAR
# =====================================================================================
DATA_DIR      = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").rstrip("/")

# Weather dosyası (güncellenecek / oluşturulacak)
WEATHER_CSV   = os.getenv("WEATHER_CSV", os.path.join(DATA_DIR, "sf_weather_5years.csv"))

# Çok-kent: virgülle ayır (örn: "sf,fr")
CITY_PREFIXES = [s.strip() for s in os.getenv("CITY_PREFIXES", "sf").split(",") if s.strip()]

# GitHub'a yalnızca istenirse yükle (hem token hem repo bilgileri lazımdır)
UPLOAD_WEATHER_TO_GH = os.getenv("UPLOAD_WEATHER_TO_GH", "0") in ("1", "true", "True")
UPLOAD_08_TO_GH      = os.getenv("UPLOAD_08_TO_GH", "0") in ("1", "true", "True")
GITHUB_TOKEN         = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
REPO_NAME            = os.getenv("REPO_NAME", "cem5113/crime_prediction_data")
WEATHER_TARGET_PATH  = os.getenv("WEATHER_TARGET_PATH", f"{DATA_DIR}/sf_weather_5years.csv")

# Meteostat ayarları
LAT, LON = float(os.getenv("WX_LAT", "37.7749")), float(os.getenv("WX_LON", "-122.4194"))  # San Francisco
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
    """Weather kolonlarını standardize eder (header-safe)."""
    d = df.copy()
    lmap = {c.lower(): c for c in d.columns}
    def has(c): return c in lmap
    def col(c): return lmap[c]

    # date/datetime/time → date
    if has("date"):
        d[col("date")] = to_date(d[col("date")])
    elif has("time"):
        d["date"] = to_date(d[col("time")])
    elif has("datetime"):
        d["date"] = to_date(d[col("datetime")])

    # yeniden adla
    ren = {}
    if has("temp_min") and not has("tmin"): ren[col("temp_min")] = "tmin"
    if has("temp_max") and not has("tmax"): ren[col("temp_max")] = "tmax"
    if has("precipitation_mm") and not has("prcp"): ren[col("precipitation_mm")] = "prcp"
    if has("prcp_mm") and not has("prcp"): ren[col("prcp_mm")] = "prcp"
    if has("taverage") and not has("tavg"): ren[col("taverage")] = "tavg"
    d.rename(columns=ren, inplace=True)

    # tipler
    for c in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Yok ise kolon oluştur
    for c in ["tavg", "tmin", "tmax", "prcp"]:
        if c not in d.columns:
            d[c] = np.nan

    # temp_range, is_rainy, is_hot_day
    d["temp_range"] = (d["tmax"] - d["tmin"]).astype(float)
    d["is_rainy"] = (pd.to_numeric(d.get("prcp", np.nan), errors="coerce").fillna(0) > 0).astype("Int64")
    d["is_hot_day"] = (pd.to_numeric(d.get("tmax", np.nan), errors="coerce") > HOT_DAY_THRESHOLD_C).astype("Int64")

    # tarih zorunlu
    if "date" not in d.columns:
        d["date"] = pd.NaT

    d["date"] = to_date(d["date"])
    d.dropna(subset=["date"], inplace=True)
    d = d.drop_duplicates(subset=["date"]).sort_values("date")

    # pencere kırp
    d = d[(d["date"] >= win_start) & (d["date"] <= win_end)].copy()

    # sadece gerekli kolonlar
    final_cols = ["date", "tavg", "tmin", "tmax", "prcp", "temp_range", "is_rainy", "is_hot_day"]
    for c in final_cols:
        if c not in d.columns:
            d[c] = np.nan
    return d[final_cols]

def fetch_weather(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    """Meteostat Daily ile veri çek; header-safe normalize et."""
    if Daily is None or Point is None:
        print("ℹ️ meteostat yok → boş DataFrame dönüyorum.")
        return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])
    start_dt = datetime(start_d.year, start_d.month, start_d.day)
    end_dt   = datetime(end_d.year, end_d.month, end_d.day)
    df = Daily(Point(lat, lon), start_dt, end_dt).fetch().reset_index()
    df.rename(columns={"time": "date"}, inplace=True)
    df = normalize_weather_columns(df)
    return df

def read_existing_weather(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])
    try:
        ex = pd.read_csv(path, low_memory=False)
        ex = normalize_weather_columns(ex)
        return ex
    except Exception as e:
        print("⚠️ Mevcut weather dosyası okunamadı, baştan çekilecek:", e)
        return pd.DataFrame(columns=["date","tavg","tmin","tmax","prcp","temp_range","is_rainy","is_hot_day"])

def upsert_github_csv(df: pd.DataFrame, target_path: str):
    if not (UPLOAD_WEATHER_TO_GH or UPLOAD_08_TO_GH):
        return
    if Github is None or not GITHUB_TOKEN:
        print("ℹ️ GitHub upload atlandı (token veya PyGithub yok).")
        return
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        csv_str = df.to_csv(index=False)
        try:
            contents = repo.get_contents(target_path)
            repo.update_file(contents.path, f"update {os.path.basename(target_path)}", csv_str, contents.sha, branch="main")
            print(f"✅ GitHub güncellendi: {target_path}")
        except Exception:
            repo.create_file(target_path, f"add {os.path.basename(target_path)}", csv_str, branch="main")
            print(f"🆕 GitHub oluşturuldu: {target_path}")
    except Exception as e:
        print("⚠️ GitHub yükleme hatası:", e)

def normalize_crime07(df: pd.DataFrame) -> pd.DataFrame:
    """sf_crime_07 için tarih kolonunu güvenle normalize et; GEOID vb. dokunma."""
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}
    date_col = None
    for cand in ("date", "datetime", "time"):
        if cand in cols:
            date_col = cols[cand]
            break
    if date_col is None:
        # hiçbir tarih kolonu yoksa bırakalım (merge edemeyiz)
        d["date"] = pd.NaT
    else:
        d["date"] = to_date(d[date_col])
    d.dropna(subset=["date"], inplace=True)
    d["date"] = d["date"].astype("object")  # date dtype
    return d

def merge_07_weather_to_08(crime07_path: str, weather_df: pd.DataFrame, out_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(crime07_path):
        print(f"❌ {crime07_path} bulunamadı; {out_path} üretilemedi.")
        return None
    try:
        c7 = pd.read_csv(crime07_path, low_memory=False)
    except Exception as e:
        print(f"❌ {crime07_path} okunamadı:", e)
        return None

    c7 = normalize_crime07(c7)

    # Weather kolonlarına wx_ prefix ekleyelim (çakışmayı önler)
    w = weather_df.copy()
    wx_cols = [c for c in w.columns if c != "date"]
    w = w.rename(columns={c: f"wx_{c}" for c in wx_cols})

    # LEFT MERGE (suç verisi master)
    merged = c7.merge(w, on="date", how="left", validate="m:1")

    # Kaydet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"💾 {out_path} yazıldı. Satır: {len(merged)}")
    return merged

# =====================================================================================
# WEATHER GÜNCELLE
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

# Kaydet (local)
os.makedirs(os.path.dirname(WEATHER_CSV), exist_ok=True)
allw.to_csv(WEATHER_CSV, index=False)
print(f"💾 Weather kaydedildi: {WEATHER_CSV} — {len(allw)} satır, {allw['date'].min()} → {allw['date'].max()}")

# İstenirse GitHub'a yükle
if UPLOAD_WEATHER_TO_GH:
    upsert_github_csv(allw, WEATHER_TARGET_PATH)

# =====================================================================================
# CRIME_07 → CRIME_08 BİRLEŞTİRME
# =====================================================================================
produced_any = False
for pref in CITY_PREFIXES:
    pref = pref.strip()
    if not pref:
        continue
    in07  = os.path.join(DATA_DIR, f"{pref}_crime_07.csv")
    out08 = os.path.join(DATA_DIR, f"{pref}_crime_08.csv")
    print(f"🔗 Birleştiriliyor: {os.path.basename(in07)} + weather → {os.path.basename(out08)}")
    m = merge_07_weather_to_08(in07, allw, out08)
    if m is not None:
        produced_any = True
        if UPLOAD_08_TO_GH:
            # her şehir için ayrı hedef yolu isimlendir
            target08 = f"{DATA_DIR}/{os.path.basename(out08)}"
            upsert_github_csv(m, target08)

if not produced_any:
    print("ℹ️ Hiçbir 07 dosyası bulunamadı; 08 üretilemedi.")
else:
    print("🎉 Weather birleştirme tamam. 08 dosyaları hazır.")
