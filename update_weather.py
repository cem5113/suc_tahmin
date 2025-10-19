# update_weather.py — robust & header-safe (Meteostat Daily, PyGithub)

from datetime import datetime, timedelta, date
import os
import pandas as pd
import numpy as np

from meteostat import Daily, Point
from github import Github

# === ENV / GITHUB AYARLARI ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
REPO_NAME    = "cem5113/crime_prediction_data"
CRIME_DIR    = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").rstrip("/")
TARGET_PATH  = f"{CRIME_DIR}/sf_weather_5years.csv"   # repo içindeki yol

# === METEOSTAT / ÇIKTI AYARLARI ===
OUT_CSV             = "sf_weather_5years.csv"         # runner'da local dosya
HOT_DAY_THRESHOLD_C = 25.0
LAT, LON            = 37.7749, -122.4194              # San Francisco

# Opsiyonel: tmin/tmax eksikse tavg etrafında basit doldurma (0/1)
FILL_TMIN_TMAX_FROM_TAVG = os.getenv("FILL_TMIN_TMAX_FROM_TAVG", "0").lower() in ("1","true","yes","on")

# === 5 yıl penceresi ===
def five_year_window(today: date):
    try:
        start = today.replace(year=today.year - 5)
    except ValueError:
        start = today - timedelta(days=365*5 + 2)
    return (start + timedelta(days=1), today)

today = date.today()
win_start, win_end = five_year_window(today)
print(f"📅 Hedef aralık: {win_start} → {win_end}")

# === Yardımcılar ===
def normalize_existing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Varsa farklı isimli kolonları standarda dönüştürür:
    - temp_min/temp_max -> tmin/tmax
    - precipitation_mm/prcp_mm -> prcp
    Saatlik 'temp' varsa günlük min/max üretir.
    """
    d = df.copy()
    lower_map = {c.lower(): c for c in d.columns}
    def has(col): return col in lower_map
    def col(col): return lower_map[col]

    # Tarih sütunu
    if has('date'):
        d[col('date')] = pd.to_datetime(d[col('date')], errors='coerce').dt.date
    elif has('time'):
        d['date'] = pd.to_datetime(d[col('time')], errors='coerce').dt.date
    elif has('datetime'):
        d['date'] = pd.to_datetime(d[col('datetime')], errors='coerce').dt.date

    # İsimleri standarda çek
    rename_pairs = {}
    if has('temp_min') and not has('tmin'): rename_pairs[col('temp_min')] = 'tmin'
    if has('temp_max') and not has('tmax'): rename_pairs[col('temp_max')] = 'tmax'
    if has('precipitation_mm') and not has('prcp'): rename_pairs[col('precipitation_mm')] = 'prcp'
    if has('prcp_mm') and not has('prcp'): rename_pairs[col('prcp_mm')] = 'prcp'
    if has('taverage') and not has('tavg'): rename_pairs[col('taverage')] = 'tavg'
    d.rename(columns=rename_pairs, inplace=True)

    # Tipler
    for c in ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    # Saatlik 'temp' → günlük min/max
    has_temp_hourly = 'temp' in d.columns
    if has_temp_hourly:
        time_col = None
        for cand in ['datetime', 'time', 'Timestamp']:
            if cand in d.columns:
                time_col = cand
                break
        if time_col is not None:
            tmp = (
                d[[time_col, 'temp']]
                .assign(date=lambda x: pd.to_datetime(x[time_col], errors='coerce').dt.date)
                .dropna(subset=['date'])
                .groupby('date')['temp'].agg(tmin='min', tmax='max', tavg='mean')
                .reset_index()
            )
            if 'date' not in d.columns:
                d['date'] = pd.to_datetime(d[time_col], errors='coerce').dt.date
            d = d.drop_duplicates(subset=['date'])
            d = d.merge(tmp, on='date', how='left')

    # Beklenen kolonlar yoksa oluştur
    for c in ['tavg', 'tmin', 'tmax', 'prcp']:
        if c not in d.columns:
            d[c] = np.nan

    if 'date' not in d.columns:
        d['date'] = pd.NaT

    return d

def fetch_daily(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    start_dt = datetime(start_d.year, start_d.month, start_d.day)
    end_dt   = datetime(end_d.year, end_d.month, end_d.day)
    pt = Point(lat, lon)

    df = Daily(pt, start_dt, end_dt).fetch().reset_index()
    df.rename(columns={'time': 'date'}, inplace=True)
    df = normalize_existing_columns(df)
    keep = ['date', 'tavg', 'tmin', 'tmax', 'prcp']
    df = df[keep]
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    return df

# === Mevcut dosyayı oku (varsa) ===
base_df = pd.DataFrame()
if os.path.exists(OUT_CSV):
    try:
        existing = pd.read_csv(OUT_CSV)
        existing = normalize_existing_columns(existing)
        existing['date'] = pd.to_datetime(existing['date'], errors='coerce').dt.date
        last_date = existing['date'].max()
        fetch_start = (last_date + timedelta(days=1)) if pd.notna(last_date) else win_start
        base_df = existing
        print(f"🗂️ Mevcut veri bulundu: {len(existing)} satır, son gün={last_date}")
    except Exception as e:
        print("⚠️ Dosya okunamadı, baştan çekilecek:", e)
        fetch_start = win_start
else:
    fetch_start = win_start

fetch_end = win_end

# === Yeni veri indir ===
if fetch_start <= fetch_end:
    print(f"📥 İndiriliyor: {fetch_start} → {fetch_end}")
    new_df = fetch_daily(LAT, LON, fetch_start, fetch_end)
    print(f"✅ Yeni çekilen satır: {len(new_df)}")
else:
    print("ℹ️ Güncel: indirilecek yeni gün yok.")
    new_df = pd.DataFrame(columns=['date','tavg','tmin','tmax','prcp'])

# === Birleştir + normalize ===
all_df = pd.concat([base_df, new_df], ignore_index=True) if len(base_df) else new_df.copy()
all_df = normalize_existing_columns(all_df)
all_df['date'] = pd.to_datetime(all_df['date'], errors='coerce').dt.date
all_df.dropna(subset=['date'], inplace=True)

# Opsiyonel hafif backfill (tavg varsa)
if FILL_TMIN_TMAX_FROM_TAVG:
    all_df['tavg'] = pd.to_numeric(all_df['tavg'], errors='coerce')
    all_df['tmin'] = pd.to_numeric(all_df['tmin'], errors='coerce')
    all_df['tmax'] = pd.to_numeric(all_df['tmax'], errors='coerce')
    m = all_df['tmin'].isna() & all_df['tavg'].notna()
    all_df.loc[m, 'tmin'] = all_df.loc[m, 'tavg'] - 3.0
    m = all_df['tmax'].isna() & all_df['tavg'].notna()
    all_df.loc[m, 'tmax'] = all_df.loc[m, 'tavg'] + 3.0

# --- NA-güvenli bayraklar ---
pr = pd.to_numeric(all_df.get('prcp'), errors='coerce')
tx = pd.to_numeric(all_df.get('tmax'), errors='coerce')

all_df['is_rainy'] = (pr > 0).astype('Int64')
all_df.loc[pr.isna(), 'is_rainy'] = pd.NA

all_df['is_hot_day'] = (tx > HOT_DAY_THRESHOLD_C).astype('Int64')
all_df.loc[tx.isna(), 'is_hot_day'] = pd.NA

# temp_range
if 'tmax' in all_df.columns and 'tmin' in all_df.columns:
    all_df['temp_range'] = all_df['tmax'] - all_df['tmin']
else:
    all_df['temp_range'] = np.nan

# Kolon sırası
final_cols = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'temp_range', 'is_rainy', 'is_hot_day']
for c in final_cols:
    if c not in all_df.columns:
        all_df[c] = np.nan
all_df = all_df[final_cols]

# Tekilleştir, sırala, pencere kırp
all_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
all_df.sort_values('date', inplace=True)
mask = (all_df['date'] >= win_start) & (all_df['date'] <= win_end)
all_df = all_df.loc[mask].copy()

# === Kaydet (local + CRIME_DIR/sf_crime_08.csv) ===
all_df.to_csv(OUT_CSV, index=False)
print(f"💾 Kaydedildi: {OUT_CSV} — {len(all_df)} satır, {all_df['date'].min()} → {all_df['date'].max()}")

os.makedirs(CRIME_DIR, exist_ok=True)
out_08 = os.path.join(CRIME_DIR, "sf_crime_08.csv")
all_df.to_csv(out_08, index=False)
print(f"💾 Kaydedildi: {out_08}")

# === GitHub’a yükle ===
if not GITHUB_TOKEN:
    raise SystemExit("❌ GITHUB_TOKEN tanımlı değil. Secrets’e ekleyin: GITHUB_TOKEN")

print("🚀 GitHub’a gönderiliyor...")
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)
csv_str = all_df.to_csv(index=False)

try:
    contents = repo.get_contents(TARGET_PATH)
    repo.update_file(contents.path, "update weather data", csv_str, contents.sha, branch="main")
    print(f"✅ Güncellendi: {TARGET_PATH}")
except Exception:
    repo.create_file(TARGET_PATH, "add weather data", csv_str, branch="main")
    print(f"🆕 Oluşturuldu: {TARGET_PATH}")

print("🎉 Hava verisi başarıyla güncellendi ve GitHub’a yüklendi.")
