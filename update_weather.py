# update_weather.py (https://meteostat.net/en/)

from datetime import datetime, timedelta, date
import pandas as pd
import os
from meteostat import Daily, Point
from github import Github

# === GITHUB AYARLARI ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "cem5113/crime_prediction_data"
TARGET_PATH = "main/sf_weather_5years.csv"

# === METEOSTAT AYARLARI ===
OUT_CSV = "sf_weather_5years.csv"
HOT_DAY_THRESHOLD_C = 25.0
LAT, LON = 37.7749, -122.4194  # San Francisco koordinatları

# === 5 yıl penceresi hesapla ===
def five_year_window(today: date):
    try:
        start = today.replace(year=today.year - 5)
    except ValueError:
        start = today - timedelta(days=365*5 + 2)
    return (start + timedelta(days=1), today)

today = date.today()
win_start, win_end = five_year_window(today)

print(f"📅 Hedef aralık: {win_start} → {win_end}")

# === Mevcut dosya varsa oku ===
if os.path.exists(OUT_CSV):
    try:
        existing = pd.read_csv(OUT_CSV)
        existing['date'] = pd.to_datetime(existing['date'], errors='coerce').dt.date
        last_date = existing['date'].max()
        fetch_start = last_date + timedelta(days=1) if last_date else win_start
        base_df = existing
        print(f"🗂️ Mevcut veri bulundu: {len(existing)} satır, son gün={last_date}")
    except Exception as e:
        print("⚠️ Dosya okunamadı:", e)
        base_df = pd.DataFrame()
        fetch_start = win_start
else:
    base_df = pd.DataFrame()
    fetch_start = win_start

fetch_end = win_end

# === Veri çek fonksiyonu ===
def fetch_daily(lat, lon, start_d: date, end_d: date) -> pd.DataFrame:
    start_dt = datetime(start_d.year, start_d.month, start_d.day)
    end_dt   = datetime(end_d.year, end_d.month, end_d.day)
    pt = Point(lat, lon)
    df = Daily(pt, start_dt, end_dt).fetch().reset_index()
    df.rename(columns={'time': 'date'}, inplace=True)
    keep = ['date', 'tavg', 'tmin', 'tmax', 'prcp']
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep]
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    for c in ['tavg', 'tmin', 'tmax', 'prcp']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# === Yeni veri indir ===
new_df = pd.DataFrame()
if fetch_start <= fetch_end:
    print(f"📥 İndiriliyor: {fetch_start} → {fetch_end}")
    new_df = fetch_daily(LAT, LON, fetch_start, fetch_end)
    print(f"✅ Yeni çekilen satır: {len(new_df)}")
else:
    print("ℹ️ Güncel: indirilecek yeni gün yok.")

# === Birleştir + türev sütunlar ===
all_df = pd.concat([base_df, new_df], ignore_index=True) if len(base_df) else new_df.copy()
all_df['date'] = pd.to_datetime(all_df['date'], errors='coerce').dt.date
all_df.dropna(subset=['date'], inplace=True)
all_df['temp_range'] = all_df['tmax'] - all_df['tmin']
all_df['is_rainy'] = (all_df['prcp'] > 0).astype('Int64')
all_df['is_hot_day'] = (all_df['tmax'] > HOT_DAY_THRESHOLD_C).astype('Int64')

final_cols = ['date', 'tavg', 'tmax', 'prcp', 'temp_range', 'is_rainy', 'is_hot_day']
all_df = all_df[final_cols]
all_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
all_df.sort_values('date', inplace=True)

# === 5 yılı aşanı temizle ===
mask = (all_df['date'] >= win_start) & (all_df['date'] <= win_end)
all_df = all_df.loc[mask].copy()

# === Kaydet lokal olarak ===
all_df.to_csv(OUT_CSV, index=False)
print(f"💾 Kaydedildi: {OUT_CSV} — {len(all_df)} satır, {all_df['date'].min()} → {all_df['date'].max()}")

# === GitHub’a yükle ===
print("🚀 GitHub’a gönderiliyor...")

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

try:
    contents = repo.get_contents(TARGET_PATH)
    repo.update_file(contents.path, "update weather data", all_df.to_csv(index=False), contents.sha, branch="main")
    print(f"✅ Güncellendi: {TARGET_PATH}")
except Exception as e:
    # Dosya yoksa oluştur
    repo.create_file(TARGET_PATH, "add weather data", all_df.to_csv(index=False), branch="main")
    print(f"🆕 Oluşturuldu: {TARGET_PATH}")

print("🎉 Hava verisi başarıyla güncellendi ve GitHub’a yüklendi.")
