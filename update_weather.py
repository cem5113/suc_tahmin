# update_weather.py — Meteostat günlük veriyi çek, normalize et, 07 ile JOIN → 08 üret
from datetime import datetime, timedelta, date
from pathlib import Path
import os
import pandas as pd
import numpy as np

from meteostat import Daily, Point
from github import Github

# === ENV / GITHUB ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
REPO_NAME    = "cem5113/crime_prediction_data"
CRIME_DIR    = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").rstrip("/")
TARGET_PATH  = f"{CRIME_DIR}/sf_weather_5years.csv"   # repo içindeki yol

# === METEOSTAT / ÇIKTI ===
OUT_CSV             = "sf_weather_5years.csv"         # runner lokali
HOT_DAY_THRESHOLD_C = 25.0
LAT, LON            = 37.7749, -122.4194              # San Francisco

# Opsiyonel: tmin/tmax eksikse tavg etrafında hafif doldurma
FILL_TMIN_TMAX_FROM_TAVG = os.getenv("FILL_TMIN_TMAX_FROM_TAVG", "0").lower() in ("1","true","yes","on")

# === 5 yıllık pencere ===
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

    # Tarih
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
    if 'temp' in d.columns:
        time_col = next((c for c in ['datetime', 'time', 'Timestamp'] if c in d.columns), None)
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
wx = pd.concat([base_df, new_df], ignore_index=True) if len(base_df) else new_df.copy()
wx = normalize_existing_columns(wx)
wx['date'] = pd.to_datetime(wx['date'], errors='coerce').dt.date
wx.dropna(subset=['date'], inplace=True)

# Opsiyonel hafif backfill (tavg varsa)
if FILL_TMIN_TMAX_FROM_TAVG:
    wx['tavg'] = pd.to_numeric(wx['tavg'], errors='coerce')
    wx['tmin'] = pd.to_numeric(wx['tmin'], errors='coerce')
    wx['tmax'] = pd.to_numeric(wx['tmax'], errors='coerce')
    m = wx['tmin'].isna() & wx['tavg'].notna()
    wx.loc[m, 'tmin'] = wx.loc[m, 'tavg'] - 3.0
    m = wx['tmax'].isna() & wx['tavg'].notna()
    wx.loc[m, 'tmax'] = wx.loc[m, 'tavg'] + 3.0

# --- NA-güvenli bayraklar ---
pr = pd.to_numeric(wx.get('prcp'), errors='coerce')
tx = pd.to_numeric(wx.get('tmax'), errors='coerce')

wx['is_rainy'] = (pr > 0).astype('Int64')
wx.loc[pr.isna(), 'is_rainy'] = pd.NA

wx['is_hot_day'] = (tx > HOT_DAY_THRESHOLD_C).astype('Int64')
wx.loc[tx.isna(), 'is_hot_day'] = pd.NA

# temp_range
if 'tmax' in wx.columns and 'tmin' in wx.columns:
    wx['temp_range'] = wx['tmax'] - wx['tmin']
else:
    wx['temp_range'] = np.nan

# Kolon sırası + pencere kırp
final_cols = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'temp_range', 'is_rainy', 'is_hot_day']
for c in final_cols:
    if c not in wx.columns:
        wx[c] = np.nan
wx = wx[final_cols]

wx.drop_duplicates(subset=['date'], keep='last', inplace=True)
wx.sort_values('date', inplace=True)
mask = (wx['date'] >= win_start) & (wx['date'] <= win_end)
wx = wx.loc[mask].copy()

# === Kaydet weather master (local)
wx.to_csv(OUT_CSV, index=False)
print(f"💾 Kaydedildi: {OUT_CSV} — {len(wx)} satır, {wx['date'].min()} → {wx['date'].max()}")

# === 07 ile birleştir ve 08'i üret ===
Path(CRIME_DIR).mkdir(parents=True, exist_ok=True)

candidates_07 = [
    Path(CRIME_DIR) / "sf_crime_07.csv",
    Path("crime_prediction_data") / "sf_crime_07.csv",
    Path("crime_data") / "sf_crime_07.csv",
    Path("sf_crime_07.csv"),
    Path("outputs") / "sf_crime_07.csv",
]
src07 = next((p for p in candidates_07 if p.exists()), None)

if src07 is None:
    # fallback: 07 yoksa 08 olarak sadece weather yaz (pipeline kırılmasın)
    out_08 = Path(CRIME_DIR) / "sf_crime_08.csv"
    wx.to_csv(out_08, index=False)
    print(f"⚠️ sf_crime_07.csv bulunamadı. Sadece hava verisi {out_08} olarak yazıldı.")
else:
    df7 = pd.read_csv(src07, low_memory=False)
    # 07'nin tarihini normalize et
    if 'date' in df7.columns:
        df7['date'] = pd.to_datetime(df7['date'], errors='coerce').dt.date
    elif 'datetime' in df7.columns:
        df7['date'] = pd.to_datetime(df7['datetime'], errors='coerce').dt.date

    merged = df7.merge(wx, on='date', how='left')
    out_08 = Path(CRIME_DIR) / "sf_crime_08.csv"
    merged.to_csv(out_08, index=False)
    print(f"✅ Birleştirildi: {src07} + weather → {out_08} (rows={len(merged)})")

# === Weather master'ı repoya push et ===
if not GITHUB_TOKEN:
    raise SystemExit("❌ GITHUB_TOKEN tanımlı değil. Secrets’e ekleyin: GITHUB_TOKEN")

print("🚀 GitHub’a gönderiliyor...")
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)
csv_str = wx.to_csv(index=False)

try:
    contents = repo.get_contents(TARGET_PATH)
    repo.update_file(contents.path, "update weather data", csv_str, contents.sha, branch="main")
    print(f"✅ Güncellendi: {TARGET_PATH}")
except Exception:
    repo.create_file(TARGET_PATH, "add weather data", csv_str, branch="main")
    print(f"🆕 Oluşturuldu: {TARGET_PATH}")

print("🎉 Weather tamam, 08 üretildi ve master CSV GitHub’a yüklendi.")
