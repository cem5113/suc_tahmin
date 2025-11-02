# update_crime_weather_daily.py
import os
import pandas as pd

BASE = "crime_prediction_data"
CRIME_IN  = os.path.join(BASE, "daily_crime_07.csv")
WEATHER_IN = os.path.join(BASE, "sf_weather_5years.csv")
CRIME_OUT = os.path.join(BASE, "daily_crime_08.csv")

print("⏳ load…")
df = pd.read_csv(CRIME_IN, low_memory=False)
wx = pd.read_csv(WEATHER_IN, low_memory=False)

# date normalizasyonu
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
wx["date"] = pd.to_datetime(wx["date"], errors="coerce").dt.date

before = df.shape

print("🔗 merge (left) …")
out = df.merge(wx, on="date", how="left")

print(f"Δ rows: {before[0]} → {out.shape[0]}")
print(f"Δ cols: {before[1]} → {out.shape[1]}")

out.to_csv(CRIME_OUT, index=False)
print(f"✅ saved → {CRIME_OUT}")
