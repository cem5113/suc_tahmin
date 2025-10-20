import os, io, sys, requests, pandas as pd
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from urllib.parse import quote

API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")
if not API_KEY:
    sys.exit("Missing VISUAL_CROSSING_API_KEY")

LOCATION = os.getenv("WX_LOCATION", "San Francisco,CA")
UNIT = os.getenv("WX_UNIT", "metric")           # "metric" önerilir
OUTDIR = os.getenv("CRIME_DATA_DIR", "crime_data")
HOT_THRESHOLD = float(os.getenv("HOT_THRESHOLD_C", "30"))

BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

def fetch_days_csv(location: str, start: date, end: date) -> pd.DataFrame:
    url = (
        f"{BASE}/{quote(location)}/{start.isoformat()}/{end.isoformat()}"
        f"?unitGroup={UNIT}&include=days"
        f"&elements=datetime,temp,tempmin,tempmax,precip"
        f"&contentType=csv&key={API_KEY}"
    )
    r = requests.get(url, timeout=60)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print("HTTP status:", r.status_code)
        print("Response preview:", r.text[:300])
        raise
    df = pd.read_csv(io.StringIO(r.text))
    need = {"datetime","temp","tempmin","tempmax","precip"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns: {miss} | got: {list(df.columns)}")
    df = df.rename(columns={"datetime":"date","temp":"tavg","tempmin":"tmin","tempmax":"tmax","precip":"prcp"})
    tr_days = {0:"Pazartesi",1:"Salı",2:"Çarşamba",3:"Perşembe",4:"Cuma",5:"Cumartesi",6:"Pazar"}
    dd = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = dd.dt.weekday.map(tr_days)
    df["temp_range"] = df["tmax"] - df["tmin"]
    df["is_rainy"] = (df["prcp"] > 0).astype(int)
    df["is_hot"] = (df["tmax"] >= HOT_THRESHOLD).astype(int)
    return df[["date","tavg","tmin","tmax","prcp","temp_range","day","is_rainy","is_hot"]]

def main():
    SF = ZoneInfo("America/Los_Angeles")
    today_sf = datetime.now(SF).date()
    tomorrow = today_sf + timedelta(days=1)
    week_end = tomorrow + timedelta(days=6)

    os.makedirs(OUTDIR, exist_ok=True)

    df_t = fetch_days_csv(LOCATION, tomorrow, tomorrow)
    df_t.to_csv(os.path.join(OUTDIR, "yarin.csv"), index=False, encoding="utf-8")
    print("✓ yarin.csv")

    df_w = fetch_days_csv(LOCATION, tomorrow, week_end)
    df_w.to_csv(os.path.join(OUTDIR, "week.csv"), index=False, encoding="utf-8")
    print("✓ week.csv")

if __name__ == "__main__":
    main()
