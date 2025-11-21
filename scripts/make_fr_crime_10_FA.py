# scripts/make_fr_crime_10_FA.py
import os
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
BASE.mkdir(parents=True, exist_ok=True)

def read_fr_09() -> pd.DataFrame:
    # Öncelik: parquet, sonra csv
    pq = BASE / "fr_crime_09.parquet"
    csv = BASE / "fr_crime_09.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError("fr_crime_09.(csv|parquet) bulunamadı")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # kritik kolon alias'ları
    lower = {c.lower(): c for c in df.columns}
    def alias(opts, target):
        for o in opts:
            if o in df.columns:
                if target not in df.columns: df[target] = df[o]
                return
            lo = o.lower()
            if lo in lower:
                if target not in df.columns: df[target] = df[lower[lo]]
                return

    alias(["GEOID","geoid","Geoid","id","cell_id"], "GEOID")
    alias(["hour","event_hour","event_hour_x","event_hour_y"], "hour")
    alias(["latitude","lat","Latitude"], "latitude")
    alias(["longitude","lon","Longitude"], "longitude")
    alias(["risk_score","p_crime","prob","score"], "risk_score")
    alias(["date","Date"], "date")
    alias(["datetime","ts","timestamp","Datetime"], "datetime")
    return df

def build_fr_10(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_columns(df)

    # hour yoksa datetime'dan çıkar
    if "hour" not in df.columns and "datetime" in df.columns:
        try:
            df["hour"] = pd.to_datetime(df["datetime"], errors="coerce").dt.hour
        except Exception:
            pass

    # GEOID normalize
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.replace(r"\D","",regex=True).str.zfill(11)

    # temel temizleme / dedup
    keep = [c for c in ["GEOID","hour","Category","Subcategory","latitude","longitude",
                        "risk_score","date","datetime"] if c in df.columns]
    if keep:
        df = df[keep].copy()
        df = df.drop_duplicates()

    # risk_score yoksa basit fallback
    if "risk_score" not in df.columns:
        df["risk_score"] = 0.5

    return df

def main():
    df9 = read_fr_09()
    out = build_fr_10(df9)
    out_path = BASE / "fr_crime_10_FA.parquet"
    out.to_parquet(out_path, index=False)
    print(f"✅ yazıldı: {out_path} | satır: {len(out)}")

if __name__ == "__main__":
    main()
