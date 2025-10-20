#!/usr/bin/env python3
# scripts/update_neighbors.py
from __future__ import annotations
import os, re
from pathlib import Path
import pandas as pd

# ---------------------
# Parametreler / ENV
# ---------------------
ROOT = Path(__file__).resolve().parent.parent  # proje kökü (app.py ile aynı seviye varsayımı)
DATA_DIR = ROOT / "crime_prediction_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Girdi/Çıktı dosyaları
IN_CSV  = Path(os.environ.get("NEIGHBOR_INPUT_CSV", str(DATA_DIR / "sf_crime_09.csv")))
OUT_CSV = Path(os.environ.get("NEIGHBOR_OUTPUT_CSV", str(DATA_DIR / "sf_crime_09.csv")))  # in-place varsayılan

# Komşuluk dosyası (2 kolonlu beklenir: GEOID, NEIGHBOR_GEOID)
NEIGHBOR_FILE = Path(os.environ.get("NEIGHBOR_FILE", str(DATA_DIR / "neighbors.csv")))

# GEOID uzunluğu (pad/normalize için)
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

# Pencere ve gecikme (gün)
WINDOW_DAYS = int(os.environ.get("NEIGHBOR_WINDOW_DAYS", "7"))
LAG_DAYS    = int(os.environ.get("NEIGHBOR_LAG_DAYS", "1"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .str[:L]
         .str.zfill(L)
    )

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")
    if not NEIGHBOR_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası bulunamadı: {NEIGHBOR_FILE}")

    # 1) Veri yükle
    df = pd.read_csv(IN_CSV, low_memory=False, dtype={"GEOID": str})
    if "date" not in df.columns and "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    df["GEOID"] = _norm_geoid(df["GEOID"], GEOID_LEN)

    # 2) Günlük toplam (GEOID × date) — crime_count yoksa 0 kabul
    if "crime_count" not in df.columns:
        df["crime_count"] = 0

    daily = (df[["GEOID", "date", "crime_count"]]
             .groupby(["GEOID", "date"], as_index=False)["crime_count"].sum())
    daily["date"] = pd.to_datetime(daily["date"])

    # 3) Komşuluk matrisi
    nbr = pd.read_csv(NEIGHBOR_FILE, dtype={"GEOID": str, "NEIGHBOR_GEOID": str})
    for c in ["GEOID", "NEIGHBOR_GEOID"]:
        nbr[c] = _norm_geoid(nbr[c], GEOID_LEN)

    # 4) Komşu günlük serileri ile genişlet
    d2 = nbr.merge(
        daily.rename(columns={"GEOID": "NEIGHBOR_GEOID"}),
        on="NEIGHBOR_GEOID",
        how="left",
        copy=False,
    )

    # 5) Rolling pencere (WINDOW_DAYS) ve gecikme (LAG_DAYS)
    d2 = d2.sort_values(["GEOID", "date"])
    def _agg_nei(x: pd.DataFrame) -> pd.DataFrame:
        x = x.set_index("date").asfreq("D", fill_value=0)  # boş günleri 0 doldur
        # önce komşu suçlarını topla → sonra lag uygula
        roll = x["crime_count"].rolling(f"{WINDOW_DAYS}D").sum().shift(LAG_DAYS)
        x["nei_7d_sum"] = roll
        return x.reset_index()

    d3 = (d2.groupby("GEOID", group_keys=False)
            .apply(_agg_nei)
            .reset_index())
    d3["date"] = d3["date"].dt.date

    # Aynı GEOID-date için birden çok komşudan gelen katkıyı toplayın
    d4 = (d3.groupby(["GEOID", "date"], as_index=False)["nei_7d_sum"].sum())
    d4["nei_7d_sum"] = pd.to_numeric(d4["nei_7d_sum"], errors="coerce").fillna(0.0).astype(float)

    # 6) Orijinal tabloya merge (in-place güncelle)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    out = df.merge(d4, on=["GEOID", "date"], how="left")
    out["nei_7d_sum"] = pd.to_numeric(out["nei_7d_sum"], errors="coerce").fillna(0.0).astype(float)

    # 7) Kaydet
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ neighbors eklendi → {OUT_CSV} (satır: {len(out)})")
    # opsiyonel debug: günlük komşu özetini de yaz
    (DATA_DIR / "sf_neighbors_daily.csv").write_text("")  # varlık işareti
    d4.to_csv(DATA_DIR / "sf_neighbors_daily.csv", index=False)

if __name__ == "__main__":
    main()
