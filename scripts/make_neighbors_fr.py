#!/usr/bin/env python3
# make_neighbors_fr.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "crime_prediction_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- I/O (FR) ----
IN_CSV  = Path(os.environ.get("NEIGHBOR_INPUT_CSV",  str(DATA_DIR / "fr_crime_08.csv")))
OUT_CSV = Path(os.environ.get("NEIGHBOR_OUTPUT_CSV", str(DATA_DIR / "fr_crime_09.csv")))  # 08 → 09

NEIGHBOR_FILE = Path(os.environ.get("NEIGHBOR_FILE", str(DATA_DIR / "neighbors.csv")))
GEOID_LEN   = int(os.environ.get("GEOID_LEN", "11"))     # gerekirse env ile değiştir
WINDOW_DAYS = int(os.environ.get("NEIGHBOR_WINDOW_DAYS", "7"))
LAG_DAYS    = int(os.environ.get("NEIGHBOR_LAG_DAYS", "1"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L))

def _pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")
    if not NEIGHBOR_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası bulunamadı: {NEIGHBOR_FILE}")

    # ---- fr_crime_08 yükle ----
    df = pd.read_csv(IN_CSV, low_memory=False)
    # tarih alanı
    dcol = _pick_col(df.columns, "date", "datetime", "time", "timestamp")
    if not dcol:
        raise RuntimeError("Tarih kolonu bulunamadı (date/datetime/time/timestamp)")
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date

    # GEOID alanı
    gcol = _pick_col(df.columns, "GEOID", "geoid", "geography_id", "geoid10", "iris", "insee")
    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı")
    df["GEOID"] = _norm_geoid(df[gcol])

    # Olay sayısı (crime_count yoksa Y_label fallback)
    ccol = _pick_col(df.columns, "crime_count")
    if not ccol:
        ycol = _pick_col(df.columns, "Y_label", "y_label", "label", "target")
        if ycol:
            ccol = "crime_count"
            df[ccol] = pd.to_numeric(df[ycol], errors="coerce").fillna(0).astype(int)
        else:
            ccol = "crime_count"
            df[ccol] = 0  # boş ise 0

    df[ccol] = pd.to_numeric(df[ccol], errors="coerce").fillna(0).astype(int)

    # ---- günlük toplam ----
    daily = (df[["GEOID", "date", ccol]]
             .groupby(["GEOID", "date"], as_index=False)[ccol].sum()
             .rename(columns={ccol: "crime_count"}))
    daily["date"] = pd.to_datetime(daily["date"])

    # ---- neighbors.csv ----
    nbr = pd.read_csv(NEIGHBOR_FILE, dtype=str)
    s = _pick_col(nbr.columns, "geoid", "GEOID", "src", "source")
    t = _pick_col(nbr.columns, "neighbor", "NEIGHBOR_GEOID", "neighbor_geoid", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nbr.columns.tolist()}")
    nbr = nbr.rename(columns={s: "geoid", t: "neighbor"})[["geoid", "neighbor"]].dropna()
    for c in ("geoid", "neighbor"):
        nbr[c] = _norm_geoid(nbr[c])

    # ---- komşu günlük serilerini bağla ----
    d2 = nbr.merge(daily.rename(columns={"GEOID": "neighbor"}),
                   left_on="neighbor", right_on="neighbor", how="left")
    d2 = d2.rename(columns={"geoid": "GEOID"})  # ana GEOID
    d2 = d2.sort_values(["GEOID", "date"])

    # Rolling + lag: komşu toplamlarının gecikmeli 7g hareketli toplamı
    def _agg(x: pd.DataFrame) -> pd.DataFrame:
        x = x.set_index("date").asfreq("D", fill_value=0)
        roll = x["crime_count"].rolling(WINDOW_DAYS).sum().shift(LAG_DAYS)
        x["nei_7d_sum"] = roll
        return x.reset_index()

    d3 = d2.groupby("GEOID", group_keys=False).apply(_agg).reset_index(drop=True)
    d3["date"] = d3["date"].dt.date
    d4 = (d3.groupby(["GEOID", "date"], as_index=False)["nei_7d_sum"].sum())
    d4["nei_7d_sum"] = pd.to_numeric(d4["nei_7d_sum"], errors="coerce").fillna(0.0)

    # ---- orijinal tabloya merge → fr_crime_09.csv ----
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.date
    df_out = df_out.merge(d4, on=["GEOID", "date"], how="left")
    df_out["nei_7d_sum"] = pd.to_numeric(df_out["nei_7d_sum"], errors="coerce").fillna(0.0)
    # uyumluluk: önceki adımlarda kullanılan isim
    df_out["neighbor_crime_7d"] = df_out["nei_7d_sum"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"✅ 08 → 09 tamam: {IN_CSV.name} → {OUT_CSV.name} | rows={len(df_out)}")

if __name__ == "__main__":
    main()
