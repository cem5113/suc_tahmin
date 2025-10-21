#make_neighbors.py
"""
make_neighbors.py
sf_crime_08.csv + neighbors.csv → sf_crime_09.csv
Zaman sütunu (datetime) kullanılmaz. GEOID bazlı toplam suç etkisi hesaplanır.
"""

import os
import pandas as pd
from pathlib import Path

CRIME_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
SRC = CRIME_DIR / "sf_crime_08.csv"
DST = CRIME_DIR / "sf_crime_09.csv"
NEIGH_FILE = CRIME_DIR / "neighbors.csv"
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L=GEOID_LEN):
    return s.astype(str).str.extract(r"(\d+)")[0].str.zfill(L)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Girdi dosyası yok: {SRC}")
    if not NEIGH_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası yok: {NEIGH_FILE}")

    print(f"▶︎ Komşuluk bazlı zenginleştirme başlıyor: {SRC.name} + neighbors.csv → {DST.name}")

    # --- Veri yükle
    df = pd.read_csv(SRC, low_memory=False)
    nb = pd.read_csv(NEIGH_FILE, dtype=str).dropna()

    # GEOID kolonlarını normalize et
    gcol = [c for c in df.columns if "geoid" in c.lower()]
    if not gcol:
        raise RuntimeError("sf_crime_08.csv içinde GEOID kolonu bulunamadı")
    gcol = gcol[0]
    df["GEOID"] = _norm_geoid(df[gcol])

    ncol_g = [c for c in nb.columns if "geoid" in c.lower()]
    ncol_n = [c for c in nb.columns if "neighbor" in c.lower()]
    nb = nb.rename(columns={ncol_g[0]: "GEOID", ncol_n[0]: "neighbor"})
    nb["GEOID"] = _norm_geoid(nb["GEOID"])
    nb["neighbor"] = _norm_geoid(nb["neighbor"])

    # Suç sayısı sütunu
    ccol = None
    for c in df.columns:
        if "crime_count" in c.lower():
            ccol = c
            break
        if "y_label" in c.lower():
            ccol = c
            break
    if ccol is None:
        raise RuntimeError("crime_count veya Y_label kolonu bulunamadı")

    # --- Komşularla birleştir
    # Komşuların suç sayısını ana GEOID'e bağla
    joined = nb.merge(df[["GEOID", ccol]], left_on="neighbor", right_on="GEOID", how="left", suffixes=("", "_nei"))
    grouped = (joined.groupby("GEOID")[[f"{ccol}"]]
               .agg(["sum", "mean", "count"])
               .reset_index())
    grouped.columns = ["GEOID", "neighbor_crime_total", "neighbor_crime_mean", "neighbor_count"]

    # --- Ana tabloya ekle
    out = df.merge(grouped, on="GEOID", how="left")
    out[["neighbor_crime_total", "neighbor_crime_mean"]] = out[["neighbor_crime_total", "neighbor_crime_mean"]].fillna(0)
    out["neighbor_count"] = out["neighbor_count"].fillna(0).astype(int)

    out.to_csv(DST, index=False)
    print(f"✅ {DST.name} oluşturuldu → {len(out):,} satır, {out.shape[1]} sütun")

if __name__ == "__main__":
    main()
