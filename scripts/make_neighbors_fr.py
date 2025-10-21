# make_neighbors_fr.py
"""
make_neighbors_fr.py
--------------------
fr_crime_08.csv + neighbors.csv → fr_crime_09.csv

Komşuluklar yeniden hesaplanmaz; son neighbors.csv kullanılır.
Her GEOID için komşu bölgelerdeki son 24h / 72h / 7d suç yoğunluğu hesaplanır.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from datetime import timedelta

# === Klasör yapısı ===
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "crime_prediction_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Girdi / Çıktı ===
IN_CSV  = Path(os.environ.get("NEIGHBOR_INPUT_CSV",  str(DATA_DIR / "fr_crime_08.csv")))
OUT_CSV = Path(os.environ.get("NEIGHBOR_OUTPUT_CSV", str(DATA_DIR / "fr_crime_09.csv")))
NEIGHBOR_FILE = Path(os.environ.get("NEIGHBOR_FILE", str(DATA_DIR / "neighbors.csv")))

# === Parametreler ===
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L))

def _pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

# === Ana akış ===
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")
    if not NEIGHBOR_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası bulunamadı: {NEIGHBOR_FILE}")

    print(f"▶︎ {IN_CSV.name} + neighbors.csv → {OUT_CSV.name}")

    # 1️⃣ Veri yükleme
    df = pd.read_csv(IN_CSV, low_memory=False)
    dcol = _pick_col(df.columns, "datetime", "date", "timestamp", "time")
    gcol = _pick_col(df.columns, "geoid", "GEOID", "geography_id", "insee", "iris")

    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı")

    df["geoid"] = _norm_geoid(df[gcol])

    if not dcol:
        raise RuntimeError("Tarih kolonu (datetime/date/timestamp) bulunamadı")
    df["datetime"] = pd.to_datetime(df[dcol], errors="coerce")

    # Suç sayısı
    ccol = _pick_col(df.columns, "crime_count", "Y_label", "label", "target")
    if not ccol:
        ccol = "crime_count"
        df[ccol] = 0
    df["crime_count"] = pd.to_numeric(df[ccol], errors="coerce").fillna(0).astype(int)

    # 2️⃣ Komşuluk verisi
    nb = pd.read_csv(NEIGHBOR_FILE, dtype=str).dropna()
    s = _pick_col(nb.columns, "geoid", "src", "source")
    t = _pick_col(nb.columns, "neighbor", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")

    nb = nb.rename(columns={s: "geoid", t: "neighbor"})[["geoid", "neighbor"]].dropna()
    for c in ("geoid", "neighbor"):
        nb[c] = _norm_geoid(nb[c])

    if nb.empty:
        raise RuntimeError("❌ neighbors.csv boş görünüyor — komşuluk hesaplanamamış.")

    # 3️⃣ Sadece suç işlenmiş satırlar (verimlilik için)
    crimes = df[df["crime_count"] > 0][["geoid", "datetime"]].copy()

    # 4️⃣ Tüm komşu suçlarını (merge + filtreleme) ile hesapla
    merged = nb.merge(crimes, left_on="neighbor", right_on="geoid", suffixes=("", "_nei"))
    merged = merged.rename(columns={"datetime": "neighbor_time", "geoid": "GEOID"}).drop(columns=["geoid_nei"])

    # 5️⃣ Her GEOID için 24h, 72h, 7d filtreleri
    results = []
    for g, group in df.groupby("geoid"):
        base_times = group["datetime"]
        if g not in merged["GEOID"].values:
            results.extend([(g, t, 0, 0, 0) for t in base_times])
            continue

        neigh_events = merged.loc[merged["GEOID"] == g, "neighbor_time"]
        for t in base_times:
            t1, t3, t7 = t - timedelta(hours=24), t - timedelta(hours=72), t - timedelta(days=7)
            n24 = ((neigh_events <= t) & (neigh_events >= t1)).sum()
            n72 = ((neigh_events <= t) & (neigh_events >= t3)).sum()
            n7d = ((neigh_events <= t) & (neigh_events >= t7)).sum()
            results.append((g, t, n24, n72, n7d))

    nei_df = pd.DataFrame(results, columns=["geoid", "datetime", "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d"])

    # 6️⃣ Birleştir ve kaydet
    out = df.merge(nei_df, on=["geoid", "datetime"], how="left")
    for c in ["neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d"]:
        out[c] = out[c].fillna(0).astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ {OUT_CSV.name} oluşturuldu → {len(out):,} satır, {out.shape[1]} sütun")

if __name__ == "__main__":
    main()
