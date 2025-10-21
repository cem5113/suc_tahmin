#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_neighbors_fr.py
--------------------
fr_crime_08.csv + neighbors.csv → fr_crime_09.csv
Komşuluklar yeniden hesaplanmaz; mevcut neighbors.csv doğrudan kullanılır.
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
GEOID_LEN   = int(os.environ.get("GEOID_LEN", "11"))

# === Yardımcı fonksiyonlar ===
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

    if not dcol or not gcol:
        raise RuntimeError("datetime veya GEOID kolonu bulunamadı")

    df["datetime"] = pd.to_datetime(df[dcol], errors="coerce")
    df["geoid"] = _norm_geoid(df[gcol])

    # Suç sayısı (varsa crime_count, yoksa Y_label)
    ccol = _pick_col(df.columns, "crime_count", "Y_label", "label", "target")
    if not ccol:
        df["crime_count"] = 0
        ccol = "crime_count"
    else:
        df["crime_count"] = pd.to_numeric(df[ccol], errors="coerce").fillna(0).astype(int)

    # 2️⃣ Komşuluk verisini yükle
    nb = pd.read_csv(NEIGHBOR_FILE, dtype=str)
    s = _pick_col(nb.columns, "geoid", "src", "source")
    t = _pick_col(nb.columns, "neighbor", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")

    nb = nb.rename(columns={s: "geoid", t: "neighbor"})[["geoid", "neighbor"]].dropna()
    for c in ("geoid", "neighbor"):
        nb[c] = _norm_geoid(nb[c])

    # 3️⃣ Yalnızca suç işlenmiş kayıtları ayır
    crimes = df[df["crime_count"] > 0][["geoid", "datetime"]].copy()

    # 4️⃣ Fonksiyon: Belirli GEOID için komşularda 24h/72h/7d suç sayısı
    def get_neighbor_counts(base_time, geoid):
        neighbors = nb.loc[nb["geoid"] == geoid, "neighbor"].tolist()
        if not neighbors:
            return (0, 0, 0)
        t1 = base_time - timedelta(hours=24)
        t3 = base_time - timedelta(hours=72)
        t7 = base_time - timedelta(days=7)
        recent = crimes[(crimes["datetime"] <= base_time) & crimes["geoid"].isin(neighbors)]
        n24 = recent[recent["datetime"] >= t1].shape[0]
        n72 = recent[recent["datetime"] >= t3].shape[0]
        n7d = recent[recent["datetime"] >= t7].shape[0]
        return (n24, n72, n7d)

    # 5️⃣ Özellikleri hesapla
    recs = []
    for i, row in df.iterrows():
        g, t = row["geoid"], row["datetime"]
        n24, n72, n7d = get_neighbor_counts(t, g)
        recs.append((g, t, n24, n72, n7d))
        if i % 10000 == 0 and i > 0:
            print(f"  → {i:,} satır işlendi...")

    nei_df = pd.DataFrame(recs, columns=["geoid", "datetime", "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d"])

    # 6️⃣ Birleştir ve kaydet
    out = df.merge(nei_df, on=["geoid", "datetime"], how="left")
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ {OUT_CSV.name} oluşturuldu → {len(out):,} satır, {out.shape[1]} sütun")

if __name__ == "__main__":
    main()
