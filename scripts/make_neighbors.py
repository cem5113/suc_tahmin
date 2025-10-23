#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_neighbors.py (zaman bağımsız)
sf_crime_07.csv + neighbors.csv → sf_crime_08.csv

Amaç:
- Her GEOID için, KOMŞU GEOID'LERDEKİ TOPLAM SUÇ SAYISINI hesaplamak.
- datetime gerektirmez. Zaman penceresi yoktur.
- Toplam suç: öncelik "crime_count", yoksa "Y_label" (veya "label"/"target").

Ortam değişkenleri:
- CRIME_DATA_DIR (varsayılan: "crime_prediction_data")
- NEIGH_FILE (varsayılan: "neighbors.csv")
- GEOID_LEN (varsayılan: 11)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

# --- I/O yolları ---
CRIME_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
SRC = CRIME_DIR / "sf_crime_07.csv"
DST = CRIME_DIR / "sf_crime_08.csv"
NEIGH_FILE = Path(os.environ.get("NEIGH_FILE", str(CRIME_DIR / "neighbors.csv")))
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L=GEOID_LEN) -> pd.Series:
    return (s.astype(str).str.extract(r"(\d+)")[0].str.zfill(L))

def _pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def main():
    # 0) Girdi kontrolleri
    if not SRC.exists():
        raise FileNotFoundError(f"Girdi dosyası yok: {SRC}")
    if not NEIGH_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası yok: {NEIGH_FILE.resolve()}")

    print(f"▶︎ SF komşuluk (zaman bağımsız) başlıyor: {SRC.name} + {NEIGH_FILE.name} → {DST.name}")

    # 1) Veri yükle
    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        df.assign(neighbor_crime_total=0).to_csv(DST, index=False)
        print(f"⚠️ {SRC.name} boş; {DST.name} yazıldı (0 toplam).")
        return

    # 2) GEOID ve suç sayısı kolonu tespiti
    gcol = _pick_col(df.columns, "geoid", "GEOID", "geography_id")
    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı (örn. geoid/GEOID/geography_id).")
    df["GEOID"] = _norm_geoid(df[gcol])

    crime_col = _pick_col(df.columns, "crime_count")
    if not crime_col:
        crime_col = _pick_col(df.columns, "Y_label", "label", "target")
        if not crime_col:
            raise RuntimeError("crime_count veya Y_label/label/target kolonu bulunamadı.")
    df["_crime_used"] = pd.to_numeric(df[crime_col], errors="coerce").fillna(0)

    # 3) Komşuluk verisi
    nb = pd.read_csv(NEIGH_FILE, dtype=str)
    # sütun isimlerini esnekçe bul
    s = _pick_col(nb.columns, "geoid", "src", "source")
    t = _pick_col(nb.columns, "neighbor", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")

    nb = nb[[s, t]].dropna().rename(columns={s: "SRC_GEOID", t: "NEI_GEOID"})
    nb["SRC_GEOID"] = _norm_geoid(nb["SRC_GEOID"])
    nb["NEI_GEOID"] = _norm_geoid(nb["NEI_GEOID"])
    # olası tekrar/öz yineleme temizliği
    nb = nb.drop_duplicates()
    # kendine komşuluğu varsa (SRC==NEI) istersek hariç tutabiliriz:
    nb = nb[nb["SRC_GEOID"] != nb["NEI_GEOID"]]

    if nb.empty:
        # Komşuluk yoksa sütunu 0 ile ekleyip çık
        out = df.copy()
        out["neighbor_crime_total"] = 0
        DST.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(DST, index=False)
        print("⚠️ neighbors.csv boş; tüm neighbor_crime_total = 0 yazıldı.")
        return

    # 4) GEOID toplam suç (zaman bağımsız toplam)
    geo_totals = (
        df.groupby("GEOID", as_index=False)["_crime_used"]
          .sum()
          .rename(columns={"_crime_used": "GEOID_CRIME_TOTAL"})
    )

    # 5) Komşu toplamlarını kaynak GEOID'e projekte et
    #    NEI_GEOID ile geo_totals'ı eşleyip SRC_GEOID bazında topla
    nei_with_counts = nb.merge(
        geo_totals.rename(columns={"GEOID": "NEI_GEOID"}),
        on="NEI_GEOID",
        how="left"
    )
    nei_with_counts["GEOID_CRIME_TOTAL"] = nei_with_counts["GEOID_CRIME_TOTAL"].fillna(0)

    nei_sum = (
        nei_with_counts.groupby("SRC_GEOID", as_index=False)["GEOID_CRIME_TOTAL"]
        .sum()
        .rename(columns={"SRC_GEOID": "GEOID", "GEOID_CRIME_TOTAL": "neighbor_crime_total"})
    )

    # 6) Ana tabloya geri yaz
    out = df.merge(nei_sum, on="GEOID", how="left")
    out["neighbor_crime_total"] = out["neighbor_crime_total"].fillna(0).astype(int)

    # 7) Temizlik ve çıktı
    out = out.drop(columns=["_crime_used"])
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"✅ {DST.name} oluşturuldu → {len(out):,} satır, {out.shape[1]} sütun")
    print("ℹ️ Eklenen sütun: neighbor_crime_total (zaman bağımsız toplam)")

if __name__ == "__main__":
    main()
