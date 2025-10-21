#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_neighbors.py
sf_crime_07.csv + neighbors.csv → sf_crime_08.csv

Her satırın kendi zamanına (datetime) göre, aynı GEOID’in komşularında
son 24 saat / 72 saat / 7 günde gerçekleşen suç sayıları hesaplanır ve eklenir.
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import timedelta
import pandas as pd

# --- I/O yolları ---
CRIME_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
SRC = CRIME_DIR / "sf_crime_07.csv"
DST = CRIME_DIR / "sf_crime_08.csv"
# Önemli: neighbors yolunu env ile esnek yapıyoruz; default: repo kökü (neighbors.csv)
NEIGH_FILE = Path(os.environ.get("NEIGH_FILE", "neighbors.csv"))
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

# --- yardımcılar ---
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

    print(f"▶︎ SF komşuluk zenginleştirme başlıyor: {SRC.name} + {NEIGH_FILE.name} → {DST.name}")

    # 1) Veri yükle
    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        # Boş girdi → boş çıktı ama şema korunur
        df.assign(neighbor_crime_24h=0, neighbor_crime_72h=0, neighbor_crime_7d=0)\
          .to_csv(DST, index=False)
        print(f"⚠️ {SRC.name} boş; {DST.name} 0 satır yazıldı.")
        return

    # 2) datetime & geoid tespiti
    dcol = _pick_col(df.columns, "datetime", "date", "timestamp", "time")
    gcol = _pick_col(df.columns, "geoid", "GEOID", "geography_id")
    if not dcol:
        raise RuntimeError("Tarih kolonu (datetime/date/timestamp/time) bulunamadı")
    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı")

    df["datetime"] = pd.to_datetime(df[dcol], errors="coerce")
    df["GEOID"] = _norm_geoid(df[gcol])

    # 3) Suç sayısı kolonu (öncelik crime_count, yoksa Y_label)
    crime_col = _pick_col(df.columns, "crime_count")
    if not crime_col:
        crime_col = _pick_col(df.columns, "Y_label", "label", "target")
        if not crime_col:
            raise RuntimeError("crime_count veya Y_label kolonu bulunamadı")
    df["crime_count_used"] = pd.to_numeric(df[crime_col], errors="coerce").fillna(0).astype(int)

    # 4) Komşuluk verisi
    nb = pd.read_csv(NEIGH_FILE, dtype=str).dropna(how="any")
    s = _pick_col(nb.columns, "geoid", "src", "source")
    t = _pick_col(nb.columns, "neighbor", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")
    nb = nb.rename(columns={s: "GEOID", t: "neighbor"})[["GEOID", "neighbor"]]
    nb["GEOID"] = _norm_geoid(nb["GEOID"])
    nb["neighbor"] = _norm_geoid(nb["neighbor"])

    if nb.empty:
        raise RuntimeError("❌ neighbors.csv boş görünüyor — komşuluk hesaplanamamış.")

    # 5) Yalnızca suç olan kayıtları olay tablosu olarak ayır (ağırlık = crime_count_used)
    events = df.loc[df["crime_count_used"] > 0, ["GEOID", "datetime", "crime_count_used"]].copy()

    # 6) Komşu olaylarını ana GEOID’e bağla (neighbor → target GEOID)
    #    merged: [GEOID(target), neighbor, neighbor_time, neighbor_count]
    merged = nb.merge(
        events.rename(columns={"GEOID": "neighbor", "datetime": "neighbor_time", "crime_count_used": "neighbor_count"}),
        on="neighbor",
        how="left"
    )
    # şimdi target GEOID kolonunu ekleyelim (nb içinden geldi)
    merged = merged.rename(columns={"GEOID": "TARGET_GEOID"})

    # 7) Her TARGET_GEOID için, df’deki baz zamanlara göre pencere sayıları
    #    performans için, önce neighbor olaylarını GEOID kırılımında bir dict’e alıyoruz
    neigh_map = {}
    grp = merged.dropna(subset=["neighbor_time", "neighbor_count"])
    if not grp.empty:
        for g, sub in grp.groupby("TARGET_GEOID"):
            # komşu olay zamanları ve ağırlıkları (count)
            neigh_map[g] = sub[["neighbor_time", "neighbor_count"]].sort_values("neighbor_time").reset_index(drop=True)

    results = []
    # df’yi GEOID’e göre gez; her GEOID’deki tüm base zamanlarını tek seferde işle
    for g, gdf in df.groupby("GEOID"):
        base_times = gdf["datetime"].to_numpy()
        if g not in neigh_map:
            # hiç komşu olayı yok → hepsi 0
            results.extend([(g, t, 0, 0, 0) for t in base_times])
            continue

        nevents = neigh_map[g]
        times = nevents["neighbor_time"].to_numpy()
        counts = nevents["neighbor_count"].to_numpy()

        # her base time için zaman pencereleri
        for t in base_times:
            if pd.isna(t):
                results.append((g, t, 0, 0, 0))
                continue
            t1 = t - timedelta(hours=24)
            t3 = t - timedelta(hours=72)
            t7 = t - timedelta(days=7)
            mask_le_t = times <= t
            n24 = counts[(times >= t1) & mask_le_t].sum()
            n72 = counts[(times >= t3) & mask_le_t].sum()
            n7d = counts[(times >= t7) & mask_le_t].sum()
            results.append((g, t, int(n24), int(n72), int(n7d)))

    nei_df = pd.DataFrame(
        results,
        columns=["GEOID", "datetime", "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d"]
    )

    # 8) Ana tabloya birleştir ve yaz
    out = df.merge(nei_df, on=["GEOID", "datetime"], how="left")
    for c in ["neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d"]:
        out[c] = out[c].fillna(0).astype(int)

    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"✅ {DST.name} oluşturuldu → {len(out):,} satır, {out.shape[1]} sütun")

if __name__ == "__main__":
    main()
