#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_neighbors_daily.py 
daily_crime_08.csv + neighbors.csv → daily_crime_09.csv

Üretir:
- neighbor_crime_alltime          (zaman bağımsız toplam)
- neighbor_crime_{1d,3d,7d,30d}   (rolling pencereler; varsayılan: 1D,3D,7D,30D)

ENV:
- CRIME_DATA_DIR  (default: crime_prediction_data)
- NEIGH_FILE      (default: {CRIME_DATA_DIR}/neighbors.csv)
- GEOID_LEN       (default: 11)
- NEIGH_WINDOWS   (default: "1D,3D,7D,30D")
- DATE_TZ         (sadece tarih normalizasyonu gerekirse; default: "UTC")
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- I/O & ENV ----------------
CRIME_DIR    = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
SRC          = CRIME_DIR / "daily_crime_08.csv"
DST          = CRIME_DIR / "daily_crime_09.csv"
NEIGH_FILE   = Path(os.environ.get("NEIGH_FILE", str(CRIME_DIR / "neighbors.csv")))
GEOID_LEN    = int(os.environ.get("GEOID_LEN", "11"))
NEIGH_WINDOWS = os.environ.get("NEIGH_WINDOWS", "1D,3D,7D,30D")
DATE_TZ      = os.environ.get("DATE_TZ", "UTC")

DATE_CANDS = ["date", "event_date", "dt", "datetime", "timestamp", "t0", "t"]
COUNT_CANDS = ["crime_count", "count", "n"]
LABEL_CANDS = ["Y_label", "label", "target"]

def _norm_geoid(s: pd.Series, L=GEOID_LEN) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(L)

def _pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:  # birebir eşleşme
            return low[c.lower()]
    # gevşek eşleşme (ör. "event_date" vs "EventDate")
    for c in cols:
        for cand in cands:
            if cand.lower() == str(c).lower():
                return c
    return None

def _detect_date_col(df: pd.DataFrame) -> str | None:
    c = _pick_col(df.columns, *DATE_CANDS)
    return c

def _to_date_col(df: pd.DataFrame, dcol: str) -> pd.Series:
    # Güvenli: aware/naive her iki olasılık için sadece "gün" çıkart
    try:
        x = pd.to_datetime(df[dcol], errors="coerce", utc=True).dt.tz_convert(DATE_TZ).dt.date
    except Exception:
        x = pd.to_datetime(df[dcol], errors="coerce")
        if getattr(x.dtype, "tz", None) is None:
            x = x.dt.tz_localize("UTC")
        x = x.dt.tz_convert(DATE_TZ).dt.date
    return x.astype("string")

def _parse_windows(s: str) -> list[tuple[str, int]]:
    """
    "1D,3D,7D,30D" → [("1d",1),("3d",3),("7d",7),("30d",30)]
    Sadece gün bazlı integer pencereleri destekliyoruz.
    """
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower().endswith("d"):
            try:
                k = int(tok[:-1])
                if k >= 1:
                    out.append((f"{k}d", k))
            except Exception:
                pass
    # tekillik ve sıraya koy
    seen, uniq = set(), []
    for name, k in out:
        if k not in seen:
            seen.add(k)
            uniq.append((name, k))
    uniq.sort(key=lambda t: t[1])
    return uniq

def main():
    # 0) Girdi kontrolleri
    if not SRC.exists():
        raise FileNotFoundError(f"Girdi dosyası yok: {SRC}")
    if not NEIGH_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası yok: {NEIGH_FILE.resolve()}")

    print(f"▶︎ Komşu pencereler: {NEIGH_WINDOWS}")
    win_spec = _parse_windows(NEIGH_WINDOWS)
    if not win_spec:
        print("⚠️ Geçerli pencere bulunamadı; varsayılan 1D,3D,7D,30D kullanılacak.")
        win_spec = _parse_windows("1D,3D,7D,30D")

    # 1) Veri yükle
    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        df.assign(
            neighbor_crime_alltime=0,
            **{f"neighbor_crime_{nm}": 0 for nm, _ in win_spec}
        ).to_csv(DST, index=False)
        print(f"⚠️ {SRC.name} boş; {DST.name} yazıldı (komşu sütunları 0).")
        return

    # 2) GEOID ve tarih/suç kolonları
    gcol = _pick_col(df.columns, "geoid", "GEOID", "geography_id")
    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı (örn. geoid/GEOID/geography_id).")
    df["GEOID"] = _norm_geoid(df[gcol])

    dcol = _detect_date_col(df)
    if not dcol:
        raise RuntimeError(f"Tarih kolonu bulunamadı. Adaylar: {DATE_CANDS}")
    df["__date__"] = _to_date_col(df, dcol)   # string (YYYY-MM-DD)

    crime_col = _pick_col(df.columns, *COUNT_CANDS)
    if crime_col:
        base_count = pd.to_numeric(df[crime_col], errors="coerce").fillna(0).astype(float)
    else:
        # Y_label / label / target → olay sayımı gibi toplanır
        ycol = _pick_col(df.columns, *LABEL_CANDS)
        if not ycol:
            raise RuntimeError("crime_count veya Y_label/label/target kolonu bulunamadı.")
        base_count = pd.to_numeric(df[ycol], errors="coerce").fillna(0).astype(float)

    df["_crime_used"] = base_count

    # 3) Komşuluk verisi (SRC → NEI)
    nb = pd.read_csv(NEIGH_FILE, dtype=str)
    s = _pick_col(nb.columns, "geoid", "src", "source")
    t = _pick_col(nb.columns, "neighbor", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")
    nb = nb[[s, t]].dropna().rename(columns={s: "SRC_GEOID", t: "NEI_GEOID"})
    nb["SRC_GEOID"] = _norm_geoid(nb["SRC_GEOID"])
    nb["NEI_GEOID"] = _norm_geoid(nb["NEI_GEOID"])
    nb = nb.drop_duplicates()
    nb = nb[nb["SRC_GEOID"] != nb["NEI_GEOID"]]

    if nb.empty:
        # Komşuluk yoksa tüm neighbor sütunları 0
        out = df.copy()
        out["neighbor_crime_alltime"] = 0
        for nm, _ in win_spec:
            out[f"neighbor_crime_{nm}"] = 0
        out.drop(columns=["_crime_used", "__date__"], errors="ignore").to_csv(DST, index=False)
        print("⚠️ neighbors.csv boş; tüm komşu metrikler 0 yazıldı.")
        return

    # 4) All-time komşu toplam (zaman bağımsız) — eski davranış
    geo_totals_all = (
        df.groupby("GEOID", as_index=False)["_crime_used"]
          .sum()
          .rename(columns={"_crime_used": "GEOID_CRIME_TOTAL"})
    )
    nei_with_counts_all = nb.merge(
        geo_totals_all.rename(columns={"GEOID": "NEI_GEOID"}),
        on="NEI_GEOID",
        how="left"
    )
    nei_with_counts_all["GEOID_CRIME_TOTAL"] = nei_with_counts_all["GEOID_CRIME_TOTAL"].fillna(0)
    nei_sum_all = (
        nei_with_counts_all.groupby("SRC_GEOID", as_index=False)["GEOID_CRIME_TOTAL"]
        .sum()
        .rename(columns={"SRC_GEOID": "GEOID", "GEOID_CRIME_TO_
