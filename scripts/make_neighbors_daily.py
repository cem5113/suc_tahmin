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
- DATE_TZ         (default: "UTC")
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- I/O & ENV ----------
CRIME_DIR     = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))
SRC           = CRIME_DIR / "daily_crime_08.csv"
DST           = CRIME_DIR / "daily_crime_09.csv"
NEIGH_FILE    = Path(os.environ.get("NEIGH_FILE", str(CRIME_DIR / "neighbors.csv")))
GEOID_LEN     = int(os.environ.get("GEOID_LEN", "11"))
NEIGH_WINDOWS = os.environ.get("NEIGH_WINDOWS", "1D,3D,7D,30D")
DATE_TZ       = os.environ.get("DATE_TZ", "UTC")

DATE_CANDS  = ["date", "event_date", "dt", "datetime", "timestamp", "t0", "t"]
COUNT_CANDS = ["crime_count", "count", "n"]
LABEL_CANDS = ["Y_label", "label", "target"]

pd.options.mode.copy_on_write = True

# ---------- helpers ----------
def _norm_geoid(s: pd.Series, L=GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str[:L]
         .str.zfill(L)
    )

def _pick_col(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    # gevşek (case-insensitive) zaten yukarıda
    return None

def _detect_date_col(df: pd.DataFrame) -> str | None:
    return _pick_col(df.columns, *DATE_CANDS)

def _to_date_col(df: pd.DataFrame, dcol: str) -> pd.Series:
    # aware/naive her iki olasılık için "gün" çıkarılır
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
    (Gün bazlı tam sayı pencereler)
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
    # tekilleştir ve sırala
    seen, uniq = set(), []
    for name, k in out:
        if k not in seen:
            seen.add(k)
            uniq.append((name, k))
    uniq.sort(key=lambda t: t[1])
    return uniq

def _safe_save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

# ---------- main ----------
def main():
    if not SRC.exists():
        raise FileNotFoundError(f"❌ Girdi dosyası yok: {SRC}")
    if not NEIGH_FILE.exists():
        raise FileNotFoundError(f"❌ Komşuluk dosyası yok: {NEIGH_FILE.resolve()}")

    print(f"▶︎ Komşu pencereler: {NEIGH_WINDOWS}")
    win_spec = _parse_windows(NEIGH_WINDOWS)
    if not win_spec:
        print("⚠️ Geçerli pencere bulunamadı; varsayılan 1D,3D,7D,30D kullanılacak.")
        win_spec = _parse_windows("1D,3D,7D,30D")

    # 1) Veri yükle
    df = pd.read_csv(SRC, low_memory=False)
    if df.empty:
        out = df.copy()
        out["neighbor_crime_alltime"] = 0
        for nm, _ in win_spec:
            out[f"neighbor_crime_{nm}"] = 0
        _safe_save_csv(out, DST)
        print(f"⚠️ {SRC.name} boş; {DST.name} yazıldı (komşu sütunları 0).")
        return

    # 2) GEOID & DATE & COUNT
    gcol = _pick_col(df.columns, "geoid", "GEOID", "geography_id")
    if not gcol:
        raise RuntimeError("❌ GEOID kolonu bulunamadı (örn. geoid/GEOID/geography_id).")
    df["GEOID"] = _norm_geoid(df[gcol])

    dcol = _detect_date_col(df)
    if not dcol:
        raise RuntimeError(f"❌ Tarih kolonu bulunamadı. Adaylar: {DATE_CANDS}")
    df["__date__"] = _to_date_col(df, dcol)  # string YYYY-MM-DD

    crime_col = _pick_col(df.columns, *COUNT_CANDS)
    if crime_col:
        base_count = pd.to_numeric(df[crime_col], errors="coerce").fillna(0).astype(float)
    else:
        ycol = _pick_col(df.columns, *LABEL_CANDS)
        if not ycol:
            raise RuntimeError("❌ crime_count veya Y_label/label/target kolonu bulunamadı.")
        base_count = pd.to_numeric(df[ycol], errors="coerce").fillna(0).astype(float)
    df["_crime_used"] = base_count

    # 3) Komşuluk verisi
    nb = pd.read_csv(NEIGH_FILE, dtype=str)
    s = _pick_col(nb.columns, "geoid", "src", "source", "SRC_GEOID")
    t = _pick_col(nb.columns, "neighbor", "dst", "target", "NEI_GEOID")
    # Bazı komşuluk dosyalarında hem 'geoid' hem 'neighbor' olabilir
    if s and t and s.lower() == t.lower():
        # tek kolon kullanılmışsa (ör. 'geoid'), anlamlı değil → hata
        raise RuntimeError("❌ neighbors.csv: kaynak ve komşu için farklı kolonlar gerekli (örn. src/dst).")
    if not s:
        s = _pick_col(nb.columns, "src", "source")
    if not t:
        t = _pick_col(nb.columns, "dst", "target", "neighbor")
    if not s or not t:
        raise RuntimeError(f"❌ neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")

    nb = nb[[s, t]].dropna().rename(columns={s: "SRC_GEOID", t: "NEI_GEOID"})
    nb["SRC_GEOID"] = _norm_geoid(nb["SRC_GEOID"])
    nb["NEI_GEOID"] = _norm_geoid(nb["NEI_GEOID"])
    nb = nb.drop_duplicates()
    # self-loop'ları at
    nb = nb[nb["SRC_GEOID"] != nb["NEI_GEOID"]]

    if nb.empty:
        out = df.copy()
        out["neighbor_crime_alltime"] = 0
        for nm, _ in win_spec:
            out[f"neighbor_crime_{nm}"] = 0
        out.drop(columns=["_crime_used", "__date__"], errors="ignore", inplace=True)
        _safe_save_csv(out, DST)
        print("⚠️ neighbors.csv boş; tüm komşu metrikler 0 yazıldı.")
        return

    # 4) Günlük GEOID × date olay sayısı
    daily = (
        df.groupby(["GEOID", "__date__"], as_index=False)["_crime_used"]
          .sum()
          .rename(columns={"_crime_used": "cnt"})
    )
    # Tüm tarih evreni → eksik günleri 0’la dolduracağız
    dates = pd.Index(sorted(daily["__date__"].unique()), name="__date__")
    geoids = pd.Index(sorted(daily["GEOID"].unique()), name="GEOID")

    # Pivot (T×C): T=tarih sayısı, C=GEOID sayısı
    pivot = (
        daily.pivot(index="__date__", columns="GEOID", values="cnt")
             .reindex(index=dates, columns=geoids, fill_value=0.0)
             .astype(float)
    )  # shape: (T, C)

    # 5) Adjacency matrisi (S×C): S=src sayısı, C=GEOID sayısı
    src_list = sorted(nb["SRC_GEOID"].unique())
    # Kaynaklar df’de yoksa (ör. boş gün) yine de adjacency’de yer alabilir;
    # kolon evreni pivot’un geoids’idir.
    idx_geo = {g: j for j, g in enumerate(geoids)}
    idx_src = {g: i for i, g in enumerate(src_list)}
    A = np.zeros((len(src_list), len(geoids)), dtype=np.float32)
    for _, r in nb.iterrows():
        si = idx_src.get(r["SRC_GEOID"])
        tj = idx_geo.get(r["NEI_GEOID"])
        if si is not None and tj is not None:
            A[si, tj] += 1.0  # ağırlıksız komşuluk; isterseniz ağırlıklandırılabilir

    # 6) All-time komşu toplamı (zaman bağımsız)
    geo_totals = np.asarray(pivot.sum(axis=0))  # (C,)
    neigh_alltime = A @ geo_totals  # (S,)
    # GEOID→alltime map
    map_alltime = {src_list[i]: float(neigh_alltime[i]) for i in range(len(src_list))}

    # 7) Rolling pencereler (sızıntısız: shift(1))
    # pivot: (T, C); rolled: (T, C)
    feats = {}  # (date, src_geoid) → {win_name: value}
    rolled_cache = {}
    for win_name, k in win_spec:
        rolled = pivot.rolling(window=k, min_periods=1).sum().shift(1).fillna(0.0)
        rolled_cache[win_name] = rolled  # gerekirse debug için sakla

    # 8) Her pencere için komşu toplamları (T×S) = (T×C) @ (C×S)
    # Adjacency transpozu: C×S
    AT = A.T  # (C, S)
    neigh_frames = {}
    for win_name, _ in win_spec:
        rolled = rolled_cache[win_name]  # (T, C)
        TS = rolled.values @ AT  # (T, S)
        neigh_frames[win_name] = pd.DataFrame(
            TS, index=dates, columns=src_list, dtype="float32"
        )

    # 9) Orijinal tabloya merge
    out = df.copy()
    # alltime (GEOID-only, tarih bağımsız)
    out["neighbor_crime_alltime"] = out["GEOID"].map(map_alltime).fillna(0).astype("float32")

    # rollingler (GEOID + date)
    # hızlı map için (date, geoid) → index yaklaşımı
    # önce tarih başına küçük bir dict üretmek yerine direkt merge yapalım:
    out["_key_date"] = out["__date__"]
    out["_key_geoid"] = out["GEOID"]

    for win_name, _ in win_spec:
        g = neigh_frames[win_name].reset_index().melt(
            id_vars="__date__", var_name="GEOID", value_name=f"neighbor_crime_{win_name}"
        )
        g["__date__"] = g["__date__"].astype("string")
        g["GEOID"] = g["GEOID"].astype("string")
        before = out.shape
        out = out.merge(
            g.rename(columns={"__date__": "_key_date"}),
            left_on=["_key_date", "_key_geoid"],
            right_on=["_key_date", "GEOID"],
            how="left"
        )
        # sağdan gelen GEOID (melt’ten) gereksiz
        out.drop(columns=["GEOID_y"], inplace=True, errors="ignore")
        # orijinal GEOID'i geri adlandır
        if "GEOID_x" in out.columns:
            out = out.rename(columns={"GEOID_x": "GEOID"})
        # boş kalanlar 0
        out[f"neighbor_crime_{win_name}"] = (
            pd.to_numeric(out[f"neighbor_crime_{win_name}"], errors="coerce")
              .fillna(0)
              .astype("float32")
        )
        print(f"🔗 merge {win_name}: {before} → {out.shape}")

    # 10) temizlik ve yaz
    out.drop(columns=["_crime_used", "__date__", "_key_date", "_key_geoid"], errors="ignore", inplace=True)
    _safe_save_csv(out, DST)

    # kısa özet
    cols = ["neighbor_crime_alltime"] + [f"neighbor_crime_{nm}" for nm, _ in win_spec]
    try:
        print("Örnek (ilk 5 satır):")
        print(out[["GEOID"] + cols].head(5).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
