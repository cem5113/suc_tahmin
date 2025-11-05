#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_neighbors_daily.py
- neighbors.csv (GEOID komşuluk listesi) + günlük suç tabanı → komşu-suçu pencereleri.
- GRID (GEOID×date) ve EVENTS (olay satırları) dosyalarına ekler.
- Sızıntı yok: tüm pencereler shift(1) ile 'dün'e kadar.

ENV (varsayılanlar):
  CRIME_DATA_DIR          (crime_prediction_data)
  NEIGH_FILE              (neighbors.csv)

  FR_GRID_DAILY_IN        (fr_crime_grid_daily.csv)
  FR_GRID_DAILY_OUT       (fr_crime_grid_daily.csv)   # üzerine yazar
  FR_EVENTS_DAILY_IN      (fr_crime_events_daily.csv)
  FR_EVENTS_DAILY_OUT     (fr_crime_events_daily.csv) # üzerine yazar
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ---------- utils ----------
def log(m: str): print(m, flush=True)

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"ℹ️ Yok: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p} ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    # küçük downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p} ({len(df):,}×{df.shape[1]})")

def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str[:L].str.zfill(L)
    )

# --- tarih yardımcıları: HER ZAMAN datetime64[ns] ---
def _as_date64(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cand = None
    for c in ["date", "incident_date", "incident_datetime", "datetime", "time", "timestamp"]:
        if c in d.columns:
            cand = c; break
    if cand is None and {"year","month","day"}.issubset(d.columns):
        d["date"] = _as_date64(pd.to_datetime(d[["year","month","day"]], errors="coerce"))
    elif cand is not None:
        d["date"] = _as_date64(d[cand])
    else:
        d["date"] = pd.NaT
    return d

def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in low: return low[k.lower()]
    return None

# küçük yardımcı: merge anahtarlarını normalize et
def _normalize_keys(df: pd.DataFrame, geoid_len: int = 11) -> pd.DataFrame:
    d = df.copy()
    if "GEOID" in d.columns:
        d["GEOID"] = _norm_geoid(d["GEOID"], geoid_len)
    d = _ensure_date_col(d)
    return d

# ---------- config ----------
BASE_DIR   = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
NEIGH_FILE_ENV = os.getenv("NEIGH_FILE", "neighbors.csv")
NEIGH_PATH = (BASE_DIR / NEIGH_FILE_ENV) if not Path(NEIGH_FILE_ENV).is_absolute() else Path(NEIGH_FILE_ENV)

GRID_IN  = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN    = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT   = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ---------- core ----------
def build_daily_base(grid: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    GEOID×date bazında günlük suç sayısı (base_cnt) üretir.
    Öncelik: GRID'de 'crime_count' veya 'Y_day'/'Y_label' → sum.
    Yoksa EVENTS'i GEOID×date gruplayıp count alınır.
    """
    # 1) GRID varsa ve en az bir hedef kolonu içeriyorsa onu kullan
    if not grid.empty:
        g = _normalize_keys(grid, GEOID_LEN)
        base_cols = [c for c in ["crime_count","Y_day","Y_label"] if c in g.columns]
        if base_cols:
            for c in base_cols:
                g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype("int64")
            g["__base__"] = g[base_cols].sum(axis=1)
            gb = (
                g.groupby(["GEOID","date"], dropna=False)["__base__"]
                 .sum().reset_index(name="base_cnt")
            )
            gb["date"] = _as_date64(gb["date"])
            return gb

    # 2) Events'ten günlük üret
    if not events.empty:
        ev = _normalize_keys(events, GEOID_LEN).dropna(subset=["date"])
        if "GEOID" not in ev.columns:
            raise RuntimeError("EVENTS içinde GEOID yok.")
        if "crime_count" in ev.columns:
            ev["crime_count"] = pd.to_numeric(ev["crime_count"], errors="coerce").fillna(0).astype("int64")
            gb = (
                ev.groupby(["GEOID","date"], dropna=False)["crime_count"]
                  .sum().reset_index(name="base_cnt")
            )
        else:
            gb = (
                ev.groupby(["GEOID","date"], dropna=False)
                  .size().reset_index(name="base_cnt")
            )
        gb["date"] = _as_date64(gb["date"])
        return gb

    raise RuntimeError("GRID ve EVENTS boş; günlük taban oluşturulamadı.")

def neighbor_daily_features(base: pd.DataFrame, neigh: pd.DataFrame) -> pd.DataFrame:
    """
    base: GEOID×date×base_cnt
    neigh: geoid, neighbor
    Adımlar:
      - neighbors ⨯ base (neighbor→date→count)
      - geoid×date toplamla → nb_cnt_day
      - geoid bazında tarihe göre sırala, shift(1) ve rolling(3,7)
    """
    b = base.copy()
    b["GEOID"]    = _norm_geoid(b["GEOID"], GEOID_LEN)
    b["date"]     = _as_date64(b["date"])
    b["base_cnt"] = pd.to_numeric(b["base_cnt"], errors="coerce").fillna(0).astype("int64")

    # Tam tarih kapsaması: her GEOID için min→max arası tüm günler (eksikler 0)
    full = []
    for g, gdf in b.groupby("GEOID", dropna=False):
        gdf = gdf.sort_values("date")
        rng = pd.date_range(gdf["date"].min(), gdf["date"].max(), freq="D")  # datetime64[ns]
        aux = pd.DataFrame({"GEOID": g, "date": rng})
        aux = aux.merge(gdf[["date","base_cnt"]], on="date", how="left")
        aux["base_cnt"] = aux["base_cnt"].fillna(0).astype("int64")
        full.append(aux)
    b2 = pd.concat(full, ignore_index=True)

    # Komşuluk: geoid (src) → neighbor (dst)
    nb = neigh.rename(columns={
        _pick(neigh.columns, "geoid","src","source"): "geoid",
        _pick(neigh.columns, "neighbor","dst","target"): "neighbor"
    }).copy()
    nb["geoid"]    = _norm_geoid(nb["geoid"], GEOID_LEN)
    nb["neighbor"] = _norm_geoid(nb["neighbor"], GEOID_LEN)
    nb = nb.dropna()

    # Komşu günlük sayıları: nb (geoid, neighbor) ⨯ base (neighbor, date, base_cnt)
    nb_merge = nb.merge(b2.rename(columns={"GEOID":"neighbor"}), on="neighbor", how="left")
    day_sum = (
        nb_merge.groupby(["geoid","date"], dropna=False)["base_cnt"]
                .sum().reset_index(name="neighbor_cnt_day")
    )

    # Rolling pencereler: geoid bazında tarih sırasıyla, dünü dahil (shift1)
    day_sum = day_sum.sort_values(["geoid","date"])
    day_sum["nb_last1d"] = day_sum.groupby("geoid")["neighbor_cnt_day"].shift(1).fillna(0)
    for W in (3,7):
        day_sum[f"nb_last{W}d"] = (
            day_sum.groupby("geoid")["neighbor_cnt_day"]
                   .shift(1)  # sızıntı yok
                   .rolling(W, min_periods=1).sum()
        ).fillna(0)

    out = day_sum.rename(columns={
        "geoid": "GEOID",
        "nb_last1d": "neighbor_crime_1d",
        "nb_last3d": "neighbor_crime_3d",
        "nb_last7d": "neighbor_crime_7d",
    })[["GEOID","date","neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]]

    out["date"] = _as_date64(out["date"])
    for c in ["neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    return out

def enrich_targets(grid: pd.DataFrame, events: pd.DataFrame, feats: pd.DataFrame):
    # merge anahtarlarını her iki tarafta normalize et (kritik!)
    f2 = feats.copy()
    f2["GEOID"] = _norm_geoid(f2["GEOID"], GEOID_LEN)
    f2["date"]  = _as_date64(f2["date"])

    g2 = grid
    e2 = events

    if not grid.empty:
        g_norm = _normalize_keys(grid, GEOID_LEN)
        g_norm["date"] = _as_date64(g_norm["date"])
        f2["date"]     = _as_date64(f2["date"])
        g2 = g_norm.merge(f2, on=["GEOID","date"], how="left")
        for c in ["neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]:
            if c in g2.columns:
                g2[c] = pd.to_numeric(g2[c], errors="coerce").fillna(0).astype("int64")

    if not events.empty:
        ev_norm = _normalize_keys(events, GEOID_LEN)
        ev_norm["date"] = _as_date64(ev_norm["date"])
        f2["date"]      = _as_date64(f2["date"])
        e2 = ev_norm.merge(f2, on=["GEOID","date"], how="left")
        for c in ["neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]:
            if c in e2.columns:
                e2[c] = pd.to_numeric(e2[c], errors="coerce").fillna(0).astype("int64")
    return g2, e2

def main() -> int:
    log("🚀 enrich_with_neighbors_daily.py")

    # Dosya yollarını çöz
    grid_in  = BASE_DIR / GRID_IN  if not GRID_IN.is_absolute()  else GRID_IN
    grid_out = BASE_DIR / GRID_OUT if not GRID_OUT.is_absolute() else GRID_OUT
    ev_in    = BASE_DIR / EV_IN    if not EV_IN.is_absolute()    else EV_IN
    ev_out   = BASE_DIR / EV_OUT   if not EV_OUT.is_absolute()   else EV_OUT

    # Girdi oku
    grid = _read_csv(grid_in)
    ev   = _read_csv(ev_in)

    # Komşuluk dosyası
    if not NEIGH_PATH.exists():
        raise FileNotFoundError(f"❌ neighbors.csv bulunamadı: {NEIGH_PATH.resolve()}")
    neigh = pd.read_csv(NEIGH_PATH, low_memory=False).dropna()
    if neigh.empty:
        raise RuntimeError("❌ neighbors.csv boş.")

    # Günlük taban (GEOID×date×base_cnt)
    base = build_daily_base(grid, ev)
    log(f"🧮 base_cnt hazır: {len(base):,} satır (GEOID×date)")

    # Komşu pencereleri
    feats = neighbor_daily_features(base, neigh)
    log(f"✨ neighbor feats: {len(feats):,} satır (GEOID×date) — eklenecek kolonlar: "
        f"[neighbor_crime_1d, _3d, _7d]")

    # Hedef dosyaları zenginleştir
    g2, e2 = enrich_targets(grid, ev, feats)

    # Yaz
    if not g2.empty:
        _safe_write_csv(g2, grid_out)
    else:
        log("ℹ️ GRID yok → yazılmadı.")
    if not e2.empty:
        _safe_write_csv(e2, ev_out)
    else:
        log("ℹ️ EVENTS yok → yazılmadı.")

    # Küçük önizleme
    try:
        cols = ["GEOID","date","neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]
        if not g2.empty:
            log("— GRID preview —")
            log(g2[cols].head(6).to_string(index=False))
        if not e2.empty:
            log("— EVENTS preview —")
            log(e2[cols].head(6).to_string(index=False))
    except Exception:
        pass

    log("✅ Tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
