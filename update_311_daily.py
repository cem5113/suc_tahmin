#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_311_daily.py — 311 özetini günlük GRID & EVENTS'e sızıntısız ekler.

ENV
----
CRIME_DATA_DIR           (default: ".")
FR_GRID_DAILY_IN         (girdi grid)   default: "crime_prediction_data/fr_crime_grid_daily.csv"
FR_GRID_DAILY_OUT        (çıktı grid)   default: "crime_prediction_data/fr_crime_grid_daily.csv"
FR_EVENTS_DAILY_IN       (girdi events) default: "crime_prediction_data/fr_crime_events_daily.csv"
FR_EVENTS_DAILY_OUT      (çıktı events) default: "crime_prediction_data/fr_crime_events_daily.csv"

FR_311_DAILY_IN          (311 özet/ham CSV/Parquet)  Örn: "sf_311_last_5_years.csv"
FR_311_DATE_COL          (311’de tarih/datetime alanı; env ile "date" verilebilir)
FR_311_GEOID_COL         (311’de GEOID alanı) default: "GEOID"
FR_311_WINSOR_Q          (winsor üst quantile: "0" kapalı, "0.999" gibi) default: "0"
FR_311_EMA_ALPHAS        (virgülle "0.3,0.6") default: "0.3,0.6"

GEOID_LEN                (default: "11")
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ---------------- ENV ----------------
BASE = Path(os.getenv("CRIME_DATA_DIR", ".")).resolve()

GRID_IN  = os.getenv("FR_GRID_DAILY_IN",  "crime_prediction_data/fr_crime_grid_daily.csv").strip()
GRID_OUT = os.getenv("FR_GRID_DAILY_OUT", "crime_prediction_data/fr_crime_grid_daily.csv").strip()

EV_IN    = os.getenv("FR_EVENTS_DAILY_IN",  "crime_prediction_data/fr_crime_events_daily.csv").strip()
EV_OUT   = os.getenv("FR_EVENTS_DAILY_OUT", "crime_prediction_data/fr_crime_events_daily.csv").strip()

N311_IN  = os.getenv("FR_311_DAILY_IN", "").strip()
DATE_COL_ENV = (os.getenv("FR_311_DATE_COL", "") or "").strip()  # kullanıcı "date" ya da başka bir ad geçebilir
GEO_COL  = (os.getenv("FR_311_GEOID_COL", "GEOID") or "GEOID").strip()

def _parse_float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "")
    s = str(raw).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

WIN_Q = _parse_float_env("FR_311_WINSOR_Q", 0.0)   # 0 → kapalı
EMA_ALPHAS = []
_raw_alphas = os.getenv("FR_311_EMA_ALPHAS", "0.3,0.6")
for tok in str(_raw_alphas).split(","):
    tok = tok.strip()
    if not tok:
        continue
    try:
        a = float(tok)
        if 0.0 < a < 1.0:
            EMA_ALPHAS.append(a)
    except Exception:
        pass
if not EMA_ALPHAS:
    EMA_ALPHAS = [0.3, 0.6]

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ---------------- Utils ----------------
def log(msg: str): 
    print(msg, flush=True)

def _norm_geoid(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:GEOID_LEN].str.zfill(GEOID_LEN)

def _read_any(p: Path, nrows: int | None = None) -> pd.DataFrame:
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        return df if nrows is None else df.head(nrows)
    return pd.read_csv(p, low_memory=False, nrows=nrows)

def _winsorize_upper(series: pd.Series, q: float) -> pd.Series:
    if not (0.0 < q < 1.0):
        return series
    try:
        ub = series.quantile(q)
        log(f"🔧 Winsorize: q={q} üst={ub}")
        return np.minimum(series, ub)
    except Exception:
        return series

def _choose_date_column(df: pd.DataFrame) -> str:
    """
    Env’de verilen DATE_COL varsa onu kullanır, yoksa yaygın adaylar arasından
    veri çakışması olmadan en fazla non-null sağlayanı seçer.
    """
    candidates_ordered = []
    # 1) Env öncelik
    if DATE_COL_ENV:
        candidates_ordered.append(DATE_COL_ENV)
    # 2) Yaygın varyantlar
    common = [
        "date", "created_date", "createddate", "created time", "created_time",
        "request_datetime", "requested_datetime", "open_dt", "opened",
        "timestamp", "datetime"
    ]
    for c in common:
        if c not in candidates_ordered:
            candidates_ordered.append(c)
    # Sadece df'de var olanları filtrele
    in_df = [c for c in candidates_ordered if c in df.columns]
    if not in_df:
        raise SystemExit(f"❌ 311 kaynağında tarih kolonu bulunamadı. Env FR_311_DATE_COL='{DATE_COL_ENV}',"
                         f" adaylar: {common}, mevcut kolonlar: {list(df.columns)}")
    # En çok non-null sayan kolonu seç
    scores = {c: df[c].notna().sum() for c in in_df}
    picked = max(scores, key=scores.get)
    log(f"📅 311 tarih kolonu: '{picked}' (env='{DATE_COL_ENV or '-'}', adaylar={in_df}, non_null={scores[picked]})")
    return picked

# ---------------- Load inputs ----------------
def load_grid_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    g_path = BASE / GRID_IN
    e_path = BASE / EV_IN
    if not g_path.exists() or not e_path.exists():
        raise SystemExit(f"❌ GRID/EVENTS yok: {g_path} / {e_path}")
    grid = _read_any(g_path)
    evts = _read_any(e_path)
    log(f"📖 Okundu: {GRID_IN} ({len(grid):,}×{len(grid.columns)})")
    log(f"📖 Okundu: {EV_IN} ({len(evts):,}×{len(evts.columns)})")
    return grid, evts

def load_311_source() -> pd.DataFrame:
    if not N311_IN:
        raise SystemExit("❌ FR_311_DAILY_IN env boş.")
    p = Path(N311_IN)
    if not p.exists():
        raise SystemExit(f"❌ 311 kaynağı yok: {p}")
    df = _read_any(p)
    log(f"📖 Okundu: {p} ({len(df):,}×{len(df.columns)})")
    return df

# ---------------- Calendar mask ----------------
def build_calendar_dates(grid: pd.DataFrame) -> pd.DataFrame:
    need = ["GEOID", "date"]
    for c in need:
        if c not in grid.columns:
            raise SystemExit(f"❌ GRID'de zorunlu kolon yok: {c}")
    cal = grid[need].copy()
    cal["GEOID"] = _norm_geoid(cal["GEOID"])
    cal["date"]  = pd.to_datetime(cal["date"], errors="coerce").dt.date
    cal = cal.dropna().drop_duplicates().reset_index(drop=True)
    log(f"🗓️  Takvim boyutu: {len(cal):,} (GEOID×date)")
    return cal

# ---------------- Feature building ----------------
def make_311_daily_counts(df311: pd.DataFrame) -> pd.DataFrame:
    date_col = _choose_date_column(df311)
    if GEO_COL not in df311.columns:
        raise SystemExit(f"❌ 311 kaynağında {GEO_COL} yok.")

    # Datetime parse → sadece gün
    s = pd.to_datetime(df311[date_col], errors="coerce", utc=True)
    if s.notna().any():
        dcol = s.dt.tz_convert(None).dt.date if getattr(s.dt, "tz", None) is not None else s.dt.date
    else:
        dcol = pd.to_datetime(df311[date_col], errors="coerce").dt.date

    out = pd.DataFrame({
        "GEOID": _norm_geoid(df311[GEO_COL]),
        "date":  dcol
    }).dropna()

    g = (out.groupby(["GEOID","date"])
             .size()
             .rename("n311")
             .reset_index())

    # winsor
    if 0.0 < WIN_Q < 1.0:
        g["n311"] = _winsorize_upper(pd.to_numeric(g["n311"], errors="coerce"), WIN_Q).astype(float)
    else:
        g["n311"] = pd.to_numeric(g["n311"], errors="coerce").astype(float)
        log("🔧 Winsorize kapalı (q=0 veya q>=1).")

    return g

def make_311_features(g_daily: pd.DataFrame, cal_dates: pd.DataFrame) -> pd.DataFrame:
    cal = cal_dates.copy()
    cal["GEOID"] = _norm_geoid(cal["GEOID"])
    cal["date"]  = pd.to_datetime(cal["date"], errors="coerce").dt.date
    cal = cal.dropna()

    g = g_daily.copy()
    g["GEOID"] = _norm_geoid(g["GEOID"])
    g["date"]  = pd.to_datetime(g["date"], errors="coerce").dt.date
    g = g.dropna()

    merged = cal.merge(g, on=["GEOID","date"], how="left")
    merged["n311"] = pd.to_numeric(merged["n311"], errors="coerce").fillna(0.0)

    merged = merged.sort_values(["GEOID","date"]).reset_index(drop=True)
    m = merged.set_index(["GEOID","date"]).sort_index()

    # EMA — hizalama + güvenli atama
    for a in EMA_ALPHAS:
        ema = (m["n311"].groupby(level=0)
               .apply(lambda s: s.ewm(alpha=a, adjust=False).mean()))
        m[f"n311_ema_a{int(a*10)}"] = ema.reindex(m.index).to_numpy()

    # Basit pencereler
    m["n311_prev_1d"] = (m["n311"].groupby(level=0).shift(1)).fillna(0.0)
    m["n311_sum_3d"]  = (m["n311"].groupby(level=0)
                         .rolling(window=3, min_periods=1).sum()
                         .reset_index(level=0, drop=True))

    return m.reset_index()

# ---------------- Merge into GRID & EVENTS ----------------
def merge_into_grid_events(feats: pd.DataFrame, grid: pd.DataFrame, evts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats["GEOID"] = _norm_geoid(feats["GEOID"])
    feats["date"]  = pd.to_datetime(feats["date"], errors="coerce").dt.date

    g2 = grid.copy()
    g2["GEOID"] = _norm_geoid(g2["GEOID"])
    g2["date"]  = pd.to_datetime(g2["date"], errors="coerce").dt.date
    k = ["GEOID","date"]
    before = g2.shape
    g2 = g2.merge(feats, on=k, how="left")
    for c in [c for c in g2.columns if c.startswith("n311")]:
        g2[c] = pd.to_numeric(g2[c], errors="coerce").fillna(0.0)
    after = g2.shape
    log(f"🔗 GRID merge: {before} → {after}")

    e2 = evts.copy()
    e2["GEOID"] = _norm_geoid(e2["GEOID"])
    if "date" in e2.columns:
        e2["date"] = pd.to_datetime(e2["date"], errors="coerce").dt.date
        before_e = e2.shape
        e2 = e2.merge(feats, on=["GEOID","date"], how="left")
        for c in [c for c in e2.columns if c.startswith("n311")]:
            e2[c] = pd.to_numeric(e2[c], errors="coerce").fillna(0.0)
        after_e = e2.shape
        log(f"🔗 EVENTS merge (date-based): {before_e} → {after_e}")
    else:
        agg = feats.groupby("GEOID", as_index=False).median(numeric_only=True)
        before_e = e2.shape
        e2 = e2.merge(agg, on="GEOID", how="left")
        for c in [c for c in e2.columns if c.startswith("n311")]:
            e2[c] = pd.to_numeric(e2[c], errors="coerce").fillna(0.0)
        after_e = e2.shape
        log(f"🔗 EVENTS merge (calendar-fallback): {before_e} → {after_e}")

    return g2, e2

# ---------------- Save ----------------
def safe_save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.as_posix() + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ---------------- MAIN ----------------
def main():
    log("🚀 enrich_with_311.py (GRID + EVENTS, günlük-only, sızıntısız)")

    grid, evts = load_grid_events()
    df311 = load_311_source()

    g_daily = make_311_daily_counts(df311)      # tarih sütunu otomatik seçiliyor
    cal     = build_calendar_dates(grid)
    feats   = make_311_features(g_daily, cal) if len(cal) else pd.DataFrame()

    if feats.empty:
        log("ℹ️ 311 feature set boş → passthrough (kolon eklemeden kaydet).")
        safe_save_csv(grid, BASE / GRID_OUT)
        safe_save_csv(evts, BASE / EV_OUT)
        return 0

    g2, e2 = merge_into_grid_events(feats, grid, evts)

    safe_save_csv(g2, BASE / GRID_OUT)
    safe_save_csv(e2, BASE / EV_OUT)
    log(f"✅ Yazıldı: {GRID_OUT} ({len(g2):,} satır, {len(g2.columns)} sütun)")
    log(f"✅ Yazıldı: {EV_OUT} ({len(e2):,} satır, {len(e2.columns)} sütun)")

    try:
        log(g2.head(5).to_string(index=False))
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
