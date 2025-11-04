#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_with_311.py — Günlük 311 (GEOID×date) özelliklerini hem GRID'e hem EVENTS'e ekler.
- Saatlik YOK. Sadece günlük agregasyon.
- Leakage YOK: tüm lag/rolling/EMA hesapları shift(1) ile (gün t için yalnızca t-1..t-k kullanılır).
- [1] Tam takvim reindex (eksik günleri 0'la)
- [2] EMA/decay (α=0.3, 0.5 — ENV ile değişir)
- [3] Trend (7g - 30g)
- [8] Kalite: dedup (id varsa), winsorize (opsiyonel), downcast

ENV (varsayılanlar):
  FR_311_PATH           (fr_311.csv)               # 311 ham olay verisi (tercih edilen kaynak)
  FR_311_DAILY_IN       ("")                       # (opsiyonel) doğrudan günlük 311 dosyası (GEOID,date,n311_d)
  FR_311_DATE_COL       (incident_datetime)        # 311 zaman kolonu (auto-detect var)
  FR_311_GEOID_COL      (GEOID)                    # 311 geoid kolonu
  FR_311_WINSOR_Q       (0.999)                    # 0→kapat; aksi winsor üst quantile
  FR_311_EMA_ALPHAS     ("0.3,0.5")                # EMA α değerleri virgüllü

  FR_GRID_DAILY_IN      (fr_crime_grid_daily.csv)  # GRID giriş
  FR_GRID_DAILY_OUT     (fr_crime_grid_daily.csv)  # GRID çıkış (üzerine yazar)
  FR_EVENTS_DAILY_IN    (fr_crime_events_daily.csv)# EVENTS giriş
  FR_EVENTS_DAILY_OUT   (fr_crime_events_daily.csv)# EVENTS çıkış (üzerine yazar)
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ----------- ENV -----------
P_311_RAW   = Path(os.getenv("FR_311_PATH", "fr_311.csv"))
P_311_DAILY = os.getenv("FR_311_DAILY_IN", "").strip()  # varsa direkt günlük dosya
COL_DT_311  = os.getenv("FR_311_DATE_COL", "incident_datetime")
COL_G_311   = os.getenv("FR_311_GEOID_COL", "GEOID")

GRID_IN     = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT    = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN       = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT      = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

WIN_Q       = float(os.getenv("FR_311_WINSOR_Q", "0.999"))  # 0 kapatır
EMA_ALPHAS  = [float(x) for x in os.getenv("FR_311_EMA_ALPHAS", "0.3,0.5").split(",") if x.strip()]

def log(x): print(x, flush=True)

# ----------- I/O -----------
def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"❌ Bulunamadı: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p} ({len(df):,}×{df.shape[1]})")
    return df

def _save_csv(df: pd.DataFrame, p: Path):
    # downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p.with_suffix(p.suffix + ".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p} ({len(df):,}×{df.shape[1]})")

# ----------- Utils -----------
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)", expand=False)
    return x.str[:L].str.zfill(L)

def autodetect_dt_col(df: pd.DataFrame, pref: str) -> str | None:
    if pref in df.columns: return pref
    for c in ["requested_datetime","closed_date","updated_datetime","created_date",
              "incident_datetime","datetime","timestamp","date"]:
        if c in df.columns: return c
    return None

def ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = _to_date(out["date"])
        return out
    for c in ("event_date","dt","day","incident_datetime","datetime","timestamp","created_datetime"):
        if c in out.columns:
            out["date"] = _to_date(out[c])
            return out
    raise ValueError("date kolonu türetilemedi.")

# ----------- 311 → günlük -----------
def load_311_daily_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    # dedup (kalite) — varsa id sütunu
    for cid in ("request_id","service_request_id","id"):
        if cid in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[cid]).copy()
            if len(df) != before:
                log(f"🧹 311 dedup: {before - len(df)} satır çıkarıldı ({cid})")
            break

    # GEOID
    col_g = COL_G_311 if COL_G_311 in df.columns else next((c for c in df.columns if "geoid" in c.lower()), None)
    if col_g is None:
        log("⚠️ 311 verisinde GEOID yok → tüm 311 atlanacak.")
        return pd.DataFrame(columns=["GEOID","date","n311_d"])
    df["GEOID"] = _norm_geoid(df[col_g])

    # tarih
    dt_col = autodetect_dt_col(df, COL_DT_311)
    if dt_col is None:
        log("⚠️ 311 zaman kolonu bulunamadı → atlanıyor.")
        return pd.DataFrame(columns=["GEOID","date","n311_d"])
    df["date"] = _to_date(df[dt_col])

    # günlük sayım
    daily = (df.dropna(subset=["GEOID","date"])
               .groupby(["GEOID","date"], as_index=False)
               .size().rename(columns={"size":"n311_d"}))
    return daily

def load_311_daily() -> pd.DataFrame:
    # 1) doğrudan günlük dosya
    if P_311_DAILY:
        p = Path(P_311_DAILY)
        d = _read_csv(p)
        if d.empty:
            return d
        # normalize
        if "GEOID" in d.columns:
            d["GEOID"] = _norm_geoid(d["GEOID"])
        d = ensure_date_col(d)
        # sütun adı yakala
        if "n311_d" not in d.columns:
            cand = [c for c in d.columns if "311" in c.lower() and "daily" in c.lower()]
            if cand:
                d = d.rename(columns={cand[0]: "n311_d"})
            elif "count" in d.columns:
                d = d.rename(columns={"count":"n311_d"})
            else:
                # satır olayı ise say
                d = (d.groupby(["GEOID","date"], dropna=False).size()
                       .reset_index(name="n311_d"))
        return d[["GEOID","date","n311_d"]].copy()

    # 2) ham dosyadan üret
    raw = _read_csv(P_311_RAW)
    if raw.empty:
        return raw
    return load_311_daily_from_raw(raw)

# ----------- Özellikler (reindex + rolling + EMA + trend) -----------
def make_311_features(daily: pd.DataFrame, calendar_dates: np.ndarray) -> pd.DataFrame:
    """
    daily: GEOID,date,n311_d
    calendar_dates: referans takvim (GRID/EV'den toplanmış tüm günler)
    """
    if daily is None or daily.empty:
        return pd.DataFrame(columns=[
            "GEOID","date","n311_d","n311_prev_1d","n311_roll_3d","n311_roll_7d","n311_roll_30d",
            *[f"n311_ema_a{int(a*10)}" for a in EMA_ALPHAS], "n311_trend_7v30"
        ])

    # winsorize (opsiyonel)
    if WIN_Q and 0 < WIN_Q < 1:
        q = daily["n311_d"].quantile(WIN_Q)
        daily["n311_d"] = daily["n311_d"].clip(upper=max(q, 1))
        log(f"🔧 Winsorize: q={WIN_Q} üst={q:.1f}")

    # tüm GEOID'ler için tam takvim reindex
    geoids = daily["GEOID"].dropna().unique()
    all_days = pd.to_datetime(pd.Series(calendar_dates)).dt.date.unique()
    idx = pd.MultiIndex.from_product([geoids, all_days], names=["GEOID","date"])
    g = (daily.set_index(["GEOID","date"])
              .reindex(idx, fill_value=0)
              .reset_index()
              .sort_values(["GEOID","date"]))

    # shift(1) + rolling
    g["n311_prev_1d"]  = g.groupby("GEOID")["n311_d"].shift(1)
    g["n311_roll_3d"]  = g.groupby("GEOID")["n311_d"].shift(1).rolling(3,  min_periods=1).sum()
    g["n311_roll_7d"]  = g.groupby("GEOID")["n311_d"].shift(1).rolling(7,  min_periods=1).sum()
    g["n311_roll_30d"] = g.groupby("GEOID")["n311_d"].shift(1).rolling(30, min_periods=1).sum()

    # EMA
    for a in EMA_ALPHAS:
        g[f"n311_ema_a{int(a*10)}"] = (
            g.groupby("GEOID")["n311_d"].apply(lambda s: s.shift(1).ewm(alpha=a, adjust=False).mean())
        ).astype("float32")

    # doldurma ve tipler
    int_cols = ["n311_d","n311_prev_1d","n311_roll_3d","n311_roll_7d","n311_roll_30d"]
    g[int_cols] = g[int_cols].fillna(0).astype("int32")

    # Trend
    g["n311_trend_7v30"] = (g["n311_roll_7d"] - g["n311_roll_30d"]).astype("float32")

    return g

# ----------- Enrich ----------- 
def enrich_grid(grid: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    if "date" in out.columns and not np.issubdtype(pd.Series(out["date"]).dtype, np.datetime64):
        out["date"] = _to_date(out["date"])
    out = out.merge(feats, on=["GEOID","date"], how="left")
    # fill NA
    for c in ["n311_d","n311_prev_1d","n311_roll_3d","n311_roll_7d","n311_roll_30d"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int32")
    for c in [f"n311_ema_a{int(a*10)}" for a in EMA_ALPHAS] + ["n311_trend_7v30"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
    return out

def enrich_events(ev: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = ev.copy()
    # GEOID + date üret/normalize
    if "GEOID" in out.columns:
        out["GEOID"] = _norm_geoid(out["GEOID"])
    if "date" not in out.columns:
        if "incident_datetime" in out.columns:
            out["date"] = _to_date(out["incident_datetime"])
        else:
            raise ValueError("EVENTS içinde 'date' yok ve 'incident_datetime' yok.")
    # sadece geçmiş pencereler
    keep = ["GEOID","date","n311_prev_1d","n311_roll_3d","n311_roll_7d","n311_roll_30d",
            *[f"n311_ema_a{int(a*10)}" for a in EMA_ALPHAS], "n311_trend_7v30"]
    feats2 = feats[keep].drop_duplicates()
    out = out.merge(feats2, on=["GEOID","date"], how="left")
    # fill
    for c in ["n311_prev_1d","n311_roll_3d","n311_roll_7d","n311_roll_30d"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int32")
    for c in [f"n311_ema_a{int(a*10)}" for a in EMA_ALPHAS] + ["n311_trend_7v30"]:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
    return out

# ----------- MAIN -----------
def main() -> int:
    log("🚀 enrich_with_311.py (GRID + EVENTS, günlük-only, sızıntısız)")
    # 1) Dosyaları oku
    grid = _read_csv(GRID_IN)
    ev   = _read_csv(EV_IN)

    if grid.empty and ev.empty:
        log("ℹ️ Ne GRID ne EVENTS bulundu — çıkılıyor.")
        return 0

    # Referans takvim: GRID varsa ondan, yoksa EVENTS'ten topla
    cal_dates = np.array([])
    if not grid.empty:
        g2 = ensure_date_col(grid)
        cal_dates = g2["date"].values
    elif not ev.empty:
        e2 = ev.copy()
        if "date" not in e2.columns and "incident_datetime" in e2.columns:
            e2["date"] = _to_date(e2["incident_datetime"])
        cal_dates = e2["date"].values
    cal_dates = pd.to_datetime(pd.Series(cal_dates), errors="coerce").dt.date.dropna().unique()

    # 2) 311 günlük yükle
    d311 = load_311_daily()
    if d311.empty:
        log("ℹ️ Günlük 311 bulunamadı → sadece 0 kolonları eklenecek.")

    # 3) Özellikleri üret
    feats = make_311_features(d311, cal_dates) if len(cal_dates) else pd.DataFrame()

    # 4) Enrich & yaz
    if not grid.empty:
        grid2 = enrich_grid(ensure_date_col(grid), feats) if not feats.empty else grid.assign(
            n311_d=0, n311_prev_1d=0, n311_roll_3d=0, n311_roll_7d=0, n311_roll_30d=0,
            **{f"n311_ema_a{int(a*10)}":0.0 for a in EMA_ALPHAS}, n311_trend_7v30=0.0
        )
        _save_csv(grid2, GRID_OUT)

    if not ev.empty:
        # events dosyasında aynı gün bilgisi kullanılmaz; merge edilen tüm kolonlar geçmiş pencerelerdir
        ev2 = enrich_events(ev, feats) if not feats.empty else ev.assign(
            n311_prev_1d=0, n311_roll_3d=0, n311_roll_7d=0, n311_roll_30d=0,
            **{f"n311_ema_a{int(a*10)}":0.0 for a in EMA_ALPHAS}, n311_trend_7v30=0.0
        )
        _save_csv(ev2, EV_OUT)

    # 5) Kısa önizleme
    try:
        if not grid.empty:
            cols = ["GEOID","date","n311_d","n311_prev_1d","n311_roll_7d","n311_roll_30d","n311_trend_7v30"]
            cols += [c for c in feats.columns if c.startswith("n311_ema_")] if not feats.empty else [f"n311_ema_a{int(a*10)}" for a in EMA_ALPHAS]
            head_df = (grid2 if 'grid2' in locals() else grid)[[c for c in cols if c in (grid2 if 'grid2' in locals() else grid).columns]].head(8)
            log(head_df.to_string(index=False))
        if not ev.empty:
            cols = ["GEOID","date","n311_prev_1d","n311_roll_7d","n311_roll_30d","n311_trend_7v30"]
            cols += [c for c in (feats.columns if not feats.empty else []) if c.startswith("n311_ema_")] if not feats.empty else [f"n311_ema_a{int(a*10)}" for a in EMA_ALPHAS]
            head_df = (ev2 if 'ev2' in locals() else ev)[[c for c in cols if c in (ev2 if 'ev2' in locals() else ev).columns]].head(8)
            log(head_df.to_string(index=False))
    except Exception:
        pass

    log("✅ Tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
