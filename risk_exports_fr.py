#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_exports_fr.py — İki çıktı üretir (başlıkları AYNI):
  1) crime_prediction_data/risk_hourly_grid_full_labeled.csv   (≤7 gün, saatlik)
  2) crime_prediction_data/risk_daily_grid_full_labeled.csv    (≤365 gün, günlük, hour_range="")
Kolonlar:
  GEOID,date,hour_range,risk_score,risk_level,risk_decile,expected_count,
  top1_category,top1_prob,top1_expected,
  top2_category,top2_prob,top2_expected,
  top3_category,top3_prob,top3_expected
"""

import os, sys, re, math, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Ortam / Sabitler
# ------------------------------------------------------------
CRIME_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
CRIME_DIR.mkdir(parents=True, exist_ok=True)

REQ_COLS = [
    "GEOID","date","hour_range","risk_score","risk_level","risk_decile","expected_count",
    "top1_category","top1_prob","top1_expected",
    "top2_category","top2_prob","top2_expected",
    "top3_category","top3_prob","top3_expected",
]

HOUR_WINDOW_DAYS   = int(os.getenv("RISK_HOURLY_WINDOW_DAYS", "7"))      # Saatlikte üretilecek görünüm penceresi
DAILY_WINDOW_DAYS  = int(os.getenv("RISK_DAILY_WINDOW_DAYS", "365"))     # Günlükte üretilecek görünüm penceresi
BASELINE_WINDOW    = int(os.getenv("RISK_BASELINE_WINDOW_DAYS", "30"))   # Poisson tabanlı beklenen için referans

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def _ensure_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    """GEOID, date (YYYY-MM-DD), hour_range(int 0-23) türetir/normalize eder."""
    d = df.copy()
    low = {c.lower(): c for c in d.columns}

    # GEOID normalize
    if "GEOID" in d.columns:
        L = int(os.getenv("GEOID_LEN","11"))
        d["GEOID"] = d["GEOID"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(L)

    # date
    if "date" not in d.columns:
        for c in ("datetime","incident_datetime","date_time"):
            if c in low:
                dc = pd.to_datetime(d[low[c]], errors="coerce", utc=False)
                d["date"] = dc.dt.date.astype("string")
                break
    else:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype("string")

    # hour_range (int 0..23)
    if "hour_range" not in d.columns:
        hc = None
        for c in ("hour","hour_of_day","event_hour","hourrange"):
            if c in low:
                hc = low[c]; break
        if hc is not None:
            d["hour_range"] = pd.to_numeric(d[hc], errors="coerce").fillna(0).astype(int)
        else:
            ref = None
            for c in ("datetime","incident_datetime","date_time"):
                if c in low:
                    ref = pd.to_datetime(d[low[c]], errors="coerce", utc=False); break
            if ref is not None:
                d["hour_range"] = ref.dt.hour.fillna(0).astype(int)
            else:
                d["hour_range"] = 0
    else:
        d["hour_range"] = pd.to_numeric(d["hour_range"], errors="coerce").fillna(0).astype(int)

    return d

def _quantile_deciles(x: pd.Series) -> pd.Series:
    """1..10 decile; tekil dağılımda 1 döndür."""
    try:
        ranks = x.rank(method="first", pct=True)
        if ranks.nunique() < 2:
            return pd.Series(np.ones(len(x), dtype=int), index=x.index)
        q = pd.qcut(ranks, 10, labels=False, duplicates="drop") + 1
        return q.astype(int)
    except Exception:
        return pd.Series(np.ones(len(x), dtype=int), index=x.index)

def _risk_level_from_score(x: float, q80: float, q90: float, threshold: float) -> str:
    if x >= max(threshold, q90):        return "high"
    if x >= max(threshold*0.8, q80):    return "medium"
    return "low"

def _pick_first(*paths):
    for p in paths:
        if p and Path(p).is_file():
            return str(p)
    return None

def _read_proba_path(pth: str) -> np.ndarray:
    p = str(pth).lower()
    if p.endswith(".npy"):
        return np.load(pth)
    dfp = pd.read_csv(pth)
    for c in ("proba","prob","p","y_pred_proba","risk_score"):
        if c in dfp.columns:
            v = pd.to_numeric(dfp[c], errors="coerce").fillna(0.0).to_numpy()
            return np.clip(v, 0.0, 1.0)
    if dfp.shape[1] == 1:
        v = pd.to_numeric(dfp.iloc[:,0], errors="coerce").fillna(0.0).to_numpy()
        return np.clip(v, 0.0, 1.0)
    raise ValueError("Proba dosyasında uygun kolon bulunamadı.")

def _risk_from_df_or_riskfile(df: pd.DataFrame, risk_csv: str | None) -> np.ndarray | None:
    """DF içinden risk_score ya da risk dosyasından join."""
    if "risk_score" in df.columns:
        v = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0).to_numpy()
        return np.clip(v, 0.0, 1.0)
    if risk_csv and Path(risk_csv).is_file():
        try:
            r = pd.read_csv(risk_csv, low_memory=False, dtype={"GEOID": str})
            r = _ensure_date_hour(r)
            need = {"GEOID","date","hour_range","risk_score"}
            if need.issubset(r.columns):
                key = ["GEOID","date","hour_range"]
                m = df.merge(r[key + ["risk_score"]], on=key, how="left")
                v = pd.to_numeric(m["risk_score"], errors="coerce").fillna(0.0).to_numpy()
                return np.clip(v, 0.0, 1.0)
        except Exception as e:
            print(f"[WARN] risk file join failed: {e}", file=sys.stderr)
    return None

def _load_recent_crime_for_baseline(window_days=30) -> pd.DataFrame | None:
    """Poisson tabanlı beklenen hesap için sf_crime.csv'den son penceriyi yükle."""
    raw_path = CRIME_DIR / "sf_crime.csv"
    if not raw_path.is_file():
        return None
    dfc = pd.read_csv(raw_path, low_memory=False, dtype={"GEOID": str})
    # tarih türet
    if "date" in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc["date"], errors="coerce")
    elif "datetime" in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc["datetime"], errors="coerce")
    else:
        return None
    dfc = dfc.dropna(subset=["date"])
    latest = dfc["date"].max()
    if pd.isna(latest):
        return None
    start = latest.normalize() - pd.Timedelta(days=window_days-1)
    dfc = dfc[(dfc["date"] >= start) & (dfc["date"] <= latest)].copy()

    # hour_range üret
    if "hour_range" not in dfc.columns:
        if "event_hour" in dfc.columns:
            h = pd.to_numeric(dfc["event_hour"], errors="coerce").fillna(0).astype(int)
        else:
            h = dfc["date"].dt.hour.fillna(0).astype(int)
        dfc["hour_range"] = h
    else:
        dfc["hour_range"] = pd.to_numeric(dfc["hour_range"], errors="coerce").fillna(0).astype(int)
    # kategori yoksa "all"
    if "category" not in dfc.columns:
        dfc["category"] = "all"
    # date string
    dfc["date"] = dfc["date"].dt.date.astype("string")
    return dfc

def _build_poisson_baselines(window_days=30):
    """λ_base & p_base & kategori payları (GEOID, hour) + şehir fallback."""
    dfc = _load_recent_crime_for_baseline(window_days=window_days)
    if dfc is None or len(dfc)==0:
        return {}, {}, {}, ({}, {}, {}), 1

    # ndays
    dts = pd.to_datetime(dfc["date"], errors="coerce")
    ndays = max(1, (dts.max().normalize() - dts.min().normalize()).days + 1)

    key = ["GEOID","hour_range"]
    tot = dfc.groupby(key).size().rename("events").reset_index()
    tot["lambda_base"] = tot["events"] / ndays
    tot["p_base"] = 1.0 - np.exp(-tot["lambda_base"])

    cat = (dfc.groupby(key+["category"]).size().rename("n").reset_index())
    cat_tot = cat.groupby(key)["n"].sum().rename("n_tot").reset_index()
    cat = cat.merge(cat_tot, on=key, how="left")
    cat["share"] = cat["n"] / cat["n_tot"].replace(0, np.nan)

    lam_map = {(r["GEOID"], r["hour_range"]): float(r["lambda_base"]) for _, r in tot.iterrows()}
    p_map   = {(r["GEOID"], r["hour_range"]): float(r["p_base"]) for _, r in tot.iterrows()}
    shares_map = {}
    for (g, hr), grp in cat.groupby(["GEOID","hour_range"]):
        shares_map[(g, hr)] = {row["category"]: float(row["share"])
                               for _, row in grp.iterrows() if not pd.isna(row["share"])}

    # Şehir fallback (hour bazında)
    hr_tot = dfc.groupby(["hour_range"]).size().rename("events").reset_index()
    hr_tot["lambda_hr_city"] = hr_tot["events"] / (ndays * max(1, dfc["GEOID"].nunique()))
    hr_tot["p_hr_city"] = 1.0 - np.exp(-hr_tot["lambda_hr_city"])
    lam_city = {r["hour_range"]: float(r["lambda_hr_city"]) for _, r in hr_tot.iterrows()}
    p_city   = {r["hour_range"]: float(r["p_hr_city"]) for _, r in hr_tot.iterrows()}

    # Şehir kategori payı (hour bazında)
    city_cat = (dfc.groupby(["hour_range","category"]).size().rename("n").reset_index())
    city_cat_tot = city_cat.groupby(["hour_range"])["n"].sum().rename("n_tot").reset_index()
    city_cat = city_cat.merge(city_cat_tot, on="hour_range", how="left")
    city_cat["share"] = city_cat["n"] / city_cat["n_tot"].replace(0, np.nan)
    shares_city = {}
    for hr, grp in city_cat.groupby("hour_range"):
        shares_city[hr] = {row["category"]: float(row["share"])
                           for _, row in grp.iterrows() if not pd.isna(row["share"])}

    return lam_map, p_map, shares_map, (lam_city, p_city, shares_city), ndays

def _load_topk_categories(key_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    A) proba_multiclass.(parquet|csv) long: GEOID,date,hour_range,category,prob
    B) proba_multiclass_wide.(parquet|csv): prob_<cat> sütunları
    C) yoksa boş
    """
    # A) long
    for name in ("proba_multiclass.parquet","proba_multiclass.csv"):
        p = _pick_first(base_dir/name, Path(".")/name)
        if p:
            try:
                t = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
                need = {"GEOID","date","hour_range","category","prob"}
                if need.issubset(t.columns):
                    t = t.copy()
                    t["prob"] = pd.to_numeric(t["prob"], errors="coerce").fillna(0.0)
                    # Top-3
                    t = t.sort_values(["GEOID","date","hour_range","prob"], ascending=[True,True,True,False])
                    top3 = t.groupby(["GEOID","date","hour_range"]).head(3)
                    def _expand(g):
                        cats = g["category"].tolist()
                        probs = g["prob"].tolist()
                        row = {}
                        for i in range(3):
                            c = cats[i] if i < len(cats) else ""
                            p_ = float(probs[i]) if i < len(probs) else 0.0
                            row[f"top{i+1}_category"] = c
                            row[f"top{i+1}_prob"] = p_
                            row[f"top{i+1}_expected"] = p_  # approx
                        return pd.Series(row)
                    exp = top3.groupby(["GEOID","date","hour_range"]).apply(_expand).reset_index()
                    return exp
            except Exception as e:
                print(f"[WARN] read {p} failed: {e}", file=sys.stderr)
    # B) wide
    for name in ("proba_multiclass_wide.parquet","proba_multiclass_wide.csv"):
        p = _pick_first(base_dir/name, Path(".")/name)
        if p:
            try:
                t = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
                prob_cols = [c for c in t.columns if c.startswith("prob_")]
                need_key = {"GEOID","date","hour_range"}
                if need_key.issubset(t.columns) and prob_cols:
                    key = ["GEOID","date","hour_range"]
                    def _top3_row(row):
                        probs = {c[5:]: float(row[c]) if pd.notna(row[c]) else 0.0 for c in prob_cols}
                        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                        out = {}
                        for i in range(3):
                            c, p_ = items[i] if i < len(items) else ("", 0.0)
                            out[f"top{i+1}_category"] = c
                            out[f"top{i+1}_prob"] = p_
                            out[f"top{i+1}_expected"] = p_
                        return pd.Series(out)
                    exp = t[key + prob_cols].copy()
                    exp = pd.concat([exp[key], exp.apply(_top3_row, axis=1)], axis=1)
                    return exp
            except Exception as e:
                print(f"[WARN] read {p} failed: {e}", file=sys.stderr)

    # C) boş
    k = key_df[["GEOID","date","hour_range"]].drop_duplicates().copy()
    for i in (1,2,3):
        k[f"top{i}_category"] = ""
        k[f"top{i}_prob"] = 0.0
        k[f"top{i}_expected"] = 0.0
    return k

# ------------------------------------------------------------
# Çekirdek üretimler
# ------------------------------------------------------------
def build_hourly(df_in: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Saatlik (≤7 gün) tabloyu üretir; REQ_COLS döner."""
    df = _ensure_date_hour(df_in)
    # pencere ≤7 gün
    dt = pd.to_datetime(df["date"], errors="coerce")
    last = dt.max()
    if pd.notna(last):
        start = (last.normalize() - pd.Timedelta(days=HOUR_WINDOW_DAYS-1)).date().isoformat()
        df = df[dt.dt.date.astype("string") >= start].copy()

    key = ["GEOID","date","hour_range"]
    # Proba → risk_score
    risk_score = np.clip(np.asarray(proba).astype(float), 0.0, 1.0)
    out = df.copy()
    out["risk_score"] = risk_score

    # decile & level
    q80 = out["risk_score"].quantile(0.8)
    q90 = out["risk_score"].quantile(0.9)
    out["risk_decile"] = _quantile_deciles(pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0))
    out["risk_level"]  = out["risk_score"].apply(lambda x: _risk_level_from_score(x, q80, q90, threshold))

    # expected_count — Poisson baseline (varsa), yoksa fallback = risk_score
    lam_map, p_map, shares_map, city_fb, _ = _build_poisson_baselines(window_days=BASELINE_WINDOW)
    lam_city, p_city, shares_city = city_fb
    exp_vals = []
    top_cols = {f"top{i}_{k}": [] for i in (1,2,3) for k in ("category","prob","expected")}

    # Top-3 (multiclass) — saatlik join
    tops = _load_topk_categories(out[key].drop_duplicates(), CRIME_DIR)
    tops = _ensure_date_hour(tops)
    out = out.merge(tops, on=key, how="left")

    # expected_count ve top boşsa doldur
    for idx, row in out.iterrows():
        g  = row.get("GEOID","")
        hr = int(row.get("hour_range", 0))
        p_pred = float(np.clip(row.get("risk_score", 0.0), 0.0, 1.0))

        lam_base = lam_map.get((g,hr), lam_city.get(hr, 0.0))
        p_base   = p_map.get((g,hr), p_city.get(hr, 0.0))
        EPS = 1e-6
        if p_base <= EPS and lam_base <= 0:
            exp_vals.append(p_pred)  # fallback
        else:
            adj = (p_pred / max(p_base, EPS))
            adj = max(0.2, min(3.0, adj))
            lam_h = max(0.0, lam_base * adj)
            exp_vals.append(lam_h)

        # Eğer tops merge boş bıraktıysa default 0/"" doldur
        for i in (1,2,3):
            ccol = f"top{i}_category"; pcol = f"top{i}_prob"; ecol = f"top{i}_expected"
            if pd.isna(row.get(ccol, np.nan)):
                out.at[idx, ccol] = ""
            if pd.isna(row.get(pcol, np.nan)):
                out.at[idx, pcol] = 0.0
            if pd.isna(row.get(ecol, np.nan)):
                out.at[idx, ecol] = float(out.at[idx, pcol])

    out["expected_count"] = exp_vals

    # REQ_COLS sırala + eksik varsa doldur
    for c in REQ_COLS:
        if c not in out.columns:
            out[c] = "" if c.endswith("_category") else (0 if c in ("risk_decile",) else 0.0)
    out = out[REQ_COLS].copy()
    return out

def build_daily_from_hourly(hourly_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Günlük (≤365 gün) tabloyu üretir; REQ_COLS döner; hour_range boş ('')."""
    h = hourly_df.copy()

    # pencere ≤365 gün
    dt = pd.to_datetime(h["date"], errors="coerce")
    last = dt.max()
    if pd.notna(last):
        start = (last.normalize() - pd.Timedelta(days=DAILY_WINDOW_DAYS-1)).date().isoformat()
        h = h[dt.dt.date.astype("string") >= start].copy()

    # Günlük risk_score: 1 - ∏(1 - p_h)  (aynı GEOID, date için)
    h["_one_minus_p"] = 1.0 - pd.to_numeric(h["risk_score"], errors="coerce").clip(0,1).fillna(0.0)
    daily = (
        h.groupby(["GEOID","date"], as_index=False)
         .agg(
             risk_score=(" _one_minus_p ".replace(" ",""), lambda s: 1.0 - float(np.prod(s.fillna(1.0).values))),
             expected_count=("expected_count","sum")
         )
    )

    # Günlük Top-3 (beklenen toplamına göre)
    stack = pd.concat([
        h[["GEOID","date","top1_category","top1_expected"]].rename(columns={"top1_category":"cat","top1_expected":"lam"}),
        h[["GEOID","date","top2_category","top2_expected"]].rename(columns={"top2_category":"cat","top2_expected":"lam"}),
        h[["GEOID","date","top3_category","top3_expected"]].rename(columns={"top3_category":"cat","top3_expected":"lam"}),
    ], ignore_index=True)
    stack = stack[stack["cat"].astype(str) != ""]
    sumlam = stack.groupby(["GEOID","date","cat"], as_index=False)["lam"].sum()

    # En yüksek 3 kategori
    tmp = sumlam.copy()
    tmp["rank"] = tmp.groupby(["GEOID","date"])["lam"].rank(method="first", ascending=False)
    tmp = tmp[tmp["rank"] <= 3].copy()
    tmp["rank"] = tmp["rank"].astype(int)
    tmp["prob"] = 1.0 - np.exp(-tmp["lam"])

    cat_w  = tmp.pivot_table(index=["GEOID","date"], columns="rank", values="cat",  aggfunc="first")
    lam_w  = tmp.pivot_table(index=["GEOID","date"], columns="rank", values="lam",  aggfunc="first")
    prob_w = tmp.pivot_table(index=["GEOID","date"], columns="rank", values="prob", aggfunc="first")

    cat_w  = cat_w.rename(columns={1:"top1_category", 2:"top2_category", 3:"top3_category"})
    lam_w  = lam_w.rename(columns={1:"top1_expected", 2:"top2_expected", 3:"top3_expected"})
    prob_w = prob_w.rename(columns={1:"top1_prob",     2:"top2_prob",     3:"top3_prob"})

    daily = daily.merge(cat_w,  on=["GEOID","date"], how="left")
    daily = daily.merge(lam_w,  on=["GEOID","date"], how="left")
    daily = daily.merge(prob_w, on=["GEOID","date"], how="left")

    # risk_level & decile
    q80 = daily["risk_score"].quantile(0.8)
    q90 = daily["risk_score"].quantile(0.9)
    daily["risk_decile"] = _quantile_deciles(pd.to_numeric(daily["risk_score"], errors="coerce").fillna(0.0))
    daily["risk_level"]  = daily["risk_score"].apply(lambda x: _risk_level_from_score(x, q80, q90, threshold))

    # hour_range = "" (başlık aynı kalsın)
    daily["hour_range"] = ""

    # REQ_COLS sırala + eksikleri doldur
    for c in REQ_COLS:
        if c not in daily.columns:
            daily[c] = "" if c.endswith("_category") or c=="hour_range" else (0 if c in ("risk_decile",) else 0.0)
    daily = daily[REQ_COLS].copy()
    return daily

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True, help="Özellik/kimlik DF (GEOID,date,hour_range içermeli/çıkarılabilir).")
    ap.add_argument("--proba", default=None, help="Harici proba (csv tek kolon / npy).")
    ap.add_argument("--proba-from-risk", default=None, help="risk_hourly_grid_full_labeled.csv yolu (join ile risk_score).")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    # DF yükle & normalize
    df = pd.read_csv(args.df, low_memory=False, dtype={"GEOID": str})
    df = _ensure_date_hour(df)

    # PROBA çöz
    proba = None
    if args.proba:
        proba = _read_proba_path(args.proba)
    if proba is None:
        proba = _risk_from_df_or_riskfile(df, args.proba_from_risk)
    if proba is None:
        print("❌ PROBA/risk_score bulunamadı (DF veya risk dosyası).", file=sys.stderr)
        sys.exit(2)

    # Saatlik tablo (≤7 gün)
    hourly = build_hourly(df, proba, threshold=args.threshold)
    hourly_path = CRIME_DIR / "risk_hourly_grid_full_labeled.csv"
    hourly.to_csv(hourly_path, index=False)
    try: hourly.to_parquet(hourly_path.with_suffix(".parquet"), index=False)
    except Exception: pass
    print(f"✅ Hourly yazıldı → {hourly_path} (rows={len(hourly)})")

    # Günlük tablo (≤365 gün)
    daily = build_daily_from_hourly(hourly, threshold=args.threshold)
    daily_path = CRIME_DIR / "risk_daily_grid_full_labeled.csv"
    daily.to_csv(daily_path, index=False)
    try: daily.to_parquet(daily_path.with_suffix(".parquet"), index=False)
    except Exception: pass
    print(f"✅ Daily  yazıldı → {daily_path} (rows={len(daily)})")

if __name__ == "__main__":
    main()
