#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_exports_fr.py — İki çıktı üretir (başlıkları AYNI):
  1) crime_prediction_data/risk_hourly_grid_full_labeled.csv   (≤7 gün, saatlik; hour_range="HH-HH")
  2) crime_prediction_data/risk_daily_grid_full_labeled.csv    (≤365 gün, günlük; hour_range="")
Kolonlar:
  GEOID,date,hour_range,risk_score,risk_level,risk_decile,expected_count,
  top1_category,top1_prob,top1_expected,
  top2_category,top2_prob,top2_expected,
  top3_category,top3_prob,top3_expected
"""

import os, sys, re, math, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ──────────────────────────────────────────────────────────────
# Ortam / Sabitler
# ──────────────────────────────────────────────────────────────
CRIME_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
CRIME_DIR.mkdir(parents=True, exist_ok=True)

REQ_COLS = [
    "GEOID","date","hour_range","risk_score","risk_level","risk_decile","expected_count",
    "top1_category","top1_prob","top1_expected",
    "top2_category","top2_prob","top2_expected",
    "top3_category","top3_prob","top3_expected",
]

HOUR_WINDOW_DAYS   = int(os.getenv("RISK_HOURLY_WINDOW_DAYS", "7"))       # Saatlik (≤7 gün)
DAILY_WINDOW_DAYS  = int(os.getenv("RISK_DAILY_WINDOW_DAYS", "365"))      # Günlük (≤365 gün)
BASELINE_WINDOW    = int(os.getenv("RISK_BASELINE_WINDOW_DAYS", "30"))    # Beklenen için Poisson pencere
GEOID_LEN          = int(os.getenv("GEOID_LEN","11"))

# ──────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────
def _hr_slot_from_hour(h: int) -> str:
    h = int(pd.to_numeric(h, errors="coerce")) if pd.notna(h) else 0
    a = (h // 3) * 3
    b = (a + 3) % 24
    return f"{a:02d}-{b:02d}"

def _now_sf_slot() -> str:
    now = datetime.now(ZoneInfo("America/Los_Angeles")) if ZoneInfo else datetime.utcnow()
    return _hr_slot_from_hour(now.hour)

def _now_sf_date_str() -> str:
    now = datetime.now(ZoneInfo("America/Los_Angeles")) if ZoneInfo else datetime.utcnow()
    return now.date().isoformat()

def _normalize_hour_range_series(s: pd.Series) -> pd.Series:
    """Karışık saat/hh-hh değerlerini 'HH-HH' slotuna dönüştürür; metin 'HH-HH' ise korur."""
    s = s.astype(str)
    mask_slot = s.str.fullmatch(r"\s*\d{1,2}\s*-\s*\d{1,2}\s*")
    mask_num  = ~mask_slot & s.str.fullmatch(r"\s*\d{1,2}\s*")

    out = pd.Series(index=s.index, dtype="string")
    # metin slot → boşlukları temizle, '12 - 15' → '12-15'
    out[mask_slot] = s[mask_slot].str.replace(r"\s*", "", regex=True)
    # tek sayı → slota çevir
    out[mask_num]  = s[mask_num].apply(lambda x: _hr_slot_from_hour(int(x)))
    # geriye kalanlar boş kalır (dışarıda dolduracağız)
    return out

def _ensure_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    GEOID, date(YYYY-MM-DD), hour_range('HH-HH') üretir/sağlamlaştırır.
    - 'HH-HH' metinleri KORUNUR.
    - Sayı → 'HH-HH'
    - Eksik/bozuksa: post-processing ANINDAKİ SF slotu yazılır.
    - date yoksa: DF'teki zamandan türet; yoksa sf_crime.csv max(date); o da yoksa SF bugünü.
    """
    d = df.copy()
    low = {c.lower(): c for c in d.columns}

    # GEOID normalize
    if "GEOID" in d.columns:
        d["GEOID"] = (
            d["GEOID"].astype(str)
              .str.extract(r"(\d+)", expand=False)
              .fillna("")
              .str.zfill(GEOID_LEN)
        )

    # date türet
    got_date = False
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype("string")
        got_date = True
    else:
        for c in ("datetime","incident_datetime","date_time","dt","event_time"):
            if c in low:
                dc = pd.to_datetime(d[low[c]], errors="coerce", utc=False)
                if dc.notna().any():
                    d["date"] = dc.dt.date.astype("string")
                    got_date = True
                    break
    if (not got_date) or d["date"].isna().all() or (d["date"].astype(str)=="").all():
        # sf_crime.csv max(date)
        maxd = None
        p = CRIME_DIR / "sf_crime.csv"
        if p.is_file():
            try:
                tmp = pd.read_csv(p, low_memory=False)
                cand_cols = [c for c in ("date","datetime","incident_datetime") if c in tmp.columns]
                if cand_cols:
                    td = pd.to_datetime(tmp[cand_cols[0]], errors="coerce")
                    maxd = td.max().date() if td.notna().any() else None
            except Exception:
                pass
        d["date"] = str(maxd) if maxd else _now_sf_date_str()

    # hour_range → 'HH-HH'
    if "hour_range" in d.columns:
        hr = _normalize_hour_range_series(d["hour_range"])
        remain = hr.isna()
        # event_hour/hour vb.
        if remain.any():
            for c in ("event_hour","hour","hour_of_day"):
                if c in low:
                    tmp = pd.to_numeric(d.loc[remain, low[c]], errors="coerce")
                    has = tmp.notna()
                    hr.loc[remain & has] = tmp[has].astype(int).apply(_hr_slot_from_hour)
                    remain = hr.isna()
                    if not remain.any(): break
        # datetime'dan
        if remain.any():
            for c in ("datetime","incident_datetime","date_time"):
                if c in low:
                    dt = pd.to_datetime(d.loc[remain, low[c]], errors="coerce", utc=False)
                    has = dt.notna()
                    hr.loc[remain & has] = dt[has].dt.hour.astype(int).apply(_hr_slot_from_hour)
                    remain = hr.isna()
                    if not remain.any(): break
        # hâlâ boşsa SF anı
        if remain.any():
            hr.loc[remain] = _now_sf_slot()
        d["hour_range"] = hr.astype("string")
    else:
        # kolon yoksa aynı mantık
        filled = None
        for c in ("event_hour","hour","hour_of_day"):
            if c in low:
                tmp = pd.to_numeric(d[low[c]], errors="coerce")
                if tmp.notna().any():
                    filled = tmp.astype(int).apply(_hr_slot_from_hour)
                    break
        if filled is None:
            ref = None
            for c in ("datetime","incident_datetime","date_time"):
                if c in low:
                    ref = pd.to_datetime(d[low[c]], errors="coerce", utc=False)
                    break
            if ref is not None and ref.notna().any():
                filled = ref.dt.hour.astype(int).apply(_hr_slot_from_hour)
            else:
                filled = pd.Series([_now_sf_slot()]*len(d), index=d.index)
        d["hour_range"] = filled.astype("string")

    return d

def _quantile_deciles(x: pd.Series) -> pd.Series:
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

# ── Baseline için son 30 gün: hour_range anahtarı 'HH-HH'
def _load_recent_crime_for_baseline(window_days=30) -> pd.DataFrame | None:
    p = CRIME_DIR / "sf_crime.csv"
    if not p.is_file(): return None
    dfc = pd.read_csv(p, low_memory=False, dtype={"GEOID": str})

    # tarih
    if "date" in dfc.columns:
        dt = pd.to_datetime(dfc["date"], errors="coerce")
    elif "datetime" in dfc.columns:
        dt = pd.to_datetime(dfc["datetime"], errors="coerce")
    else:
        return None

    dfc = dfc.assign(_dt=dt).dropna(subset=["_dt"]).copy()
    latest = dfc["_dt"].max()
    start = latest.normalize() - pd.Timedelta(days=window_days-1)
    dfc = dfc[(dfc["_dt"] >= start) & (dfc["_dt"] <= latest)].copy()

    # hour_range → HH-HH
    if "hour_range" in dfc.columns:
        hr = _normalize_hour_range_series(dfc["hour_range"])
        # tek sayı vb. için tamamla
        remain = hr.isna()
        if remain.any():
            if "event_hour" in dfc.columns:
                tmp = pd.to_numeric(dfc.loc[remain, "event_hour"], errors="coerce")
                hr.loc[remain & tmp.notna()] = tmp[tmp.notna()].astype(int).apply(_hr_slot_from_hour)
        dfc["hour_range"] = hr.fillna(_now_sf_slot()).astype("string")
    else:
        src = None
        if "event_hour" in dfc.columns:
            src = pd.to_numeric(dfc["event_hour"], errors="coerce")
            dfc["hour_range"] = src.fillna(0).astype(int).apply(_hr_slot_from_hour)
        else:
            dfc["hour_range"] = dfc["_dt"].dt.hour.astype(int).apply(_hr_slot_from_hour)

    if "category" not in dfc.columns:
        dfc["category"] = "all"
    dfc["date"] = dfc["_dt"].dt.date.astype("string")
    return dfc.drop(columns=["_dt"])

def _build_poisson_baselines(window_days=30):
    dfc = _load_recent_crime_for_baseline(window_days=window_days)
    if dfc is None or len(dfc)==0:
        return {}, {}, {}, ({}, {}, {}), 1

    dts = pd.to_datetime(dfc["date"], errors="coerce")
    ndays = max(1, (dts.max().normalize() - dts.min().normalize()).days + 1)

    key = ["GEOID","hour_range"]  # hour_range: 'HH-HH'
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

    hr_tot = dfc.groupby(["hour_range"]).size().rename("events").reset_index()
    # şehir bazında saat dilimi başına ortalama yoğunluk
    n_geo = max(1, dfc["GEOID"].nunique())
    hr_tot["lambda_hr_city"] = hr_tot["events"] / (ndays * n_geo)
    hr_tot["p_hr_city"] = 1.0 - np.exp(-hr_tot["lambda_hr_city"])
    lam_city = {r["hour_range"]: float(r["lambda_hr_city"]) for _, r in hr_tot.iterrows()}
    p_city   = {r["hour_range"]: float(r["p_hr_city"]) for _, r in hr_tot.iterrows()}

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
    """Multiclass proba kaynaklarından Top-3'ü 'HH-HH' anahtarıyla getirir."""
    key_df = key_df.copy()
    key_df["hour_range"] = _normalize_hour_range_series(key_df["hour_range"]).fillna(_now_sf_slot())

    # A) long
    for name in ("proba_multiclass.parquet","proba_multiclass.csv"):
        p = _pick_first(base_dir/name, Path(".")/name)
        if p:
            try:
                t = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
                need = {"GEOID","date","hour_range","category","prob"}
                if need.issubset(t.columns):
                    t = t.copy()
                    t["hour_range"] = _normalize_hour_range_series(t["hour_range"]).fillna(_now_sf_slot())
                    t["prob"] = pd.to_numeric(t["prob"], errors="coerce").fillna(0.0)
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
                            row[f"top{i+1}_expected"] = p_
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
                t["hour_range"] = _normalize_hour_range_series(t["hour_range"]).fillna(_now_sf_slot())
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

    # C) kaynak yoksa: boş top alanlarıyla anahtar setini döndür
    k = key_df[["GEOID","date","hour_range"]].drop_duplicates().copy()
    for i in (1,2,3):
        k[f"top{i}_category"] = ""
        k[f"top{i}_prob"] = 0.0
        k[f"top{i}_expected"] = 0.0
    return k

# ──────────────────────────────────────────────────────────────
# Çekirdek üretimler
# ──────────────────────────────────────────────────────────────
def build_hourly(df_in: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    df = _ensure_date_hour(df_in)

    # ≤7 gün filtresi
    dt = pd.to_datetime(df["date"], errors="coerce")
    last = dt.max()
    if pd.notna(last):
        start = (last.normalize() - pd.Timedelta(days=HOUR_WINDOW_DAYS-1)).date().isoformat()
        df = df[dt.dt.date.astype("string") >= start].copy()

    key = ["GEOID","date","hour_range"]  # hour_range: 'HH-HH'
    out = df.copy()
    out["risk_score"] = np.clip(np.asarray(proba).astype(float), 0.0, 1.0)

    # decile & level
    q80 = out["risk_score"].quantile(0.8)
    q90 = out["risk_score"].quantile(0.9)
    out["risk_decile"] = _quantile_deciles(pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0))
    out["risk_level"]  = out["risk_score"].apply(lambda x: _risk_level_from_score(x, q80, q90, threshold))

    # Poisson baseline (HH-HH anahtarıyla)
    lam_map, p_map, shares_map, city_fb, _ = _build_poisson_baselines(window_days=BASELINE_WINDOW)
    lam_city, p_city, shares_city = city_fb

    # Top-3 (multiclass)
    tops = _load_topk_categories(out[key].drop_duplicates(), CRIME_DIR)
    tops = _ensure_date_hour(tops)
    out = out.merge(tops, on=key, how="left")

    # expected + boş top alanlarını doldur
    EPS = 1e-6
    exp_vals = []
    for idx, row in out.iterrows():
        g  = row.get("GEOID","")
        hr = row.get("hour_range", _now_sf_slot())  # 'HH-HH'
        p_pred = float(np.clip(row.get("risk_score", 0.0), 0.0, 1.0))

        lam_base = lam_map.get((g,hr), lam_city.get(hr, 0.0))
        p_base   = p_map.get((g,hr), p_city.get(hr, 0.0))
        if (p_base <= EPS and lam_base <= 0):
            exp_vals.append(p_pred)  # minimal fallback
        else:
            adj = p_pred / max(p_base, EPS)
            adj = max(0.2, min(3.0, adj))
            lam_h = max(0.0, lam_base * adj)
            exp_vals.append(lam_h)

        for i in (1,2,3):
            ccol = f"top{i}_category"; pcol = f"top{i}_prob"; ecol = f"top{i}_expected"
            if pd.isna(row.get(ccol, np.nan)): out.at[idx, ccol] = ""
            if pd.isna(row.get(pcol, np.nan)): out.at[idx, pcol] = 0.0
            if pd.isna(row.get(ecol, np.nan)): out.at[idx, ecol] = float(out.at[idx, pcol])

    out["expected_count"] = exp_vals

    # REQ_COLS sırala + eksikleri tamamla
    for c in REQ_COLS:
        if c not in out.columns:
            out[c] = "" if c.endswith("_category") else (0 if c in ("risk_decile",) else 0.0)
    out = out[REQ_COLS].copy()
    return out

def build_daily_from_hourly(hourly_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    h = hourly_df.copy()

    # ≤365 gün filtresi
    dt = pd.to_datetime(h["date"], errors="coerce")
    last = dt.max()
    if pd.notna(last):
        start = (last.normalize() - pd.Timedelta(days=DAILY_WINDOW_DAYS-1)).date().isoformat()
        h = h[dt.dt.date.astype("string") >= start].copy()

    # Günlük risk_score = 1 - ∏(1 - p_h)
    p = pd.to_numeric(h["risk_score"], errors="coerce").clip(0,1).fillna(0.0)
    h["_one_minus_p"] = 1.0 - p
    daily = (
        h.groupby(["GEOID","date"], as_index=False)
         .agg(
             risk_score=("_one_minus_p", lambda s: 1.0 - float(np.prod(s.fillna(1.0).values))),
             expected_count=("expected_count","sum")
         )
    )

    # Günlük Top-3 (λ toplamına göre)
    stack = pd.concat([
        h[["GEOID","date","top1_category","top1_expected"]].rename(columns={"top1_category":"cat","top1_expected":"lam"}),
        h[["GEOID","date","top2_category","top2_expected"]].rename(columns={"top2_category":"cat","top2_expected":"lam"}),
        h[["GEOID","date","top3_category","top3_expected"]].rename(columns={"top3_category":"cat","top3_expected":"lam"}),
    ], ignore_index=True)
    stack = stack[stack["cat"].astype(str) != ""]
    sumlam = stack.groupby(["GEOID","date","cat"], as_index=False)["lam"].sum()

    if len(sumlam):
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
    else:
        for i in (1,2,3):
            daily[f"top{i}_category"] = ""
            daily[f"top{i}_prob"]     = 0.0
            daily[f"top{i}_expected"] = 0.0

    # risk_level & decile
    q80 = daily["risk_score"].quantile(0.8) if len(daily) else 0.8
    q90 = daily["risk_score"].quantile(0.9) if len(daily) else 0.9
    daily["risk_decile"] = _quantile_deciles(pd.to_numeric(daily["risk_score"], errors="coerce").fillna(0.0))
    daily["risk_level"]  = daily["risk_score"].apply(lambda x: _risk_level_from_score(x, q80, q90, threshold))

    # hour_range = "" (başlık sabit)
    daily["hour_range"] = ""

    # REQ_COLS sırala + eksikleri tamamla
    for c in REQ_COLS:
        if c not in daily.columns:
            daily[c] = "" if c.endswith("_category") or c=="hour_range" else (0 if c in ("risk_decile",) else 0.0)
    daily = daily[REQ_COLS].copy()
    return daily

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True, help="Özellik/kimlik DF (GEOID,date/hour_range olabilir).")
    ap.add_argument("--proba", default=None, help="Harici proba (csv tek kolon / npy).")
    ap.add_argument("--proba-from-risk", default=None, help="risk_hourly_grid_full_labeled.csv (join ile risk_score).")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.df, low_memory=False, dtype={"GEOID": str})
    df = _ensure_date_hour(df)

    # PROBA
    proba = None
    if args.proba:
        proba = _read_proba_path(args.proba)
    if proba is None:
        proba = _risk_from_df_or_riskfile(df, args.proba_from_risk)
    if proba is None:
        print("❌ PROBA/risk_score bulunamadı (DF veya risk dosyası).", file=sys.stderr)
        sys.exit(2)

    # Saatlik
    hourly = build_hourly(df, proba, threshold=args.threshold)
    hourly_path = CRIME_DIR / "risk_hourly_grid_full_labeled.csv"
    hourly.to_csv(hourly_path, index=False)
    try: hourly.to_parquet(hourly_path.with_suffix(".parquet"), index=False)
    except Exception: pass
    print(f"✅ Hourly yazıldı → {hourly_path} (rows={len(hourly)})")

    # Günlük
    daily = build_daily_from_hourly(hourly, threshold=args.threshold)
    daily_path = CRIME_DIR / "risk_daily_grid_full_labeled.csv"
    daily.to_csv(daily_path, index=False)
    try: daily.to_parquet(daily_path.with_suffix(".parquet"), index=False)
    except Exception: pass
    print(f"✅ Daily  yazıldı → {daily_path} (rows={len(daily)})")

if __name__ == "__main__":
    main()
