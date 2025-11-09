# risk_exports.py
# Saatlik + günlük risk exportları + opsiyonel Top-3 suç türü tablosu
import os, math, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

CRIME_DIR = os.getenv("CRIME_DATA_DIR", os.getcwd())

# ------------ yardımcılar ------------
def _normalize_hour_range(hr):
    if pd.isna(hr): return "00-03"
    s = str(hr)
    m = pd.Series([s]).str.extract(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
    if not m.notna().all(axis=1).iloc[0]:
        try: h = int(float(s))
        except: h = 0
        a, b = int((h//3)*3), int(((h//3)*3 + 3) % 24)
        return f"{a:02d}-{b:02d}"
    a = int(float(m.iloc[0,0])); b = int(float(m.iloc[0,1]))
    return f"{a:02d}-{b:02d}"

def _ensure_date_hour_on_df(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date.astype("string")
    else:
        from zoneinfo import ZoneInfo
        df["date"] = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    if "hour_range" not in df.columns:
        if "event_hour" in df.columns:
            h = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int)
        elif "datetime" in df.columns:
            h = pd.to_datetime(df["datetime"], errors="coerce").dt.hour.fillna(0).astype(int)
        else:
            h = pd.Series(np.zeros(len(df), dtype=int))
        a = (h // 3) * 3; b = (a + 3) % 24
        df["hour_range"] = a.map("{:02d}".format) + "-" + pd.Series(b).map("{:02d}".format)
    else:
        df["hour_range"] = df["hour_range"].apply(_normalize_hour_range)
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    return df

def _load_recent_crime(window_days=30):
    raw_path = os.path.join(CRIME_DIR, "sf_crime.csv")
    if not os.path.exists(raw_path): return None
    dfc = pd.read_csv(raw_path, low_memory=False, dtype={"GEOID": str})
    # tarih sütunu
    if "date" in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc["date"], errors="coerce")
    elif "datetime" in dfc.columns:
        dfc["date"] = pd.to_datetime(dfc["datetime"], errors="coerce")
    else:
        return None
    dfc = dfc.dropna(subset=["date"])
    latest = dfc["date"].max()
    if pd.isna(latest): return None
    start = latest.normalize() - pd.Timedelta(days=window_days-1)
    dfc = dfc[(dfc["date"] >= start) & (dfc["date"] <= latest)].copy()
    # hour_range
    if "hour_range" not in dfc.columns:
        if "event_hour" in dfc.columns:
            h = pd.to_numeric(dfc["event_hour"], errors="coerce").fillna(0).astype(int)
        else:
            h = dfc["date"].dt.hour.fillna(0).astype(int)
        a = (h // 3) * 3; b = (a + 3) % 24
        dfc["hour_range"] = a.map("{:02d}".format) + "-" + pd.Series(b).map("{:02d}".format)
    else:
        dfc["hour_range"] = dfc["hour_range"].apply(_normalize_hour_range)
    # kategori yoksa "all"
    if "category" not in dfc.columns:
        dfc["category"] = "all"
    return dfc

def _compute_baselines(window_days=30):
    dfc = _load_recent_crime(window_days=window_days)
    if dfc is None or len(dfc)==0:
        return {}, {}, {}, ({}, {}, {}), 1
    ndays = max(1, (dfc["date"].max().normalize() - dfc["date"].min().normalize()).days + 1)

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

    hr_tot = dfc.groupby(["hour_range"]).size().rename("events").reset_index()
    hr_tot["lambda_hr_city"] = hr_tot["events"] / (ndays * max(1, dfc["GEOID"].nunique()))
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

def _clip(v, lo, hi): return max(lo, min(hi, v))

# ------------ ana API ------------
def export_risk_tables(df, y, proba, threshold, out_prefix=""):
    """
    Saatlik + günlük risk çıktıları üretir:
      - risk_hourly{out_prefix}.csv
      - risk_daily{out_prefix}.csv
      - patrol_recs{out_prefix}.csv (aynı gün)
    """
    df = _ensure_date_hour_on_df(df)
    cols = [c for c in ["GEOID","date","hour_range"] if c in df.columns]
    risk = df[cols].copy()
    risk["risk_score"] = np.asarray(proba).astype(float)
    risk["hour_range"] = risk["hour_range"].apply(_normalize_hour_range)

    if set(["GEOID","date","hour_range"]).issubset(risk.columns):
        risk = risk.drop_duplicates(["GEOID","date","hour_range"], keep="last")

    q80 = risk["risk_score"].quantile(0.8); q90 = risk["risk_score"].quantile(0.9)
    def _level(x):
        if x >= max(threshold, q90): return "critical"
        if x >= max(threshold*0.8, q80): return "high"
        if x >= 0.5*threshold: return "medium"
        return "low"
    risk["risk_level"]  = risk["risk_score"].apply(_level)

    _r = pd.to_numeric(risk["risk_score"], errors="coerce").fillna(0.0)
    _ranks = _r.rank(method="first", pct=True)
    if _ranks.nunique() < 2:
        risk["risk_decile"] = 1
    else:
        risk["risk_decile"] = pd.qcut(_ranks, q=10, labels=False, duplicates="drop") + 1
    risk["risk_decile"] = pd.to_numeric(risk["risk_decile"], errors="coerce").fillna(1).astype(int)

    WINDOW_DAYS = int(os.getenv("TOP3_WINDOW_DAYS", "30"))
    lam_map, p_map, shares_map, city_fb, _ = _compute_baselines(window_days=WINDOW_DAYS)
    lam_city, p_city, shares_city = city_fb
    EPS = 1e-6

    exp_vals = []
    tops = {f"top{i}_{k}": [] for i in [1,2,3] for k in ["category","prob","expected"]}

    for _, row in risk.iterrows():
        g = row.get("GEOID",""); hr = row.get("hour_range","00-03")
        p_pred = float(np.clip(row["risk_score"], 0.001, 0.999))
        lam_base = lam_map.get((g,hr), lam_city.get(hr, 0.0))
        p_base   = p_map.get((g,hr), p_city.get(hr, 0.0))
        shares   = shares_map.get((g,hr), shares_city.get(hr, {}))
        if p_base < EPS and lam_base <= 0:
            lam_base = lam_city.get(hr, 0.0); p_base = p_city.get(hr, 0.0)

        adj  = (p_pred / max(p_base, EPS)) if p_base > 0 else (p_pred / max(p_city.get(hr, EPS), EPS))
        adj  = _clip(adj, 0.2, 3.0)   
        lamh = max(0.0, lam_base * adj)
        exp_vals.append(lamh)

        if not shares: shares = {"all": 1.0}
        cat_items = sorted(shares.items(), key=lambda x: x[1], reverse=True)[:3]
        pad = 3 - len(cat_items)
        cat_items += [("", 0.0)] * max(0, pad)

        for i, (cat, sh) in enumerate(cat_items, start=1):
            lam_k = lamh * float(sh)
            p_k   = 1.0 - math.exp(-lam_k)
            tops[f"top{i}_category"].append(cat)
            tops[f"top{i}_prob"].append(p_k)
            tops[f"top{i}_expected"].append(lam_k)

    risk["expected_count"] = exp_vals
    for k, v in tops.items(): risk[k] = v
    
    risk["date"] = pd.to_datetime(risk.get("date"), errors="coerce").dt.date.astype("string")
    hourly_path = os.path.join(CRIME_DIR, f"risk_hourly{out_prefix}.csv")
    risk.to_csv(hourly_path, index=False)

    # Günlük birleşik olasılık
    p = pd.to_numeric(risk["risk_score"], errors="coerce").clip(0, 1).fillna(0.0)
    risk["_one_minus_p"] = 1.0 - p
    
    daily = (
        risk.groupby(["GEOID","date"], as_index=False)
            .agg(
                risk_score_day=("_one_minus_p", lambda s: 1.0 - float(np.prod(s.fillna(1.0).values))),
                expected_count_day=("expected_count","sum")
            )
    )
    
    q80d = daily["risk_score_day"].quantile(0.8)
    q90d = daily["risk_score_day"].quantile(0.9)
    def _level_d(x):
        if x >= max(threshold, q90d): return "critical"
        if x >= max(threshold*0.8, q80d): return "high"
        if x >= 0.5*threshold: return "medium"
        return "low"
    daily["risk_level_day"]  = daily["risk_score_day"].apply(_level_d)
    
    _vals  = pd.to_numeric(daily["risk_score_day"], errors="coerce").fillna(0.0)
    _ranks = _vals.rank(method="first", pct=True)
    if _ranks.nunique() < 2:
        daily["risk_decile_day"] = 1
    else:
        daily["risk_decile_day"] = pd.qcut(_ranks, q=10, labels=False, duplicates="drop") + 1
    daily["risk_decile_day"] = pd.to_numeric(daily["risk_decile_day"], errors="coerce").fillna(1).astype(int)

    # Günlük top-3 (λ toplamına göre)
    _stack = pd.concat([
        risk[["GEOID","date","top1_category","top1_expected"]].rename(columns={"top1_category":"cat","top1_expected":"lam"}),
        risk[["GEOID","date","top2_category","top2_expected"]].rename(columns={"top2_category":"cat","top2_expected":"lam"}),
        risk[["GEOID","date","top3_category","top3_expected"]].rename(columns={"top3_category":"cat","top3_expected":"lam"}),
    ], ignore_index=True)
    
    _stack = _stack[_stack["cat"].astype(str) != ""]
    _sumlam = _stack.groupby(["GEOID","date","cat"], as_index=False)["lam"].sum()
    
    def _expand_top3(grp):
        g = grp.sort_values("lam", ascending=False).head(3).reset_index(drop=True)
        out = {}
        for j in range(3):
            if j < len(g):
                lamj = float(g.loc[j, "lam"])
                catj = str(g.loc[j, "cat"])
                pj   = 1.0 - math.exp(-lamj)
            else:
                lamj, catj, pj = 0.0, "", 0.0
            out[f"top{j+1}_category_day"] = catj
            out[f"top{j+1}_expected_day"] = lamj
            out[f"top{j+1}_prob_day"]     = pj
        return pd.Series(out)

    tmp = _sumlam.copy()
    tmp["rank"] = tmp.groupby(["GEOID","date"])["lam"].rank(method="first", ascending=False)
    tmp = tmp[tmp["rank"] <= 3].copy()
    tmp["rank"] = tmp["rank"].astype(int)
    tmp["prob"] = 1.0 - np.exp(-tmp["lam"])
    
    cat_w  = tmp.pivot_table(index=["GEOID","date"], columns="rank", values="cat",  aggfunc="first")
    lam_w  = tmp.pivot_table(index=["GEOID","date"], columns="rank", values="lam",  aggfunc="first")
    prob_w = tmp.pivot_table(index=["GEOID","date"], columns="rank", values="prob", aggfunc="first")
    
    cat_w  = cat_w.rename(columns={1:"top1_category_day", 2:"top2_category_day", 3:"top3_category_day"})
    lam_w  = lam_w.rename(columns={1:"top1_expected_day", 2:"top2_expected_day", 3:"top3_expected_day"})
    prob_w = prob_w.rename(columns={1:"top1_prob_day",     2:"top2_prob_day",     3:"top3_prob_day"})
    
    daily_extra = pd.concat([cat_w, lam_w, prob_w], axis=1).reset_index()
    daily = daily.merge(daily_extra, on=["GEOID","date"], how="left")

    daily_path = os.path.join(CRIME_DIR, f"risk_daily{out_prefix}.csv")
    daily.to_csv(daily_path, index=False)

    # Aynı güne Top-K devriye
    latest_date = pd.to_datetime(risk["date"], errors="coerce").max()
    if pd.notna(latest_date):
        latest_date = latest_date.date()
        day = risk[pd.to_datetime(risk["date"], errors="coerce").dt.date == latest_date].copy()
        rec_rows = []
        TOP_K = int(os.getenv("PATROL_TOP_K","50"))
        for hr in sorted(day["hour_range"].dropna().unique()):
            slot = day[day["hour_range"]==hr].sort_values("risk_score", ascending=False)
            for _, r in slot.head(TOP_K).iterrows():
                rec_rows.append({
                    "date": latest_date,
                    "hour_range": hr,
                    "GEOID": r.get("GEOID",""),
                    "risk_score": float(r["risk_score"]),
                    "risk_level": r["risk_level"],
                    "expected_count": float(r.get("expected_count",0.0)),
                    "top1_category": r.get("top1_category",""),
                    "top1_prob": float(r.get("top1_prob",0.0)),
                })
        pd.DataFrame(rec_rows).to_csv(os.path.join(CRIME_DIR, f"patrol_recs{out_prefix}.csv"), index=False)

    print(f"Hourly risk  → {hourly_path}")
    print(f"Daily  risk  → {daily_path}")
    return hourly_path, os.path.join(CRIME_DIR, f"patrol_recs{out_prefix}.csv")

# ------------ Opsiyonel Top-3 suç türü ------------
def optional_top_crime_types(window_days: int = 365, out_name: str = "risk_types_top3.csv"):
    """
    sf_crime.csv mevcutsa, son `window_days` penceresinde:
      - GENEL (overall) Top-3 suç türü
      - 3-saatlik dilimler (hour_range) için Top-3 suç türü
    tablolarını üretir ve CRIME_DIR/out_name olarak kaydeder.
    Yoksa None döner.
    """
    try:
        df = _load_recent_crime(window_days=window_days)
        if df is None or len(df) == 0 or "category" not in df.columns:
            return None

        # overall
        over = (df.groupby("category").size()
                  .rename("count").reset_index()
                  .sort_values("count", ascending=False)
                  .head(3))
        over["rank"] = np.arange(1, len(over)+1)
        over["scope"] = "overall"
        over["hour_range"] = ""

        # hour_range bazında
        df["hour_range"] = df["hour_range"].apply(_normalize_hour_range)
        by_hr = (df.groupby(["hour_range","category"]).size()
                   .rename("count").reset_index())
        # her hour_range için top3
        rows = []
        for hr, grp in by_hr.groupby("hour_range"):
            g = grp.sort_values("count", ascending=False).head(3).copy()
            g["rank"] = np.arange(1, len(g)+1)
            g["scope"] = "by_hour_range"
            g["hour_range"] = hr
            rows.append(g)
        hr_top = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["hour_range","category","count","rank","scope"])

        out = pd.concat([over, hr_top], ignore_index=True)
        # oran bilgisi
        total_over = float(df.shape[0]) if df.shape[0] > 0 else 1.0
        out["share_overall"] = out["count"] / total_over

        # meta bilgiler
        start = df["date"].min().date() if hasattr(df["date"].min(), "date") else pd.to_datetime(df["date"].min(), errors="coerce").date()
        end   = df["date"].max().date() if hasattr(df["date"].max(), "date") else pd.to_datetime(df["date"].max(), errors="coerce").date()
        out["period_start"] = str(start)
        out["period_end"]   = str(end)
        out["ndays"]        = max(1, (pd.to_datetime(str(end)) - pd.to_datetime(str(start))).days + 1)

        out_path = os.path.join(CRIME_DIR, out_name)
        out[["scope","hour_range","rank","category","count","share_overall","period_start","period_end","ndays"]].to_csv(out_path, index=False)
        print(f"Top crime types → {out_path}")
        return out_path
    except Exception as e:
        # Sessiz başarısızlık: pipeline bu fonksiyonu opsiyonel çağırıyor
        print(f"[WARN] optional_top_crime_types failed: {e}")
        return None

# ------------ CLI (opsiyonel) ------------
def _read_proba(proba_path):
    ext = str(proba_path).lower()
    if ext.endswith(".npy"):
        return np.load(proba_path)
    dfp = pd.read_csv(proba_path)
    for c in ["proba","prob","p","y_pred_proba"]:
        if c in dfp.columns: return dfp[c].values
    if dfp.shape[1]==1: return dfp.iloc[:,0].values
    raise ValueError("Proba dosyasında uygun kolon bulunamadı.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--df", required=True, help="Özellik/kimlik DF CSV (GEOID,date/hour_range içerebilir)")
    p.add_argument("--proba", required=True, help="Tahmin olasılıkları (CSV tek kolon veya .npy)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-prefix", default="")
    p.add_argument("--top3-window-days", type=int, default=30, help="Top crime types penceresi (gün)")
    args = p.parse_args()

    df = pd.read_csv(args.df, low_memory=False, dtype={"GEOID": str})
    proba = _read_proba(args.proba)
    export_risk_tables(df, y=None, proba=proba, threshold=args.threshold, out_prefix=args.out_prefix)

    risk = df[cols].copy()
    risk["risk_score"] = np.asarray(proba).astype(float)
    risk["hour_range"] = risk["hour_range"].apply(_normalize_hour_range)
    risk["risk_score"] = np.clip(risk["risk_score"], 0.001, 0.999)

    risk["date"] = pd.to_datetime(risk.get("date"), errors="coerce").dt.date
    risk = risk[~pd.isna(risk["date"])].copy()      # NaT olan satırları at
    risk["date"] = risk["date"].astype("string")

    # CLI'dan çağrılırsa top-3 hesaplamasını da deneyebilir
    optional_top_crime_types(window_days=args.top3_window_days, out_name="risk_types_top3.csv")

if __name__ == "__main__":
    main()
