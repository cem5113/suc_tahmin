#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_exports_fr.py — İki dosya yazar:
  1) crime_prediction_data/risk_hourly_grid_full_labeled.csv  (son 7 gün, saatlik)
  2) crime_prediction_data/risk_daily_grid_full_labeled.csv   (son 365 gün, günlük; hour_range boş)

İki dosyanın da şeması AYNIDIR:
GEOID,date,hour_range,risk_score,risk_level,risk_decile,expected_count,
top1_category,top1_prob,top1_expected,top2_category,top2_prob,top2_expected,
top3_category,top3_prob,top3_expected

TEMEL İLKELER
- Skorlar DF ile anahtarlı (GEOID, date, hour_range) merge edilir; sıra/uzunluk varsayımı yoktur.
- Çok-sınıflı (top3) bilgiler varsa anahtarlarla join edilir; yoksa eksik alanlar 0/"" ile doldurulur.
- Günlük dosyada hour_range kolonu boş string ("") olarak yazılır ama şema korunur.
"""

import os, sys, argparse, math, warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Ortam ve sabitler
# ---------------------------------------------------------------------
CRIME_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
CRIME_DIR.mkdir(parents=True, exist_ok=True)

REQ_COLS = [
    "GEOID","date","hour_range","risk_score","risk_level","risk_decile","expected_count",
    "top1_category","top1_prob","top1_expected",
    "top2_category","top2_prob","top2_expected",
    "top3_category","top3_prob","top3_expected",
]
KEY = ["GEOID", "date", "hour_range"]

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _zfill_geoid(s: pd.Series, geolen: int = None) -> pd.Series:
    L = int(os.getenv("GEOID_LEN", str(geolen if geolen else 11)))
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str.zfill(L)
    )

def _ensure_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    DF içinde GEOID, date (YYYY-MM-DD), hour_range (0..23) sütunlarını mümkün olan en iyi şekilde üretir.
    """
    d = df.copy()

    # GEOID
    if "GEOID" not in d.columns:
        raise ValueError("DF içinde 'GEOID' kolonu bulunmalı.")
    d["GEOID"] = _zfill_geoid(d["GEOID"])

    # date
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date.astype("string")
    else:
        # en olası tarih/saat kolonları
        cands = [c for c in ["datetime","incident_datetime","timestamp","created_at"] if c in d.columns]
        if cands:
            dt = pd.to_datetime(d[cands[0]], errors="coerce")
            d["date"] = dt.dt.date.astype("string")
        else:
            raise ValueError("DF için 'date' veya bir datetime kolonu (datetime/incident_datetime) gerekli.")

    # hour_range (saat)
    if "hour_range" in d.columns:
        d["hour_range"] = pd.to_numeric(d["hour_range"], errors="coerce").fillna(0).astype(int)
    else:
        # hour / event_hour / datetime türevi
        hc = None
        for c in ("hour","event_hour","hour_of_day"):
            if c in d.columns:
                hc = c; break
        if hc is not None:
            d["hour_range"] = pd.to_numeric(d[hc], errors="coerce").fillna(0).astype(int)
        else:
            # datetime bazlı
            cands = [c for c in ["datetime","incident_datetime","timestamp","created_at"] if c in d.columns]
            if cands:
                dt = pd.to_datetime(d[cands[0]], errors="coerce")
                d["hour_range"] = dt.dt.hour.fillna(0).astype(int)
            else:
                # Günlük veri olabilir; boş bırakmak istemiyoruz → 0 kabul
                d["hour_range"] = 0

    # Anahtarların NULL olmaması
    d = d.dropna(subset=["GEOID","date"])
    d["hour_range"] = d["hour_range"].fillna(0).astype(int)

    return d

def _quantile_deciles(x: pd.Series) -> pd.Series:
    """
    1..10 arasında decile. Tüm değerler aynı ise 1 döndürür.
    """
    s = pd.to_numeric(x, errors="coerce").fillna(0.0)
    ranks = s.rank(method="first", pct=True)
    if ranks.nunique() < 2:
        return pd.Series(np.ones(len(s), dtype=int), index=x.index)
    # duplicates='drop' kullanmadan rank üstünden qcut güvenli
    return pd.qcut(ranks, 10, labels=False, duplicates="drop") + 1

def _risk_level_from_decile(dec: pd.Series) -> pd.Series:
    """
    1-4 low, 5-7 medium, 8-10 high
    """
    dec = pd.to_numeric(dec, errors="coerce").fillna(1).astype(int)
    out = np.where(dec >= 8, "high", np.where(dec >= 5, "medium", "low"))
    return pd.Series(out, index=dec.index, dtype=object)

def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, on=KEY, how="left", right_name="right"):
    """
    Merge öncesi right tarafında anahtar duplikasyonlarını kontrol et.
    """
    if right is None or right.empty:
        return left.copy()
    if not set(on).issubset(right.columns):
        missing = [c for c in on if c not in right.columns]
        raise ValueError(f"{right_name} içinde eksik anahtar kolon(lar): {missing}")
    # Dupe kontrol
    dup = right.duplicated(on=on, keep=False)
    if dup.any():
        sample = right.loc[dup, on].head(5).to_dict(orient="records")
        raise ValueError(f"{right_name} anahtarlarında çoğulluk var (örnekler): {sample}")
    return left.merge(right, on=on, how=how)

# ---------------------------------------------------------------------
# Proba okuma (anahtarlı)
# ---------------------------------------------------------------------
def read_proba_df(proba_path: str | None, base_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Beklenen ideal format: CSV/Parquet ve anahtarlar + bir skor kolonu:
      GEOID,date,hour_range,[proba|prob|p|y_pred_proba|risk_score]
    Eğer .npy verilirse VE base_df uzunluğu eşitse, sıralı olarak bağlanır (uyarı verilir).
    """
    if not proba_path:
        return pd.DataFrame(columns=KEY+["risk_score"])

    p = Path(proba_path)
    if not p.exists():
        raise FileNotFoundError(f"Proba dosyası yok: {proba_path}")

    if str(p).lower().endswith(".npy"):
        if base_df is None:
            raise ValueError(".npy kullanılacaksa base_df gereklidir.")
        arr = np.load(p)
        if len(arr) != len(base_df):
            raise ValueError(f".npy uzunluğu ({len(arr)}) base_df uzunluğuyla ({len(base_df)}) eşleşmiyor.")
        print("[WARN] .npy proba sıralı bindiriliyor (anahtarsız). Bu, filtre/sıra değişince kırılabilir.")
        out = base_df[KEY].copy()
        out["risk_score"] = np.clip(pd.to_numeric(arr, errors="coerce").fillna(0.0), 0.0, 1.0)
        return out

    # CSV/Parquet
    if str(p).lower().endswith(".parquet"):
        dfp = pd.read_parquet(p)
    else:
        dfp = pd.read_csv(p, low_memory=False)

    # Anahtarları normalize et
    if "GEOID" not in dfp.columns:
        raise ValueError("Proba dosyasında GEOID bulunmalı.")
    dfp["GEOID"] = _zfill_geoid(dfp["GEOID"])

    if "date" in dfp.columns:
        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce").dt.date.astype("string")
    else:
        # datetime'dan üret
        cand = None
        for c in ("datetime","incident_datetime","timestamp","created_at"):
            if c in dfp.columns:
                cand = c; break
        if cand is None:
            raise ValueError("Proba dosyasında 'date' yok ve datetime da bulunamadı.")
        dt = pd.to_datetime(dfp[cand], errors="coerce")
        dfp["date"] = dt.dt.date.astype("string")

    if "hour_range" in dfp.columns:
        dfp["hour_range"] = pd.to_numeric(dfp["hour_range"], errors="coerce").fillna(0).astype(int)
    else:
        # hour vb.?
        hc = None
        for c in ("hour","event_hour","hour_of_day"):
            if c in dfp.columns:
                hc = c; break
        if hc is not None:
            dfp["hour_range"] = pd.to_numeric(dfp[hc], errors="coerce").fillna(0).astype(int)
        else:
            # datetime bazlı saat
            cand = None
            for c in ("datetime","incident_datetime","timestamp","created_at"):
                if c in dfp.columns:
                    cand = c; break
            if cand is None:
                raise ValueError("Proba dosyasında hour_range türetilemedi.")
            dt = pd.to_datetime(dfp[cand], errors="coerce")
            dfp["hour_range"] = dt.dt.hour.fillna(0).astype(int)

    # skor kolonu
    score_col = None
    for c in ("risk_score","proba","prob","p","y_pred_proba"):
        if c in dfp.columns:
            score_col = c; break
    if score_col is None:
        if dfp.shape[1] == 1:
            raise ValueError("Proba dosyası tek kolon ve isimlendirilmemiş; anahtarla birleştirilemez.")
        raise ValueError("Proba dosyasında skor kolonu (risk_score/proba/prob/p/y_pred_proba) bulunamadı.")

    dfp["risk_score"] = np.clip(pd.to_numeric(dfp[score_col], errors="coerce").fillna(0.0), 0.0, 1.0)
    dfp = dfp[KEY + ["risk_score"]].dropna(subset=["date"]).copy()
    return dfp

# ---------------------------------------------------------------------
# Çok-sınıflı (Top-3) okuma
# ---------------------------------------------------------------------
def load_top3_df(base_dir: Path) -> pd.DataFrame:
    """
    Desteklenen kaynaklar (öncelik sırası):
      1) proba_multiclass.(parquet|csv)  -> long: GEOID,date,hour_range,category,prob
      2) proba_multiclass_wide.(parquet|csv) -> wide: prob_<cat> sütunları
      3) risk_types_top3.csv -> yalnız date bazlı (hour boş kalabilir)
    Çıkış: KEY + top1/2/3_category, top1/2/3_prob
    """
    # 1) long
    for name in ("proba_multiclass.parquet","proba_multiclass.csv"):
        p = (base_dir / name)
        if p.exists():
            try:
                t = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
                need = {"GEOID","date","hour_range","category","prob"}
                if not need.issubset(t.columns):
                    raise ValueError("proba_multiclass dosyasında kolonlar eksik.")
                t = t.copy()
                t["GEOID"] = _zfill_geoid(t["GEOID"])
                t["date"]  = pd.to_datetime(t["date"], errors="coerce").dt.date.astype("string")
                t["hour_range"] = pd.to_numeric(t["hour_range"], errors="coerce").fillna(0).astype(int)
                t["prob"] = np.clip(pd.to_numeric(t["prob"], errors="coerce").fillna(0.0), 0.0, 1.0)

                # Her KEY için en yüksek 3 kategori
                t = t.sort_values(KEY + ["prob"], ascending=[True,True,True,False])
                top3 = t.groupby(KEY, as_index=False).head(3)

                def _expand(grp):
                    cats = grp["category"].tolist()
                    probs = grp["prob"].tolist()
                    row = {}
                    for i in range(3):
                        c = cats[i] if i < len(cats) else ""
                        p_ = float(probs[i]) if i < len(probs) else 0.0
                        row[f"top{i+1}_category"] = c
                        row[f"top{i+1}_prob"]     = p_
                    return pd.Series(row)

                exp = top3.groupby(KEY).apply(_expand).reset_index()
                return exp
            except Exception as e:
                print(f"[WARN] {name} okunamadı: {e}", file=sys.stderr)

    # 2) wide
    for name in ("proba_multiclass_wide.parquet","proba_multiclass_wide.csv"):
        p = (base_dir / name)
        if p.exists():
            try:
                t = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
                if not set(KEY).issubset(t.columns):
                    raise ValueError("proba_multiclass_wide anahtar kolonları eksik.")
                t["GEOID"] = _zfill_geoid(t["GEOID"])
                t["date"]  = pd.to_datetime(t["date"], errors="coerce").dt.date.astype("string")
                t["hour_range"] = pd.to_numeric(t["hour_range"], errors="coerce").fillna(0).astype(int)
                prob_cols = [c for c in t.columns if c.startswith("prob_")]
                if not prob_cols:
                    raise ValueError("proba_multiclass_wide içinde prob_* kolonları yok.")
                key_cols = KEY.copy()

                def _top3_row(row):
                    probs = {c[5:]: float(row[c]) if pd.notna(row[c]) else 0.0 for c in prob_cols}
                    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    out = {}
                    for i in range(3):
                        cat, p_ = items[i] if i < len(items) else ("", 0.0)
                        out[f"top{i+1}_category"] = cat
                        out[f"top{i+1}_prob"]     = float(np.clip(p_, 0.0, 1.0))
                    return pd.Series(out)

                exp = pd.concat([t[key_cols], t.apply(_top3_row, axis=1)], axis=1)
                return exp
            except Exception as e:
                print(f"[WARN] {name} okunamadı: {e}", file=sys.stderr)

    # 3) risk_types_top3.csv (date bazlı; hour_range yok)
    p = base_dir / "risk_types_top3.csv"
    if p.exists():
        try:
            r = pd.read_csv(p)
            # Beklenen kolonlar (minimum): date, top1_category/top1_prob, ...
            have = set(r.columns)
            base = ["top1_category","top1_prob","top2_category","top2_prob","top3_category","top3_prob"]
            if "date" not in have:
                raise ValueError("risk_types_top3.csv içinde 'date' yok.")
            rr = r[["date"] + [c for c in base if c in have]].copy()
            rr["date"] = pd.to_datetime(rr["date"], errors="coerce").dt.date.astype("string")
            # KEY için GEOID ve hour_range yok; merge sırasında sadece 'date' ile eşleşecek
            return rr
        except Exception as e:
            print(f"[WARN] risk_types_top3.csv okunamadı: {e}", file=sys.stderr)

    # Yoksa boş
    return pd.DataFrame(columns=[
        "GEOID","date","hour_range",
        "top1_category","top1_prob",
        "top2_category","top2_prob",
        "top3_category","top3_prob",
    ])

# ---------------------------------------------------------------------
# Çekirdek inşa fonksiyonları
# ---------------------------------------------------------------------
def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zorunlu kolonları garanti eder ve sıralar.
    expected_count: float
    risk_decile  : int
    """
    out = df.copy()
    for c in REQ_COLS:
        if c not in out.columns:
            if c.endswith("_category"):
                out[c] = ""
            elif c in ("risk_decile",):
                out[c] = 1
            else:
                out[c] = 0.0

    # Tipler
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0)
    out["expected_count"] = pd.to_numeric(out["expected_count"], errors="coerce").fillna(0.0)
    out["risk_decile"] = pd.to_numeric(out["risk_decile"], errors="coerce").fillna(1).astype(int)

    # Prob alanları [0,1]
    for c in ["top1_prob","top2_prob","top3_prob"]:
        out[c] = np.clip(pd.to_numeric(out[c], errors="coerce").fillna(0.0), 0.0, 1.0)

    # Sıralama
    out = out[REQ_COLS].copy()
    return out

def build_hourly(df_all: pd.DataFrame, proba_df: pd.DataFrame, top3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Son 7 güne ait (GEOID,date,hour_range) saatlik çıktı.
    """
    df = _ensure_date_hour(df_all)
    # Zaman filtresi (son 7 gün)
    dt = pd.to_datetime(df["date"], errors="coerce")
    maxd = dt.max()
    if pd.isna(maxd):
        raise ValueError("DF içinde geçerli tarih yok.")
    start = (maxd - pd.Timedelta(days=6)).date()
    df7 = df[dt.dt.date >= start].copy()

    # Anahtarlı proba merge
    proba_df = proba_df.copy()
    if not proba_df.empty:
        proba_df = _ensure_date_hour(proba_df)
        proba_df = proba_df[KEY + ["risk_score"]].drop_duplicates(KEY)
    out = _safe_merge(df7[KEY].drop_duplicates(), proba_df, on=KEY, how="left", right_name="proba_df")

    # risk_score
    out["risk_score"] = np.clip(pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0), 0.0, 1.0)

    # decile & risk_level
    dec = _quantile_deciles(out["risk_score"])
    out["risk_decile"] = dec
    out["risk_level"]  = _risk_level_from_decile(dec)

    # expected_count (basit: risk_score)
    out["expected_count"] = out["risk_score"].astype(float)

    # Top-3 (opsiyonel)
    if top3_df is not None and not top3_df.empty:
        t = top3_df.copy()

        # Join anahtarı: mümkünse KEY, değilse alt kümeler
        join_keys = [c for c in KEY if c in t.columns]
        if not join_keys:
            # sadece date varsa yine de birleştir
            if "date" in t.columns:
                join_keys = ["date"]
            else:
                join_keys = []

        if join_keys:
            tcols = [c for c in t.columns if c in (
                "GEOID","date","hour_range",
                "top1_category","top1_prob",
                "top2_category","top2_prob",
                "top3_category","top3_prob",
            )]
            t = t[tcols].copy()
            out = out.merge(t, on=join_keys, how="left")

    # Eksik top3 alanları
    for i in (1,2,3):
        if f"top{i}_category" not in out.columns: out[f"top{i}_category"] = ""
        if f"top{i}_prob" not in out.columns:     out[f"top{i}_prob"]     = 0.0

    # top*_expected = top*_prob * expected_count
    for i in (1,2,3):
        out[f"top{i}_expected"] = pd.to_numeric(out[f"top{i}_prob"], errors="coerce").fillna(0.0) * out["expected_count"]

    # Şema sonlandır
    out = _finalize_schema(out)

    # Log
    print(f"[Hourly] rows={len(out)}, uniq_keys={out[KEY].drop_duplicates().shape[0]}")
    return out

def build_daily(df_all: pd.DataFrame, proba_df: pd.DataFrame, top3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Son 365 güne ait günlük çıktı. Şema korunur, 'hour_range' boş string yazılır.
    Günlük risk_score, saatlik olasılıklardan birleşik olasılık olarak hesaplanmaz;
    burada basit yaklaşım: aynı gün-saha için ortalama risk (veya max) yerine,
    doğrudan DF içindeki risk_score'ların saatlik birleşimini kullanmak da mümkündür.
    Ancak şema eşitliği gereği burada şu strateji kullanılıyor:
      - Günlük satırlar oluşturulur (GEOID, date)
      - risk_score = saatlik skorların 1 - ∏(1 - p) birleşimi (varsa)
      - hour_range = "" (boş)
      - Top3: saatlik top3'lerden λ ~ p * expected_count ile birikimli yaklaşım;
              bulunamazsa günlükte boş bırakılır.
    """
    base = _ensure_date_hour(df_all)

    # Tarih aralığı (365 gün)
    dt = pd.to_datetime(base["date"], errors="coerce")
    maxd = dt.max()
    if pd.isna(maxd):
        raise ValueError("DF içinde geçerli tarih yok.")
    start = (maxd - pd.Timedelta(days=364)).date()
    base365 = base[dt.dt.date >= start].copy()

    # Anahtarlı proba
    if proba_df is not None and not proba_df.empty:
        p = _ensure_date_hour(proba_df)
        p = p[KEY + ["risk_score"]].drop_duplicates(KEY)
    else:
        p = pd.DataFrame(columns=KEY+["risk_score"])

    # Saatlik görünüm (KEY + risk_score)
    hourly = _safe_merge(base365[KEY].drop_duplicates(), p, on=KEY, how="left", right_name="proba_df")
    hourly["risk_score"] = np.clip(pd.to_numeric(hourly["risk_score"], errors="coerce").fillna(0.0), 0.0, 1.0)

    # Günlük birleşik olasılık: 1 - ∏(1 - p_hour)
    tmp = hourly.copy()
    tmp["_one_minus_p"] = 1.0 - tmp["risk_score"]
    daily = (
        tmp.groupby(["GEOID","date"], as_index=False)
           .agg(risk_score=("_one_minus_p", lambda s: 1.0 - float(np.prod(s.fillna(1.0).values))))
    )

    # hour_range boş string, expected_count = risk_score (basit)
    daily["hour_range"] = ""
    daily["expected_count"] = daily["risk_score"].astype(float)

    # decile & risk_level (günlük için ayrı dağılım)
    dec = _quantile_deciles(daily["risk_score"])
    daily["risk_decile"] = dec
    daily["risk_level"]  = _risk_level_from_decile(dec)

    # Top-3 (saatlikten günlük özet)
    # Mantık: Saatlik top3 varsa, günlükte λ ~= p_hour * expected_hour (= p_hour * p_hour) gibi
    # sadeleştirilmiş bir yaklaşım yerine; saatlik p_hour'u ağırlık olarak kullanıp gün boyunca
    # kategori skorlarını toplayalım ve en büyük 3'ü seçelim.
    if top3_df is not None and not top3_df.empty:
        t = top3_df.copy()

        # KEY'e kadar geniş; fakat bazı kaynaklarda sadece 'date' olabilir.
        if not set(["date"]).issubset(t.columns):
            t = pd.DataFrame(columns=["date"])
        # Saatlik top3 varsa:
        if set(KEY).issubset(t.columns):
            tt = _safe_merge(hourly[KEY + ["risk_score"]], t, on=KEY, how="left", right_name="top3_df")
            # Top3 prob'larını topla → gün bazında en baskın 3 kategori
            rows = []
            for i in (1,2,3):
                if f"top{i}_category" in tt.columns and f"top{i}_prob" in tt.columns:
                    part = tt[["GEOID","date",f"top{i}_category",f"top{i}_prob"]].rename(
                        columns={f"top{i}_category":"cat", f"top{i}_prob":"prob"}
                    )
                    rows.append(part)
            if rows:
                stack = pd.concat(rows, ignore_index=True)
                stack = stack[stack["cat"].astype(str) != ""].copy()
                agg = (stack.groupby(["GEOID","date","cat"], as_index=False)["prob"].sum())
                # Her (GEOID,date) için en büyük 3
                agg["rank"] = agg.groupby(["GEOID","date"])["prob"].rank(method="first", ascending=False)
                agg = agg[agg["rank"] <= 3].copy()
                # Pivot gibi genişlet
                agg = agg.sort_values(["GEOID","date","prob"], ascending=[True,True,False])
                def _expand(gr):
                    g = gr.head(3).reset_index(drop=True)
                    out = {}
                    for j in range(3):
                        if j < len(g):
                            out[f"top{j+1}_category"] = str(g.loc[j,"cat"])
                            out[f"top{j+1}_prob"]     = float(np.clip(g.loc[j,"prob"], 0.0, 1.0))
                        else:
                            out[f"top{j+1}_category"] = ""
                            out[f"top{j+1}_prob"]     = 0.0
                    return pd.Series(out)
                tops_day = agg.groupby(["GEOID","date"]).apply(_expand).reset_index()
                daily = daily.merge(tops_day, on=["GEOID","date"], how="left")

        # Aksi hâlde yalnız 'date' bazlı top3 varsa, günlüğe direkt basılır
        elif "date" in t.columns:
            keep = ["date","top1_category","top1_prob","top2_category","top2_prob","top3_category","top3_prob"]
            t2 = t[[c for c in keep if c in t.columns]].copy()
            daily = daily.merge(t2, on="date", how="left")

    # Eksik top3 alanları tamamla ve expected
    for i in (1,2,3):
        if f"top{i}_category" not in daily.columns: daily[f"top{i}_category"] = ""
        if f"top{i}_prob" not in daily.columns:     daily[f"top{i}_prob"]     = 0.0
        daily[f"top{i}_expected"] = pd.to_numeric(daily[f"top{i}_prob"], errors="coerce").fillna(0.0) * daily["expected_count"]

    # Şema sonlandır
    daily = _finalize_schema(daily)

    # Log
    print(f"[Daily] rows={len(daily)}, uniq_pairs={daily[['GEOID','date']].drop_duplicates().shape[0]}")
    return daily

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True, help="Özellik/kimlik DF CSV/Parquet (GEOID,date,hour_range türetilebilir).")
    ap.add_argument("--proba", default=None, help="Anahtarlı proba (CSV/Parquet: KEY+score) veya .npy (sıralı, riskli).")
    ap.add_argument("--top3", default=None, help="Opsiyonel: top3 kaynak dizini (varsayılan CRIME_DATA_DIR).")
    args = ap.parse_args()

    # DF yükle
    src = Path(args.df)
    if not src.exists():
        raise FileNotFoundError(f"DF bulunamadı: {args.df}")
    if str(src).lower().endswith(".parquet"):
        df_all = pd.read_parquet(src)
    else:
        df_all = pd.read_csv(src, low_memory=False)

    # Proba DF
    proba_df = read_proba_df(args.proba, base_df=df_all)

    # Top3 DF (varsa)
    top_dir = Path(args.top3).resolve() if args.top3 else CRIME_DIR
    top3_df = load_top3_df(top_dir)

    # ÇIKTILAR
    hourly = build_hourly(df_all=df_all, proba_df=proba_df, top3_df=top3_df)
    daily  = build_daily(df_all=df_all,  proba_df=proba_df, top3_df=top3_df)

    # Yaz
    hourly_path = CRIME_DIR / "risk_hourly_grid_full_labeled.csv"
    daily_path  = CRIME_DIR / "risk_daily_grid_full_labeled.csv"

    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)

    # Parquet de üret (opsiyonel)
    try:
        hourly.to_parquet(hourly_path.with_suffix(".parquet"), index=False)
        daily.to_parquet(daily_path.with_suffix(".parquet"), index=False)
    except Exception as e:
        print(f"[WARN] Parquet yazılamadı: {e}", file=sys.stderr)

    print(f"✅ Saatlik (≤7g) → {hourly_path}  rows={len(hourly)}")
    print(f"✅ Günlük  (≤365g) → {daily_path}  rows={len(daily)}")

if __name__ == "__main__":
    main()
