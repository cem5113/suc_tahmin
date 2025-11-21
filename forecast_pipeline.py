#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_pipeline.py — SUTAM Modern Forecast (risk + type + count)

- stacking_risk_pipeline.py sadece TRAIN eder.
- Bu dosya geleceğe ait GRID üretir ve forecast export eder.
"""

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta
from zoneinfo import ZoneInfo
from joblib import load

from build_future_grid import build_future_grid
from forecast_exports import top_k_types

CRIME_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
FR_TZ     = ZoneInfo("America/Los_Angeles")

HORIZON_HOURS_24H  = 24
HORIZON_HOURS_7D   = 7 * 24
HORIZON_HOURS_365D = 365 * 24

TYPE_COL = os.getenv("CRIME_TYPE_COL", "crime_category")  # sende hangi kolon ise

def load_models():
    mdir = Path(CRIME_DIR) / "models"
    pre  = load(mdir / "preprocessor.joblib")
    base = load(mdir / "base_pipes.joblib")
    stack_files = sorted(mdir.glob("stacking_*.joblib"))
    thr_files   = sorted(mdir.glob("threshold_*.json"))
    if not stack_files:
        raise FileNotFoundError("stacking_*.joblib yok.")
    stack_obj = load(stack_files[0])
    names = stack_obj["names"]; meta = stack_obj["meta"]
    thr = 0.5
    if thr_files:
        try:
            thr = json.loads(thr_files[0].read_text())["threshold"]
        except Exception:
            pass
    return pre, base, meta, names, thr

def stacking_predict_proba(pre, base_pipes, meta, proba_cols, X):
    mats = []
    for n in proba_cols:
        mdl = base_pipes[n]
        p = mdl.predict_proba(X)[:,1] if hasattr(mdl,"predict_proba") else mdl.decision_function(X)
        mats.append(p.reshape(-1,1))
    Z = np.hstack(mats)
    p_stack = meta.predict_proba(Z)[:,1] if hasattr(meta,"predict_proba") else meta.decision_function(Z)
    return p_stack

def train_type_model(train_df, feature_cols, type_col):
    """Y=1 satırlarda çok sınıflı type modeli. Basit LightGBM."""
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None

    df = train_df[train_df["Y_label"]==1].dropna(subset=[type_col]).copy()
    if df.empty:
        return None

    X = df[feature_cols]
    y = df[type_col].astype(str)

    mdl = LGBMClassifier(
        objective="multiclass",
        num_class=len(y.unique()),
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    mdl.fit(X, y)
    return mdl

def make_and_forecast(train_df, horizon_hours, feature_cols):
    geoid_list = train_df["GEOID"].unique()
    future_df  = build_future_grid(train_df, geoid_list, horizon_hours=horizon_hours)

    pre, base_pipes, meta, proba_cols, thr = load_models()

    # aynı preprocessing kolon seti ile çalış
    X_future = future_df[feature_cols].copy()
    p_stack  = stacking_predict_proba(pre, base_pipes, meta, proba_cols, X_future)
    future_df["risk_p"] = p_stack
    future_df["risk_hat"] = (p_stack >= thr).astype(int)

    return future_df, thr

def export_forecast(fut_df, tag):
    outdir = Path(CRIME_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # hourly risk export
    hourly_path = outdir / f"risk_hourly_next{tag}.parquet"
    fut_df.to_parquet(hourly_path, index=False)

    # daily aggregation
    daily = (fut_df
             .assign(date=pd.to_datetime(fut_df["datetime"]).dt.date)
             .groupby(["GEOID","date"])["risk_p"]
             .mean()
             .reset_index()
             .rename(columns={"risk_p":"risk_p_daily"}))
    daily_path = outdir / f"risk_daily_next{tag}.parquet"
    daily.to_parquet(daily_path, index=False)

    return hourly_path, daily_path

if __name__ == "__main__":
    data_path = Path(CRIME_DIR) / "fr_crime_10_FA.csv"
    df = pd.read_csv(data_path, low_memory=False, dtype={"GEOID": str})

    if "datetime" not in df.columns:
        # senin sistemde ensure_date_hour_on_df bunu üretiyor; yoksa fallback:
        df["datetime"] = pd.to_datetime(df["date"])

    # TRAIN için kullanılan feature listeni aynen oku
    # (stacking_risk_pipeline sonunda manifest'e yazdırıyorsun, oradan çek)
    feat_path = Path(CRIME_DIR) / "selected_features_fr.csv"
    if feat_path.exists():
        feature_cols = pd.read_csv(feat_path)["feature"].tolist()
    else:
        # last resort: Y_label dışındaki tüm sütunlar
        feature_cols = [c for c in df.columns if c not in {"Y_label","crime_id","datetime"}]

    # forecast 24h / 7d / 365d
    for hh, tag in [(HORIZON_HOURS_24H,"24h"), (HORIZON_HOURS_7D,"7d"), (HORIZON_HOURS_365D,"365d")]:
        fut_df, thr = make_and_forecast(df, hh, feature_cols)
        hpath, dpath = export_forecast(fut_df, tag)
        print(f"✅ Forecast exported: {hpath.name}, {dpath.name} | thr={thr:.4f}")

    # --- Top crime types (isteğe bağlı)
    # type modelin varsa burada devreye alınır
    type_model = train_type_model(df, feature_cols, TYPE_COL)
    if type_model is not None:
        # sadece 7d için top-3, 365d için top-5 çıkar
        fut7, _ = make_and_forecast(df, HORIZON_HOURS_7D, feature_cols)
        q = type_model.predict_proba(fut7[feature_cols])
        labels = type_model.classes_
        for i,c in enumerate(labels):
            fut7[f"type_{c}_risk"] = fut7["risk_p"] * q[:,i]
        top3 = top_k_types(fut7, k=3)
        top3.to_csv(Path(CRIME_DIR)/"top3_types_next7d.csv")

        fut365, _ = make_and_forecast(df, HORIZON_HOURS_365D, feature_cols)
        q2 = type_model.predict_proba(fut365[feature_cols])
        for i,c in enumerate(labels):
            fut365[f"type_{c}_risk"] = fut365["risk_p"] * q2[:,i]
        top5 = top_k_types(fut365, k=5)
        top5.to_csv(Path(CRIME_DIR)/"top5_types_next365d.csv")

        print("✅ Top crime types exported (next7d top3, next365d top5).")
