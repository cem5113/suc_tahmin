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
from sklearn.pipeline import Pipeline

from build_future_grid import build_future_grid
from forecast_exports import top_k_types

CRIME_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
FR_TZ     = ZoneInfo("America/Los_Angeles")

HORIZON_HOURS_24H  = 24
HORIZON_HOURS_7D   = 7 * 24
HORIZON_HOURS_365D = 365 * 24

TYPE_COL = os.getenv("CRIME_TYPE_COL", "crime_category")  # sende hangi kolon ise


# ---------------------------------------------------------
# MODEL LOAD / PREDICT
# ---------------------------------------------------------
def load_models():
    """
    stacking_risk_pipeline.py çıktısı:
      - models/preprocessor.joblib
      - models/base_pipes.joblib   (prep+clf pipeline'ları)
      - models/stacking_*.joblib   ({"names": [...], "meta": meta_model})
      - models/threshold_*.json
    """
    mdir = Path(CRIME_DIR) / "models"
    pre  = load(mdir / "preprocessor.joblib")
    base = load(mdir / "base_pipes.joblib")

    stack_files = sorted(mdir.glob("stacking_*.joblib"))
    thr_files   = sorted(mdir.glob("threshold_*.json"))
    if not stack_files:
        raise FileNotFoundError("stacking_*.joblib yok.")

    stack_obj = load(stack_files[0])
    names = stack_obj["names"]
    meta  = stack_obj["meta"]

    thr = 0.5
    if thr_files:
        try:
            thr = json.loads(thr_files[0].read_text())["threshold"]
        except Exception:
            pass

    return pre, base, meta, names, thr


def stacking_predict_proba(base_pipes, meta, proba_cols, X):
    """
    base_pipes zaten Pipeline(prep+clf) olduğu için X ham feature DF olur.
    """
    mats = []
    for n in proba_cols:
        mdl = base_pipes[n]
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)[:, 1]
        else:
            d = mdl.decision_function(X)
            p = (d - d.min()) / (d.max() - d.min() + 1e-9)
        mats.append(p.reshape(-1, 1))

    Z = np.hstack(mats)
    if hasattr(meta, "predict_proba"):
        p_stack = meta.predict_proba(Z)[:, 1]
    else:
        d = meta.decision_function(Z)
        p_stack = (d - d.min()) / (d.max() - d.min() + 1e-9)

    return p_stack


# ---------------------------------------------------------
# TYPE MODEL (MULTICLASS) — PREPROCESSOR İLE PIPELINE
# ---------------------------------------------------------
def train_type_model(train_df, feature_cols, type_col):
    """Y=1 satırlarda çok sınıflı type modeli. Preprocessor + LGBM multiclass."""
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None

    df_pos = train_df[train_df["Y_label"] == 1].dropna(subset=[type_col]).copy()
    if df_pos.empty:
        return None

    # stacking tarafındaki preprocessor’ı reuse et
    pre, _, _, _, _ = load_models()

    usable_cols = [c for c in feature_cols if c in df_pos.columns]
    if not usable_cols:
        return None

    X = df_pos[usable_cols]
    y = df_pos[type_col].astype(str)

    type_clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    pipe = Pipeline([("prep", pre), ("clf", type_clf)])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------
# FUTURE GRID + FORECAST
# ---------------------------------------------------------
def make_and_forecast(train_df, horizon_hours, feature_cols):
    geoid_list = train_df["GEOID"].unique()
    future_df  = build_future_grid(train_df, geoid_list, horizon_hours=horizon_hours)

    pre, base_pipes, meta, proba_cols, thr = load_models()

    # future grid’de olmayan feature’ları güvenle at
    usable_cols = [c for c in feature_cols if c in future_df.columns]
    missing = [c for c in feature_cols if c not in future_df.columns]
    if missing:
        print(
            f"⚠️ future grid’de olmayan feature’lar atlandı: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    if not usable_cols:
        raise ValueError("future_df içinde kullanılabilir feature kalmadı.")

    X_future = future_df[usable_cols].copy()

    p_stack  = stacking_predict_proba(base_pipes, meta, proba_cols, X_future)
    future_df["risk_p"]   = p_stack
    future_df["risk_hat"] = (p_stack >= thr).astype(int)

    return future_df, thr, usable_cols


def export_forecast(fut_df, tag):
    outdir = Path(CRIME_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # hourly risk export
    hourly_path = outdir / f"risk_hourly_next{tag}.parquet"
    fut_df.to_parquet(hourly_path, index=False)

    # daily aggregation
    daily = (
        fut_df
        .assign(date=pd.to_datetime(fut_df["datetime"]).dt.date)
        .groupby(["GEOID", "date"], as_index=False)["risk_p"]
        .mean()
        .rename(columns={"risk_p": "risk_p_daily"})
    )
    daily_path = outdir / f"risk_daily_next{tag}.parquet"
    daily.to_parquet(daily_path, index=False)

    return hourly_path, daily_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    data_path = Path(CRIME_DIR) / "fr_crime_10_FA.csv"
    df = pd.read_csv(data_path, low_memory=False, dtype={"GEOID": str})

    if "datetime" not in df.columns:
        # ensure_date_hour_on_df bunu üretiyor; yoksa fallback:
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise ValueError("df içinde ne datetime ne date var.")

    # TRAIN için kullanılan feature listeni aynen oku (FA çıktısı)
    feat_path = Path(CRIME_DIR) / "selected_features_fr.csv"
    if feat_path.exists():
        feature_cols = pd.read_csv(feat_path)["feature"].astype(str).tolist()
    else:
        feature_cols = [c for c in df.columns if c not in {"Y_label", "crime_id", "datetime"}]

    # güvenlik: hedef/etiket/type vs. kesin çıkar
    drop_always = {"Y_label", "crime_id", "crime_category", "datetime"}
    feature_cols = [c for c in feature_cols if c in df.columns and c not in drop_always]

    if not feature_cols:
        raise ValueError("feature_cols boş kaldı (selected_features_fr.csv kontrol et).")

    # forecast 24h / 7d / 365d
    for hh, tag in [
        (HORIZON_HOURS_24H,  "24h"),
        (HORIZON_HOURS_7D,   "7d"),
        (HORIZON_HOURS_365D, "365d")
    ]:
        fut_df, thr, usable_cols = make_and_forecast(df, hh, feature_cols)
        hpath, dpath = export_forecast(fut_df, tag)
        print(f"✅ Forecast exported: {hpath.name}, {dpath.name} | thr={thr:.4f}")

    # --- Top crime types (isteğe bağlı)
    type_model = train_type_model(df, feature_cols, TYPE_COL)
    if type_model is not None:
        # sadece 7d için top-3, 365d için top-5 çıkar
        fut7, thr7, cols7 = make_and_forecast(df, HORIZON_HOURS_7D, feature_cols)

        q = type_model.predict_proba(fut7[cols7])
        labels = type_model.named_steps["clf"].classes_

        for i, c in enumerate(labels):
            fut7[f"type_{c}_risk"] = fut7["risk_p"] * q[:, i]

        top3 = top_k_types(fut7, k=3)
        top3.to_csv(Path(CRIME_DIR) / "top3_types_next7d.csv", index=False)

        fut365, thr365, cols365 = make_and_forecast(df, HORIZON_HOURS_365D, feature_cols)

        q2 = type_model.predict_proba(fut365[cols365])
        for i, c in enumerate(labels):
            fut365[f"type_{c}_risk"] = fut365["risk_p"] * q2[:, i]

        top5 = top_k_types(fut365, k=5)
        top5.to_csv(Path(CRIME_DIR) / "top5_types_next365d.csv", index=False)

        print("✅ Top crime types exported (next7d top3, next365d top5).")
    else:
        print("ℹ️ Type model kurulamadı (LightGBM yok / Y=1 type boş). Top types atlandı.")
