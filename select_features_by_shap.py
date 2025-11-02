#!/usr/bin/env python3
# select_features_by_shap.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

REQ = ["GEOID","date","event_hour","Y_label","latitude","longitude"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    miss = [c for c in REQ if c not in df.columns]
    if miss: raise SystemExit(f"Eksik kolonlar: {miss}")
    df = df.copy()
    df["GEOID"] = df["GEOID"].astype(str)
    df["date"] = df["date"].astype(str)
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce")
    if df["event_hour"].isna().any():
        df["event_hour"] = df["event_hour"].fillna(df["event_hour"].mode(dropna=True).iloc[0])
    df["event_hour"] = df["event_hour"].astype("int16")
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    df["_t"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def load_shap(shap_csv: str, top_k: int) -> list[str]:
    """Beklenen kolon adları: feature, importance (veya value)."""
    imp = pd.read_csv(shap_csv)
    cand_cols = None
    for c in ["importance","mean_abs_shap","value","importance_mean","shap_value"]:
        if c in imp.columns:
            cand_cols = c; break
    if cand_cols is None:
        # importance kolonu bulunamazsa ilk sütunun skoru olduğunu varsay
        cand_cols = imp.columns[1]
    name_col = "feature" if "feature" in imp.columns else imp.columns[0]
    imp = imp[[name_col, cand_cols]].dropna()
    imp = imp.sort_values(cand_cols, ascending=False)
    feats = imp[name_col].astype(str).tolist()
    # GEOID ve event_hour yoksa ekleyelim (genelde gerekli)
    base = ["GEOID","event_hour"]
    for b in base:
        if b in feats:
            feats.remove(b)
    feats = base + feats
    return feats[:top_k]

def make_pipeline(X_cols: list[str], df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    X = df[X_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X.drop(columns=all_nan, inplace=True)
        X_cols = [c for c in X_cols if c not in all_nan]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if "GEOID" in num_cols:
        num_cols.remove("GEOID"); cat_cols.append("GEOID")

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols),
    ])

    # hızlı ve kararlı tek model (stacking yerine tarama için ideal)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42
    )

    pipe = Pipeline([("pre", pre), ("clf", xgb)])
    return pipe, X_cols

def evaluate(df: pd.DataFrame, feats: list[str]) -> float:
    # Train/Test: son %20 test (zaman)
    cutoff = df["_t"].quantile(0.80)
    tr = df["_t"] < cutoff
    te = df["_t"] >= cutoff

    X_cols = [c for c in feats if c in df.columns]
    if not X_cols:
        return -1.0

    pipe, X_cols = make_pipeline(X_cols, df)
    Xtr, ytr = df.loc[tr, X_cols], df.loc[tr, "Y_label"].to_numpy()
    Xte, yte = df.loc[te, X_cols], df.loc[te, "Y_label"].to_numpy()
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:,1]
    return float(average_precision_score(yte, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="fr_crime_09.parquet (veya CSV)")
    ap.add_argument("--shap_csv", required=True, help="SHAP importance CSV (feature, importance)")
    ap.add_argument("--kmin", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=70)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", default="features_selected.json")
    args = ap.parse_args()

    df = load_data(args.data)

    results = []
    best_ap, best_k, best_feats = -1.0, None, None

    for k in range(args.kmin, args.kmax+1, args.step):
        feats_k = load_shap(args.shap_csv, k)
        ap_k = evaluate(df, feats_k)
        results.append({"k": k, "ap": ap_k, "used": len([c for c in feats_k if c in df.columns])})
        if ap_k > best_ap:
            best_ap, best_k, best_feats = ap_k, k, feats_k

        print(f"[K={k:>2}] AP={ap_k:.4f} (kullanılan={results[-1]['used']})")

    out = {
        "best_k": best_k,
        "best_ap": best_ap,
        "features": best_feats,
        "grid": results
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n✅ Seçilen K={best_k} | AP={best_ap:.4f}")
    print(f"→ Kaydedildi: {args.out}")

if __name__ == "__main__":
    main()
