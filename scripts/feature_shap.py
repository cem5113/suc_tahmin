#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FA v2 — ONLY Feature Analysis + SHAP
Girdi : fr_crime_09.csv
Çıktı :
  - feature_importances_fr09.csv
  - shap_feature_importance_fr09.csv
  - features_fr09.json
"""

import os, json, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# GEOID normalize
# ============================================================
def normalize_geoid(val, length=11):
    if val is None or (isinstance(val,float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    try:
        if "e" in s.lower(): s = str(int(float(s)))
    except: pass
    return s.zfill(length) if s.isdigit() else s


def main():
    # ------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------
    base_dir = Path(os.environ.get("CRIME_DATA_DIR", ".")).resolve()
    out_dir  = Path(os.environ.get("FR_OUTPUT_DIR", base_dir / "fr_outputs")).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    in_path  = base_dir / "fr_crime_09.csv"

    print("📌 INPUT  :", in_path)
    print("📌 OUTPUT :", out_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"fr_crime_09.csv bulunamadı → {in_path}")

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    df = pd.read_csv(in_path, low_memory=False)
    print("📊 Veri shape:", df.shape)

    # temel colonlar
    if "GEOID" in df:
        df["GEOID"] = df["GEOID"].astype(str).apply(normalize_geoid)
        df["geoid"] = df["GEOID"]
    elif "geoid" in df:
        df["geoid"] = df["geoid"].astype(str).apply(normalize_geoid)
    else:
        raise Exception("CSV içinde GEOID/geoid yok.")

    if "date" not in df:
        raise Exception("'date' kolonu yok.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if "Y_label" not in df:
        raise Exception("Y_label yok.")

    # ML hazırlanışı
    y = df["Y_label"]
    X = df.drop(columns=["Y_label", "date"], errors="ignore")

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    print(f"🔢 num={len(num_cols)}, cat={len(cat_cols)}")

    # ------------------------------------------------------------
    # PIPELINE (RF importance için)
    # ------------------------------------------------------------
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n⏳ RF importance eğitiliyor...")
    model.fit(X, y)

    feat_names = model.named_steps["prep"].get_feature_names_out()
    imps = model.named_steps["rf"].feature_importances_

    imp_df = pd.DataFrame({"feature": feat_names, "importance": imps})
    imp_df = imp_df.sort_values("importance", ascending=False)

    imp_df.to_csv(out_dir/"feature_importances_fr09.csv", index=False)
    print("💾 RF importance yazıldı.")

    # ------------------------------------------------------------
    # SHAP IMPORTANCE
    # ------------------------------------------------------------
    try:
        import shap
        print("\n⏳ SHAP açıklanabilirlik hesaplanıyor...")

        rf_only  = model.named_steps["rf"]
        prep_only = model.named_steps["prep"]

        # örneklem
        N = min(3000, len(X))
        Xs = X.sample(N, random_state=42)
        Xt = prep_only.transform(Xs)

        explainer = shap.TreeExplainer(rf_only)
        shap_vals = explainer.shap_values(Xt)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # binary

        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        shap_df = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)

        shap_df.to_csv(out_dir/"shap_feature_importance_fr09.csv", index=False)
        print("💾 SHAP importance yazıldı.")

    except Exception as e:
        print("⚠️ SHAP atlandı:", e)

    # ------------------------------------------------------------
    # FEATURE LIST
    # ------------------------------------------------------------
    with open(out_dir/"features_fr09.json", "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    print("\n✅ SADE FA + SHAP Tamamlandı.")


if __name__ == "__main__":
    main()
