#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_fr.py — SUTAM için global SHAP feature analizi (tez / makale için)

Girdi :
  - FR_OUTPUT_DIR/fr_crime_09_clean.csv  (yoksa CRIME_DATA_DIR)

Çıktı :
  - FR_OUTPUT_DIR/shap_feature_importance_fr.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import shap


def main() -> None:
    base_dir_env = os.environ.get("FR_OUTPUT_DIR") or os.environ.get("CRIME_DATA_DIR") or "."
    BASE_DIR = Path(base_dir_env).resolve()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = BASE_DIR / "fr_crime_09_clean.csv"
    OUT_PATH = BASE_DIR / "shap_feature_importance_fr.csv"

    print("📥 Veri:", DATA_PATH)

    if not DATA_PATH.exists():
        raise RuntimeError(f"❌ fr_crime_09_clean.csv bulunamadı: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if "Y_label" not in df.columns:
        raise RuntimeError("❌ 'Y_label' bulunamadı!")

    drop_cols = []
    for c in df.columns:
        if c == "Y_label":
            continue
        uniq = df[c].dropna().unique()
        if len(uniq) <= 1:
            drop_cols.append(c)

    if drop_cols:
        print("⚠️ Sabit/boş kolonlar atılıyor:", drop_cols)
        df = df.drop(columns=drop_cols)

    y = df["Y_label"].astype(int)
    X = df.drop(columns=["Y_label"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print(f"🔢 Numeric: {len(num_cols)} | 🔠 Categorical: {len(cat_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
    )

    clf = Pipeline([
        ("pre", pre),
        ("xgb", model),
    ])

    print("🤖 Model eğitiliyor (XGBoost + basit preprocess)...")
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"🎯 Test ROC-AUC: {auc:.3f}")

    # ---------------------------------------------------------
    # SHAP ANALİZİ (GLOBAL)
    # ---------------------------------------------------------
    sample_size = min(5000, len(X_train))
    print(f"🧪 SHAP için örnek sayısı: {sample_size}")

    X_sample = X_train.sample(n=sample_size, random_state=42)

    X_sample_trans = clf.named_steps["pre"].transform(X_sample)
    xgb_model = clf.named_steps["xgb"]

    print("🧠 TreeExplainer oluşturuluyor...")
    explainer = shap.TreeExplainer(xgb_model)

    print("🧠 SHAP değerleri hesaplanıyor...")
    shap_values = explainer.shap_values(X_sample_trans)

    shap_abs = np.abs(shap_values)
    mean_abs = shap_abs.mean(axis=0)

    feature_names_pre = clf.named_steps["pre"].get_feature_names_out()

    shap_importance = pd.DataFrame({
        "feature_pre": feature_names_pre,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    def map_back(colname: str) -> str:
        if "__" in colname:
            return colname.split("__", 1)[1]
        return colname

    shap_importance["feature_raw_guess"] = shap_importance["feature_pre"].apply(map_back)

    shap_importance.to_csv(OUT_PATH, index=False)
    print("✅ SHAP feature importance kaydedildi →", OUT_PATH)


if __name__ == "__main__":
    main()
