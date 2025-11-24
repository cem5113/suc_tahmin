#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FA v1 — ONLY Feature Analysis + SHAP
------------------------------------
Girdi : fr_crime_09.csv
Çıktı :
  - fr_crime_09_clean.csv
  - feature_importances_fr09.csv
  - shap_feature_importance_fr09.csv
  - features_fr09.json
"""

import os
import csv
import math
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


# ================================
# GEOID normalize
# ================================
def normalize_geoid(val, length: int = 11) -> str | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    try:
        if "e" in s.lower():
            s = str(int(float(s)))
    except:
        pass
    if s.isdigit():
        s = s.zfill(length)
    return s


def main():

    # ============================================================
    # PATH AYARLARI
    # ============================================================
    base_dir = Path(os.environ.get("CRIME_DATA_DIR", ".")).resolve()
    output_dir = Path(os.environ.get("FR_OUTPUT_DIR", base_dir / "fr_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path   = base_dir / "fr_crime_09.csv"
    clean_path = output_dir / "fr_crime_09_clean.csv"

    print("📂 CRIME_DATA_DIR :", base_dir)
    print("📂 FR_OUTPUT_DIR  :", output_dir)
    print("📄 RAW_PATH       :", raw_path)
    print("📄 CLEAN_PATH     :", clean_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"❌ fr_crime_09.csv bulunamadı: {raw_path}")

    # ============================================================
    # 1) CSV TEMİZLEME (ParserError önleme)
    # ============================================================
    print(f"\n📥 Orijinal CSV temizleniyor...")

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(clean_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        expected_cols = len(header)
        writer.writerow(header)

        bad = 0

        for row in reader:
            if len(row) == expected_cols:
                writer.writerow(row)
            else:
                bad += 1

    print(f"🧹 Temizleme tamam — hatalı satır: {bad}")
    print("✔ Temiz CSV yazıldı:", clean_path)

    # ============================================================
    # 2) VERİ YÜKLE & FEATURE SET HAZIRLA
    # ============================================================
    df = pd.read_csv(clean_path, low_memory=False)
    print("📊 Temiz veri:", df.shape)

    df.columns = [c.strip() for c in df.columns]

    # GEOID
    if "geoid" in df:
        df["geoid"] = df["geoid"].astype(str)
    elif "GEOID" in df:
        df["geoid"] = df["GEOID"].astype(str)
    else:
        raise Exception("❌ GEOID/geoid kolonu yok.")

    df["geoid"] = df["geoid"].str.replace(r"\.0$", "", regex=True).str.zfill(11)

    # datetime → date
    if "date" not in df.columns:
        raise Exception("❌ 'date' kolonu yok.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if "Y_label" not in df.columns:
        raise Exception("❌ Y_label yok.")

    # ==========================
    # ML subset
    # ==========================
    if "hour" not in df:
        df["hour"] = df["date"].dt.hour

    # sadece regression-like düşmeyelim → binary Y
    y = df["Y_label"]
    drop_cols = ["Y_label", "date"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    print(f"🔢 Feature sayısı → num={len(num_cols)}, cat={len(cat_cols)}")

    # Preprocess
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    print("\n⏳ Preprocess + RF importance eğitiliyor...")

    rf = Pipeline([
        ("prep", pre),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        ))
    ])

    rf.fit(X, y)

    # Feature names
    feat_names = rf.named_steps["prep"].get_feature_names_out()
    rf_imp_raw = rf.named_steps["rf"].feature_importances_

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": rf_imp_raw
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(output_dir / "feature_importances_fr09.csv", index=False)
    print("💾 RF feature importance kaydedildi.")

    # Ayrıca orijinal kolon listesi:
    with open(output_dir / "features_fr09.json", "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    print("💾 Feature list yazıldı (features_fr09.json).")

    # ============================================================
    # 3) SHAP IMPORTANCE
    # ============================================================
    print("\n⏳ SHAP hesaplanıyor (RF bazlı)...")

    try:
        import shap

        rf_only = rf.named_steps["rf"]
        pre_only = rf.named_steps["prep"]

        N = min(3000, len(X))
        Xs = X.sample(N, random_state=42)
        Xt = pre_only.transform(Xs)

        explainer = shap.TreeExplainer(rf_only)
        shap_vals = explainer.shap_values(Xt)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # binary pos-class

        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        shap_df = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)

        shap_df.to_csv(output_dir / "shap_feature_importance_fr09.csv", index=False)

        print("💾 SHAP feature importance kaydedildi.")

    except Exception as e:
        print("⚠️ SHAP hesaplamasında hata:", e)

    print("\n✅ Feature Analysis + SHAP TAMAMLANDI.")


if __name__ == "__main__":
    main()
