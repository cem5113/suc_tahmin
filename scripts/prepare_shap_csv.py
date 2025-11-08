#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OA (outputs_feature_analysis) ve BASE (CRIME_DATA_DIR) altında SHAP dosyasını bul/üret:
Öncelik:
  0) BASE/shap_feature_importance.csv varsa bırak
  1) OA/shap_feature_importance.csv → BASE/
  2) OA/feature_importance_shap.csv → kanoniğe çevir
  3) OA/feature_importance_ensemble_base.csv → SHAP vekili
  4) BASE/feature_importance_xgb.csv → SHAP vekili
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
OA   = BASE / "outputs_feature_analysis"
OUT  = BASE / "shap_feature_importance.csv"

def write(df: pd.DataFrame) -> None:
    df = df.copy()
    if "feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature"})
    if "mean_abs_shap" not in df.columns:
        # ikinci kolonu değer kabul et
        val = df.columns[1]
        df["mean_abs_shap"] = pd.to_numeric(df[val], errors="coerce")
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce").clip(lower=0)
    df = df[df["mean_abs_shap"] > 0]
    s = df["mean_abs_shap"].sum()
    if s > 0:
        df["mean_abs_shap"] = df["mean_abs_shap"] / s
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["feature", "mean_abs_shap"]].to_csv(OUT, index=False)
    print(f"✅ wrote {OUT}")

def main() -> int:
    # 0) BASE’de hazırsa çık
    if OUT.exists():
        print(f"✅ SHAP already present at {OUT}")
        return 0

    # 1) OA/shap_feature_importance.csv
    p = OA / "shap_feature_importance.csv"
    if p.exists():
        p.replace(OUT) if OUT.exists() else OUT.write_bytes(p.read_bytes())
        print(f"✅ Found canonical SHAP at {p} → {OUT}")
        return 0

    # 2) OA/feature_importance_shap.csv
    p = OA / "feature_importance_shap.csv"
    if p.exists():
        df = pd.read_csv(p)
        write(df)
        return 0

    # 3) OA/feature_importance_ensemble_base.csv (proxy)
    p = OA / "feature_importance_ensemble_base.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "feature" not in df.columns:
            if "ensemble" in df.columns:
                others = [c for c in df.columns if c != "ensemble"]
                if len(others) == 1:
                    df = df.rename(columns={others[0]: "feature"})
            if "feature" not in df.columns:
                df = df.rename(columns={df.columns[0]: "feature", df.columns[1]: "ensemble"})
        if "ensemble" not in df.columns:
            # ikinci kolonu kullan
            val = df.columns[1]
            df = df.rename(columns={val: "ensemble"})
        df = df.rename(columns={"ensemble": "mean_abs_shap"})
        write(df[["feature", "mean_abs_shap"]])
        return 0

    # 4) BASE/feature_importance_xgb.csv (proxy)
    p = BASE / "feature_importance_xgb.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "importance" not in df.columns:
            val = df.columns[1]
            df = df.rename(columns={val: "importance"})
        df = df.rename(columns={"importance": "mean_abs_shap"})
        write(df[["feature", "mean_abs_shap"]])
        return 0

    print(f"❌ No SHAP/ensemble/xgb importance found under {OA} or {BASE}", file=sys.stderr)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())

