#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_feature_analysis.py

Tek script ile:
- sf_crime_08.csv (GEOID×zaman + Y_label) => MODELLEME + SHAP + ÖNEMLER
- fr_crime_09.csv (olay/suç_id bazlı, genelde Y=1) => EDA (model yok)

Kullanım:
  # sf (modelleme)
  python unified_feature_analysis.py \
      --csv crime_prediction_data/sf_crime_08.csv \
      --target Y_label \
      --outdir outputs_feature_analysis/sf_08 \
      --group_by_geoid

  # fr (EDA)
  python unified_feature_analysis.py \
      --csv crime_prediction_data/fr_crime_09.csv \
      --outdir outputs_feature_analysis/fr_09

Notlar:
- XGBoost yoksa LightGBM, o da yoksa RandomForest kullanılır (sf için).
- SHAP opsiyonel; kurulu değilse atlanır (sf için).
- Kantil etiketler ("Q1 (...)") otomatik sayıya çevrilir.
- GEOID/ID/lat-lon/date gibi sızıntı alanları (sf için) düşürülür.
"""

import argparse
import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# ML & viz
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# opsiyonel modeller
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# SHAP opsiyonel
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib


# ------------------------- yardımcılar -------------------------

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = df.columns.tolist()
    new_cols = []
    seen = {}
    for c in original_cols:
        nc = re.sub(r"[^\w]+", "_", str(c), flags=re.UNICODE)
        nc = re.sub(r"_+", "_", nc).strip("_")
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}_dup{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    df.columns = new_cols
    return df

def extract_quantile_label(val):
    if pd.isna(val): return np.nan
    m = re.match(r"Q(\d+)", str(val).strip())
    if m:
        try: return int(m.group(1))
        except Exception: return np.nan
    return np.nan

def mixed_text_to_tokens(val: str) -> List[str]:
    if pd.isna(val): return []
    parts = [p.strip() for p in str(val).split(",")]
    tokens = []
    for p in parts:
        t = re.sub(r"\(.*?\)", "", p).strip()
        if t: tokens.append(t)
    return tokens

def make_mixed_text_indicators(series: pd.Series, top_k: int = 15) -> pd.DataFrame:
    all_tokens = []
    for v in series.fillna("").tolist():
        all_tokens.extend(mixed_text_to_tokens(v))
    if not all_tokens:
        return pd.DataFrame(index=series.index)
    vc = pd.Series(all_tokens).value_counts()
    top = vc.index[:top_k].tolist()
    out = pd.DataFrame(index=series.index)
    for token in top:
        out[f"mix_{re.sub(r'[^A-Za-z0-9_]+','_',token)}"] = series.fillna("").apply(
            lambda s: 1 if token in mixed_text_to_tokens(s) else 0
        )
    return out

def coerce_numeric(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c in exclude_cols:
            categorical_cols.append(c)
            continue
        if df[c].dtype == object:
            qvals = df[c].map(extract_quantile_label)
            if qvals.notna().mean() > 0.6:
                df[c] = qvals
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.8:
                df[c] = conv; numeric_cols.append(c)
            else:
                categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    return df, numeric_cols, categorical_cols

def pick_model():
    if HAS_XGB:
        return "xgb", XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", objective="binary:logistic",
            eval_metric="auc", n_jobs=-1
        )
    if HAS_LGBM:
        return "lgbm", LGBMClassifier(
            n_estimators=1000, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1
        )
    return "rf", RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42
    )

def compute_class_weight(y: pd.Series) -> float:
    if isinstance(y, pd.DataFrame): y = y.iloc[:,0]
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).clip(0,1)
    pos = int((y==1).sum()); neg = int((y==0).sum())
    return float(neg)/float(pos) if pos>0 else 1.0

def plot_and_save_importance(importances: pd.Series, out_png: Path, top_n: int = 30, title: str = "Feature Importance"):
    top = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, int(0.25*top_n))))
    top.iloc[::-1].plot(kind="barh")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def log_shape(df, label):
    print(f"📊 {label}: {df.shape[0]} satır × {df.shape[1]} sütun")


# ------------------------- EDA (fr_* için) -------------------------

def run_eda_only(df_raw: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = sanitize_columns(df_raw.copy())
    log_shape(df, "EDA (ham)")

    # temel özetler
    df.describe(include="all").to_csv(outdir / "eda_describe.csv")

    # NA oranları
    na_rates = df.isna().mean().sort_values(ascending=False)
    na_rates.to_csv(outdir / "eda_missing_rates.csv")
    plt.figure(figsize=(8, max(4, int(0.25 * min(25, len(na_rates))))))
    na_rates.head(25).iloc[::-1].plot(kind="barh")
    plt.title("Eksik Değer Oranları (İlk 25)")
    plt.tight_layout(); plt.savefig(outdir / "eda_missing_top25.png", dpi=200); plt.close()

    # sayısal sütunlar için korelasyon (ilk 40)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        corr.to_csv(outdir / "eda_corr_numeric.csv")

    # kategorik en çok tekrar edenler (ilk 15 sütun)
    obj_cols = [c for c in df.columns if df[c].dtype==object][:15]
    cat_summary = {}
    for c in obj_cols:
        vc = df[c].value_counts(dropna=False).head(20)
        cat_summary[c] = vc.to_dict()
    with open(outdir / "eda_top_categories.json", "w", encoding="utf-8") as f:
        json.dump(cat_summary, f, ensure_ascii=False, indent=2)

    # sayısal histogramlar (ilk 20)
    for c in num_cols[:20]:
        try:
            plt.figure()
            df[c].dropna().plot(kind="hist", bins=40)
            plt.title(f"Histogram: {c}")
            plt.tight_layout(); plt.savefig(outdir / f"eda_hist_{c}.png", dpi=200); plt.close()
        except Exception:
            pass

    print("✅ EDA bitti (fr_*). Çıktılar:", outdir.resolve())


# ------------------------- Modelleme (sf_* için) -------------------------

def run_model_pipeline(df_raw: pd.DataFrame, target: str, outdir: Path, group_by_geoid: bool, csv_path: str):
    outdir.mkdir(parents=True, exist_ok=True)
    original_cols = df_raw.columns.tolist()
    df = sanitize_columns(df_raw.copy())
    if target not in df.columns:
        raise ValueError(f"Hedef sütun '{target}' bulunamadı.")

    # sızıntıları/sorunlu alanları düş
    drop_exact = [c for c in ["date","time","datetime"] if c in df.columns]
    geo_keys = [c for c in df.columns if c.lower() in {
        "geoid","geo_id","tract","tract_geoid","geoid10","blockgroup","block","census_block","block_group"
    }]
    drop_exact += geo_keys
    id_like = [c for c in df.columns if re.fullmatch(r"(id|incident_id|case_id|row_id|index)", c, flags=re.I)]
    drop_exact += id_like
    coord_cols = [c for c in df.columns if re.search(r"^(latitude|longitude|centroid_lat|centroid_lon)", c, flags=re.I)]
    drop_exact += coord_cols
    leakage_cols = [c for c in df.columns if c.startswith(target) and c != target]
    drop_exact += leakage_cols

    # karma metinlerden dummy
    text_like_cols = [c for c in ["crime_mix","crime_mix_x","crime_mix_y"] if c in df.columns]
    mix_df_list = []
    for col in text_like_cols:
        md = make_mixed_text_indicators(df[col], top_k=15)
        if md.shape[1] > 0: mix_df_list.append(md.add_prefix(f"{col}_"))
    if mix_df_list:
        mix_block = pd.concat(mix_df_list, axis=1)
        df = pd.concat([df, mix_block], axis=1)

    to_drop = set(drop_exact + text_like_cols)
    keep_cols = [c for c in df.columns if c not in to_drop]
    df = df[keep_cols + [target]]

    # hedef
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).clip(0,1)
    X = df.drop(columns=[target]).copy()

    # tür ayrımı
    force_categorical = [c for c in ["category","subcategory","season","season_x","season_y",
                                     "day_of_week","day_of_week_x","day_of_week_y",
                                     "hour_range","hour_range_x","hour_range_y"] if c in X.columns]
    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)

    for col in X.columns:
        if X[col].dtype == object and set(map(str, X[col].dropna().unique())) <= {"0","1","0.0","1.0"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # split
    groups = None
    if group_by_geoid:
        # orijinal dosyada geoid benzeri bir kolon varsa grupla
        _df_geoid = None
        geo_candidates = [c for c in original_cols if str(c).lower() in {"geoid","geo_id","tract","tract_geoid","geoid10"}]
        if geo_candidates:
            try:
                _df_geoid = pd.read_csv(csv_path, usecols=geo_candidates, low_memory=False)
            except Exception:
                _df_geoid = None
        if _df_geoid is not None and len(_df_geoid)==len(X):
            groups = _df_geoid[geo_candidates[0]].astype(str)

    if group_by_geoid and groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test, y_train, y_test = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42, stratify=y)

    # pipeline
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    pre = ColumnTransformer([("num", num_pipe, [c for c in numeric_cols if c in X.columns]),
                             ("cat", cat_pipe, [c for c in categorical_cols if c in X.columns])],
                            remainder="drop")
    model_name, base_model = pick_model()
    spw = compute_class_weight(y_train)
    if model_name=="xgb":
        base_model.set_params(scale_pos_weight=spw)
    clf = Pipeline([("pre", pre), ("model", base_model)])
    clf.fit(X_train, y_train)

    # değerlendirme
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf.named_steps["model"], "predict_proba") else clf.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba.ndim==1 else np.nan
    pr_auc  = average_precision_score(y_test, y_proba) if y_proba.ndim==1 else np.nan
    f1_def  = f1_score(y_test, y_pred)
    p_def, r_def, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    # en iyi eşik (F1)
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1, best_th, best_p, best_r = -1, 0.5, None, None
    for th in thresholds:
        y_hat = (y_proba >= th).astype(int)
        f1_tmp = f1_score(y_test, y_hat)
        if f1_tmp > best_f1:
            best_f1, best_th = f1_tmp, th
            best_p, best_r, _, _ = precision_recall_fscore_support(y_test, y_hat, average="binary", zero_division=0)

    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc) if roc_auc==roc_auc else None,
        "pr_auc": float(pr_auc) if pr_auc==pr_auc else None,
        "f1_default_0.50": float(f1_def),
        "precision_default_0.50": float(p_def),
        "recall_default_0.50": float(r_def),
        "best_threshold": float(best_th),
        "best_f1_at_threshold": float(best_f1),
        "best_precision_at_threshold": float(best_p) if best_p is not None else None,
        "best_recall_at_threshold": float(best_r) if best_r is not None else None,
        "scale_pos_weight": float(spw),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "group_split_geoid": bool(group_by_geoid and (groups is not None))
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # feature names
    try:
        pre_fit = clf.named_steps["pre"]
        ohe = pre_fit.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre_fit.transformers_[0][2]
        cat_cols_final = pre_fit.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names = list(num_cols_final) + ohe_names
    except Exception:
        feature_names = [f"f{i}" for i in range(clf.named_steps["pre"].transform(X_train).shape[1])]

    # model-based importance
    importances = None
    try:
        booster = clf.named_steps["model"]
        if hasattr(booster, "get_booster"):  # xgb
            try:
                raw = booster.get_booster().get_score(importance_type="gain")
                imp = pd.Series({int(k[1:]): v for k,v in raw.items()})
                imp = imp.reindex(range(len(feature_names))).fillna(0.0)
                importances = pd.Series(imp.values, index=feature_names)
            except Exception:
                importances = pd.Series(getattr(booster, "feature_importances_", np.zeros(len(feature_names))),
                                        index=feature_names)
        elif hasattr(booster, "feature_importances_"):
            importances = pd.Series(booster.feature_importances_, index=feature_names)
    except Exception:
        importances = None

    if importances is not None:
        importances.sort_values(ascending=False).to_csv(outdir / "feature_importance_model.csv")
        plot_and_save_importance(importances, outdir / "feature_importance_model.png", 30,
                                 title=f"Model Importance ({metrics['model']})")

    # permutation
    try:
        pi = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        pi_series = pd.Series(pi.importances_mean, index=feature_names).sort_values(ascending=False)
        pi_series.to_csv(outdir / "feature_importance_permutation.csv")
        plot_and_save_importance(pi_series, outdir / "feature_importance_permutation.png", 30,
                                 title="Permutation Importance")
    except Exception:
        pass

    # SHAP (opsiyonel)
    if HAS_SHAP and metrics["model"] in ("xgb","lgbm"):
        try:
            Xt = clf.named_steps["pre"].transform(X_train)
            Xs = Xt.toarray() if hasattr(Xt,"toarray") else Xt
            explainer = shap.TreeExplainer(clf.named_steps["model"])
            shap_values = explainer.shap_values(Xs)
            plt.figure()
            shap.summary_plot(shap_values, Xs, feature_names=feature_names, show=False, max_display=30)
            plt.tight_layout(); plt.savefig(outdir / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()

            if isinstance(shap_values, list):
                sv = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                sv = np.mean(np.abs(shap_values), axis=0)
            shap_imp = pd.Series(sv, index=feature_names).sort_values(ascending=False)
            shap_imp.to_csv(outdir / "feature_importance_shap.csv")
            plot_and_save_importance(shap_imp, outdir / "feature_importance_shap.png", 30,
                                     title="SHAP Global Importance")
        except Exception as e:
            with open(outdir / "shap_error.txt","w",encoding="utf-8") as f: f.write(str(e))

    # NA oranları
    na_rates = X.isna().mean().sort_values(ascending=False)
    na_rates.to_csv(outdir / "missing_rates.csv")
    plt.figure(figsize=(8, max(4, int(0.25 * min(25, len(na_rates))))))
    na_rates.head(25).iloc[::-1].plot(kind="barh")
    plt.title("Eksik Değer Oranları (İlk 25)")
    plt.tight_layout(); plt.savefig(outdir / "missing_rates_top25.png", dpi=200); plt.close()

    # kaydet
    joblib.dump(clf, outdir / "model_pipeline.joblib")
    with open(outdir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print("✅ Modelleme bitti (sf_*). Çıktılar:", outdir.resolve())


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Girdi CSV (fr_crime_09.csv veya sf_crime_09.csv)")
    parser.add_argument("--target", type=str, default="Y_label", help="sf için hedef sütun adı (binary 0/1)")
    parser.add_argument("--outdir", type=str, default="outputs_feature_analysis", help="Çıktı klasörü")
    parser.add_argument("--group_by_geoid", action="store_true", help="sf için GEOID gruplu split")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv, low_memory=False)
    fname = Path(args.csv).name.lower()

    # tip tespiti
    has_target = args.target in df_raw.columns
    has_both_classes = (has_target and df_raw[args.target].dropna().nunique() >= 2)

    if fname.startswith("sf_") or (has_target and has_both_classes):
        print("🔎 Veri tipi: sf (grid, Y_label mevcut) → MODELLEME moduna geçiliyor.")
        run_model_pipeline(df_raw, target=args.target, outdir=outdir, group_by_geoid=args.group_by_geoid, csv_path=args.csv)
    else:
        print("🔎 Veri tipi: fr (olay/suç_id bazlı, genelde tek sınıf) → EDA moduna geçiliyor.")
        run_eda_only(df_raw, outdir)

if __name__ == "__main__":
    main()
