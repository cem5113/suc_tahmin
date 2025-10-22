#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_feature_analysis_sf_only.py

Girdi:  sf_crime_08.csv (veya hedefi olan başka bir CSV)
Çıktı:  varsayılan sf_crime_09.csv (veya --out_csv ile belirtilen)

Öz: Y_label hedefiyle denetimli özellik seçimi.
- Model: XGB/LGBM yoksa RF
- Önem: model / permutation / (varsa) SHAP
- OHE alt-kolonlarını ana sütunda toplayıp ileri seçim (F1) ile en iyi seti yazar.
"""

import argparse, json, re, warnings, joblib
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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


# ---------- yardımcılar ----------
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for c in df.columns:
        nc = re.sub(r"[^\w]+", "_", str(c), flags=re.UNICODE)
        nc = re.sub(r"_+", "_", nc).strip("_")
        if nc in seen:
            seen[nc] += 1; nc = f"{nc}_dup{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    df.columns = new_cols
    return df

def extract_quantile_label(val):
    if pd.isna(val): return np.nan
    m = re.match(r"Q(\d+)", str(val).strip()); 
    return int(m.group(1)) if m else np.nan

def coerce_numeric(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c in exclude_cols: categorical_cols.append(c); continue
        if df[c].dtype == object:
            qvals = df[c].map(extract_quantile_label)
            if qvals.notna().mean() > 0.6: df[c] = qvals
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.8: df[c] = conv; numeric_cols.append(c)
            else: categorical_cols.append(c)
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
            n_estimators=1000, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1
        )
    return "rf", RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42
    )

def compute_class_weight(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).clip(0,1)
    pos = int((y==1).sum()); neg = int((y==0).sum())
    return float(neg)/float(pos) if pos>0 else 1.0

def plot_and_save_importance(importances: pd.Series, out_png: Path, top_n: int = 30, title: str = "Feature Importance"):
    top = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, int(0.28*len(top)))))
    top.iloc[::-1].plot(kind="barh")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def collapse_to_base_feature(feature_names_transformed: List[str]) -> Dict[str, List[int]]:
    mapping = {}
    for i, name in enumerate(feature_names_transformed):
        base = name.split("_", 1)[0] if ("_" in name) else name
        mapping.setdefault(base, []).append(i)
    return mapping


def forward_select_by_rank(clf_template: Pipeline, X, y, base_ranked_features: List[str],
                           feature_to_columns: Dict[str, List[int]], metric: str = "f1"):
    def transform_and_select(pre, Xdf, chosen_bases):
        Xt = pre.transform(Xdf); 
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        keep_idx = sorted(set(sum((feature_to_columns[b] for b in chosen_bases), [])))
        return Xt[:, keep_idx]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pre = clf_template.named_steps["pre"]; pre.fit(X_tr)
    model_name, base_model = pick_model()
    if model_name == "xgb": base_model.set_params(scale_pos_weight=compute_class_weight(y_tr))
    chosen, best_set, best_score, curve = [], [], -1, []
    for k, b in enumerate(base_ranked_features, start=1):
        chosen.append(b)
        Xtr_sel = transform_and_select(pre, X_tr, chosen)
        Xva_sel = transform_and_select(pre, X_va, chosen)
        base_model.fit(Xtr_sel, y_tr)
        proba = base_model.predict_proba(Xva_sel)[:,1] if hasattr(base_model,"predict_proba") else base_model.predict(Xva_sel)
        yhat = (proba >= 0.5).astype(int) if proba.ndim==1 else proba
        f1 = f1_score(y_va, yhat)
        auc = roc_auc_score(y_va, proba) if proba.ndim==1 else np.nan
        curve.append({"k": k, "f1": float(f1), "roc_auc": float(auc) if auc==auc else None})
        score = f1
        if score > best_score: best_score, best_set = score, chosen.copy()
    return best_set, {"forward_curve": curve, "best_k": len(best_set), "metric": metric}


# ---------- çekirdek: denetimli seçim ----------
def supervised_feature_selection(df_raw: pd.DataFrame, target: str, outdir: Path,
                                 group_by_geoid: bool, csv_path: str,
                                 desired_output_csv: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    original_cols = df_raw.columns.tolist()
    df = sanitize_columns(df_raw.copy())
    if target not in df.columns: raise ValueError(f"Hedef sütun '{target}' bulunamadı.")

    # sızıntı/id/geo/koordinat
    drop_exact = [c for c in ["date","time","datetime"] if c in df.columns]
    geo_keys = [c for c in df.columns if c.lower() in {"geoid","geo_id","tract","tract_geoid","geoid10","blockgroup","block"}]
    drop_exact += geo_keys
    id_like = [c for c in df.columns if re.fullmatch(r"(id|incident_id|case_id|row_id|index)", c, flags=re.I)]
    drop_exact += id_like
    coord_cols = [c for c in df.columns if re.search(r"^(latitude|longitude|centroid_lat|centroid_lon)", c, flags=re.I)]
    drop_exact += coord_cols
    leakage_cols = [c for c in df.columns if c.startswith(target) and c != target]
    drop_exact += leakage_cols

    keep_cols = [c for c in df.columns if c not in set(drop_exact)]
    df = df[keep_cols + [target]]

    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).clip(0,1)
    X = df.drop(columns=[target]).copy()

    force_categorical = [c for c in ["category","subcategory","season","season_x","season_y",
                                     "day_of_week","day_of_week_x","day_of_week_y",
                                     "hour_range","hour_range_x","hour_range_y"] if c in X.columns]
    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)

    # binary string -> sayısal
    for col in X.columns:
        if X[col].dtype == object and set(map(str, X[col].dropna().unique())) <= {"0","1","0.0","1.0"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # split (opsiyonel GEOID gruplu)
    groups = None
    if group_by_geoid:
        geo_candidates = [c for c in original_cols if str(c).lower() in {"geoid","geo_id","tract","tract_geoid","geoid10"}]
        if geo_candidates:
            try:
                _df_geoid = pd.read_csv(csv_path, usecols=geo_candidates, low_memory=False)
                if len(_df_geoid)==len(X): groups = _df_geoid[geo_candidates[0]].astype(str)
            except Exception:
                pass
    if group_by_geoid and (groups is not None):
        tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=groups))
        X_train, X_test, y_train, y_test = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
    if model_name=="xgb": base_model.set_params(scale_pos_weight=spw)
    clf = Pipeline([("pre", pre), ("model", base_model)])
    clf.fit(X_train, y_train)

    # metrikler
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(base_model, "predict_proba") else clf.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int) if y_proba.ndim==1 else y_proba
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba.ndim==1 else np.nan
    f1_def  = f1_score(y_test, y_pred)
    p_def, r_def, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    # --------- önemler ----------
    try:
        pre_fit = clf.named_steps["pre"]
        ohe = pre_fit.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre_fit.transformers_[0][2]
        cat_cols_final = pre_fit.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names_transformed = list(num_cols_final) + ohe_names
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(clf.named_steps["pre"].transform(X_train).shape[1])]

    importances_model = None
    booster = clf.named_steps["model"]
    try:
        if hasattr(booster, "get_booster"):  # xgb
            raw = booster.get_booster().get_score(importance_type="gain")
            imp = pd.Series({int(k[1:]): v for k,v in raw.items()})
            imp = imp.reindex(range(len(feature_names_transformed))).fillna(0.0)
            importances_model = pd.Series(imp.values, index=feature_names_transformed)
        elif hasattr(booster, "feature_importances_"):
            importances_model = pd.Series(booster.feature_importances_, index=feature_names_transformed)
    except Exception:
        pass

    importances_perm = None
    try:
        pi = permutation_importance(clf, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
        importances_perm = pd.Series(pi.importances_mean, index=feature_names_transformed)
    except Exception:
        pass

    importances_shap = None
    if HAS_SHAP and model_name in ("xgb","lgbm"):
        try:
            Xt = clf.named_steps["pre"].transform(X_train)
            Xs = Xt.toarray() if hasattr(Xt,"toarray") else Xt
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(Xs)
            sv = np.mean(np.abs(shap_values[0]), axis=0) if isinstance(shap_values, list) else np.mean(np.abs(shap_values), axis=0)
            importances_shap = pd.Series(sv, index=feature_names_transformed)
            plt.figure()
            shap.summary_plot(shap_values, Xs, feature_names=feature_names_transformed, show=False, max_display=30)
            plt.tight_layout(); plt.savefig(outdir / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
        except Exception as e:
            with open(outdir / "shap_error.txt","w",encoding="utf-8") as f: f.write(str(e))

    sources = []
    for name, ser in [("model", importances_model), ("perm", importances_perm), ("shap", importances_shap)]:
        if ser is None: continue
        ser = ser.clip(lower=0)
        if ser.sum() == 0: continue
        (ser / ser.sum()).sort_values(ascending=False).to_csv(outdir / f"feature_importance_{name}.csv")
        plot_and_save_importance(ser, outdir / f"feature_importance_{name}.png", 30, f"{name} importance")
        s_norm = ser / ser.sum(); s_norm.name = name; sources.append(s_norm)
    if not sources: raise RuntimeError("Özellik önemi hesaplanamadı.")

    df_imp = pd.concat(sources, axis=1).fillna(0.0)
    df_imp["ensemble"] = df_imp.mean(axis=1)
    df_imp.sort_values("ensemble", ascending=False).to_csv(outdir / "feature_importance_ensemble_transformed.csv")

    base_map = collapse_to_base_feature(df_imp.index.tolist())
    base_scores = {b: float(df_imp.iloc[[i for i,_ in enumerate(df_imp.index) if i in idxs]]["ensemble"].sum())
                   for b, idxs in base_map.items()}
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_ensemble_base.csv")

    template = Pipeline([("pre", pre), ("model", pick_model()[1])])
    best_bases, forward_info = forward_select_by_rank(template, X, y, base_rank.index.tolist(), base_map, metric="f1")
    with open(outdir / "forward_selection.json","w",encoding="utf-8") as f: json.dump(forward_info, f, ensure_ascii=False, indent=2)

    selected_cols = [c for c in best_bases if c in X.columns]
    pd.concat([X[selected_cols], y.rename(target)], axis=1).to_csv(desired_output_csv, index=False)
    with open(outdir / "selected_columns.json","w",encoding="utf-8") as f: json.dump(selected_cols, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": model_name, "roc_auc": float(roc_auc) if roc_auc==roc_auc else None,
        "f1_default_0.50": float(f1_def), "precision_default_0.50": float(p_def), "recall_default_0.50": float(r_def),
        "scale_pos_weight": float(spw), "n_selected_features": len(selected_cols),
        "output_csv": str(desired_output_csv)
    }
    with open(outdir / "metrics.json","w",encoding="utf-8") as f: json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(clf, outdir / "model_pipeline.joblib")
    print(f"✅ Özellik seçimi bitti. Seçilmiş veri → {desired_output_csv}")


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Girdi CSV (örn. sf_crime_08.csv)")
    p.add_argument("--target", type=str, default="Y_label", help="Hedef sütun adı (vars: Y_label)")
    p.add_argument("--outdir", type=str, default="outputs_feature_analysis", help="Çıktı klasörü")
    p.add_argument("--group_by_geoid", action="store_true", help="GEOID gruplu split")
    p.add_argument("--out_csv", type=str, default="", help="Çıktı CSV yolu (vars: sf_crime_09.csv)")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(args.csv, low_memory=False)
    fname = Path(args.csv).name.lower()

    # otomatik çıktı
    desired_out = Path(args.out_csv) if args.out_csv else (
        Path(args.csv).with_name("sf_crime_09.csv") if fname.startswith("sf_") else Path(args.csv).with_name("selected_output.csv")
    )

    if args.target not in df_raw.columns or df_raw[args.target].dropna().nunique() < 2:
        raise ValueError("Hedef (Y_label) yok ya da en az 2 sınıf içermiyor. Bu sf (denetimli) sürüm etiket gerektirir.")

    print("🔎 Tip: sf → Denetimli özellik seçimi.")
    supervised_feature_selection(df_raw, target=args.target, outdir=outdir,
                                 group_by_geoid=args.group_by_geoid, csv_path=args.csv,
                                 desired_output_csv=desired_out)

if __name__ == "__main__":
    main()
