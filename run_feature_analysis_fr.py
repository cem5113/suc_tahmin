#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_feature_analysis_fr.py

Girdi : fr_crime_09.csv (etiketsiz) ya da hedefi olan bir CSV
Çıktı : fr_crime_10.csv (veya --out_csv ile belirtilen)

Akış (etiketsiz):
- Preprocess (impute, OHE, scale)
- KMeans k=2..4 -> en iyi silhouette
- RF feature_importances_ (dönüşmüş uzayda)
- OHE alt-kolonlarını ana sütuna toplayıp ileri seçim (proxy=küme)
- Seçilen ana sütunlarla çıktı CSV yazılır

Akış (hedef varsa):
- Basit denetimli: aynı preprocess + RF/XGB/LGBM, önem + ileri seçim (F1)
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

# opsiyonel modeller (denetimli dal için)
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


# ---------------- yardımcılar ----------------
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for c in df.columns:
        nc = re.sub(r"[^\w]+", "_", str(c), flags=re.UNICODE)
        nc = re.sub(r"_+", "_", nc).strip("_")
        if nc in seen: seen[nc]+=1; nc=f"{nc}_dup{seen[nc]}"
        else: seen[nc]=0
        new_cols.append(nc)
    df.columns = new_cols
    return df

def extract_quantile_label(val):
    if pd.isna(val): return np.nan
    m = re.match(r"Q(\d+)", str(val).strip())
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
            n_estimators=500, max_depth=6, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", objective="binary:logistic",
            eval_metric="auc", n_jobs=-1
        )
    if HAS_LGBM:
        return "lgbm", LGBMClassifier(
            n_estimators=800, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1
        )
    return "rf", RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42
    )

def compute_class_weight(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).clip(0,1)
    pos, neg = int((y==1).sum()), int((y==0).sum())
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


# ---------------- çekirdek: etiketsiz ----------------
def unsupervised_feature_selection(df_raw: pd.DataFrame, outdir: Path, desired_output_csv: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = sanitize_columns(df_raw.copy())

    # sızıntı/id/geo/koordinat kaldır
    drop_exact = [c for c in ["date","time","datetime"] if c in df.columns]
    drop_exact += [c for c in df.columns if re.fullmatch(r"(id|incident_id|case_id|row_id|index)", c, flags=re.I)]
    drop_exact += [c for c in df.columns if c.lower() in {"geoid","geo_id","tract","tract_geoid","geoid10","blockgroup","block"}]
    drop_exact += [c for c in df.columns if re.search(r"^(latitude|longitude|centroid_lat|centroid_lon)", c, flags=re.I)]
    keep_cols = [c for c in df.columns if c not in set(drop_exact)]
    X = df[keep_cols].copy()

    # tür ayrımı
    force_categorical = [c for c in ["category","subcategory","season","day_of_week","hour_range",
                                     "season_x","season_y","day_of_week_x","day_of_week_y",
                                     "hour_range_x","hour_range_y"] if c in X.columns]
    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)

    # pipeline (yalnız preprocess)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    pre = ColumnTransformer([("num", num_pipe, [c for c in numeric_cols if c in X.columns]),
                             ("cat", cat_pipe, [c for c in categorical_cols if c in X.columns])],
                            remainder="drop")
    Xt = pre.fit_transform(X)
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    # KMeans k=2..4, silhouette ile seçim
    best_k, best_sil, best_labels = 2, -1e9, None
    for k in [2,3,4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xt_dense)
        sil = silhouette_score(Xt_dense, labels) if k>1 else -1e9
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels

    # RF ile önem (dönüşmüş ad uzayında)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(Xt_dense, best_labels)

    # Dönüşmüş isimleri çıkar
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre.transformers_[0][2]
        cat_cols_final = pre.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names_transformed = list(num_cols_final) + ohe_names
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(Xt_dense.shape[1])]

    imp = pd.Series(getattr(rf, "feature_importances_", np.zeros(len(feature_names_transformed))),
                    index=feature_names_transformed).clip(lower=0.0)
    if imp.sum() == 0:
        raise RuntimeError("RF önemleri hesaplanamadı.")
    imp_norm = (imp / imp.sum()).sort_values(ascending=False)
    imp_norm.to_csv(outdir / "feature_importance_unsupervised_transformed.csv")
    plot_and_save_importance(imp_norm, outdir / "feature_importance_unsupervised.png", 30, "Unsupervised RF Importance")

    # OHE -> ana sütun
    base_map = collapse_to_base_feature(imp_norm.index.tolist())
    base_scores = {b: float(imp_norm.iloc[[i for i,_ in enumerate(imp_norm.index) if i in idxs]].sum())
                   for b, idxs in base_map.items()}
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_unsupervised_base.csv")

    # İleri seçim (proxy hedef = best_labels), sınıflandırıcı = pick_model()
    model_name, base_model = pick_model()
    chosen, best_set, best_score, curve = [], [], -1, []
    X_tr, X_va, y_tr, y_va = train_test_split(X, best_labels, test_size=0.25, random_state=42, stratify=best_labels)

    def transform_and_select(pre, Xdf, chosen_bases):
        Xt = pre.transform(Xdf)
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        keep_idx = []
        for b in chosen_bases: keep_idx.extend(base_map[b])
        keep_idx = sorted(set(keep_idx))
        return Xt[:, keep_idx]

    pre.fit(X_tr)
    for k, base_feat in enumerate(base_rank.index.tolist(), start=1):
        chosen.append(base_feat)
        Xtr_sel = transform_and_select(pre, X_tr, chosen)
        Xva_sel = transform_and_select(pre, X_va, chosen)
        base_model.fit(Xtr_sel, y_tr)
        score = base_model.score(Xva_sel, y_va)  # proxy doğruluk
        curve.append({"k": k, "acc_proxy": float(score)})
        if score > best_score:
            best_score, best_set = score, chosen.copy()

    with open(outdir / "forward_selection_unsupervised.json","w",encoding="utf-8") as f:
        json.dump({"forward_curve": curve, "best_k": len(best_set)}, f, ensure_ascii=False, indent=2)

    selected_cols = [c for c in best_set if c in X.columns]
    X[selected_cols].to_csv(desired_output_csv, index=False)
    with open(outdir / "selected_columns.json","w",encoding="utf-8") as f:
        json.dump(selected_cols, f, ensure_ascii=False, indent=2)

    meta = {"kmeans_k": int(best_k), "silhouette": float(best_sil),
            "n_selected_features": len(selected_cols), "output_csv": str(desired_output_csv)}
    with open(outdir / "metrics_unsupervised.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ (Unsupervised) Özellik seçimi bitti. Seçilmiş veri → {desired_output_csv}")


# ---------------- opsiyonel: hedef varsa denetimli ----------------
def supervised_if_target(df_raw: pd.DataFrame, target: str, outdir: Path, desired_output_csv: Path) -> bool:
    if target not in df_raw.columns or df_raw[target].dropna().nunique() < 2:
        return False

    df = sanitize_columns(df_raw.copy())
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).clip(0,1)
    X = df.drop(columns=[target]).copy()

    # sızıntı/id/geo/koordinat
    drop_exact = [c for c in ["date","time","datetime"] if c in X.columns]
    drop_exact += [c for c in X.columns if re.fullmatch(r"(id|incident_id|case_id|row_id|index)", c, flags=re.I)]
    drop_exact += [c for c in X.columns if c.lower() in {"geoid","geo_id","tract","tract_geoid","geoid10","blockgroup","block"}]
    drop_exact += [c for c in X.columns if re.search(r"^(latitude|longitude|centroid_lat|centroid_lon)", c, flags=re.I)]
    X = X[[c for c in X.columns if c not in set(drop_exact)]].copy()

    force_categorical = [c for c in ["category","subcategory","season","day_of_week","hour_range",
                                     "season_x","season_y","day_of_week_x","day_of_week_y",
                                     "hour_range_x","hour_range_y"] if c in X.columns]
    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)
    for col in X.columns:
        if X[col].dtype == object and set(map(str, X[col].dropna().unique())) <= {"0","1","0.0","1.0"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]),
         [c for c in numeric_cols if c in X.columns]),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]),
         [c for c in categorical_cols if c in X.columns])
    ], remainder="drop")

    model_name, base_model = pick_model()
    if model_name == "xgb":
        base_model.set_params(scale_pos_weight=compute_class_weight(y_tr))

    clf = Pipeline([("pre", pre), ("model", base_model)])
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:,1] if hasattr(base_model, "predict_proba") else clf.predict(X_te)
    yhat = (proba >= 0.5).astype(int) if proba.ndim==1 else proba
    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_te, proba)) if proba.ndim==1 else None,
        "f1_default_0.50": float(f1_score(y_te, yhat)),
        "precision_default_0.50": float(precision_recall_fscore_support(y_te, yhat, average="binary", zero_division=0)[0]),
        "recall_default_0.50": float(precision_recall_fscore_support(y_te, yhat, average="binary", zero_division=0)[1])
    }
    with open(outdir / "metrics_supervised.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # önem isimleri
    try:
        pre_fit = clf.named_steps["pre"]
        ohe = pre_fit.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre_fit.transformers_[0][2]
        cat_cols_final = pre_fit.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names_transformed = list(num_cols_final) + ohe_names
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(clf.named_steps["pre"].transform(X_tr).shape[1])]

    imp = None
    model = clf.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names_transformed).clip(lower=0.0)
    elif hasattr(model, "get_booster"):
        raw = model.get_booster().get_score(importance_type="gain")
        ser = pd.Series({int(k[1:]): v for k,v in raw.items()})
        ser = ser.reindex(range(len(feature_names_transformed))).fillna(0.0)
        imp = pd.Series(ser.values, index=feature_names_transformed)
    if imp is None or imp.sum()==0:
        joblib.dump(clf, outdir / "model_pipeline.joblib")
        return True

    imp_norm = (imp/imp.sum()).sort_values(ascending=False)
    imp_norm.to_csv(outdir / "feature_importance_supervised_transformed.csv")
    plot_and_save_importance(imp_norm, outdir / "feature_importance_supervised.png", 30, "Supervised Importance")

    base_map = collapse_to_base_feature(imp_norm.index.tolist())
    base_scores = {b: float(imp_norm.iloc[[i for i,_ in enumerate(imp_norm.index) if i in idxs]].sum())
                   for b, idxs in base_map.items()}
    base_rank = pd.Series(base_scores).sort_values(ascending=False)

    # ileri seçim (F1)
    template_model = pick_model()[1]
    chosen, best_set, best_score = [], [], -1
    def transform_and_select(pre, Xdf, chosen_bases):
        Xt = pre.transform(Xdf);  Xt = Xt.toarray() if hasattr(Xt,"toarray") else Xt
        keep_idx = sorted(set(sum((base_map[b] for b in chosen_bases), [])))
        return Xt[:, keep_idx]
    pre.fit(X_tr)
    for b in base_rank.index.tolist():
        chosen.append(b)
        Xtr_sel = transform_and_select(pre, X_tr, chosen)
        Xte_sel = transform_and_select(pre, X_te, chosen)
        template_model.fit(Xtr_sel, y_tr)
        p = template_model.predict_proba(Xte_sel)[:,1] if hasattr(template_model,"predict_proba") else template_model.predict(Xte_sel)
        yhat = (p >= 0.5).astype(int) if p.ndim==1 else p
        f1 = f1_score(y_te, yhat)
        if f1 > best_score: best_score, best_set = f1, chosen.copy()

    selected_cols = [c for c in best_set if c in X.columns]
    pd.concat([X[selected_cols], y.rename(target)], axis=1).to_csv(desired_output_csv, index=False)
    with open(outdir / "selected_columns.json","w",encoding="utf-8") as f:
        json.dump(selected_cols, f, ensure_ascii=False, indent=2)

    joblib.dump(clf, outdir / "model_pipeline.joblib")
    print(f"✅ (Supervised) Özellik seçimi bitti. Seçilmiş veri → {desired_output_csv}")
    return True


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Girdi CSV (vars: fr_crime_09.csv)")
    p.add_argument("--target", type=str, default="Y_label", help="Hedef sütun adı (varsa)")
    p.add_argument("--outdir", type=str, default="outputs_feature_analysis_fr", help="Çıktı klasörü")
    p.add_argument("--out_csv", type=str, default="", help="Çıktı CSV yolu (vars: fr_crime_10.csv)")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(args.csv, low_memory=False)
    fname = Path(args.csv).name.lower()

    desired_out = Path(args.out_csv) if args.out_csv else (
        Path(args.csv).with_name("fr_crime_10.csv") if fname.startswith("fr_") else Path(args.csv).with_name("selected_output.csv")
    )

    # önce hedef var mı diye bak; varsa denetimli yoldan ilerle
    if supervised_if_target(df_raw, target=args.target, outdir=outdir, desired_output_csv=desired_out):
        return

    print("🔎 Tip: fr (etiketsiz) → Denetimsiz özellik seçimi.")
    unsupervised_feature_selection(df_raw, outdir=outdir, desired_output_csv=desired_out)

if __name__ == "__main__":
    main()
