#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_feature_analysis.py  (revized)

- sf_crime_08.csv  -> özellik seçimi -> sf_crime_09.csv
- fr_crime_09.csv  -> özellik seçimi -> fr_crime_10.csv

Özet:
- sf (etiketli): XGB/LGBM/RF + (Permutation + SHAP*varsa) önemleri, ileri seçimle en iyi AUC/F1 seti.
- fr (etiketsiz): KMeans(2..4) -> RF önemleri, ileri seçim (proxy hedef = küme). Eğer hedef varsa denetimli yol.
- One-hot önemleri ana sütuna toplanır; sızıntı/id/koordinat düşülür.
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support,
    silhouette_score
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

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
    original_cols = df.columns.tolist()
    new_cols, seen = [], {}
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
    """
    OHE isimlerini ana özellik ismine indirger ve index listesini tutar.
    Örn: 'day_of_week_Mon' -> 'day_of_week'
    """
    mapping = {}
    for i, name in enumerate(feature_names_transformed):
        base = name.split("_", 1)[0] if ("_" in name) else name
        mapping.setdefault(base, []).append(i)
    return mapping

def forward_select_by_rank(clf_template: Pipeline, X, y, base_ranked_features: List[str],
                           feature_to_columns: Dict[str, List[int]],
                           feature_names_transformed: List[str],
                           metric: str = "f1") -> Tuple[List[str], Dict]:
    """
    Ranke göre ileri seçim: her adımda bir ana özellik eklenir (OHE kolonlarının tamamı).
    En iyi metrik değeri veren k noktası seçilir.
    """
    # küçük yardımcı: dönüştürülmüş matriste sadece seçili base feature kolonlarını bırak
    def transform_and_select(pre, Xdf, chosen_bases):
        Xt = pre.transform(Xdf)
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        keep_idx = []
        for b in chosen_bases:
            keep_idx.extend(feature_to_columns[b])
        keep_idx = sorted(set(keep_idx))
        return Xt[:, keep_idx], keep_idx

    # eğitim/valid ayrımı (stratify)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pre = clf_template.named_steps["pre"]
    model_name, base_model = pick_model()
    spw = compute_class_weight(y_tr)
    if model_name == "xgb":
        base_model.set_params(scale_pos_weight=spw)
    clf_core = base_model

    best_score, best_set, best_k = -1, [], 0
    curve = []

    chosen = []
    for k, base_feat in enumerate(base_ranked_features, start=1):
        chosen.append(base_feat)
        # pre-fit (sadece preprocessor'u tüm kolonlara fitliyoruz, selection transform sonrası)
        if k == 1:
            pre.fit(X_tr)
        Xtr_sel, idx_tr = transform_and_select(pre, X_tr, chosen)
        Xva_sel, idx_va = transform_and_select(pre, X_va, chosen)

        clf_core.fit(Xtr_sel, y_tr)
        if hasattr(clf_core, "predict_proba"):
            proba = clf_core.predict_proba(Xva_sel)[:,1]
            auc = roc_auc_score(y_va, proba)
            yhat = (proba >= 0.5).astype(int)
        else:
            yhat = clf_core.predict(Xva_sel)
            auc = np.nan
        f1 = f1_score(y_va, yhat)
        pr = precision_recall_fscore_support(y_va, yhat, average="binary", zero_division=0)
        score = f1 if metric=="f1" else (auc if not np.isnan(auc) else f1)
        curve.append({"k": k, "f1": float(f1), "roc_auc": float(auc) if auc==auc else None,
                      "precision": float(pr[0]), "recall": float(pr[1])})

        if score > best_score:
            best_score, best_set, best_k = score, chosen.copy(), k

    return best_set, {"forward_curve": curve, "best_k": best_k, "metric": metric}


# ---------- çekirdek: modelleme + önem birleştirme ----------
def supervised_feature_selection(df_raw: pd.DataFrame, target: str, outdir: Path,
                                 group_by_geoid: bool, csv_path: str,
                                 desired_output_csv: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    original_cols = df_raw.columns.tolist()
    df = sanitize_columns(df_raw.copy())
    if target not in df.columns:
        raise ValueError(f"Hedef sütun '{target}' bulunamadı.")

    # sızıntı/id/geo/koordinat
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

    keep_cols = [c for c in df.columns if c not in set(drop_exact)]
    df = df[keep_cols + [target]]

    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).clip(0,1)
    X = df.drop(columns=[target]).copy()

    force_categorical = [c for c in ["category","subcategory","season","season_x","season_y",
                                     "day_of_week","day_of_week_x","day_of_week_y",
                                     "hour_range","hour_range_x","hour_range_y"] if c in X.columns]
    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)

    # binary stringleri sayıya çevir
    for col in X.columns:
        if X[col].dtype == object and set(map(str, X[col].dropna().unique())) <= {"0","1","0.0","1.0"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # split (isteğe bağlı GEOID gruplu)
    groups = None
    if group_by_geoid:
        geo_candidates = [c for c in original_cols if str(c).lower() in {"geoid","geo_id","tract","tract_geoid","geoid10"}]
        if geo_candidates:
            try:
                _df_geoid = pd.read_csv(csv_path, usecols=geo_candidates, low_memory=False)
                if len(_df_geoid)==len(X):
                    groups = _df_geoid[geo_candidates[0]].astype(str)
            except Exception:
                pass

    if group_by_geoid and (groups is not None):
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

    # temel metrikler
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(base_model, "predict_proba") else clf.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba.ndim==1 else np.nan
    f1_def  = f1_score(y_test, y_pred)
    p_def, r_def, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    # ------------- önemleri çıkar -------------
    # 1) Dönüşmüş feature isimleri
    try:
        pre_fit = clf.named_steps["pre"]
        ohe = pre_fit.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre_fit.transformers_[0][2]
        cat_cols_final = pre_fit.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names_transformed = list(num_cols_final) + ohe_names
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(clf.named_steps["pre"].transform(X_train).shape[1])]

    # 2) Model internal importance
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

    # 3) Permutation
    importances_perm = None
    try:
        pi = permutation_importance(clf, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
        importances_perm = pd.Series(pi.importances_mean, index=feature_names_transformed)
    except Exception:
        pass

    # 4) SHAP (opsiyonel)
    importances_shap = None
    if HAS_SHAP and model_name in ("xgb","lgbm"):
        try:
            Xt = clf.named_steps["pre"].transform(X_train)
            Xs = Xt.toarray() if hasattr(Xt,"toarray") else Xt
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(Xs)
            if isinstance(shap_values, list):
                sv = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                sv = np.mean(np.abs(shap_values), axis=0)
            importances_shap = pd.Series(sv, index=feature_names_transformed)
            plt.figure()
            shap.summary_plot(shap_values, Xs, feature_names=feature_names_transformed, show=False, max_display=30)
            plt.tight_layout(); plt.savefig(outdir / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
        except Exception as e:
            with open(outdir / "shap_error.txt","w",encoding="utf-8") as f: f.write(str(e))

    # 5) normalize & ensemble
    sources = []
    for s in [("model", importances_model), ("perm", importances_perm), ("shap", importances_shap)]:
        name, ser = s
        if ser is None: continue
        ser = ser.clip(lower=0)
        if ser.sum() == 0: continue
        ser_norm = ser / ser.sum()
        ser_norm.name = name
        sources.append(ser_norm)
        ser.sort_values(ascending=False).to_csv(outdir / f"feature_importance_{name}.csv")
        plot_and_save_importance(ser, outdir / f"feature_importance_{name}.png", 30, f"{name} importance")

    if not sources:
        raise RuntimeError("Özellik önemi hesaplanamadı.")

    df_imp = pd.concat(sources, axis=1).fillna(0.0)
    df_imp["ensemble"] = df_imp.mean(axis=1)
    df_imp.sort_values("ensemble", ascending=False).to_csv(outdir / "feature_importance_ensemble_transformed.csv")

    # 6) OHE'den ana sütuna toplama
    base_map = collapse_to_base_feature(df_imp.index.tolist())
    base_scores = {}
    for base, idxs in base_map.items():
        base_scores[base] = float(df_imp.iloc[[i for i,_ in enumerate(df_imp.index) if i in idxs]]["ensemble"].sum())
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_ensemble_base.csv")

    # 7) Ranke göre ileri seçim (best k)
    template = Pipeline([("pre", pre), ("model", pick_model()[1])])
    best_bases, forward_info = forward_select_by_rank(
        template, X, y, base_rank.index.tolist(), base_map, df_imp.index.tolist(), metric="f1"
    )
    with open(outdir / "forward_selection.json","w",encoding="utf-8") as f:
        json.dump(forward_info, f, ensure_ascii=False, indent=2)

    # 8) Seçilmiş sütunlarla final CSV
    selected_cols = [c for c in best_bases if c in X.columns]  # ana sütun isimleri
    df_selected = pd.concat([X[selected_cols], y.rename(target)], axis=1)
    df_selected.to_csv(desired_output_csv, index=False)
    with open(outdir / "selected_columns.json","w",encoding="utf-8") as f:
        json.dump(selected_cols, f, ensure_ascii=False, indent=2)

    # 9) Metrikler
    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc) if roc_auc==roc_auc else None,
        "f1_default_0.50": float(f1_def),
        "precision_default_0.50": float(p_def),
        "recall_default_0.50": float(r_def),
        "scale_pos_weight": float(spw),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "group_split_geoid": bool(group_by_geoid and (groups is not None)),
        "n_selected_features": len(selected_cols),
        "output_csv": str(desired_output_csv)
    }
    with open(outdir / "metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 10) modeli kaydet
    joblib.dump(clf, outdir / "model_pipeline.joblib")
    print(f"✅ Özellik seçimi bitti. Seçilmiş veri → {desired_output_csv}")

def unsupervised_feature_selection(df_raw: pd.DataFrame, outdir: Path,
                                   desired_output_csv: Path,
                                   target_if_exists: str | None = None) -> None:
    """
    Etiket yoksa: KMeans(2..4) ile proxy hedef, RF önemleri + ileri seçim.
    Etiket varsa denetimli yolu kullanır.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = sanitize_columns(df_raw.copy())

    # hedef varsa denetimli
    if target_if_exists and target_if_exists in df.columns and df[target_if_exists].nunique() >= 2:
        print("⚠️ Hedef bulundu (fr). Denetimli yola geçiliyor.")
        return supervised_feature_selection(df_raw, target_if_exists, outdir,
                                            group_by_geoid=False, csv_path="", desired_output_csv=desired_output_csv)

    # sızıntı/id/geo/koordinat kaldır
    drop_exact = [c for c in ["date","time","datetime"] if c in df.columns]
    drop_exact += [c for c in df.columns if re.fullmatch(r"(id|incident_id|case_id|row_id|index)", c, flags=re.I)]
    drop_exact += [c for c in df.columns if c.lower() in {"geoid","geo_id","tract","tract_geoid","geoid10"}]
    drop_exact += [c for c in df.columns if re.search(r"^(latitude|longitude|centroid_lat|centroid_lon)", c, flags=re.I)]
    keep_cols = [c for c in df.columns if c not in set(drop_exact)]
    X = df[keep_cols].copy()

    # tür ayrımı
    force_categorical = [c for c in ["category","subcategory","season","day_of_week","hour_range"] if c in X.columns]
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

    # en iyi küme sayısı (2..4) - silhouette
    best_k, best_sil, best_labels = 2, -1e9, None
    for k in [2,3,4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xt_dense)
        sil = silhouette_score(Xt_dense, labels) if k>1 else -1e9
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels

    # RF ile önem
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(Xt_dense, best_labels)
    # isimler
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre.transformers_[0][2]
        cat_cols_final = pre.transformers_[1][2]
        ohe_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names_transformed = list(num_cols_final) + ohe_names
    except Exception:
        feature_names_transformed = [f"f{i}" for i in range(Xt_dense.shape[1])]

    imp = pd.Series(getattr(rf, "feature_importances_", np.zeros(len(feature_names_transformed))),
                    index=feature_names_transformed)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.to_csv(outdir / "feature_importance_unsupervised_transformed.csv")
    plot_and_save_importance(imp, outdir / "feature_importance_unsupervised.png", 30, "Unsupervised RF Importance")

    # OHE'den ana sütuna topla
    base_map = collapse_to_base_feature(imp.index.tolist())
    base_scores = {b: float(imp.iloc[[i for i,_ in enumerate(imp.index) if i in idxs]].sum())
                   for b, idxs in base_map.items()}
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_unsupervised_base.csv")

    # ileri seçim (proxy hedef = best_labels)
    template_model = pick_model()[1]
    chosen = []
    curve = []
    X_tr, X_va, y_tr, y_va = train_test_split(X, best_labels, test_size=0.25, random_state=42, stratify=best_labels)

    def transform_and_select(pre, Xdf, chosen_bases):
        Xt = pre.transform(Xdf)
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        keep_idx = []
        for b in chosen_bases:
            keep_idx.extend(base_map[b])
        keep_idx = sorted(set(keep_idx))
        return Xt[:, keep_idx]

    # pre fit
    pre.fit(X_tr)
    best_score, best_set = -1, []

    for k, base_feat in enumerate(base_rank.index.tolist(), start=1):
        chosen.append(base_feat)
        Xtr_sel = transform_and_select(pre, X_tr, chosen)
        Xva_sel = transform_and_select(pre, X_va, chosen)
        template_model.fit(Xtr_sel, y_tr)
        score = template_model.score(Xva_sel, y_va)  # proxy doğruluk
        curve.append({"k": k, "acc_proxy": float(score)})
        if score > best_score:
            best_score, best_set = score, chosen.copy()

    with open(outdir / "forward_selection_unsupervised.json","w",encoding="utf-8") as f:
        json.dump({"forward_curve": curve, "best_k": len(best_set)}, f, ensure_ascii=False, indent=2)

    selected_cols = [c for c in best_set if c in X.columns]
    df_selected = X[selected_cols].copy()
    df_selected.to_csv(desired_output_csv, index=False)
    with open(outdir / "selected_columns.json","w",encoding="utf-8") as f:
        json.dump(selected_cols, f, ensure_ascii=False, indent=2)

    meta = {
        "kmeans_k": int(best_k), "silhouette": float(best_sil),
        "n_selected_features": len(selected_cols),
        "output_csv": str(desired_output_csv)
    }
    with open(outdir / "metrics_unsupervised.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ (Unsupervised) Özellik seçimi bitti. Seçilmiş veri → {desired_output_csv}")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Girdi CSV (sf_crime_08.csv veya fr_crime_09.csv)")
    parser.add_argument("--target", type=str, default="Y_label", help="sf/fr için hedef sütun adı (varsa)")
    parser.add_argument("--outdir", type=str, default="outputs_feature_analysis", help="Çıktı klasörü")
    parser.add_argument("--group_by_geoid", action="store_true", help="sf için GEOID gruplu split")
    parser.add_argument("--out_csv", type=str, default="", help="İsteğe bağlı çıktı CSV yolu (varsayılan: sf→sf_crime_09.csv, fr→fr_crime_10.csv)")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv, low_memory=False)
    fname = Path(args.csv).name.lower()

    # otomatik çıktı isimleri
    if args.out_csv:
        desired_out = Path(args.out_csv)
    else:
        if fname.startswith("sf_"):
            desired_out = Path(Path(args.csv).parent) / "sf_crime_09.csv"
        elif fname.startswith("fr_"):
            desired_out = Path(Path(args.csv).parent) / "fr_crime_10.csv"
        else:
            # güvenli varsayılan
            desired_out = Path(Path(args.csv).parent) / "selected_output.csv"

    has_target = args.target in df_raw.columns and df_raw[args.target].dropna().nunique() >= 2

    if fname.startswith("sf_") or has_target:
        print("🔎 Tip: sf (veya hedef mevcut) → Denetimli özellik seçimi.")
        supervised_feature_selection(df_raw, target=args.target, outdir=outdir,
                                     group_by_geoid=args.group_by_geoid, csv_path=args.csv,
                                     desired_output_csv=desired_out)
    else:
        print("🔎 Tip: fr (etiketsiz) → Denetimsiz özellik seçimi.")
        unsupervised_feature_selection(df_raw, outdir=outdir,
                                       desired_output_csv=desired_out,
                                       target_if_exists=args.target if args.target in df_raw.columns else None)

if __name__ == "__main__":
    main()
