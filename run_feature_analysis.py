# file: run_feature_analysis.py
import os, sys, json, math, warnings, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, f1_score, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# =========================
# Defaults (env/backward compat)
# =========================
DEFAULT_DATA_FILES = [
    "crime_prediction_data/sf_crime_grid_full_labeled.csv",
    "crime_prediction_data/fr_crime_grid_full_labeled.csv",
]

TARGET_COL = "Y_label"
DATETIME_COL = "datetime"
GEOID_COL = "GEOID"
ID_COLS = ["id"]
DROP_ALWAYS = [TARGET_COL, DATETIME_COL, GEOID_COL] + ID_COLS

TEST_DAYS = 30
TOPK_RATIO = 0.05
PRECISION_TARGET = 0.5
RANDOM_STATE = 42

# Top-k sweep
K_GRID = np.linspace(0.01, 0.25, 25)
BENEFIT_TP = 1.0
COST_FP = 0.25
COST_FN = 2.0
PRECISION_MIN = 0.50
RECALL_MIN = None

# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Feature analysis + top-k optimization")
    p.add_argument("--files", nargs="+", default=DEFAULT_DATA_FILES,
                   help="Analiz edilecek CSV yolları (birden çok dosya verilebilir)")
    return p.parse_args()

def read_df(path):
    try:
        cols = pd.read_csv(path, nrows=1).columns.tolist()
    except Exception:
        raise
    parse_dates = [DATETIME_COL] if DATETIME_COL in cols else []
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)

def basic_profile(df, outdir):
    prof = {"n_rows": int(len(df)),
            "positive_rate": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None}
    df.isna().mean().sort_values(ascending=False)\
        .to_csv(os.path.join(outdir, "missingness.csv"), index_label="column", header=["missing_rate"])
    df.describe(include="all").to_csv(os.path.join(outdir, "describe.csv"))
    with open(os.path.join(outdir, "profile.json"), "w") as f:
        json.dump(prof, f, indent=2)
    return prof

def make_splits(df):
    if DATETIME_COL in df.columns and np.issubdtype(df[DATETIME_COL].dtype, np.datetime64):
        df_sorted = df.sort_values(DATETIME_COL)
        cutoff = df_sorted[DATETIME_COL].max() - pd.Timedelta(days=TEST_DAYS)
        train_idx = df_sorted[DATETIME_COL] <= cutoff
        test_idx  = df_sorted[DATETIME_COL] > cutoff
        return df_sorted[train_idx], df_sorted[test_idx], "time_holdout", cutoff
    if GEOID_COL in df.columns:
        unique_geoids = df[GEOID_COL].dropna().unique()
        train_geo, test_geo = train_test_split(unique_geoids, test_size=0.2, random_state=RANDOM_STATE)
        train = df[df[GEOID_COL].isin(train_geo)]
        test  = df[df[GEOID_COL].isin(test_geo)]
        return train, test, "group_holdout_geoid", None
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
    return train, test, "random_holdout", None

def build_preprocessor(X):
    num_cols = [c for c in X.columns if X[c].dtype.kind in "fc" or np.issubdtype(X[c].dtype, np.number)]
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop"
    )

def train_lgb(pre, X_train, y_train):
    if not HAS_LGB:
        model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        model.fit(X_train, y_train)
        return model, "logreg_balanced"
    clf = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, objective="binary",
        class_weight="balanced", random_state=RANDOM_STATE
    )
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)
    return model, "lightgbm_balanced"

def calibration_curve_simple(y_true, y_prob, n_bins=20):
    df = pd.DataFrame({"y":y_true, "p":y_prob}).sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    grp = df.groupby("bin")
    return grp["p"].mean().values, grp["y"].mean().values

def evaluate(model, X_test, y_test, outdir, label="base"):
    proba = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)
    fpr, tpr, roc_th = roc_curve(y_test, proba)
    pr_prec, pr_rec, pr_th = precision_recall_curve(y_test, proba)
    pm, fp = calibration_curve_simple(y_test, proba, n_bins=20)

    pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":roc_th}).to_csv(os.path.join(outdir, f"roc_curve_{label}.csv"), index=False)
    pd.DataFrame({"recall":pr_rec, "precision":pr_prec}).to_csv(os.path.join(outdir, f"pr_curve_{label}.csv"), index=False)
    pd.DataFrame({"prob_mean":pm, "frac_pos":fp}).to_csv(os.path.join(outdir, f"calibration_{label}.csv"), index=False)
    with open(os.path.join(outdir, f"metrics_{label}.json"), "w") as f:
        json.dump({"roc_auc":float(roc_auc), "auprc":float(pr_auc), "brier":float(brier_score_loss(y_test, proba))}, f, indent=2)
    return proba

def choose_thresholds(y_true, y_prob, precision_target=PRECISION_TARGET, topk_ratio=TOPK_RATIO):
    pr_prec, pr_rec, pr_th = precision_recall_curve(y_true, y_prob)
    f1_vals = 2 * (pr_prec * pr_rec) / (pr_prec + pr_rec + 1e-12)
    f1_idx = np.nanargmax(f1_vals)
    th_f1 = pr_th[max(f1_idx-1, 0)] if f1_idx < len(pr_th) else 0.5

    th_prec = None
    for p, r, th in zip(pr_prec[:-1], pr_rec[:-1], pr_th):
        if p >= precision_target:
            th_prec = float(th); break

    n = len(y_prob)
    k = max(1, int(math.ceil(topk_ratio * n)))
    topk_idx = np.argsort(-y_prob)[:k]
    th_topk = float(np.min(y_prob[topk_idx]))

    return {"threshold_f1": float(th_f1),
            "threshold_precision_target": th_prec,
            "threshold_topk": th_topk,
            "topk_k": int(k)}

def confusion_at_threshold(y_true, y_prob, th):
    y_hat = (y_prob >= th).astype(int)
    return {"precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0))}

def export_feature_importance(model, X_train, outpath):
    try:
        pre = model.named_steps["pre"]; clf = model.named_steps["clf"]
        ohe = None
        for name, trans, cols in pre.transformers_:
            if name == "cat": ohe = trans.named_steps.get("ohe")
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2] if len(pre.transformers_)>1 else []
        ohe_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else list(cat_cols)
        feat_names = list(num_cols) + ohe_names
        if hasattr(clf, "feature_importances_"):
            imp = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
            imp.sort_values("importance", ascending=False).to_csv(outpath, index=False)
        else:
            pd.DataFrame({"feature": feat_names}).to_csv(outpath, index=False)
    except Exception as e:
        pd.DataFrame({"error":[str(e)]}).to_csv(outpath.replace(".csv", "_error.csv"), index=False)

def maybe_shap(model, X_sample, outdir, filename_prefix="shap_global"):
    if not HAS_SHAP: return
    try:
        pre = model.named_steps["pre"]; clf = model.named_steps["clf"]
        X_enc = pre.transform(X_sample)
        explainer = shap.TreeExplainer(clf) if hasattr(clf, "predict_proba") else shap.Explainer(clf)
        shap_values = explainer(X_enc)
        sv = np.abs(shap_values.values).mean(axis=0)
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "num": names += list(cols)
            elif name == "cat":
                ohe = trans.named_steps.get("ohe")
                names += list(ohe.get_feature_names_out(cols)) if ohe is not None else list(cols)
        pd.DataFrame({"feature": names, "mean_abs_shap": sv}).sort_values("mean_abs_shap", ascending=False)\
          .to_csv(os.path.join(outdir, f"{filename_prefix}.csv"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, f"{filename_prefix}_error.txt"), "w") as f:
            f.write(str(e))

# ---- Top-k optimization helpers ----
def topk_metrics_at_ratio(y_true, y_prob, k_ratio):
    n = len(y_prob); k = max(1, int(math.ceil(k_ratio * n)))
    order = np.argsort(-y_prob); sel = np.zeros(n, dtype=int); sel[order[:k]] = 1
    TP = int(((sel == 1) & (y_true == 1)).sum())
    FP = int(((sel == 1) & (y_true == 0)).sum())
    FN = int(((sel == 0) & (y_true == 1)).sum())
    TN = int(((sel == 0) & (y_true == 0)).sum())
    prec = precision_score(y_true, sel, zero_division=0)
    rec  = recall_score(y_true, sel, zero_division=0)
    f1   = f1_score(y_true, sel, zero_division=0)
    denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) or 1.0
    mcc = ((TP*TN) - (FP*FN)) / denom
    utility = BENEFIT_TP * TP - COST_FP * FP - COST_FN * FN
    thr_equiv = float(np.min(y_prob[order[:k]]))
    return {"k_ratio": float(k_ratio), "k_count": int(k),
            "precision_at_k": float(prec), "recall_at_k": float(rec),
            "f1_at_k": float(f1), "mcc_at_k": float(mcc),
            "utility_at_k": float(utility), "threshold_equiv": thr_equiv}

def pick_best_k(grid_rows):
    df = pd.DataFrame(grid_rows)
    mask = np.ones(len(df), dtype=bool)
    if PRECISION_MIN is not None: mask &= (df["precision_at_k"] >= PRECISION_MIN)
    if RECALL_MIN is not None:    mask &= (df["recall_at_k"] >= RECALL_MIN)
    cand = df[mask] if mask.any() else df
    cand = cand.sort_values(["utility_at_k", "f1_at_k"], ascending=[False, False]).reset_index(drop=True)
    return cand.iloc[0].to_dict(), df

def run_one(path):
    if not os.path.isfile(path):
        print(f"↪️ Skip (missing): {path}")
        return
    print(f"=== Processing: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join("feature_analysis_outputs", name)
    os.makedirs(outdir, exist_ok=True)

    df = read_df(path)
    assert TARGET_COL in df.columns, f"{TARGET_COL} bulunamadı."
    drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    X_cols = [c for c in df.columns if c not in drop_cols]

    basic_profile(df, outdir)

    train_df, test_df, split_kind, cutoff = make_splits(df)
    with open(os.path.join(outdir, "split_info.json"), "w") as f:
        json.dump({"split_kind": split_kind, "cutoff": str(cutoff) if cutoff is not None else None,
                   "train_rows": int(len(train_df)), "test_rows": int(len(test_df))}, f, indent=2)

    X_train, y_train = train_df[X_cols], train_df[TARGET_COL].astype(int).values
    X_test,  y_test  = test_df[X_cols],  test_df[TARGET_COL].astype(int).values

    pre = build_preprocessor(X_train)
    model, model_name = train_lgb(pre, X_train, y_train)

    proba = evaluate(model, X_test, y_test, outdir, label=model_name)

    ths = choose_thresholds(y_test, proba)
    th_summ = {k: (v if v is None else float(v)) for k, v in ths.items()}
    details = {}
    for key in ["threshold_f1", "threshold_precision_target", "threshold_topk"]:
        th = ths.get(key, None)
        if th is not None:
            details[key] = confusion_at_threshold(y_test, proba, th)
    with open(os.path.join(outdir, "thresholds.json"), "w") as f:
        json.dump({"thresholds": th_summ, "metrics_at_thresholds": details}, f, indent=2)

    # Top-k optimization
    grid_rows = [topk_metrics_at_ratio(y_test, proba, kr) for kr in K_GRID]
    best_k_row, df_grid = pick_best_k(grid_rows)
    pd.DataFrame(grid_rows).to_csv(os.path.join(outdir, "k_sweep_metrics.csv"), index=False)
    with open(os.path.join(outdir, "best_k_summary.json"), "w") as f:
        json.dump(best_k_row, f, indent=2)
    jury_report = {
        "decision_rule": "Maximize expected utility under constraints",
        "constraints": {"precision_min": PRECISION_MIN, "recall_min": RECALL_MIN},
        "chosen_k_ratio": best_k_row["k_ratio"],
        "topk_size": best_k_row["k_count"],
        "equivalent_threshold": best_k_row["threshold_equiv"],
        "precision_at_k": best_k_row["precision_at_k"],
        "recall_at_k": best_k_row["recall_at_k"],
        "f1_at_k": best_k_row["f1_at_k"],
        "mcc_at_k": best_k_row["mcc_at_k"],
        "utility_at_k": best_k_row["utility_at_k"]
    }
    with open(os.path.join(outdir, "jury_friendly_summary.json"), "w") as f:
        json.dump(jury_report, f, indent=2)

    # Export top-k rows
    n = len(proba); k = best_k_row["k_count"]; order = np.argsort(-proba); top_idx = order[:k]
    export_cols = ["pred_proba"]
    if GEOID_COL in test_df.columns: export_cols.append(GEOID_COL)
    if DATETIME_COL in test_df.columns: export_cols.append(DATETIME_COL)
    top_rows = test_df.iloc[top_idx].copy(); top_rows["pred_proba"] = proba[top_idx]
    top_rows[export_cols].sort_values("pred_proba", ascending=False)\
        .to_csv(os.path.join(outdir, "top_predictions_at_best_k.csv"), index=False)

    export_feature_importance(model, X_train, os.path.join(outdir, "feature_importance.csv"))

    # SHAP (global)
    X_sample = X_test.sample(n=min(len(X_test), 5000), random_state=RANDOM_STATE)
    maybe_shap(model, X_sample, outdir, filename_prefix="shap_global")
    # SHAP (top-k)
    if HAS_SHAP and k >= 5:
        maybe_shap(model, test_df.iloc[top_idx][X_cols], outdir, filename_prefix="shap_on_best_topk")

    print(f"✔ Done: {name} | best k={best_k_row['k_ratio']:.3f} | thr≈{best_k_row['threshold_equiv']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    for f in args.files:
        try:
            run_one(f)
        except FileNotFoundError:
            print(f"↪️ Skip (not found): {f}")
        except Exception as e:
            print(f"⚠️ Error on {f}: {e}", file=sys.stderr)
