# file: run_feature_analysis.py
import os, json, math, warnings
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
# Ayarlar
# =========================
DATA_FILES = [
    "crime_prediction_data/sf_crime_grid_full_labeled.csv",
    "crime_prediction_data/fr_crime_grid_full_labeled.csv",
]

TARGET_COL = "Y_label"         # hedef sütun (0/1)
DATETIME_COL = "datetime"      # varsa zamansal split için kullanılır
GEOID_COL = "GEOID"            # gruplanmış split/raporlama için
ID_COLS = ["id"]               # varsa çıkarılacak
DROP_ALWAYS = [TARGET_COL, DATETIME_COL, GEOID_COL] + ID_COLS

TEST_DAYS = 30                 # zaman kolonu varsa: son 30 gün test
TOPK_RATIO = 0.05              # top-k seçim kapasitesi örneği: en riskli %5
PRECISION_TARGET = 0.5         # hedef hassasiyet (örnek): %50 precision eşiği
RANDOM_STATE = 42

# =========================
# Yardımcı fonksiyonlar
# =========================

def read_df(path):
    parse_dates = [DATETIME_COL] if DATETIME_COL in pd.read_csv(path, nrows=1).columns else []
    return pd.read_csv(path, parse_dates=parse_dates)

def basic_profile(df, outdir):
    prof = {}
    prof["n_rows"] = len(df)
    prof["positive_rate"] = float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(os.path.join(outdir, "missingness.csv"), index_label="column", header=["missing_rate"])
    df.describe(include="all").to_csv(os.path.join(outdir, "describe.csv"))
    with open(os.path.join(outdir, "profile.json"), "w") as f:
        json.dump(prof, f, indent=2)
    return prof

def make_splits(df):
    # Zaman kolonu varsa: son TEST_DAYS test
    if DATETIME_COL in df.columns and np.issubdtype(df[DATETIME_COL].dtype, np.datetime64):
        df_sorted = df.sort_values(DATETIME_COL)
        cutoff = df_sorted[DATETIME_COL].max() - pd.Timedelta(days=TEST_DAYS)
        train_idx = df_sorted[DATETIME_COL] <= cutoff
        test_idx  = df_sorted[DATETIME_COL] > cutoff
        return df_sorted[train_idx], df_sorted[test_idx], "time_holdout", cutoff
    # Zaman kolonu yoksa: GEOID ile grup bazlı split (mekânsal sızıntıyı azaltma)
    if GEOID_COL in df.columns:
        unique_geoids = df[GEOID_COL].dropna().unique()
        train_geo, test_geo = train_test_split(unique_geoids, test_size=0.2, random_state=RANDOM_STATE)
        train = df[df[GEOID_COL].isin(train_geo)]
        test  = df[df[GEOID_COL].isin(test_geo)]
        return train, test, "group_holdout_geoid", None
    # Hiçbiri yoksa: düz rastgele (son çare)
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
    return train, test, "random_holdout", None

def build_preprocessor(X):
    num_cols = [c for c in X.columns if X[c].dtype.kind in "fc" or np.issubdtype(X[c].dtype, np.number)]
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )
    return pre

def train_lgb(pre, X_train, y_train):
    if not HAS_LGB:
        # Alternatif küçük model (kurulum kolaylığı için)
        model = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        model.fit(X_train, y_train)
        return model, "logreg_balanced"
    # LightGBM (güçlü baseline)
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)
    return model, "lightgbm_balanced"

def evaluate(model, X_test, y_test, outdir, label="base"):
    proba = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)

    fpr, tpr, roc_th = roc_curve(y_test, proba)
    pr_prec, pr_rec, pr_th = precision_recall_curve(y_test, proba)
    cal_df = calibration_curve(y_test, proba, n_bins=20)

    pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":roc_th}).to_csv(os.path.join(outdir, f"roc_curve_{label}.csv"), index=False)
    pd.DataFrame({"recall":pr_rec, "precision":pr_prec}).to_csv(os.path.join(outdir, f"pr_curve_{label}.csv"), index=False)
    pd.DataFrame({"prob_mean":cal_df[0], "frac_pos":cal_df[1]}).to_csv(os.path.join(outdir, f"calibration_{label}.csv"), index=False)

    with open(os.path.join(outdir, f"metrics_{label}.json"), "w") as f:
        json.dump({"roc_auc":float(roc_auc), "auprc":float(pr_auc), "brier":float(brier_score_loss(y_test, proba))}, f, indent=2)
    return proba

def calibration_curve(y_true, y_prob, n_bins=20):
    # basit kalibrasyon eğrisi (isotonic öncesi)
    df = pd.DataFrame({"y":y_true, "p":y_prob}).sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    grp = df.groupby("bin")
    prob_mean = grp["p"].mean().values
    frac_pos  = grp["y"].mean().values
    return prob_mean, frac_pos

def choose_thresholds(y_true, y_prob, precision_target=PRECISION_TARGET, topk_ratio=TOPK_RATIO):
    # 1) F1'i maksimize eden eşik
    pr_prec, pr_rec, pr_th = precision_recall_curve(y_true, y_prob)
    f1_vals = 2 * (pr_prec * pr_rec) / (pr_prec + pr_rec + 1e-12)
    f1_idx = np.nanargmax(f1_vals)
    th_f1 = pr_th[max(f1_idx-1, 0)] if f1_idx < len(pr_th) else 0.5  # pr_th 1 eksik uzunlukta olur

    # 2) Hedef precision'ı sağlayan en düşük eşik
    th_prec = None
    for p, r, th in zip(pr_prec[:-1], pr_rec[:-1], pr_th):
        if p >= precision_target:
            th_prec = float(th)
            break

    # 3) Top-k (kapasite kısıtı) — en yüksek olasılıklı ilk k örnek
    n = len(y_prob)
    k = max(1, int(math.ceil(topk_ratio * n)))
    topk_idx = np.argsort(-y_prob)[:k]
    th_topk = float(np.min(y_prob[topk_idx]))

    return {
        "threshold_f1": float(th_f1),
        "threshold_precision_target": th_prec,
        "threshold_topk": th_topk,
        "topk_k": int(k)
    }

def confusion_at_threshold(y_true, y_prob, th):
    y_hat = (y_prob >= th).astype(int)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    f1   = f1_score(y_true, y_hat, zero_division=0)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

def export_feature_importance(model, X_train, outpath):
    # LightGBM ise önemleri çıkar; değilse pipeline kolon adlarını kaydet
    try:
        # çıkarılmış OHE ile birlikte feature isimleri:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        ohe = None
        for name, trans, cols in pre.transformers_:
            if name == "cat":
                ohe = trans.named_steps.get("ohe")
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2] if len(pre.transformers_)>1 else []
        feat_names = []
        if ohe is not None:
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        else:
            ohe_names = cat_cols
        feat_names = list(num_cols) + ohe_names

        if hasattr(clf, "feature_importances_"):
            imp = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
            imp.sort_values("importance", ascending=False).to_csv(outpath, index=False)
        else:
            pd.DataFrame({"feature": feat_names}).to_csv(outpath, index=False)
    except Exception as e:
        pd.DataFrame({"error":[str(e)]}).to_csv(outpath.replace(".csv", "_error.csv"), index=False)

def maybe_shap(model, X_sample, outdir):
    if not HAS_SHAP:
        return
    try:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        X_enc = pre.transform(X_sample)
        explainer = shap.TreeExplainer(clf) if hasattr(clf, "predict_proba") else shap.Explainer(clf)
        shap_values = explainer(X_enc)
        # global mean |SHAP| değerleri
        sv = np.abs(shap_values.values).mean(axis=0)
        # feature isimleri
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                names += list(cols)
            elif name == "cat":
                ohe = trans.named_steps.get("ohe")
                if ohe is not None:
                    names += list(ohe.get_feature_names_out(cols))
                else:
                    names += cols
        pd.DataFrame({"feature": names, "mean_abs_shap": sv}).sort_values("mean_abs_shap", ascending=False)\
          .to_csv(os.path.join(outdir, "shap_global.csv"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, "shap_error.txt"), "w") as f:
            f.write(str(e))

def run_one(path):
    print(f"=== Processing: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join("feature_analysis_outputs", name)
    os.makedirs(outdir, exist_ok=True)

    df = read_df(path)
    assert TARGET_COL in df.columns, f"{TARGET_COL} bulunamadı."
    # Çıkartılacak sütunlar mevcutsa ayır
    drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    X_cols = [c for c in df.columns if c not in drop_cols]
    X = df[X_cols].copy()
    y = df[TARGET_COL].astype(int).values

    # Profil
    prof = basic_profile(df, outdir)

    # Split
    train_df, test_df, split_kind, cutoff = make_splits(df)
    with open(os.path.join(outdir, "split_info.json"), "w") as f:
        json.dump({"split_kind": split_kind,
                   "cutoff": str(cutoff) if cutoff is not None else None,
                   "train_rows": int(len(train_df)),
                   "test_rows": int(len(test_df))}, f, indent=2)

    X_train = train_df[X_cols]
    y_train = train_df[TARGET_COL].astype(int).values
    X_test  = test_df[X_cols]
    y_test  = test_df[TARGET_COL].astype(int).values

    # Preprocess + Model
    pre = build_preprocessor(X_train)
    model, model_name = train_lgb(pre, X_train, y_train)

    # Değerlendirme
    proba = evaluate(model, X_test, y_test, outdir, label=model_name)

    # Eşik/Top-k
    ths = choose_thresholds(y_test, proba)
    # eşiğe göre kısa özet
    th_summ = {k: (v if v is None else float(v)) for k, v in ths.items()}
    details = {}
    for key in ["threshold_f1", "threshold_precision_target", "threshold_topk"]:
        th = ths.get(key, None)
        if th is not None:
            details[key] = confusion_at_threshold(y_test, proba, th)

    with open(os.path.join(outdir, "thresholds.json"), "w") as f:
        json.dump({"thresholds": th_summ, "metrics_at_thresholds": details}, f, indent=2)

    # Özellik önemi
    export_feature_importance(model, X_train, os.path.join(outdir, "feature_importance.csv"))

    # (Opsiyonel) SHAP (örneklem ile hızlandır)
    if len(X_test) > 5000:
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=5000, replace=False)
        X_sample = X_test.iloc[sample_idx].copy()
    else:
        X_sample = X_test.copy()
    maybe_shap(model, X_sample, outdir)

    print(f"Done: outputs in {outdir}")

if __name__ == "__main__":
    for f in DATA_FILES:
        run_one(f)
