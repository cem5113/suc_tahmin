# run_feature_analysis_fr.py

import os, argparse, json, re, warnings, joblib
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence

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
        if nc in seen:
            seen[nc] += 1; nc = f"{nc}_dup{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    df.columns = new_cols
    return df


def extract_quantile_label(val):
    if pd.isna(val): return np.nan
    m = re.match(r"Q(\d+)", str(val).strip())
    try:
        return int(m.group(1)) if m else np.nan
    except Exception:
        return np.nan


def coerce_numeric(df: pd.DataFrame, exclude_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    excl = set(exclude_cols)
    for c in df.columns:
        if c in excl:
            categorical_cols.append(c); continue
        if df[c].dtype == object:
            qvals = df[c].map(extract_quantile_label)
            if qvals.notna().mean() > 0.6: df[c] = qvals
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
            n_estimators=500, max_depth=6, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", objective="binary:logistic",
            eval_metric="auc", n_jobs=-1, random_state=42
        )
    if HAS_LGBM:
        return "lgbm", LGBMClassifier(
            n_estimators=800, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=42
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
    plt.figure(figsize=(8, max(4, int(0.28*len(top)))) )
    top.iloc[::-1].plot(kind="barh")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def make_ohe_compat(**kwargs):
    """OneHotEncoder sparse_output (>=1.2) ile sparse (<=1.1) arasında uyumluluk."""
    try:
        return OneHotEncoder(**kwargs, handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(**kwargs, handle_unknown="ignore", sparse=True)


def get_transformed_feature_names(pre: ColumnTransformer,
                                  numeric_cols_in_use: Sequence[str],
                                  categorical_cols_in_use: Sequence[str],
                                  X_sample: pd.DataFrame) -> List[str]:
    names: List[str] = []
    names.extend(list(numeric_cols_in_use))
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        ohe_raw = ohe.get_feature_names_out(categorical_cols_in_use).tolist()
        pretty = []
        for nm in ohe_raw:
            matched = False
            for base in categorical_cols_in_use:
                pref = f"{base}_"
                if nm.startswith(pref):
                    pretty.append(f"{base}={nm[len(pref):]}")
                    matched = True
                    break
            if not matched:
                pretty.append(nm)
        names.extend(pretty)
    except Exception:
        try:
            n_total = pre.transform(X_sample).shape[1]
        except Exception:
            n_total = len(numeric_cols_in_use)
        rest = n_total - len(numeric_cols_in_use)
        names.extend([f"ohe_f{i}" for i in range(rest)])
    return names


def collapse_to_base_feature(feature_names_transformed: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for i, nm in enumerate(feature_names_transformed):
        if "=" in nm:
            base = nm.split("=", 1)[0]
        else:
            base = nm
        mapping.setdefault(base, []).append(i)
    return mapping


def make_importance_series_from_model(model, pre: ColumnTransformer, X_sample: pd.DataFrame,
                                      feature_names: Sequence[str]) -> Optional[pd.Series]:
    n_trans = pre.transform(X_sample).shape[1]

    if hasattr(model, "feature_importances_"):
        vals = getattr(model, "feature_importances_", None)
        if vals is None: return None
        vals = np.asarray(vals)
        n = min(len(feature_names), len(vals), n_trans)
        if n == 0: return None
        return pd.Series(vals[:n], index=list(feature_names)[:n]).clip(lower=0.0)

    if hasattr(model, "get_booster"):  # XGB
        try:
            raw = model.get_booster().get_score(importance_type="gain")
            tmp = pd.Series({int(k[1:]): v for k, v in raw.items()})
            tmp = tmp.reindex(range(n_trans)).fillna(0.0)
            n = min(len(feature_names), n_trans)
            if n == 0: return None
            return pd.Series(tmp.values[:n], index=list(feature_names)[:n]).clip(lower=0.0)
        except Exception:
            return None

    return None


def extract_geoid_series(df_raw: pd.DataFrame) -> Optional[pd.Series]:
    candidates = ["GEOID", "geoid", "geo_id", "tract_geoid", "geoid10", "tract", "blockgroup", "block_group", "block"]
    found = None
    for cand in candidates:
        for c in df_raw.columns:
            if str(c).lower() == cand.lower():
                found = df_raw[c]; break
        if found is not None: break
    if found is None:
        return None
    s = pd.Series(found).astype(str).str.replace(r"\.0$", "", regex=True)
    s.name = "GEOID"
    return s


# --- eğri analizi (artefaktlar korunuyor; seçim kuralı aşağıda REV ile değişti) ---
def _analyze_forward_curve(curve: List[dict], key: str, eps: float = 0.0015, plateau_win: int = 3):
    if not curve:
        return {"best_k": None, "best_score": None, "elbow_k": None, "plateau_from": None}
    ks = [int(c["k"]) for c in curve]
    vs = [float(c.get(key, float("nan"))) for c in curve]
    best_idx = int(np.nanargmax(vs))
    best_k = ks[best_idx]; best_score = float(vs[best_idx])
    gains = [float("nan")] + [vs[i] - vs[i-1] for i in range(1, len(vs))]
    elbow_k = None
    for i in range(1, len(ks)):
        if gains[i] < eps:
            elbow_k = ks[i]; break
    plateau_from = None
    for i in range(plateau_win, len(ks)):
        window = gains[i-plateau_win+1:i+1]
        if all(g < eps for g in window):
            plateau_from = ks[i-plateau_win+1]; break
    return {"best_k": int(best_k), "best_score": best_score,
            "elbow_k": int(elbow_k) if elbow_k is not None else None,
            "plateau_from": int(plateau_from) if plateau_from is not None else None}


def _save_forward_curve_artifacts(curve: List[dict], key: str, outdir: Path, title: str):
    dfc = pd.DataFrame(curve)
    dfc.to_csv(outdir / "forward_selection_curve.csv", index=False)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(dfc["k"], dfc[key], marker="o")
    plt.xlabel("k (seçilen temel değişken sayısı)")
    plt.ylabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outdir / "forward_selection_curve.png", dpi=180)
    plt.close()


# ---------------- çekirdek: etiketsiz ----------------
def unsupervised_feature_selection(df_raw: pd.DataFrame, outdir: Path, desired_output_csv: Path, top_k: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = sanitize_columns(df_raw.copy())

    # sızıntı/id/geo/koordinat kaldır (eğitimden)
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

    # pipeline (yalnız preprocess) - OHE uyumlu
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", make_ohe_compat())])
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

    # RF ile önem (dönüşmüş uzayda)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(Xt_dense, best_labels)

    # Dönüşmüş isimleri güvenle al + hizalama
    feature_names_transformed = get_transformed_feature_names(pre,
        numeric_cols_in_use=[c for c in numeric_cols if c in X.columns],
        categorical_cols_in_use=[c for c in categorical_cols if c in X.columns],
        X_sample=X
    )
    imp_ser = make_importance_series_from_model(rf, pre, X, feature_names_transformed)
    if imp_ser is None or imp_ser.sum() == 0:
        raise RuntimeError("RF önemleri hesaplanamadı.")
    imp_norm = (imp_ser / imp_ser.sum()).sort_values(ascending=False)
    (outdir / "feature_importance_unsupervised_transformed.csv").write_text(
        imp_norm.to_csv(), encoding="utf-8"
    )
    plot_and_save_importance(imp_norm, outdir / "feature_importance_unsupervised.png", 30, "Unsupervised RF Importance")

    # OHE -> ana sütun eşlemesi
    base_map = collapse_to_base_feature(feature_names_transformed)

    # taban skorlarını RF önemleri üzerinden hesapla
    base_scores = {}
    for b, idxs in base_map.items():
        names_for_b = [feature_names_transformed[i] for i in idxs if 0 <= i < len(feature_names_transformed)]
        base_scores[b] = float(imp_norm.loc[imp_norm.index.intersection(names_for_b)].sum())
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_unsupervised_base.csv")

    # REV: Seçim kuralı — yalnız **>0** puanlılar, azalan sırada **en iyi top_k**.
    base_pos = base_rank[base_rank > 0.0]
    selected_bases = base_pos.head(top_k).index.tolist()
    if len(selected_bases) < top_k:
        print(f"⚠️ Pozitif önemli temel değişken sayısı {len(selected_bases)} < {top_k}. "
              f"Yalnızca mevcut pozitifler kullanılacak.")

    # (Artefakt devamlılığı için basit bir eğri üretelim — k=1..len(selected_bases) proxy acc)
    model_name, base_model = pick_model()
    chosen, curve = [], []
    X_tr, X_va, y_tr, y_va = train_test_split(X, best_labels, test_size=0.25, random_state=42, stratify=best_labels)
    pre.fit(X_tr)
    def transform_and_select(pre, Xdf, chosen_bases):
        Xt = pre.transform(Xdf)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        keep_idx = sorted(set(sum((base_map[b] for b in chosen_bases if b in base_map), [])))
        return Xt[:, keep_idx] if keep_idx else Xt
    for b in selected_bases:
        chosen.append(b)
        Xtr_sel = transform_and_select(pre, X_tr, chosen)
        Xva_sel = transform_and_select(pre, X_va, chosen)
        base_model.fit(Xtr_sel, y_tr)
        score = base_model.score(Xva_sel, y_va)
        curve.append({"k": len(chosen), "acc_proxy": float(score)})

    # Eğri artefaktları + özet
    with open(outdir / "forward_selection_unsupervised.json","w",encoding="utf-8") as f:
        json.dump({"forward_curve": curve, "best_k": len(selected_bases)}, f, ensure_ascii=False, indent=2)
    summary = _analyze_forward_curve(curve, key="acc_proxy", eps=0.002)
    with open(outdir / "forward_selection_summary.json","w",encoding="utf-8") as f:
        json.dump({"mode":"unsupervised", **summary}, f, ensure_ascii=False, indent=2)
    _save_forward_curve_artifacts(curve, key="acc_proxy", outdir=outdir,
                                  title="Forward Selection (Unsupervised, top-K>0 rule)")

    print(f"🏁 Unsupervised (REV) → seçilen temel değişken sayısı = {len(selected_bases)}")

    # --- ÇIKIŞ: GEOID passthrough + seçilen özellikler ---
    selected_cols = [c for c in selected_bases if c in X.columns]  # taban adları doğrudan sütun adıyla eşleşiyorsa
    # Not: Eğer taban adın OHE’lenmiş kategorinin base’i ise, giriş X'te base kolon zaten kategorik olduğu için vardır.

    geo = extract_geoid_series(df_raw)
    selected_df = X[selected_cols].copy().reset_index(drop=True)
    if geo is not None:
        out_df = pd.concat([geo.reset_index(drop=True), selected_df], axis=1)
    else:
        out_df = selected_df
    out_df.to_csv(desired_output_csv, index=False)

    meta = {"kmeans_k": int(best_k), "silhouette": float(best_sil),
            "n_selected_features": len(selected_cols), "output_csv": str(desired_output_csv)}
    with open(outdir / "metrics_unsupervised.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ (Unsupervised) fr_crime_10.csv üretildi → {desired_output_csv}")


# ---------------- opsiyonel: hedef varsa denetimli ----------------
def supervised_if_target(df_raw: pd.DataFrame, target: str, outdir: Path, desired_output_csv: Path, top_k: int) -> bool:
    if target not in df_raw.columns or df_raw[target].dropna().nunique() < 2:
        return False

    df = sanitize_columns(df_raw.copy())
    # Hedef adı sanitize edildiyse yakala
    tgt_candidates = [c for c in df.columns if c.lower() == target.lower()]
    target_col = tgt_candidates[0] if tgt_candidates else target
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).clip(0,1)
    X = df.drop(columns=[target_col]).copy()

    # sızıntı/id/geo/koordinat (eğitimden çıkar)
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
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe_compat())]),
         [c for c in categorical_cols if c in X.columns])
    ], remainder="drop")

    model_name, base_model = pick_model()
    if model_name == "xgb":
        base_model.set_params(scale_pos_weight=compute_class_weight(y_tr))

    clf = Pipeline([("pre", pre), ("model", base_model)])
    clf.fit(X_tr, y_tr)

    # metrikler (artefakt)
    if hasattr(base_model, "predict_proba"):
        proba = clf.predict_proba(X_te)[:,1]
        yhat = (proba >= 0.5).astype(int)
        roc = float(roc_auc_score(y_te, proba))
    else:
        pred = clf.predict(X_te)
        yhat = pred[:,0] if getattr(pred, "ndim", 1) > 1 else pred
        roc = None
    prec, rec, _, _ = precision_recall_fscore_support(y_te, yhat, average="binary", zero_division=0)
    metrics = {
        "model": model_name,
        "roc_auc": roc,
        "f1_default_0.50": float(f1_score(y_te, yhat)),
        "precision_default_0.50": float(prec),
        "recall_default_0.50": float(rec)
    }
    with open(outdir / "metrics_supervised.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Dönüşmüş isimleri güvenle al + importances hizala
    pre_fit = clf.named_steps["pre"]
    feature_names_transformed = get_transformed_feature_names(
        pre_fit,
        numeric_cols_in_use=[c for c in numeric_cols if c in X.columns],
        categorical_cols_in_use=[c for c in categorical_cols if c in X.columns],
        X_sample=X_tr
    )
    imp = make_importance_series_from_model(clf.named_steps["model"], pre_fit, X_tr, feature_names_transformed)

    if imp is None or imp.sum() == 0:
        joblib.dump(clf, outdir / "model_pipeline.joblib")
        # REV: Pozitif önem bulunamadıysa yine de Y_label + (varsa GEOID) + tüm X ile yaz.
        geo = extract_geoid_series(df_raw)
        parts = []
        if geo is not None:
            parts.append(geo.reset_index(drop=True))
        parts.extend([X.reset_index(drop=True), y.rename(target_col).reset_index(drop=True)])
        out_df = pd.concat(parts, axis=1)
        out_df.to_csv(desired_output_csv, index=False)
        print("⚠️ Önem bulunamadı; tüm özelliklerle çıktı üretildi.")
        return True

    imp_norm = (imp/imp.sum()).sort_values(ascending=False)
    imp_norm.to_csv(outdir / "feature_importance_supervised_transformed.csv")
    plot_and_save_importance(imp_norm, outdir / "feature_importance_supervised.png", 30, "Supervised Importance")

    base_map = collapse_to_base_feature(feature_names_transformed)
    # taban skoru, isim eşlemesiyle
    base_scores = {}
    for b, idxs in base_map.items():
        names_for_b = [feature_names_transformed[i] for i in idxs if 0 <= i < len(feature_names_transformed)]
        base_scores[b] = float(imp_norm.loc[imp_norm.index.intersection(names_for_b)].sum())
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_supervised_base.csv")

    # REV: Seçim kuralı — yalnız **>0** puanlılar, azalan sırada **en iyi top_k**.
    base_pos = base_rank[base_rank > 0.0]
    selected_bases = base_pos.head(top_k).index.tolist()
    if len(selected_bases) < top_k:
        print(f"⚠️ Pozitif önemli temel değişken sayısı {len(selected_bases)} < {top_k}. "
              f"Yalnızca mevcut pozitifler kullanılacak.")

    # --- ÇIKIŞ: GEOID passthrough + seçilen özellikler + hedef ---
    # Not: taban adları giriş X’teki gerçek sütunlarla eşleşir (OHE base).
    selected_cols = [c for c in selected_bases if c in X.columns]
    geo = extract_geoid_series(df_raw)
    feat = X[selected_cols].reset_index(drop=True)
    yy = y.rename(target_col).reset_index(drop=True)
    parts = []
    if geo is not None:
        parts.append(geo.reset_index(drop=True))
    parts.extend([feat, yy])
    out_df = pd.concat(parts, axis=1)
    out_df.to_csv(desired_output_csv, index=False)

    joblib.dump(clf, outdir / "model_pipeline.joblib")
    print(f"✅ (Supervised) fr_crime_10.csv üretildi → {desired_output_csv}")
    return True


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=os.getenv("CRIME_CSV_FR", os.getenv("CRIME_CSV", "")),
                   help="Girdi CSV (vars: fr_crime_09.csv). Verilmezse CRIME_CSV_FR/CRIME_CSV ortam değişkeni kullanılır.")
    p.add_argument("--target", type=str, default="Y_label", help="Hedef sütun adı (varsa)")
    p.add_argument("--outdir", type=str, default="outputs_feature_analysis_fr", help="Çıktı klasörü")
    p.add_argument("--out_csv", type=str, default="", help="Çıktı CSV yolu (vars: fr_crime_10.csv)")
    p.add_argument("--topk", type=int, default=int(os.getenv("TOP_K", "10")), help="Seçilecek pozitif önemli temel değişken sayısı")  # REV
    args = p.parse_args()

    csv_path = args.csv
    if not csv_path:
        guess = Path("fr_crime_09.csv")
        if guess.exists():
            csv_path = str(guess)
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(
            "CSV bulunamadı. '--csv <dosya>' verin ya da CRIME_CSV_FR / CRIME_CSV ortam değişkenini ayarlayın. "
            "Örn: --csv fr_crime_09.csv"
        )

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(csv_path, low_memory=False)
    fname = Path(csv_path).name.lower()

    desired_out = Path(args.out_csv) if args.out_csv else (
        Path(csv_path).with_name("fr_crime_10.csv") if fname.startswith("fr_") else Path(csv_path).with_name("selected_output.csv")
    )

    # önce hedef var mı diye bak; varsa denetimli yoldan ilerle
    if supervised_if_target(df_raw, target=args.target, outdir=outdir, desired_output_csv=desired_out, top_k=args.topk):
        mode = "supervised"
    else:
        print("🔎 Tip: fr (etiketsiz) → Denetimsiz özellik seçimi.")
        unsupervised_feature_selection(df_raw, outdir=outdir, desired_output_csv=desired_out, top_k=args.topk)
        mode = "unsupervised"

    # Çalışma özeti (varsa)
    try:
        s = json.loads((outdir / "forward_selection_summary.json").read_text(encoding="utf-8"))
        print(f"🔎 Özet ({mode}): "
              f"best_k={s.get('best_k')}, score={s.get('best_score')}, "
              f"elbow_k={s.get('elbow_k')}, plateau_from={s.get('plateau_from')}")
    except Exception:
        pass

    print(f"📥 Girdi: {csv_path}")
    print(f"📤 Çıktı CSV: {desired_out}")
    print(f"📁 Outdir: {outdir.resolve()}")

if __name__ == "__main__":
    main()
