#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crime_feature_analysis.py — Gün-bazlı (GEOID × date) veriler için denetimli özellik analizi & seçim.

Bu sürüm, aşağıdaki örnek şemayı doğrudan destekler:
GEOID, season, day_of_week, event_hour, latitude, longitude, is_holiday, crime_count, crime_mix, Y_label,
hr_key, hour_range, sf_wet_season, sf_dry_season, sf_fog_season,
crime_last_3d, crime_last_7d, neigh_crime_last_3d, neigh_crime_last_7d,
is_weekend, is_night, is_school_hour, is_business_hour,
911_request_count_hour_range, 911_request_count_daily(before_24_hours),
911_geo_last3d, 911_geo_last7d, 911_geo_hr_last3d, 911_geo_hr_last7d, 911_neighbors_last3d, 911_neighbors_last7d,
311_request_count, population,
distance_to_bus, bus_stop_count, distance_to_bus_range, bus_stop_count_range,
distance_to_train, train_stop_count, distance_to_train_range, train_stop_count_range,
poi_total_count, poi_risk_score, poi_dominant_type, poi_total_count_range, poi_risk_score_range,
_lat_, _lon_,
distance_to_police, distance_to_police_range,
distance_to_government_building, distance_to_government_building_range,
is_near_police, is_near_government, neighbor_crime_total, ...

Öz:
- Zaman-farkındalıklı split: öncelik date/datetime; yoksa hr_key içinden tarih çıkarma; o da yoksa stratified random.
- (İsteğe bağlı) GEOID gruplu split ipuçları
- Model/Permutation/(varsa) SHAP importance → ensemble
- OHE alt kolonu bazında base mapping ve ileri seçim (F1)
- Operasyonel metrik: Precision@k

Kullanım:
  python crime_feature_analysis.py --csv sf_crime_08.csv --outdir outputs_feature_analysis --group_by_geoid
  # veya
  export CRIME_CSV=sf_crime_08.csv && python crime_feature_analysis.py
"""

import argparse, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone

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


# ------------------------- yardımcılar -------------------------
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = [], {}
    for c in df.columns:
        nc = re.sub(r"[^\w]+", "_", str(c), flags=re.UNICODE)  # parantez vb. → _
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
    if pd.isna(val):
        return np.nan
    m = re.match(r"Q(\d+)", str(val).strip(), flags=re.I)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def coerce_numeric(df: pd.DataFrame, exclude_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    excl = set(exclude_cols)
    for c in df.columns:
        if c in excl:
            categorical_cols.append(c)
            continue
        if df[c].dtype == object:
            qvals = df[c].map(extract_quantile_label)
            if qvals.notna().mean() > 0.6:
                df[c] = qvals
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.8:
                df[c] = conv
                numeric_cols.append(c)
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
            eval_metric="auc", n_jobs=-1, random_state=42
        )
    if HAS_LGBM:
        return "lgbm", LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=42
        )
    return "rf", RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1,
        class_weight="balanced_subsample", random_state=42
    )


def compute_class_weight(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).clip(0, 1)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    return float(neg) / float(pos) if pos > 0 else 1.0


def plot_and_save_importance(importances: pd.Series, out_png: Path, top_n: int = 30, title: str = "Feature Importance"):
    top = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, int(0.28 * len(top)))))
    top.iloc[::-1].plot(kind="barh")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def make_ohe_compat(**kwargs):
    try:
        return OneHotEncoder(**kwargs, handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(**kwargs, handle_unknown="ignore", sparse=True)


def get_transformed_feature_names(pre: ColumnTransformer,
                                  numeric_cols_in_use: Sequence[str],
                                  categorical_cols_in_use: Sequence[str]) -> List[str]:
    names: List[str] = []
    names.extend(list(numeric_cols_in_use))
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(categorical_cols_in_use).tolist()
        pretty = []
        for nm in ohe_names:
            matched = False
            for base in categorical_cols_in_use:
                prefix = f"{base}_"
                if nm.startswith(prefix):
                    val = nm[len(prefix):]
                    pretty.append(f"{base}={val}")
                    matched = True
                    break
            if not matched:
                pretty.append(nm)
        names.extend(pretty)
    except Exception:
        try:
            n_total = pre.transform(np.zeros((1, len(numeric_cols_in_use) + len(categorical_cols_in_use)))).shape[1]
        except Exception:
            n_total = len(numeric_cols_in_use)
        rest = n_total - len(numeric_cols_in_use)
        names.extend([f"ohe_f{i}" for i in range(rest)])
    return names


def build_base_mapping(feature_names: Sequence[str],
                       numeric_cols_in_use: Sequence[str],
                       categorical_cols_in_use: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for i, nm in enumerate(feature_names):
        if nm in numeric_cols_in_use:
            mapping.setdefault(nm, []).append(i)
            continue
        if "=" in nm:
            base = nm.split("=", 1)[0]
            mapping.setdefault(base, []).append(i)
        else:
            mapping.setdefault(nm, []).append(i)
    return mapping


def select_columns_from_transformed(pre: ColumnTransformer, Xdf: pd.DataFrame,
                                   chosen_bases: Sequence[str],
                                   feature_to_columns: Dict[str, List[int]]):
    Xt = pre.transform(Xdf)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    keep_idx = sorted(set(sum((feature_to_columns[b] for b in chosen_bases if b in feature_to_columns), [])))
    if not keep_idx:
        return Xt
    return Xt[:, keep_idx]


def forward_select_by_rank(pre: ColumnTransformer, base_model,
                           X: pd.DataFrame, y: pd.Series,
                           base_ranked_features: List[str],
                           feature_to_columns: Dict[str, List[int]],
                           metric: str = "f1",
                           random_state: int = 42):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25,
                                              random_state=random_state, stratify=y)
    pre.fit(X_tr)
    try:
        spw = compute_class_weight(y_tr)
        if base_model.__class__.__name__.lower().startswith("xgb"):
            base_model.set_params(scale_pos_weight=spw)
    except Exception:
        pass

    best_set, best_score, curve, chosen = [], -1.0, [], []
    for k, b in enumerate(base_ranked_features, start=1):
        chosen.append(b)
        Xtr_sel = select_columns_from_transformed(pre, X_tr, chosen, feature_to_columns)
        Xva_sel = select_columns_from_transformed(pre, X_va, chosen, feature_to_columns)

        mdl = clone(base_model)
        mdl.fit(Xtr_sel, y_tr)
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(Xva_sel)[:, 1]
            yhat = (proba >= 0.5).astype(int)
            f1 = f1_score(y_va, yhat)
            try:
                auc = float(roc_auc_score(y_va, proba))
            except Exception:
                auc = None
        else:
            yhat = mdl.predict(Xva_sel)
            if hasattr(yhat, "ndim") and yhat.ndim > 1:
                yhat = yhat[:, 0]
            f1 = f1_score(y_va, yhat); auc = None

        curve.append({"k": k, "feature": b, "f1": float(f1), "roc_auc": auc})
        if f1 > best_score:
            best_score, best_set = f1, list(chosen)

    info = {"forward_curve": curve, "best_k": len(best_set), "metric": metric, "best_score": float(best_score)}
    return best_set, info


def precision_at_k(y_true, scores, k=50) -> Optional[float]:
    try:
        idx = np.argsort(-np.asarray(scores))[:k]
        return float((np.asarray(y_true)[idx] == 1).mean())
    except Exception:
        return None


def try_parse_datetime_from_hrkey(s: pd.Series) -> Optional[pd.Series]:
    """hr_key ör. '2024-08-12 14', '2024-08-12', '20240812-14', '2024-08-12T14' vb."""
    if s is None:
        return None
    s = s.astype(str)
    # Çok yaygın desenleri normalize etmeye çalış
    # Önce 'YYYY-MM-DD HH' / 'YYYY-MM-DD' doğrudan dene
    dt = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
    if dt.notna().mean() > 0.5:
        return dt
    # 'YYYYMMDD-HH' veya 'YYYYMMDDHH' gibi
    s2 = s.str.replace(r"[^0-9]", "", regex=True)
    # uzunluk 8 → YYYYMMDD; 10→ YYYYMMDDHH
    def parse_compact(z):
        if len(z) >= 8:
            y, m, d = z[:4], z[4:6], z[6:8]
            if len(z) >= 10:
                h = z[8:10]
                return f"{y}-{m}-{d} {h}:00:00"
            return f"{y}-{m}-{d} 00:00:00"
        return None
    s3 = s2.map(parse_compact)
    dt2 = pd.to_datetime(s3, errors="coerce", utc=False)
    if dt2.notna().mean() > 0.5:
        return dt2
    return None


# ------------------------- çekirdek -------------------------
def supervised_feature_selection(df_raw: pd.DataFrame, target: str, outdir: Path,
                                 group_by_geoid: bool, csv_path: str,
                                 desired_output_csv: Path, random_state: int = 42,
                                 split_date: Optional[str] = None,
                                 top_k: int = 50) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # GEOID (her türlü büyük/küçük) yakala
    geo_candidates_raw = [c for c in df_raw.columns if str(c).lower() in {"geoid","geo_id","tract","tract_geoid","geoid10","blockgroup"}]
    original_geoid = df_raw[geo_candidates_raw[0]].astype(str) if geo_candidates_raw else None

    # tarih/datetime kolonları
    time_cols_raw = [c for c in df_raw.columns if str(c).lower() in {"date","incident_date","day","ds","datetime"}]
    time_col_raw = time_cols_raw[0] if time_cols_raw else None
    date_ser_raw = pd.to_datetime(df_raw[time_col_raw], errors="coerce") if time_col_raw else None

    # hr_key'den tarih çıkar (fallback)
    if date_ser_raw is None and "hr_key" in df_raw.columns:
        date_from_hr = try_parse_datetime_from_hrkey(df_raw["hr_key"])
        if date_from_hr is not None:
            date_ser_raw = date_from_hr

    # sanitize
    df = sanitize_columns(df_raw.copy())
    if target not in df.columns:
        raise ValueError(f"Hedef sütun '{target}' bulunamadı.")
    if df[target].dropna().nunique() < 2:
        raise ValueError("Hedef (Y_label) en az 2 sınıf içermiyor.")

    # sanitize sonrası tarih adını tekrar bul/ekle
    time_cols = [c for c in df.columns if c.lower() in {"date","incident_date","day","ds","datetime"}]
    time_col = time_cols[0] if time_cols else None
    if (time_col is None) and (date_ser_raw is not None):
        df.insert(0, "date_sanitized", pd.to_datetime(date_ser_raw, errors="coerce"))
        time_col = "date_sanitized"

    # split için tarih serisi
    date_ser = None
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        date_ser = df[time_col].copy()

    # Hedef ve X_all (tarih şimdilik dursun)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).clip(0, 1)
    X_all = df.drop(columns=[target]).copy()

    # Tarih türevleri
    if date_ser is not None:
        if "day_of_week" not in X_all.columns:
            X_all["dow"] = date_ser.dt.dayofweek
        if "month" not in X_all.columns:
            X_all["month"] = date_ser.dt.month
        if "is_weekend" not in X_all.columns:
            X_all["is_weekend"] = date_ser.dt.dayofweek.isin([5,6]).astype(int)

    # ---- Modelden düşülecek alanlar (ID/leakage) ----
    # hr_key (ID/tarih taşıyabilir), GEOID anahtarları, koordinatlar, target-türevleri
    drop_exact = []
    geo_keys = [c for c in df.columns if c.lower() in {"geoid","geo_id","tract","tract_geoid","geoid10","blockgroup","block","census_block","block_group"}]
    drop_exact += geo_keys
    id_like = [c for c in df.columns if re.fullmatch(r"(id|incident_id|case_id|row_id|index|hr_key)", c, flags=re.I)]
    drop_exact += id_like
    coord_cols = [c for c in df.columns if re.search(r"(latitude|longitude|lat|lon|lng|_lat_|_lon_|centroid_lat|centroid_lon|geometry)", c, flags=re.I)]
    drop_exact += coord_cols
    leakage_cols = [c for c in df.columns if c.startswith(target) and c != target]
    drop_exact += leakage_cols

    # ---- ZAMAN-TABANLI SPLIT ----
    if date_ser is not None:
        cutoff = pd.to_datetime(split_date) if split_date else date_ser.quantile(0.80)
        tr_mask = date_ser <= cutoff
        te_mask = date_ser >  cutoff
        X_train_full, X_test_full = X_all.loc[tr_mask], X_all.loc[te_mask]
        y_train, y_test = y.loc[tr_mask], y.loc[te_mask]
    else:
        # tarih çıkarılamadı → stratified random
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_all, y, test_size=0.2, random_state=random_state, stratify=y
        )
        cutoff = None

    # tarih ve diğer drop'ları modelden at
    drop_for_model = list(set(drop_exact + ([time_col] if time_col is not None else [])))
    X_train = X_train_full.drop(columns=[c for c in drop_for_model if c in X_train_full.columns], errors="ignore")
    X_test  = X_test_full.drop(columns=[c for c in drop_for_model if c in X_test_full.columns],  errors="ignore")

    # ---- Tip zorlamaları ----
    # Kategorik olarak zorlanacaklar: açık listeler + tüm *_range + bilinen enum alanları
    range_like = [c for c in X_train.columns if c.endswith("_range")]
    force_categorical = [c for c in [
        "category", "subcategory", "season", "season_x", "season_y",
        "day_of_week", "day_of_week_x", "day_of_week_y",
        "hour_range", "hour_range_x", "hour_range_y",
        "sf_wet_season", "sf_dry_season", "sf_fog_season",
        "poi_dominant_type"
    ] if c in X_train.columns] + range_like

    # is_* alanlarını binary numeric'e çevir
    def coerce_binary(df_):
        for col in df_.columns:
            if col.startswith("is_") or col in ["is_weekend","is_night","is_school_hour","is_business_hour","is_near_police","is_near_government"]:
                if df_[col].dtype == object:
                    # "true/false/yes/no/0/1"
                    m = df_[col].str.lower().map({"true":1,"yes":1,"y":1,"false":0,"no":0,"n":0})
                    if m.notna().any():
                        df_[col] = m.fillna(df_[col])
                df_[col] = pd.to_numeric(df_[col], errors="coerce")
        return df_
    X_train = coerce_binary(X_train.copy())
    X_test  = coerce_binary(X_test.copy())

    # numeric/cat ayrımı ve dönüştürmeler
    X_train, numeric_cols, categorical_cols = coerce_numeric(X_train.copy(), exclude_cols=force_categorical)
    X_test,  _,             _              = coerce_numeric(X_test.copy(),  exclude_cols=force_categorical)

    # pipeline
    numeric_cols_in_use = [c for c in numeric_cols if c in X_train.columns]
    categorical_cols_in_use = [c for c in categorical_cols if c in X_train.columns]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", make_ohe_compat())])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols_in_use),
                      ("cat", cat_pipe, categorical_cols_in_use)],
        remainder="drop"
    )

    # model
    model_name, base_model = pick_model()
    spw = compute_class_weight(y_train)
    if model_name == "xgb":
        base_model.set_params(scale_pos_weight=spw)
    clf = Pipeline([("pre", pre), ("model", base_model)])
    clf.fit(X_train, y_train)

    # metrikler
    if hasattr(base_model, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            roc_auc = None
    else:
        y_pred = clf.predict(X_test)
        if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
            y_pred = y_pred[:, 0]
        roc_auc = None

    f1_def = float(f1_score(y_test, y_pred))
    p_def, r_def, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    p_def, r_def = float(p_def), float(r_def)
    p_at_k = precision_at_k(y_test.values, y_proba, k=top_k) if hasattr(base_model, "predict_proba") else None

    # önemler
    pre_fit = clf.named_steps["pre"]
    feature_names_transformed = get_transformed_feature_names(pre_fit, numeric_cols_in_use, categorical_cols_in_use)

    importances_model = None
    booster = clf.named_steps["model"]
    try:
        if hasattr(booster, "get_booster"):  # XGB
            raw = booster.get_booster().get_score(importance_type="gain")
            imp = pd.Series({int(k[1:]): v for k, v in raw.items()})
            imp = imp.reindex(range(len(feature_names_transformed))).fillna(0.0)
            importances_model = pd.Series(imp.values, index=feature_names_transformed)
        elif hasattr(booster, "feature_importances_"):
            vals = booster.feature_importances_
            if vals is not None and len(vals) == len(feature_names_transformed):
                importances_model = pd.Series(vals, index=feature_names_transformed)
    except Exception:
        pass

    importances_perm = None
    try:
        pi = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1)
        importances_perm = pd.Series(pi.importances_mean, index=feature_names_transformed)
    except Exception:
        pass

    importances_shap = None
    if HAS_SHAP and model_name in ("xgb", "lgbm"):
        try:
            Xt = pre_fit.transform(X_train)
            Xs = Xt.toarray() if hasattr(Xt, "toarray") else Xt
            try:
                booster_for_shap = booster.get_booster()
            except Exception:
                booster_for_shap = getattr(booster, "booster_", booster)
            explainer = shap.TreeExplainer(booster_for_shap)
            shap_values = explainer.shap_values(Xs)
            if isinstance(shap_values, list):
                sv = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                sv = np.mean(np.abs(shap_values), axis=0)
            importances_shap = pd.Series(sv, index=feature_names_transformed)
            plt.figure()
            shap.summary_plot(shap_values, Xs, feature_names=feature_names_transformed, show=False, max_display=30)
            plt.tight_layout()
            plt.savefig(outdir / "shap_summary.png", dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            with open(outdir / "shap_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))

    sources = []
    for name, ser in [("model", importances_model), ("perm", importances_perm), ("shap", importances_shap)]:
        if ser is None:
            continue
        ser = ser.fillna(0).clip(lower=0)
        if ser.sum() <= 0:
            continue
        s_norm = ser / ser.sum()
        (outdir / f"feature_importance_{name}.csv").write_text(s_norm.to_csv(), encoding="utf-8")
        plot_and_save_importance(s_norm, outdir / f"feature_importance_{name}.png", 30, f"{name} importance")
        s_norm.name = name
        sources.append(s_norm)

    if not sources:
        raise RuntimeError("Özellik önemi hesaplanamadı (model/perm/SHAP).")

    df_imp = pd.concat(sources, axis=1).fillna(0.0)
    df_imp["ensemble"] = df_imp.mean(axis=1)
    df_imp.sort_values("ensemble", ascending=False).to_csv(outdir / "feature_importance_ensemble_transformed.csv")

    # base mapping ve ileri seçim
    base_map = build_base_mapping(df_imp.index.tolist(), numeric_cols_in_use, categorical_cols_in_use)

    base_scores: Dict[str, float] = {}
    for b, idxs in base_map.items():
        names_for_b = [feature_names_transformed[i] for i in idxs if 0 <= i < len(feature_names_transformed)]
        base_scores[b] = float(df_imp.loc[df_imp.index.intersection(names_for_b), "ensemble"].sum())
    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_ensemble_base.csv")

    # İleri seçim, bütün veride hızlı iç doğrulama ile
    all_X = pd.concat([X_train, X_test], axis=0)
    all_y = pd.concat([y_train, y_test], axis=0)
    template_model_name, template_model = pick_model()
    best_bases, forward_info = forward_select_by_rank(
        pre_fit, template_model, all_X, all_y, base_rank.index.tolist(), base_map, metric="f1", random_state=random_state
    )
    (outdir / "forward_selection.json").write_text(json.dumps(forward_info, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_cols = [c for c in best_bases if c in all_X.columns]
    selected_df = pd.concat([all_X[selected_cols], all_y.rename(target)], axis=1)
    selected_df.to_csv(desired_output_csv, index=False)

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc,
        "f1_default_0_50": f1_def,
        "precision_default_0_50": p_def,
        "recall_default_0_50": r_def,
        "precision_at_k": float(p_at_k) if p_at_k is not None else None,
        "top_k": int(top_k),
        "scale_pos_weight": float(spw),
        "n_selected_features": int(len(selected_cols)),
        "output_csv": str(desired_output_csv),
        "time_split_cutoff": str(cutoff) if cutoff is not None else None
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    import joblib
    joblib.dump(clf, outdir / "model_pipeline.joblib")
    print("✅ Özellik seçimi tamamlandı.")
    print(f"→ Seçilmiş veri: {desired_output_csv}")
    print(f"→ Çıktılar: {outdir.resolve()}")


# ------------------------- main -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=Path((Path.cwd() / "sf_crime_08.csv")).as_posix(),
                   help="Girdi CSV (örn. sf_crime_08.csv). Varsayılan: ./sf_crime_08.csv")
    p.add_argument("--target", type=str, default="Y_label", help="Hedef sütun adı (vars: Y_label)")
    p.add_argument("--outdir", type=str, default="outputs_feature_analysis", help="Çıktı klasörü")
    p.add_argument("--group_by_geoid", action="store_true", help="GEOID gruplu split (bilgi amaçlı)")
    p.add_argument("--out_csv", type=str, default="", help="Çıktı CSV (vars: sf_crime_09.csv)")
    p.add_argument("--split_date", type=str, default="", help="Sabit zaman eşiği (örn. 2025-07-01). Boşsa %80 quantile.")
    p.add_argument("--top_k", type=int, default=50, help="Precision@k için k (vars: 50)")
    args = p.parse_args()

    csv_path = args.csv or ""
    if not Path(csv_path).exists():
        env_csv = Path(str(Path.cwd() / (os.getenv("CRIME_CSV") or "")))
        if env_csv.name and env_csv.exists():
            csv_path = env_csv.as_posix()
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError("CSV bulunamadı. '--csv <dosya>' verin ya da CRIME_CSV ortam değişkenini ayarlayın.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(csv_path, low_memory=False)
    fname = Path(csv_path).name.lower()
    desired_out = Path(args.out_csv) if args.out_csv else (
        Path(csv_path).with_name("sf_crime_09.csv") if fname.startswith("sf_") else Path(csv_path).with_name("selected_output.csv")
    )

    print("🔎 Mod: Denetimli özellik analizi + seçim (time-aware, hr_key destekli)")
    print(f"📥 Girdi: {csv_path}")
    print(f"📤 Çıktı CSV: {desired_out}")
    print(f"📁 Outdir: {outdir.resolve()}")
    if args.split_date:
        print(f"⏱  Sabit eşik tarih: {args.split_date}")
    print(f"🎯 Precision@k → k = {args.top_k}")

    supervised_feature_selection(
        df_raw, target=args.target, outdir=outdir,
        group_by_geoid=args.group_by_geoid, csv_path=csv_path,
        desired_output_csv=desired_out, random_state=42,
        split_date=(args.split_date or None), top_k=int(args.top_k)
    )


if __name__ == "__main__":
    import os
    main()
