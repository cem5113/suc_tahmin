#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crime_feature_analysis.py

Girdi:  sf_crime_08.csv (veya hedefi olan başka bir CSV)
Çıktı:  varsayılan sf_crime_09.csv (veya --out_csv ile belirtilen)

Öz: Y_label hedefiyle denetimli özellik seçimi.
- Model: XGB / LGBM, yoksa RF
- Önem: model / permutation / (varsa) SHAP
- OHE alt-kolonlarını "taban sütun" düzeyinde gruplayıp ileri seçim (F1) ile en iyi seti yazar.

Kullanım:
  python crime_feature_analysis.py --csv sf_crime_08.csv --outdir outputs_feature_analysis --group_by_geoid
  # veya
  export CRIME_CSV=sf_crime_08.csv && python crime_feature_analysis.py
"""
import os, argparse, json, re, warnings, joblib
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, Optional

import os      
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
    """Sütun adlarını güvenli/sade hale getirir."""
    new_cols, seen = [], {}
    for c in df.columns:
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
    """'Q1 (...)' → 1, 'Q5 (...)' → 5; değilse NaN."""
    if pd.isna(val):
        return np.nan
    m = re.match(r"Q(\d+)", str(val).strip())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def coerce_numeric(df: pd.DataFrame, exclude_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Sayısala çevrilebilenleri çevir; diğerlerini kategorik bırak."""
    numeric_cols, categorical_cols = [], []
    excl = set(exclude_cols)
    for c in df.columns:
        if c in excl:
            categorical_cols.append(c)
            continue
        if df[c].dtype == object:
            # Q-etiketlerini dene
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
    """Öncelik: XGB > LGBM > RF."""
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
    """XGBoost için scale_pos_weight ~ (neg/pos)."""
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).clip(0, 1)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
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
    """
    OneHotEncoder sparse_output (>=1.2) ile sparse (<=1.1) arasında uyumluluk.
    """
    try:
        return OneHotEncoder(**kwargs, handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # eski sürüm: sparse_output yok
        return OneHotEncoder(**kwargs, handle_unknown="ignore", sparse=True)


def get_transformed_feature_names(pre: ColumnTransformer,
                                  numeric_cols_in_use: Sequence[str],
                                  categorical_cols_in_use: Sequence[str]) -> List[str]:
    """
    Numeric + OHE sonrası nihai öznitelik adlarını döndürür.
    OHE için scikit-learn 1.0–1.5 uyumlu.
    """
    names: List[str] = []
    # Sayısal
    names.extend(list(numeric_cols_in_use))

    # Kategorik: get_feature_names_out dene
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(categorical_cols_in_use).tolist()
        # sklearn formatı çoğunlukla 'col_val' olur; biz daha okunur olsun diye 'col=val' formatına çevirelim
        pretty = []
        for nm in ohe_names:
            # nm örn: 'category_A' ya da 'category__A' (alt çizgili değerlerde)
            # güvenli ayırmak için ilk kategorik kolon adını prefix olarak ara
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
        # get_feature_names_out yoksa sayıyı tahmin et
        n_final = pre.transformers_[0][2].__len__() if hasattr(pre.transformers_[0][2], "__len__") else 0
        try:
            n_total = pre.transform(np.zeros((1, len(numeric_cols_in_use) + len(categorical_cols_in_use)))).shape[1]
        except Exception:
            n_total = n_final
        rest = n_total - len(numeric_cols_in_use)
        names.extend([f"ohe_f{i}" for i in range(rest)])
    return names


def build_base_mapping(feature_names: Sequence[str],
                       numeric_cols_in_use: Sequence[str],
                       categorical_cols_in_use: Sequence[str]) -> Dict[str, List[int]]:
    """
    “Taban sütun” → dönüştürülmüş indeks listesi.
    - Sayısallar: kendi adı tek indeks.
    - Kategorikler: 'col=val' ile başlayan tüm indeksler aynı tabana (col) gider.
    """
    mapping: Dict[str, List[int]] = {}
    for i, nm in enumerate(feature_names):
        # sayısal kolonda isim birebir eşleşir
        if nm in numeric_cols_in_use:
            mapping.setdefault(nm, []).append(i)
            continue
        # kategorik: "col=val" olarak ürettik
        if "=" in nm:
            base = nm.split("=", 1)[0]
            mapping.setdefault(base, []).append(i)
        else:
            # bilinmiyorsa kendisini baz al
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
        # hiç eşleşme yoksa tümünü döndür (emniyet)
        return Xt
    return Xt[:, keep_idx]


def forward_select_by_rank(pre: ColumnTransformer, base_model,
                           X: pd.DataFrame, y: pd.Series,
                           base_ranked_features: List[str],
                           feature_to_columns: Dict[str, List[int]],
                           metric: str = "f1",
                           random_state: int = 42):
    """
    Basitten karmaşığa taban-sütun ekleyerek F1 izleyip en iyi küme.
    """
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25,
                                              random_state=random_state, stratify=y)
    # pre-fit
    pre.fit(X_tr)
    # dengesizlik ayarı
    if hasattr(base_model, "set_params"):
        try:
            spw = compute_class_weight(y_tr)
            # yalnızca XGB için doğrudan set edelim; diğerlerinde etkisiz olabilir
            if base_model.__class__.__name__.lower().startswith("xgb"):
                base_model.set_params(scale_pos_weight=spw)
        except Exception:
            pass

    chosen, best_set, best_score, curve = [], [], -1.0, []
    for k, b in enumerate(base_ranked_features, start=1):
        chosen.append(b)
        Xtr_sel = select_columns_from_transformed(pre, X_tr, chosen, feature_to_columns)
        Xva_sel = select_columns_from_transformed(pre, X_va, chosen, feature_to_columns)

        mdl = base_model
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
            if yhat.ndim > 1:
                yhat = yhat[:, 0]
            f1 = f1_score(y_va, yhat)
            auc = None

        curve.append({"k": k, "feature": b, "f1": float(f1), "roc_auc": auc})
        if f1 > best_score:
            best_score, best_set = f1, list(chosen)

    info = {"forward_curve": curve, "best_k": len(best_set), "metric": metric, "best_score": float(best_score)}
    return best_set, info


# ---------- çekirdek: denetimli seçim ----------
def supervised_feature_selection(df_raw: pd.DataFrame, target: str, outdir: Path,
                                 group_by_geoid: bool, csv_path: str,
                                 desired_output_csv: Path, random_state: int = 42) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    original_cols = df_raw.columns.tolist()
    df = sanitize_columns(df_raw.copy())

    if target not in df.columns:
        raise ValueError(f"Hedef sütun '{target}' bulunamadı.")

    # sızıntı/id/geo/koordinat alanlarını düş
    drop_exact = [c for c in ["date", "time", "datetime"] if c in df.columns]
    geo_keys = [c for c in df.columns if c.lower() in {
        "geoid", "geo_id", "tract", "tract_geoid", "geoid10", "blockgroup", "block", "census_block", "block_group"
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

    # hedef & özellik
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if y.nunique() < 2:
        raise ValueError("Hedef (Y_label) tek sınıf içeriyor. Denetimli seçim için en az 2 sınıf gerekir.")
    X = df.drop(columns=[target]).copy()

    # zorla kategorik sayılacak adaylar
    force_categorical = [c for c in [
        "category", "subcategory", "season", "season_x", "season_y",
        "day_of_week", "day_of_week_x", "day_of_week_y",
        "hour_range", "hour_range_x", "hour_range_y"
    ] if c in X.columns]

    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)

    # binary string → sayısal
    for col in X.columns:
        if X[col].dtype == object and set(map(str, X[col].dropna().unique())) <= {"0", "1", "0.0", "1.0"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # split (opsiyonel GEOID gruplu)
    groups: Optional[pd.Series] = None
    if group_by_geoid:
        geo_candidates = [c for c in original_cols if str(c).lower() in {"geoid", "geo_id", "tract", "tract_geoid", "geoid10"}]
        if geo_candidates:
            try:
                _df_geoid = pd.read_csv(csv_path, usecols=geo_candidates, low_memory=False)
                if len(_df_geoid) == len(X):
                    groups = _df_geoid[geo_candidates[0]].astype(str)
            except Exception:
                groups = None

    if group_by_geoid and (groups is not None):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test, y_train, y_test = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

    # pipeline (sürüm uyumlu OHE)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe_compat())
    ])
    numeric_cols_in_use = [c for c in numeric_cols if c in X.columns]
    categorical_cols_in_use = [c for c in categorical_cols if c in X.columns]
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols_in_use),
            ("cat", cat_pipe, categorical_cols_in_use),
        ],
        remainder="drop"
    )

    model_name, base_model = pick_model()
    spw = compute_class_weight(y_train)
    if model_name == "xgb":
        base_model.set_params(scale_pos_weight=spw)

    clf = Pipeline([("pre", pre), ("model", base_model)])
    clf.fit(X_train, y_train)

    # metrikler (korumalı)
    if hasattr(base_model, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            roc_auc = None
    else:
        y_pred = clf.predict(X_test)
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]
        roc_auc = None
    f1_def = float(f1_score(y_test, y_pred))
    p_def, r_def, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    p_def, r_def = float(p_def), float(r_def)

    # --------- önemler ----------
    # nihai öznitelik adları
    pre_fit = clf.named_steps["pre"]
    feature_names_transformed = get_transformed_feature_names(pre_fit, numeric_cols_in_use, categorical_cols_in_use)

    # model importance
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

    # permutation importance
    importances_perm = None
    try:
        pi = permutation_importance(clf, X_test, y_test, n_repeats=8, random_state=random_state, n_jobs=-1)
        importances_perm = pd.Series(pi.importances_mean, index=feature_names_transformed)
    except Exception:
        pass

    # SHAP importance (opsiyonel)
    importances_shap = None
    if HAS_SHAP and model_name in ("xgb", "lgbm"):
        try:
            Xt = pre_fit.transform(X_train)
            Xs = Xt.toarray() if hasattr(Xt, "toarray") else Xt
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(Xs)
            if isinstance(shap_values, list):
                sv = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                sv = np.mean(np.abs(shap_values), axis=0)
            importances_shap = pd.Series(sv, index=feature_names_transformed)
            # SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, Xs, feature_names=feature_names_transformed, show=False, max_display=30)
            plt.tight_layout()
            plt.savefig(outdir / "shap_summary.png", dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            with open(outdir / "shap_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))

    # önemleri normalize et + kaydet
    sources = []
    for name, ser in [("model", importances_model), ("perm", importances_perm), ("shap", importances_shap)]:
        if ser is None:
            continue
        ser = ser.fillna(0).clip(lower=0)
        if ser.sum() <= 0:
            continue
        s_norm = ser / ser.sum()
        s_norm.to_csv(outdir / f"feature_importance_{name}.csv")
        plot_and_save_importance(s_norm, outdir / f"feature_importance_{name}.png", 30, f"{name} importance")
        s_norm.name = name
        sources.append(s_norm)

    if not sources:
        raise RuntimeError("Özellik önemi hesaplanamadı (model/perm/SHAP).")

    df_imp = pd.concat(sources, axis=1).fillna(0.0)
    df_imp["ensemble"] = df_imp.mean(axis=1)
    df_imp.sort_values("ensemble", ascending=False).to_csv(outdir / "feature_importance_ensemble_transformed.csv")

    # taban sütun eşlemesi (numeric kendisi, kategorik 'col=val' → base=col)
    base_map = build_base_mapping(df_imp.index.tolist(), numeric_cols_in_use, categorical_cols_in_use)

    # taban skor (ensemble toplamı)
    base_scores = {b: float(df_imp.loc[df_imp.index.intersection(
        [nm for nm in df_imp.index if nm in [f"{b}={v}" for v in []] or nm == b or nm.startswith(f"{b}=")]
    ), "ensemble"].sum()) for b in base_map.keys()}

    # daha sağlam: doğrudan indekslerden
    base_scores = {}
    for b, idxs in base_map.items():
        # idxs, df_imp.index'e göre değil; feature_names_transformed'a göre. Bu yüzden isim eşlemesi yapalım:
        names_for_b = [feature_names_transformed[i] for i in idxs if 0 <= i < len(feature_names_transformed)]
        base_scores[b] = float(df_imp.loc[df_imp.index.intersection(names_for_b), "ensemble"].sum())

    base_rank = pd.Series(base_scores).sort_values(ascending=False)
    base_rank.to_csv(outdir / "feature_importance_ensemble_base.csv")

    # ileri seçim (tabanlar üzerinden)
    template_model_name, template_model = pick_model()
    # scale_pos_weight yalnız XGB'ye set edilecek, fonksiyon içinde ele alınıyor
    best_bases, forward_info = forward_select_by_rank(
        pre_fit, template_model, X, y, base_rank.index.tolist(), base_map, metric="f1", random_state=random_state
    )
    with open(outdir / "forward_selection.json", "w", encoding="utf-8") as f:
        json.dump(forward_info, f, ensure_ascii=False, indent=2)

    # seçilen kolonlar (yalnızca taban isimleri X içinde varsa)
    selected_cols = [c for c in best_bases if c in X.columns]
    selected_df = pd.concat([X[selected_cols], y.rename(target)], axis=1)
    selected_df.to_csv(desired_output_csv, index=False)

    with open(outdir / "selected_columns.json", "w", encoding="utf-8") as f:
        json.dump(selected_cols, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc,
        "f1_default_0.50": f1_def,
        "precision_default_0.50": p_def,
        "recall_default_0.50": r_def,
        "scale_pos_weight": float(spw),
        "n_selected_features": int(len(selected_cols)),
        "output_csv": str(desired_output_csv),
        "group_split_geoid": bool(group_by_geoid and (groups is not None))
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(clf, outdir / "model_pipeline.joblib")
    print(f"✅ Özellik seçimi tamamlandı.\n→ Seçilmiş veri: {desired_output_csv}\n→ Çıktılar: {outdir.resolve()}")


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    # --csv opsiyonel: argüman yoksa ENV (CRIME_CSV) ya da anlamlı bir varsayılan denenir
    p.add_argument("--csv", type=str, default=os.getenv("CRIME_CSV", ""),
                   help="Girdi CSV (örn. sf_crime_08.csv). Verilmezse CRIME_CSV ortam değişkeni kullanılır.")
    p.add_argument("--target", type=str, default="Y_label", help="Hedef sütun adı (vars: Y_label)")
    p.add_argument("--outdir", type=str, default="outputs_feature_analysis", help="Çıktı klasörü")
    p.add_argument("--group_by_geoid", action="store_true", help="GEOID gruplu split")
    p.add_argument("--out_csv", type=str, default="", help="Çıktı CSV yolu (vars: sf_crime_09.csv)")
    args = p.parse_args()

    # CSV yolu belirleme
    csv_path = args.csv
    if not csv_path:
        # makul bir varsayılan: proje kökünde sf_crime_08.csv varsa onu dene
        guess = Path("sf_crime_08.csv")
        if guess.exists():
            csv_path = str(guess)
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(
            "CSV bulunamadı. '--csv <dosya>' verin ya da CRIME_CSV ortam değişkenini ayarlayın. "
            "Örn: --csv sf_crime_08.csv"
        )

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(csv_path, low_memory=False)
    fname = Path(csv_path).name.lower()

    # otomatik çıktı adı
    desired_out = Path(args.out_csv) if args.out_csv else (
        Path(csv_path).with_name("sf_crime_09.csv") if fname.startswith("sf_")
        else Path(csv_path).with_name("selected_output.csv")
    )

    if args.target not in df_raw.columns:
        raise ValueError(f"Hedef (target) sütunu bulunamadı: {args.target}")
    if df_raw[args.target].dropna().nunique() < 2:
        raise ValueError("Hedef (Y_label) en az 2 sınıf içermiyor. Denetimli seçim için gereklidir.")

    print("🔎 Mod: sf (denetimli özellik seçimi)")
    print(f"📥 Girdi: {csv_path}")
    print(f"📤 Çıktı CSV: {desired_out}")
    print(f"📁 Outdir: {outdir.resolve()}")
    if args.group_by_geoid:
        print("🧩 Split: GEOID gruplu")

    supervised_feature_selection(df_raw, target=args.target, outdir=outdir,
                                 group_by_geoid=args.group_by_geoid, csv_path=csv_path,
                                 desired_output_csv=desired_out, random_state=42)

if __name__ == "__main__":
    main()
