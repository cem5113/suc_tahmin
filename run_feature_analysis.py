
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crime_feature_analysis.py

San Francisco suç tahmin verisi üzerinde özellik (feature) analizi, 
ön işleme ve açıklanabilirlik (SHAP) çıktıları üreten komut satırı aracı.

Kullanım örneği:
    python crime_feature_analysis.py \
        --csv fr_crime_grid_full_labeled.csv \
        --target Y_label \
        --id_col id \
        --sample_n 300000 \
        --test_size 0.2 \
        --random_state 42 \
        --outdir outputs_feature_analysis

Notlar:
- SHAP opsiyoneldir; yüklü değilse SHAP grafiklerini atlar.
- XGBoost yüklü değilse LightGBM veya RandomForest'a otomatik düşer (kurulu olana).
- Kategorizasyon: category, subcategory, crime_mix gibi metinsel sütunlar One-Hot ile işlenir.
- "Q1 (..)" gibi kantil etiketleri sayısal sıraya çevrilir (Q1->1, ..., Q5->5).
- "Assault(67%), ..." gibi karışık metinleri baseline olarak token sayımı (n-gram değil) ile basitçe işaretler.
"""

import argparse
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple

# Model ve pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, classification_report
)
from sklearn.impute import SimpleImputer

# Alternatif algoritmalar (yüklü olanı seçilecek)
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

from sklearn.ensemble import RandomForestClassifier

# SHAP opsiyonel
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import matplotlib
matplotlib.use("Agg")  # dosyaya yazmak için
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
import joblib


# ----------------------- Yardımcı Fonksiyonlar -----------------------

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sütun adlarını modellemeye uygun hale getirir.
    - boşluk, parantez, eğik çizgi, yüzde vb. karakterleri alt çizgiye çevirir.
    - birden fazla alt çizgiyi tek alt çizgiye indirger.
    - tekrar eden sütun adlarını _dupN olarak numaralar.
    """
    original_cols = df.columns.tolist()
    new_cols = []
    seen = {}
    for c in original_cols:
        nc = re.sub(r"[^\w]+", "_", str(c), flags=re.UNICODE)  # non-word -> _
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
    """
    "Q1 (..)" -> 1, "Q5 (..)" -> 5; değilse None döner.
    """
    if pd.isna(val):
        return np.nan
    m = re.match(r"Q(\d+)", str(val).strip())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def mixed_text_to_tokens(val: str) -> List[str]:
    """
    "Assault(67%), Disorderly Conduct(33%)" gibi metinleri tokenlara böler.
    Yalnızca kategori isimlerini alır (yüzdeleri atar).
    """
    if pd.isna(val):
        return []
    parts = [p.strip() for p in str(val).split(",")]
    tokens = []
    for p in parts:
        # "Assault(67%)" -> "Assault"
        t = re.sub(r"\(.*?\)", "", p).strip()
        if t:
            tokens.append(t)
    return tokens


def make_mixed_text_indicators(series: pd.Series, top_k: int = 15) -> pd.DataFrame:
    """
    En sık geçen top_k token için 0/1 sütunlar üretir.
    """
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
    """
    Sayısal dönüştürülebilen sütunları numeric'e çevirir. Dönüşemeyenler kategorik kalır.
    exclude_cols içindekilere dokunmaz.
    """
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c in exclude_cols:
            categorical_cols.append(c)
            continue
        # kantil etiketleri varsa önce onları sayıya çevirmeyi dene
        if df[c].dtype == object:
            qvals = df[c].map(extract_quantile_label)
            if qvals.notna().mean() > 0.6:
                df[c] = qvals
        # sayıya çevirme denemesi
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.8:  # çoğunluğu sayısalsa sayısal olarak al
                df[c] = conv
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    return df, numeric_cols, categorical_cols


def pick_model():
    """
    Kurulu olan en iyi algoritmayı seç.
    Öncelik: XGBoost > LightGBM > RandomForest
    """
    if HAS_XGB:
        return "xgb", XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1
        )
    if HAS_LGBM:
        return "lgbm", LGBMClassifier(
            n_estimators=1000,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1
        )
    return "rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )


def compute_class_weight(y: pd.Series) -> float:
    """
    XGBoost için scale_pos_weight ~ (neg/pos)
    """
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def plot_and_save_importance(importances: pd.Series, out_png: Path, top_n: int = 30, title: str = "Feature Importance"):
    top = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, int(0.25 * top_n))))
    top.iloc[::-1].plot(kind="barh")  # tek plot, renk belirtme yok
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Girdi CSV yolu (örn. fr_crime_grid_full_labeled.csv)")
    parser.add_argument("--target", type=str, default="Y_label", help="Hedef sütun adı (binary 0/1)")
    parser.add_argument("--id_col", type=str, default="id", help="ID sütunu (varsa)")
    parser.add_argument("--sample_n", type=int, default=0, help="İsteğe bağlı örneklem boyutu (0=hepsi)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test oranı")
    parser.add_argument("--random_state", type=int, default=42, help="Rastgele tohum")
    parser.add_argument("--outdir", type=str, default="outputs_feature_analysis", help="Çıktı klasörü")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Veri yükleme
    df = pd.read_csv(args.csv, low_memory=False)
    df = sanitize_columns(df)

    # 2) Hedef ve potansiyel sızıntı sütunları
    target = args.target if args.target in df.columns else None
    if target is None:
        raise ValueError(f"Hedef sütun '{args.target}' bulunamadı. Mevcut sütunlar: {df.columns.tolist()[:20]} ...")

    # Bilinen zaman damgaları (modelde kullanmak istemezsek buradan çıkarabiliriz)
    drop_exact = [c for c in ["date", "time", "datetime"] if c in df.columns]

    # Coğrafi koordinatlar (opsiyonel olarak düş)
    coord_cols = [c for c in df.columns if re.search(r"^latitude|^longitude", c)]
    # Hedef kaçağı: Y_label türevleri
    leakage_cols = [c for c in df.columns if c.startswith(target) and c != target]

    # ID sütunu
    if args.id_col in df.columns:
        drop_exact.append(args.id_col)

    # 3) Basit örnekleme (isteğe bağlı)
    if args.sample_n and args.sample_n > 0 and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.random_state)

    # 4) Karışık metin alanları ve kategorik metinler
    text_like_cols = []
    for cand in ["crime_mix", "crime_mix_x", "crime_mix_y"]:
        if cand in df.columns:
            text_like_cols.append(cand)

    mix_df_list = []
    for col in text_like_cols:
        md = make_mixed_text_indicators(df[col], top_k=15)
        if md.shape[1] > 0:
            mix_df_list.append(md.add_prefix(f"{col}_"))
    if mix_df_list:
        mix_block = pd.concat(mix_df_list, axis=1)
        df = pd.concat([df, mix_block], axis=1)

    # 5) Düşülecek sütunlar
    to_drop = set(drop_exact + coord_cols + leakage_cols + text_like_cols)
    keep_cols = [c for c in df.columns if c not in to_drop]
    df = df[keep_cols + [target]]  # hedefi sonda garanti et

    # 6) Hedef/özellik ayırma
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # 7) Sayısal/kategorik ayrımı ve sayısallaştırma
    #    category, subcategory gibi sütunları kategorik olarak zorla
    force_categorical = [c for c in ["category", "subcategory", "season", "season_x", "season_y",
                                     "day_of_week", "day_of_week_x", "day_of_week_y",
                                     "hour_range", "hour_range_x", "hour_range_y"]
                         if c in X.columns]

    X, numeric_cols, categorical_cols = coerce_numeric(X, exclude_cols=force_categorical)

    # Bazı bool benzeri sütunlar 0/1 float/str olabilir -> sayıya çevir
    for col in X.columns:
        if X[col].dtype == object and set(X[col].dropna().unique()) <= {"0", "1", "0.0", "1.0"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 8) Eğitim/validasyon ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # 9) Pipeline
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in numeric_cols if c in X.columns]),
            ("cat", cat_pipe, [c for c in categorical_cols if c in X.columns]),
        ],
        remainder="drop"
    )

    model_name, base_model = pick_model()

    # Dengesizlik ayarı (XGB/LGBM desteklerse)
    spw = compute_class_weight(y_train)
    if model_name == "xgb":
        base_model.set_params(scale_pos_weight=spw)
    elif model_name == "lgbm":
        base_model.set_params(class_weight=None)  # LGBM farklı ölçekler kullanabilir
    elif model_name == "rf":
        pass

    clf = Pipeline(steps=[("pre", pre), ("model", base_model)])

    # 10) Eğitim
    clf.fit(X_train, y_train)

    # 11) Değerlendirme
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf.named_steps["model"], "predict_proba") else clf.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int) if y_proba.ndim == 1 else y_proba

    roc_auc = roc_auc_score(y_test, y_proba) if y_proba.ndim == 1 else np.nan
    pr_auc = average_precision_score(y_test, y_proba) if y_proba.ndim == 1 else np.nan
    f1 = f1_score(y_test, y_pred)
    prec, rec, f1s, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc) if roc_auc==roc_auc else None,
        "pr_auc": float(pr_auc) if pr_auc==pr_auc else None,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "scale_pos_weight": float(spw),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 12) Önem grafikleri (model-based)
    #    Not: Pipeline içindeki OHE sonrası feature isimlerini almak
    #    için ColumnTransformer'dan üretilen adları kullanacağız.
    feature_names = []
    try:
        pre_fit = clf.named_steps["pre"]
        ohe = pre_fit.named_transformers_["cat"].named_steps["ohe"]
        num_cols_final = pre_fit.transformers_[0][2]
        cat_cols_final = pre_fit.transformers_[1][2]
        ohe_feature_names = ohe.get_feature_names_out(cat_cols_final).tolist()
        feature_names = list(num_cols_final) + ohe_feature_names
    except Exception:
        # Yedek: orijinal sütunlar
        feature_names = [f"f{i}" for i in range(clf.named_steps["pre"].transform(X_train).shape[1])]

    # Model türüne göre feature importance
    importances = None
    if model_name in ("xgb", "lgbm", "rf"):
        try:
            if model_name == "xgb":
                booster = clf.named_steps["model"]
                # XGBoost: 'gain' daha bilgilendirici
                try:
                    score_type = "gain"
                    booster_importance = booster.get_booster().get_score(importance_type=score_type)
                    # get_score feature index bazlı döner: "f0","f1"...
                    imp_series = pd.Series({int(k[1:]): v for k, v in booster_importance.items()})
                    imp_series = imp_series.reindex(range(len(feature_names))).fillna(0.0)
                    importances = pd.Series(imp_series.values, index=feature_names)
                except Exception:
                    # yedek: feature_importances_
                    importances = pd.Series(booster.feature_importances_, index=feature_names)
            elif model_name == "lgbm":
                booster = clf.named_steps["model"]
                importances = pd.Series(booster.feature_importances_, index=feature_names)
            elif model_name == "rf":
                booster = clf.named_steps["model"]
                importances = pd.Series(booster.feature_importances_, index=feature_names)
        except Exception:
            importances = None

    if importances is not None:
        importances.sort_values(ascending=False).to_csv(outdir / "feature_importance_model.csv", index=True)
        plot_and_save_importance(importances, out_png=outdir / "feature_importance_model.png", top_n=30,
                                 title=f"Model Importance ({model_name})")

    # 13) Permutation importance (daha ağır ama model-agnostic)
    try:
        pi = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        pi_series = pd.Series(pi.importances_mean, index=feature_names).sort_values(ascending=False)
        pi_series.to_csv(outdir / "feature_importance_permutation.csv", index=True)
        plot_and_save_importance(pi_series, out_png=outdir / "feature_importance_permutation.png", top_n=30,
                                 title="Permutation Importance")
    except Exception:
        pass

    # 14) SHAP açıklanabilirlik (opsiyonel)
    if HAS_SHAP and model_name in ("xgb", "lgbm",):
        try:
            # Eğitim verisini transform et (sparse olabilir -> dense'e dikkat!)
            Xt = clf.named_steps["pre"].transform(X_train)
            Xs = Xt.toarray() if hasattr(Xt, "toarray") else Xt

            model = clf.named_steps["model"]
            if model_name == "xgb":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xs)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xs)

            # SHAP summary plot (top özellikler)
            plt.figure()
            shap.summary_plot(shap_values, Xs, feature_names=feature_names, show=False, max_display=30)
            plt.tight_layout()
            plt.savefig(outdir / "shap_summary.png", dpi=200, bbox_inches="tight")
            plt.close()

            # Global SHAP önemleri
            if isinstance(shap_values, list):  # multiclass ise (beklemiyoruz)
                sv = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                sv = np.mean(np.abs(shap_values), axis=0)
            shap_imp = pd.Series(sv, index=feature_names).sort_values(ascending=False)
            shap_imp.to_csv(outdir / "feature_importance_shap.csv", index=True)
            plot_and_save_importance(shap_imp, out_png=outdir / "feature_importance_shap.png", top_n=30,
                                     title="SHAP Global Importance")
        except Exception as e:
            with open(outdir / "shap_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))

    # 15) Eksik değer oranları
    na_rates = X.isna().mean().sort_values(ascending=False)
    na_rates.to_csv(outdir / "missing_rates.csv")
    # Basit bir bar grafiği (ilk 25 sütun)
    top_na = na_rates.head(25)
    plt.figure(figsize=(8, max(4, int(0.25 * len(top_na)))))
    top_na.iloc[::-1].plot(kind="barh")
    plt.title("Eksik Değer Oranları (İlk 25)")
    plt.tight_layout()
    plt.savefig(outdir / "missing_rates_top25.png", dpi=200)
    plt.close()

    # 16) Model ve öznitelik isimlerini kaydet
    joblib.dump(clf, outdir / "model_pipeline.joblib")
    with open(outdir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # 17) Rapor özetini yaz
    summary = {
        "metrics": metrics,
        "n_features_after_encoding": len(feature_names),
        "columns_numeric": [c for c in numeric_cols if c in X.columns],
        "columns_categorical": [c for c in categorical_cols if c in X.columns],
        "dropped_columns": sorted(list(to_drop)),
        "model": model_name
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Özellik Analizi Tamamlandı ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Çıktılar: {outdir.resolve()}")

if __name__ == "__main__":
    main()
