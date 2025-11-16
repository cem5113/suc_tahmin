#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stacking_feature_shap.py — fr_crime_09.csv üzerinden:
  - CSV temizleme (ParserError önlemi) → fr_crime_09_clean.csv
  - Son FR_ML_DAYS gün üzerinden class-balanced stacking modeli eğitimi
  - RandomForest ile numerik feature importance (exogenous effects)
  - Opsiyonel SHAP feature importance (shap varsa)

Çıktılar (FR_OUTPUT_DIR altında):
  - fr_crime_09_clean.csv
  - model_stacking_fr09.pkl
  - features_fr09.json
  - feature_importances_stacking_fr09.csv
  - shap_feature_importance_fr09.csv   (shap varsa)
"""

import os
import csv
import math
import json
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Geo
import geopandas as gpd
from libpysal.weights import Queen, Rook

# ML — Stacking
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib


# ------------------------------------------------------------
# Yardımcı: GEOID normalize
# ------------------------------------------------------------
def normalize_geoid(val, length: int = 11) -> str | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()

    # 6075010101.0 → 6075010101
    if s.endswith(".0"):
        s = s[:-2]

    # Bilimsel gösterim (6.07501E+09 gibi)
    try:
        if "e" in s.lower():
            s = str(int(float(s)))
    except Exception:
        pass

    # Sadece rakamsa, sıfır dolduralım
    if s.isdigit() and length:
        s = s.zfill(length)

    return s


def main() -> None:
    # ============================================================
    # 0) PATH / ENV AYARLARI
    # ============================================================
    base_dir = Path(os.environ.get("CRIME_DATA_DIR", ".")).resolve()
    output_dir = Path(os.environ.get("FR_OUTPUT_DIR", base_dir / "fr_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = base_dir / "fr_crime_09.csv"
    clean_path = output_dir / "fr_crime_09_clean.csv"

    # sf_cells.geojson yolu
    sf_cells_path_env = os.environ.get("SF_CELLS_PATH", "")
    if sf_cells_path_env:
        shp_path = Path(sf_cells_path_env).resolve()
    else:
        # varsayılan: CRIME_DATA_DIR içinde sf_cells.geojson
        shp_path = base_dir / "sf_cells.geojson"

    print("📂 CRIME_DATA_DIR :", base_dir)
    print("📂 FR_OUTPUT_DIR  :", output_dir)
    print("📄 RAW_PATH       :", raw_path)
    print("📄 CLEAN_PATH     :", clean_path)
    print("🗺  SF_CELLS_PATH :", shp_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"❌ fr_crime_09.csv bulunamadı: {raw_path}")
    if not shp_path.exists():
        raise FileNotFoundError(f"❌ sf_cells.geojson bulunamadı: {shp_path}")

    # ============================================================
    # 1) CSV TEMİZLEME → fr_crime_09_clean.csv
    # ============================================================
    bad_rows = 0
    print(f"\n📥 Orijinal CSV satır satır okunuyor: {raw_path}")

    with open(raw_path, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(clean_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader, None)
        if header is None:
            raise RuntimeError("❌ CSV boş görünüyor, header yok.")

        expected_cols = len(header)
        writer.writerow(header)
        print(f"🔧 Beklenen kolon sayısı (header'dan): {expected_cols}")

        for i, row in enumerate(reader, start=2):
            if len(row) == expected_cols:
                writer.writerow(row)
            else:
                bad_rows += 1
                # GitHub logları şişmesin diye satır print'ini kapalı tutuyoruz
                # print(f"⚠️ Hatalı satır: {i}  → kolon sayısı: {len(row)} (beklenen: {expected_cols})")
                pass

    print(f"\n🧹 TEMİZLEME TAMAM. Toplam hatalı satır sayısı: {bad_rows}")
    print("✔ Temiz CSV yazıldı:", clean_path)

    # ============================================================
    # 2) VERİYİ YÜKLE & DÜZENLE
    # ============================================================
    df = pd.read_csv(clean_path, low_memory=False)
    print("📊 Temiz veri shape:", df.shape)

    df.columns = [c.strip() for c in df.columns]

    # GEOID / geoid
    if "geoid" in df.columns:
        df["geoid"] = df["geoid"].astype(str)
    elif "GEOID" in df.columns:
        df["geoid"] = df["GEOID"].astype(str)
    else:
        raise Exception("❌ CSV içinde 'GEOID' veya 'geoid' kolonu yok.")

    df["geoid"] = df["geoid"].str.replace(r"\.0$", "", regex=True).str.zfill(11)

    if "date" not in df.columns:
        raise Exception("❌ 'date' kolonu bulunamadı.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if "Y_label" not in df.columns:
        raise Exception("❌ 'Y_label' kolonu yok, model eğitimi yapılamaz.")

    print("🔎 Kolonlar:", df.columns.tolist())

    # ============================================================
    # 3) 3-SAATLİK ZAMAN ARALIĞI & PIVOT (Son 30 gün, log amaçlı)
    # ============================================================
    if "hour" not in df.columns:
        df["hour"] = df["date"].dt.hour

    df["hour_range"] = (df["hour"] // 3).astype(int)

    df_grp = df.groupby(["geoid", "date", "hour_range"]).size().reset_index(name="crime_count")

    last_date = df_grp["date"].max()
    df30 = df_grp[df_grp["date"] >= last_date - timedelta(days=30)].copy()
    df30["datetime"] = df30["date"] + pd.to_timedelta(df30["hour_range"] * 3, unit="h")

    pivot_3h = df30.pivot_table(
        index="datetime",
        columns=["geoid", "hour_range"],
        values="crime_count",
        fill_value=0
    ).sort_index(axis=1)

    print("✔ 3-hour pivot shape:", pivot_3h.shape)

    # ============================================================
    # 4) GEOMETRİ & MEKÂNSAL AĞIRLIK MATRİSİ (sf_cells.geojson)
    # ============================================================
    print("\n📥 GeoJSON okunuyor:", shp_path)
    gdf = gpd.read_file(shp_path)
    print("✔ GeoJSON yüklendi. Sütunlar:", list(gdf.columns))

    geojson_geo_col = "geoid"
    if geojson_geo_col not in gdf.columns:
        raise Exception(f"❌ GeoJSON içinde '{geojson_geo_col}' kolonu yok.")

    gdf["GEOID_norm"] = gdf[geojson_geo_col].apply(lambda x: normalize_geoid(x, 11))
    df["GEOID_norm"] = df["geoid"].apply(lambda x: normalize_geoid(x, 11))

    print("🔎 GeoJSON ilk 5 GEOID_norm:", gdf["GEOID_norm"].head().tolist())
    print("🔎 CSV ilk 5 GEOID_norm    :", df["GEOID_norm"].head().tolist())

    geojson_ids = set(gdf["GEOID_norm"].dropna().unique())
    csv_ids = set(df["GEOID_norm"].dropna().unique())
    common_ids = geojson_ids.intersection(csv_ids)

    print(f"📊 GeoJSON GEOID sayısı: {len(geojson_ids)}")
    print(f"📊 CSV GEOID sayısı    : {len(csv_ids)}")
    print(f"📊 ORTAK GEOID sayısı  : {len(common_ids)}")

    if len(common_ids) == 0:
        raise Exception("❌ GeoJSON ile fr_crime_09.csv arasında GEOID eşleşmesi yok.")

    gdf2 = gdf[gdf["GEOID_norm"].isin(common_ids)].copy().set_index("GEOID_norm")
    print("✔ Eşleşen hücre sayısı:", len(gdf2), "/", len(geojson_ids))

    print("⏳ Queen & Rook mekânsal ağırlık matrisleri oluşturuluyor...")
    W_queen = Queen.from_dataframe(gdf2)
    W_queen.transform = "r"
    W_rook = Rook.from_dataframe(gdf2)
    W_rook.transform = "r"

    print("✔ Queen neighbors (örnek):", dict(list(W_queen.neighbors.items())[:3]))
    print("✔ Rook neighbors  (örnek):", dict(list(W_rook.neighbors.items())[:3]))
    print("📌 Mekânsal komşuluk yapısı hazır (STARIMA vb. için kullanılabilir).")

    # ============================================================
    # 5) STACKING ML MODEL (FULL FEATURE SET, Son FR_ML_DAYS gün)
    # ============================================================
    ML_DAYS = int(os.environ.get("FR_ML_DAYS", "365"))
    max_date = df["date"].max()
    cutoff_ml = max_date - pd.Timedelta(days=ML_DAYS)

    df_ml = df[df["date"] >= cutoff_ml].copy()
    df_ml = df_ml.dropna(subset=["Y_label"]).reset_index(drop=True)

    print(f"\n🧪 ML için kullanılan tarih aralığı: {cutoff_ml.date()} → {max_date.date()}")
    print(f"📏 ML veri satır sayısı (son {ML_DAYS} gün): {len(df_ml)}")

    pos = df_ml[df_ml["Y_label"] == 1]
    neg = df_ml[df_ml["Y_label"] == 0]

    NEG_FRAC = float(os.environ.get("FR_NEG_FRAC", "0.3"))
    if len(neg) > 0 and NEG_FRAC > 0:
        neg_sample = neg.sample(frac=NEG_FRAC, random_state=42)
        df_train = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=42)
    else:
        df_train = pos.copy()

    print(
        f"🔎 Denge sonrası: pozitif={ (df_train['Y_label']==1).sum() }, "
        f"negatif={ (df_train['Y_label']==0).sum() }"
    )

    y = df_train["Y_label"]
    drop_cols = ["Y_label", "date"]
    X = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include="object").columns

    print(f"🔢 Feature sayısı → numeric={len(num_cols)}, categorical={len(cat_cols)}")

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=80,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=150,
                learning_rate=0.07,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]

    meta = LogisticRegression(max_iter=300)

    stack_model = Pipeline([
        ("prep", preprocess),
        ("stack", StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=3,
            n_jobs=-1,
            passthrough=False,
        )),
    ])

    print("\n⏳ STACKING modeli (hızlı mod) eğitiliyor...")
    stack_model.fit(X, y)
    print("✔ STACKING model fit edildi (hızlı mod).")

    # ============================================================
    # 6) EXOGENOUS EFFECTS (RF IMPORTANCE)
    # ============================================================
    print("\n⏳ RandomForest ile numerik feature importance hesaplanıyor...")
    rf_imp = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )
    rf_imp.fit(X[num_cols], y)

    imp_norm = rf_imp.feature_importances_ / rf_imp.feature_importances_.sum()
    effects = dict(zip(num_cols, imp_norm))

    print("✔ Exogenous feature etkileri hesaplandı.")

    # ============================================================
    # 7) FEATURE IMPORTANCE CSV + MODEL & FEATURE LIST KAYIT
    # ============================================================
    feat_imp_df = pd.DataFrame({
        "feature": num_cols,
        "importance": [effects.get(c, 0.0) for c in num_cols],
    }).sort_values("importance", ascending=False)

    feat_imp_path = output_dir / "feature_importances_stacking_fr09.csv"
    feat_imp_df.to_csv(feat_imp_path, index=False)

    print("\n💾 Feature importance kaydedildi:", feat_imp_path)
    print("🔝 En önemli ilk 20 feature:")
    print(feat_imp_df.head(20))

    model_path = output_dir / "model_stacking_fr09.pkl"
    feat_path = output_dir / "features_fr09.json"

    joblib.dump(stack_model, model_path)
    print("💾 Model kaydedildi →", model_path)

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, indent=2, ensure_ascii=False)
    print("💾 Feature list kaydedildi →", feat_path)

    # ============================================================
    # 8) OPSİYONEL: SHAP FEATURE IMPORTANCE
    # ============================================================
    shap_path = output_dir / "shap_feature_importance_fr09.csv"
    try:
        import shap

        print("\n⏳ SHAP TreeExplainer ile numerik feature SHAP değeri hesaplanıyor...")
        sample_n = min(5000, len(X))
        X_sample = X[num_cols].sample(n=sample_n, random_state=42)

        explainer = shap.TreeExplainer(rf_imp)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_pos = shap_values[1]
        else:
            shap_values_pos = shap_values

        mean_abs = np.mean(np.abs(shap_values_pos), axis=0)
        shap_imp_df = pd.DataFrame({
            "feature": num_cols,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_imp_df.to_csv(shap_path, index=False)
        print("💾 SHAP feature importance kaydedildi:", shap_path)
        print("🔝 SHAP'a göre ilk 20 feature:")
        print(shap_imp_df.head(20))

    except ImportError:
        print("⚠️ 'shap' paketi yüklü değil, SHAP analizi atlandı.")
    except Exception as e:
        print(f"⚠️ SHAP analizi sırasında hata oluştu: {e}")

    print("\n✅ fr_crime_09 için feature analysis + stacking + (opsiyonel) SHAP aşaması TAMAMLANDI.")


if __name__ == "__main__":
    main()
