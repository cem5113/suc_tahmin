#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_fr.py — SUTAM için forecast engine (STACKING PIPELINE ÇIKTILARIYLA UYUMLU)

Girdi (aynı repoda stacking_risk_pipeline.py ürettiği çıktılar):
  - CRIME_DATA_DIR altında en güncel fr_crime*.csv (baseline için)
  - CRIME_DATA_DIR/models/preprocessor.joblib
  - CRIME_DATA_DIR/models/base_pipes.joblib
  - CRIME_DATA_DIR/models/stacking_meta_*.joblib   (auto-detect)
  - CRIME_DATA_DIR/models/threshold_*.json         (auto-detect)

Çıktılar (CRIME_DATA_DIR altında):
  - risk_hourly_next24h_top3.csv
  - risk_3h_next7d_top3.csv
  - risk_daily_next365d_top5.csv
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta
import zipfile
import tempfile

import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------------------
# SUÇ TÜRÜ KOLONU ADAYLARI
# -------------------------------------------------------------
CRIME_CAT_CANDIDATES = [
    "crime_category", "CRIME_CATEGORY",
    "Category", "CATEGORY",
    "primary_type", "Primary_Type",
    "Offense", "offense", "category",
]

EXPLAIN_NUM_FEATURES = [
    "neighbor_crime_7d", "neighbor_crime_24h", "neighbor_crime_72h",
    "past_7d_crimes", "crime_count_past_48h",
    "prev_crime_1h", "prev_crime_2h", "prev_crime_3h",
    "911_request_count_daily_before_24_hours",
    "911_request_count_hour_range",
    "911_geo_hr_last3d", "911_geo_hr_last7d",
    "911_geo_last3d", "911_geo_last7d",
    "311_request_count",
    "poi_risk_score", "poi_total_count",
    "poi_count_300m", "poi_risk_300m",
    "poi_count_600m", "poi_risk_600m",
    "poi_count_900m", "poi_risk_900m",
    "bus_stop_count", "train_stop_count",
    "distance_to_bus", "distance_to_train",
    "distance_to_police", "distance_to_government_building",
    "population", "population_density",
    "wx_tavg", "wx_tmin", "wx_tmax", "wx_prcp",
]

EXPLAIN_BIN_FEATURES = [
    "is_night", "is_weekend", "is_holiday",
    "is_business_hour", "is_school_hour",
    "is_near_police", "is_near_government",
    "wx_is_rainy", "wx_is_hot_day",
]

EXPLAIN_CAT_FEATURES = [
    "poi_dominant_type",
    "season_x", "day_of_week_x", "hour_range_x",
]

IGNORE_FEATURES = {
    "Y_label", "fr_snapshot_at", "fr_label_keys", "hr_key",
    "id", "date", "time", "datetime", "received_time",
}

# -------------------------------------------------------------
# ✅ SpatialTargetEncoder (joblib load için gerekli)
# -------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin

class SpatialTargetEncoder(BaseEstimator, TransformerMixin):
    """
    GEOID için hedef kodlama (komşu katkılı, Laplace smoothing):
      TE(g) = (sum_y(g) + m * global_mean + sum_y(neighbors(g))) / (n(g) + m + n_neighbors(g))
    - m = TE_ALPHA (Laplace smoothing)
    - neighbors(g) varsa eklenir, yoksa yalnız Laplace yapılır.
    Not: CV sırasında sızıntı yok; Pipeline içindeki her fold'un train'inde fit edilir.
    """
    def __init__(self, geo_col="GEOID", alpha=50.0, neighbors_dict=None):
        self.geo_col = geo_col
        self.alpha = float(alpha)
        self.neighbors_dict = neighbors_dict
        self.mapping_ = None
        self.global_mean_ = None

    def _geo_series(self, X):
        import pandas as pd, numpy as np
        if isinstance(X, pd.DataFrame):
            if self.geo_col in X.columns:
                return X[self.geo_col].astype(str)
            if X.shape[1] == 1:
                return X.iloc[:, 0].astype(str)
            raise ValueError(f"{self.geo_col} kolonu bulunamadı ve çoklu sütun geldi.")
        X = np.asarray(X)
        if X.ndim == 1:
            return pd.Series(X.ravel().astype(str))
        return pd.Series(X[:, 0].astype(str))

    def fit(self, X, y=None):
        import pandas as pd
        import numpy as np
        if y is None:
            raise ValueError("SpatialTargetEncoder.fit için y gerekli.")
        s_geo = self._geo_series(X)
        y = pd.Series(y).astype(float)
        if len(s_geo) != len(y):
            raise ValueError("X ve y uzunlukları uyumsuz.")

        grp = pd.DataFrame({"geo": s_geo, "y": y}).groupby("geo")["y"].agg(["sum", "count"])
        grp.columns = ["sum_y", "n"]
        self.global_mean_ = float(y.mean()) if len(y) else 0.5

        if self.neighbors_dict:
            sum_dict = grp["sum_y"].to_dict()
            n_dict   = grp["n"].to_dict()
            neigh_sum, neigh_n = [], []
            for g in grp.index:
                acc_s = 0.0
                acc_n = 0.0
                for nb in self.neighbors_dict.get(g, []):
                    acc_s += sum_dict.get(nb, 0.0)
                    acc_n += n_dict.get(nb, 0.0)
                neigh_sum.append(acc_s)
                neigh_n.append(acc_n)
            grp["neigh_sum"] = neigh_sum
            grp["neigh_n"]   = neigh_n
        else:
            grp["neigh_sum"] = 0.0
            grp["neigh_n"]   = 0.0

        m = self.alpha
        grp["te"] = (
            (grp["sum_y"] + m*self.global_mean_ + grp["neigh_sum"])
            / (grp["n"] + m + grp["neigh_n"])
        )
        self.mapping_ = grp["te"].to_dict()
        return self

    def transform(self, X):
        import pandas as pd, numpy as np
        s_geo = self._geo_series(X)
        te = s_geo.map(self.mapping_).fillna(
            self.global_mean_ if self.global_mean_ is not None else 0.5
        )
        return te.to_numpy().reshape(-1, 1)

# -------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------------------
def resolve_models_dir() -> Path:
    """
    Stacking pipeline'ın ürettiği models klasörünü bul.
    Öncelik:
      1) CRIME_DATA_DIR/models
      2) ARTIFACT_DIR/models
      3) ARTIFACT_ZIP içinden çıkar
      4) FALLBACK_DIRS altında ara
    """
    candidates = []

    crime_dir = os.getenv("CRIME_DATA_DIR", ".")
    candidates.append(Path(crime_dir) / "models")

    artifact_dir = os.getenv("ARTIFACT_DIR", "artifact/sf-crime-pipeline-output")
    candidates.append(Path(artifact_dir) / "models")

    fb = os.getenv("FALLBACK_DIRS", "")
    for p in [x.strip() for x in fb.split(",") if x.strip()]:
        candidates.append(Path(p) / "models")

    for c in candidates:
        if c.exists():
            print(f"✅ models dir bulundu → {c}")
            return c

    zpath = os.getenv("ARTIFACT_ZIP", "")
    if zpath and Path(zpath).exists():
        print(f"📦 ARTIFACT_ZIP bulundu → {zpath} | unzip ediliyor...")
        tmpdir = Path(tempfile.mkdtemp(prefix="models_"))
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        for root, dirs, files in os.walk(tmpdir):
            if os.path.basename(root) == "models":
                mdir = Path(root)
                print(f"✅ models dir zip içinden bulundu → {mdir}")
                return mdir

    raise RuntimeError(
        "❌ models klasörü bulunamadı. "
        "CRIME_DATA_DIR, ARTIFACT_DIR, ARTIFACT_ZIP veya FALLBACK_DIRS kontrol et."
    )

def detect_crime_category_column(df_raw: pd.DataFrame) -> str | None:
    for c in CRIME_CAT_CANDIDATES:
        if c in df_raw.columns:
            return c
    return None

def assign_risk_level(dec: float) -> str:
    if pd.isna(dec):
        return "unknown"
    d = int(dec)
    if d <= 3:
        return "low"
    elif d <= 7:
        return "medium"
    else:
        return "high"

def season_from_month(m: int) -> str:
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def make_hour_range_from_hour(h: int) -> str:
    start = (h // 3) * 3
    end = (start + 3) % 24
    return f"{start:02d}-{end:02d}"

def build_time_features_for_dt(dt: pd.Timestamp) -> dict:
    dow = dt.weekday()
    month = dt.month
    season = season_from_month(month)
    hour = dt.hour
    hour_bin = hour // 3

    is_weekend = 1 if dow >= 5 else 0
    is_night = 1 if (hour < 6 or hour >= 22) else 0
    is_business = 1 if (hour >= 9 and hour < 18 and dow < 5) else 0
    is_school = 1 if (hour >= 8 and hour < 17 and dow < 5) else 0

    return {
        "event_hour_x": hour,
        "event_hour": hour,
        "hour_range_x": hour_bin,
        "hour_range_y": hour_bin,
        "day_of_week_x": dow,
        "day_of_week_y": dow,
        "month_x": month,
        "month_y": month,
        "season_x": season,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_business_hour": is_business,
        "is_school_hour": is_school,
        "is_holiday": 0,
    }

def ensure_columns(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    return df_out[cols]

def _normalize_geoid_series(s: pd.Series, geoid_len: int = 11) -> pd.Series:
    z = s.astype(str).str.extract(r"(\d+)")[0]
    return z.str[-geoid_len:].str.zfill(geoid_len)

def _find_latest_file(base_dir: Path, glob_pat: str) -> Path | None:
    files = list(base_dir.glob(glob_pat))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _find_hist_csv(base_dir: Path) -> Path | None:
    candidates = [
        "fr_crime_09.csv",
    ]
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    p2 = _find_latest_file(base_dir, "fr_crime*.csv")
    if p2:
        return p2
    p3 = _find_latest_file(base_dir, "sf_crime*.csv")
    return p3

def _load_stacking_bundle(models_dir: Path):
    """
    stacking_meta_*.joblib + threshold_meta_*.json auto-detect.
    Stacking pipeline formatı:
      - stacking_meta_lgb.joblib / stacking_meta_logreg.joblib
      - threshold_meta_lgb.json  / threshold_meta_logreg.json
      - içerik: {"names": [...], "meta": meta_model}
    """
    stack_path = _find_latest_file(models_dir, "stacking_meta_*.joblib")
    if stack_path is None:
        raise RuntimeError(f"❌ stacking_meta_*.joblib bulunamadı: {models_dir}")

    thr_path = _find_latest_file(models_dir, "threshold_meta_*.json")
    if thr_path is None:
        raise RuntimeError(f"❌ threshold_meta_*.json bulunamadı: {models_dir}")

    stack_obj = joblib.load(stack_path)
    thr = json.loads(thr_path.read_text(encoding="utf-8")).get("threshold", 0.5)

    if not isinstance(stack_obj, dict) or "names" not in stack_obj or "meta" not in stack_obj:
        raise RuntimeError("❌ stacking_meta_*.joblib formatı beklenen dict yapısında değil.")

    return stack_obj, float(thr), stack_path, thr_path

def _get_expected_raw_cols(pre):
    """
    ColumnTransformer fit edildiyse raw input kolon listesini al.
    sklearn yeni sürüm: feature_names_in_
    fallback: empty list
    """
    cols = []
    if hasattr(pre, "feature_names_in_"):
        cols = list(pre.feature_names_in_)
    return cols

def _predict_stacking_proba(base_pipes: dict, stack_obj: dict, X_raw: pd.DataFrame):
    """
    stacking pipeline'ının kaydettiği base_pipes + meta ile p_stack üret.
    """
    names = stack_obj["names"]
    meta  = stack_obj["meta"]

    mats = []
    for nm in names:
        if nm not in base_pipes:
            raise RuntimeError(f"❌ base_pipes içinde '{nm}' yok. Mevcutlar: {list(base_pipes.keys())[:5]} ...")
        pipe = base_pipes[nm]
        if hasattr(pipe, "predict_proba"):
            p = pipe.predict_proba(X_raw)[:, 1]
        else:
            d = pipe.decision_function(X_raw)
            p = (d - d.min()) / (d.max() - d.min() + 1e-9)
        mats.append(p.reshape(-1,1))

    Z = np.hstack(mats)
    if hasattr(meta, "predict_proba"):
        p_stack = meta.predict_proba(Z)[:, 1]
    else:
        d = meta.decision_function(Z)
        p_stack = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return p_stack

# -------------------------------------------------------------
# EXPLANATION BLOĞU (aynen korunuyor)
# -------------------------------------------------------------
def build_explanations_for_row(row, num_medians: dict) -> pd.Series:
    reasons: list[str] = []

    def _num_reason(col, base_text, fmt="{:.2f}", mult_med: float | None = None):
        if col in row and pd.notna(row[col]):
            med = num_medians.get(col, None)
            if med is None:
                return
            thr = med * mult_med if mult_med is not None else med
            if row[col] > thr:
                try:
                    val_s = fmt.format(float(row[col]))
                    med_s = fmt.format(float(med))
                except Exception:
                    val_s = str(row[col])
                    med_s = str(med)
                reasons.append(
                    f"{base_text} (özellik: {col} = {val_s}, medyan = {med_s})."
                )

    _num_reason("neighbor_crime_7d","Komşu GEOID'lerde son 7 gündeki suç yoğunluğu yüksek.", fmt="{:.1f}")
    _num_reason("neighbor_crime_24h","Komşu GEOID'lerde son 24 saatte suç yoğunluğu artmış.", fmt="{:.1f}")
    _num_reason("neighbor_crime_72h","Komşu GEOID'lerde son 72 saatte tekrar eden suçlar var.", fmt="{:.1f}")
    _num_reason("past_7d_crimes","Bu GEOID'de son 7 gündeki suç sayısı yüksek.", fmt="{:.1f}")
    _num_reason("crime_count_past_48h","Son 48 saatte bu bölgede suç tekrarları gözleniyor.", fmt="{:.1f}")
    _num_reason("prev_crime_1h","Son 1 saatte bu bölgede suç olayı raporlanmış.", fmt="{:.0f}")
    _num_reason("prev_crime_2h","Son 2 saatte bu bölgede suç olayı raporlanmış.", fmt="{:.0f}")
    _num_reason("prev_crime_3h","Son 3 saatte bu bölgede suç olayı raporlanmış.", fmt="{:.0f}")

    _num_reason("911_request_count_daily_before_24_hours","Önceki 24 saatte bu GEOID'de 911 çağrıları artmış.", fmt="{:.0f}")
    _num_reason("911_request_count_hour_range","Bu saat aralığında 911 çağrıları yoğun.", fmt="{:.0f}")
    _num_reason("911_geo_hr_last3d","Son 3 günde bu GEOID-saat kombinasyonunda 911 çağrıları yüksek.", fmt="{:.0f}")
    _num_reason("911_geo_hr_last7d","Son 7 günde bu GEOID-saat kombinasyonunda 911 çağrıları yüksek.", fmt="{:.0f}")
    _num_reason("911_geo_last3d","Son 3 günde bu GEOID için 911 çağrıları artmış.", fmt="{:.0f}")
    _num_reason("911_geo_last7d","Son 7 günde bu GEOID için 911 çağrıları yüksek.", fmt="{:.0f}")
    _num_reason("311_request_count","311 asayiş şikayetleri bu bölgede yoğun.", fmt="{:.0f}")

    if "poi_risk_score" in row and pd.notna(row["poi_risk_score"]):
        med = num_medians.get("poi_risk_score", None)
        if med is not None and float(row["poi_risk_score"]) >= med:
            try:
                val_s = f"{float(row['poi_risk_score']):.2f}"
                med_s = f"{float(med):.2f}"
            except Exception:
                val_s = str(row["poi_risk_score"])
                med_s = str(med)

            if "poi_dominant_type" in row and isinstance(row["poi_dominant_type"], str):
                reasons.append(
                    f"Bölgede riskli POI yoğunluğu yüksek "
                    f"(özellik: poi_risk_score = {val_s}, medyan = {med_s}, "
                    f"baskın tür: '{row['poi_dominant_type']}')."
                )
            else:
                reasons.append(
                    f"Bölgede riskli POI yoğunluğu yüksek "
                    f"(özellik: poi_risk_score = {val_s}, medyan = {med_s})."
                )

    if "poi_total_count" in row and pd.notna(row["poi_total_count"]):
        med = num_medians.get("poi_total_count", None)
        if med is not None and float(row["poi_total_count"]) > med:
            val_s = f"{float(row['poi_total_count']):.0f}"
            med_s = f"{float(med):.0f}"
            reasons.append(
                f"Genel POI (işletme/yapı) yoğunluğu yüksek "
                f"(özellik: poi_total_count = {val_s}, medyan = {med_s})."
            )

    _num_reason("bus_stop_count","Otobüs durağı yoğunluğu yüksek; insan hareketliliği fazla.", fmt="{:.0f}")
    _num_reason("train_stop_count","Tren/metro durağı yoğun; bölge aktarma noktası konumunda.", fmt="{:.0f}")

    if "is_near_police" in row and row.get("is_near_police", 0) >= 0.5:
        reasons.append("Polis birimine yakın; devriye rotaları zaten bu bölgeyi kapsıyor.")
    elif "distance_to_police" in row and pd.notna(row["distance_to_police"]):
        med = num_medians.get("distance_to_police", None)
        if med is not None and float(row["distance_to_police"]) > med * 1.5:
            val_s = f"{float(row['distance_to_police']):.0f}"
            med_s = f"{float(med):.0f}"
            reasons.append(
                "Polis birimine görece uzak; devriye ile takviye edilmesi önemli olabilir "
                f"(özellik: distance_to_police = {val_s} m, medyan = {med_s} m)."
            )

    if "is_near_government" in row and row.get("is_near_government", 0) >= 0.5:
        reasons.append("Hükümet/kamu binasına yakın; kritik altyapı barındırıyor.")
    elif "distance_to_government_building" in row and pd.notna(row["distance_to_government_building"]):
        med = num_medians.get("distance_to_government_building", None)
        if med is not None and float(row["distance_to_government_building"]) > med * 1.5:
            val_s = f"{float(row['distance_to_government_building']):.0f}"
            med_s = f"{float(med):.0f}"
            reasons.append(
                "Kamu binalarına görece uzak; çevre güvenliği için devriye planlaması önemli "
                f"(özellik: distance_to_government_building = {val_s} m, medyan = {med_s} m)."
            )

    _num_reason(
        "population_density",
        "Nüfus yoğunluğu yüksek; kişi sayısı arttıkça suç riski de artabiliyor.",
        fmt="{:.0f}",
    )

    if row.get("wx_is_rainy", 0) >= 0.5:
        reasons.append("Yağışlı hava koşulları mevcut; suç örüntüleri bu tip günlerde farklılaşabiliyor.")
    if row.get("wx_is_hot_day", 0) >= 0.5:
        reasons.append("Sıcak gün eşiği aşılmış; açık alan kullanımı ve hareketlilik artmış olabilir.")

    if row.get("is_night", 0) >= 0.5:
        reasons.append("Gece saat aralığı; belirli suç türleri bu saatlerde artma eğiliminde.")
    if row.get("is_weekend", 0) >= 0.5:
        reasons.append("Hafta sonu; eğlence ve sosyal alan kullanımı artıyor.")
    if row.get("is_business_hour", 0) >= 0.5:
        reasons.append("Yoğun iş saatleri; işyeri ve yaya trafiği belirgin.")
    if row.get("is_school_hour", 0) >= 0.5:
        reasons.append("Okul saatleri; okul çevresi ve öğrenci yoğunluğu etkili olabilir.")

    if isinstance(row.get("season_x", None), str):
        reasons.append(f"Sezon: {row['season_x']}.")
    if isinstance(row.get("day_of_week_x", None), str):
        reasons.append(f"Haftanın günü: {row['day_of_week_x']}.")
    if isinstance(row.get("hour_range_x", None), str):
        reasons.append(f"Saat aralığı: {row['hour_range_x']}.")

    MAX_SHORT = 5
    short = reasons[:MAX_SHORT] or [
        "Bu GEOID ve zaman dilimi için model, geçmiş suç örüntülerine göre risk öngörmektedir."
    ]
    explanation_report = " ".join([s for s in short if s])
    while len(short) < MAX_SHORT:
        short.append("")

    return pd.Series({
        "reason_1": short[0],
        "reason_2": short[1],
        "reason_3": short[2],
        "reason_4": short[3],
        "reason_5": short[4],
        "explanation_report": explanation_report,
    })

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------
    # BASE DIR (aynı repo / CRIME_DATA_DIR)
    # ---------------------------------------------------------
    base_dir_env = os.environ.get("CRIME_DATA_DIR") or os.environ.get("FR_OUTPUT_DIR") or "."
    BASE_DIR = Path(base_dir_env).resolve()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # ✅ MODELS_DIR ve HIST CSV BURADA OLMALI
    # ---------------------------------------------------------
    MODELS_DIR = resolve_models_dir()

    CSV_PATH = _find_hist_csv(BASE_DIR)
    if CSV_PATH is None or not CSV_PATH.exists():
        raise RuntimeError(f"❌ Baseline için fr_crime*.csv bulunamadı: {BASE_DIR}")

    # ---------------------------------------------------------
    # MODELLERİ YÜKLE (stacking pipeline çıktıları)
    # ---------------------------------------------------------
    PRE_PATH  = MODELS_DIR / "preprocessor.joblib"
    BASE_PATH = MODELS_DIR / "base_pipes.joblib"

    if not PRE_PATH.exists():
        raise RuntimeError(f"❌ preprocessor.joblib yok: {PRE_PATH}")
    if not BASE_PATH.exists():
        raise RuntimeError(f"❌ base_pipes.joblib yok: {BASE_PATH}")

    pre = joblib.load(PRE_PATH)
    base_pipes = joblib.load(BASE_PATH)

    stack_obj, thr, STACK_PATH, THR_PATH = _load_stacking_bundle(MODELS_DIR)

    expected_raw_cols = _get_expected_raw_cols(pre)

    if not expected_raw_cols:
        any_pipe = next(iter(base_pipes.values()))
        prep_step = any_pipe.named_steps.get("prep", None)
        if hasattr(prep_step, "feature_names_in_"):
            expected_raw_cols = list(prep_step.feature_names_in_)

    if not expected_raw_cols:
        raw_cols = []
        for _, _, cols in getattr(pre, "transformers", []):
            if isinstance(cols, (list, tuple)):
                raw_cols.extend(cols)
        expected_raw_cols = sorted(set(raw_cols))

    print("📥 Veri (baseline):", CSV_PATH)
    print("📥 Preprocessor:", PRE_PATH)
    print("📥 Base pipes:", BASE_PATH)
    print("📥 Stacking:", STACK_PATH)
    print("📥 Threshold:", THR_PATH)
    print("🎯 Expected raw feature cols:", len(expected_raw_cols))
    print("🎚 Threshold =", thr)

    OUT_DIR = BASE_DIR
    RISK_HOURLY_24H_PATH = OUT_DIR / "risk_hourly_next24h_top3.csv"
    RISK_3H_7D_PATH      = OUT_DIR / "risk_3h_next7d_top3.csv"
    RISK_DAILY_365D_PATH = OUT_DIR / "risk_daily_next365d_top5.csv"

    # ---------------------------------------------------------
    # GEÇMİŞ VERİYİ YÜKLE (BASELINE)
    # ---------------------------------------------------------
    df_raw = pd.read_csv(CSV_PATH, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    if "date" not in df_raw.columns:
        if "datetime" in df_raw.columns:
            df_raw["date"] = pd.to_datetime(df_raw["datetime"], errors="coerce")
        else:
            raise RuntimeError("❌ date/datetime kolonu bulunamadı!")

    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    df = df_raw.copy()
    if "geoid" in df.columns:
        df["GEOID"] = df["geoid"].astype(str)
    elif "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    else:
        raise RuntimeError("❌ GEOID veya geoid kolonu yok!")

    geoid_len = int(os.getenv("GEOID_LEN", "11"))
    df["GEOID"] = _normalize_geoid_series(df["GEOID"], geoid_len)

    crime_cat_col = detect_crime_category_column(df_raw)
    if crime_cat_col:
        print(f"🔎 Suç türü kolonu: {crime_cat_col}")
    else:
        print("⚠️ Suç türü kolonu bulunamadı; top-N type alanları boş kalabilir.")

    # ---------------------------------------------------------
    # BASELINE PENCERESİ: SON 30 GÜN
    # ---------------------------------------------------------
    max_date_hist = df["date"].max().normalize()
    baseline_start = max_date_hist - pd.Timedelta(days=29)

    df_base = df[df["date"].between(baseline_start, max_date_hist)].copy()
    print(f"📅 Baseline aralığı: {baseline_start.date()} → {max_date_hist.date()}")
    print("   Satır sayısı:", len(df_base))

    geoids = sorted(df_base["GEOID"].unique())
    print("🌐 GEOID sayısı:", len(geoids))

    # ---------------------------------------------------------
    # GEOID BAZLI ORTALAMA PROFİL (baseline)
    # ---------------------------------------------------------
    time_like_cols = [
        "event_hour_x", "event_hour",
        "hour_range_x", "hour_range_y", "hour_range",
        "day_of_week_x", "day_of_week_y",
        "month_x", "month_y",
        "season_x", "season_y",
        "is_holiday", "is_weekend", "is_night",
        "is_school_hour", "is_business_hour",
    ]

    base_num_cols, base_cat_cols = [], []
    for c in df_base.columns:
        if c in ["GEOID", "date", crime_cat_col]:
            continue
        if c in time_like_cols:
            continue
        if df_base[c].dtype == object:
            base_cat_cols.append(c)
        else:
            base_num_cols.append(c)

    agg_dict: dict = {}
    for c in base_num_cols:
        agg_dict[c] = "mean"
    for c in base_cat_cols:
        agg_dict[c] = lambda x: x.value_counts().index[0] if len(x.dropna()) else np.nan

    geo_profile = (
        df_base[["GEOID"] + base_num_cols + base_cat_cols]
        .groupby("GEOID", as_index=False)
        .agg(agg_dict)
    )

    print("🧩 Geo profile shape:", geo_profile.shape)

    num_medians: dict = {}
    for c in EXPLAIN_NUM_FEATURES:
        if c in df_base.columns:
            num_medians[c] = df_base[c].median()

    def get_dynamic_base_row(gid: str):
        dfg = df_base[df_base["GEOID"] == gid]
        if len(dfg) == 0:
            base_row = geo_profile[geo_profile["GEOID"] == gid]
            if base_row.empty:
                return None
            return base_row.iloc[0].to_dict()

        dfg7 = dfg.tail(168)
        num_cols2 = dfg7.select_dtypes(include="number").columns
        base = dfg7[num_cols2].mean().to_dict()
        base["GEOID"] = gid
        return base

    # ---------------------------------------------------------
    # 1) 24 SAAT İLERİ – SAATLİK
    # ---------------------------------------------------------
    print("\n⏱ 24 saatlik (saatlik) forecast üretiliyor...")

    start_dt = max_date_hist
    future_dts_24h = [start_dt + pd.Timedelta(hours=h) for h in range(24)]

    rows_24h = []
    for gid in geoids:
        base_vals = get_dynamic_base_row(gid)
        if base_vals is None:
            continue
        for dt in future_dts_24h:
            t_feats = build_time_features_for_dt(dt)
            row = base_vals.copy()
            row.update(t_feats)
            row["GEOID"] = gid
            row["date"] = dt.normalize()
            row["forecast_datetime"] = dt
            rows_24h.append(row)

    df_24h = pd.DataFrame(rows_24h)
    print("   24h forecast satır sayısı:", len(df_24h))

    # ---------------------------------------------------------
    # 2) 7 GÜN İLERİ – 3 SAATLİK SLOT
    # ---------------------------------------------------------
    print("\n📆 7 günlük (3 saatlik slot) forecast üretiliyor...")

    future_dts_7d = []
    for d in range(7):
        day0 = max_date_hist + pd.Timedelta(days=d)
        for h in range(0, 24, 3):
            future_dts_7d.append(day0 + pd.Timedelta(hours=h))

    rows_3h = []
    for gid in geoids:
        base_vals = get_dynamic_base_row(gid)
        if base_vals is None:
            continue
        for dt in future_dts_7d:
            t_feats = build_time_features_for_dt(dt)
            row = base_vals.copy()
            row.update(t_feats)
            row["GEOID"] = gid
            row["date"] = dt.normalize()
            row["forecast_datetime"] = dt
            rows_3h.append(row)

    df_3h_raw = pd.DataFrame(rows_3h)
    print("   7g/3h forecast satır sayısı:", len(df_3h_raw))

    # ---------------------------------------------------------
    # 3) 365 GÜN İLERİ – GÜNLÜK
    # ---------------------------------------------------------
    print("\n📅 365 günlük forecast üretiliyor...")

    future_dates_365 = [max_date_hist + pd.Timedelta(days=d) for d in range(365)]

    rows_365 = []
    for gid in geoids:
        base_vals = get_dynamic_base_row(gid)
        if base_vals is None:
            continue
        for dt in future_dates_365:
            t_feats = build_time_features_for_dt(dt + pd.Timedelta(hours=12))
            row = base_vals.copy()
            row.update(t_feats)
            row["GEOID"] = gid
            row["date"] = dt.normalize()
            row["forecast_datetime"] = dt + pd.Timedelta(hours=12)
            rows_365.append(row)

    df_365 = pd.DataFrame(rows_365)
    print("   365gün forecast satır sayısı:", len(df_365))

    # ---------------------------------------------------------
    # MODEL İÇİN RAW FEATURE MATRİSLERİ
    # ---------------------------------------------------------
    print("\n🤖 Model tahmini başlıyor (base_pipes + stacking_meta)…")

    if not expected_raw_cols:
        raise RuntimeError("❌ expected_raw_cols boş. preprocessor fit kolonları okunamadı.")

    X_24h_raw  = ensure_columns(df_24h, expected_raw_cols)
    X_3h_raw   = ensure_columns(df_3h_raw, expected_raw_cols)
    X_365_raw  = ensure_columns(df_365, expected_raw_cols)

    p_24h  = _predict_stacking_proba(base_pipes, stack_obj, X_24h_raw)
    p_3h   = _predict_stacking_proba(base_pipes, stack_obj, X_3h_raw)
    p_365  = _predict_stacking_proba(base_pipes, stack_obj, X_365_raw)

    df_24h["risk_score"] = p_24h
    df_24h["expected_count"] = df_24h["risk_score"]

    df_3h_raw["risk_score"] = p_3h
    df_3h_raw["expected_count"] = df_3h_raw["risk_score"]

    df_365["risk_score"] = p_365
    df_365["expected_count"] = df_365["risk_score"]

    # ---------------------------------------------------------
    # RISK_HOURLY (24h)
    # ---------------------------------------------------------
    risk_hourly_24 = df_24h.copy()
    risk_hourly_24["risk_decile"] = pd.qcut(
        risk_hourly_24["risk_score"].rank(method="first"),
        10, labels=False, duplicates="drop"
    ) + 1
    risk_hourly_24["risk_level"] = risk_hourly_24["risk_decile"].apply(assign_risk_level)
    risk_hourly_24["risk_prob"] = risk_hourly_24["risk_score"]
    risk_hourly_24["expected_crimes"] = risk_hourly_24["expected_count"]

    # ---------------------------------------------------------
    # RISK_3H (7 gün)
    # ---------------------------------------------------------
    df_3h_raw["hour_range_3h"] = df_3h_raw["forecast_datetime"].dt.hour.apply(
        lambda h: make_hour_range_from_hour(h)
    )

    agg_3h: dict = {"risk_score": "mean", "expected_count": "sum"}
    for c in EXPLAIN_NUM_FEATURES:
        if c in df_3h_raw.columns:
            agg_3h[c] = "mean"
    for c in EXPLAIN_BIN_FEATURES:
        if c in df_3h_raw.columns:
            agg_3h[c] = "max"
    for c in EXPLAIN_CAT_FEATURES:
        if c in df_3h_raw.columns:
            agg_3h[c] = lambda x: x.value_counts().index[0] if len(x.dropna()) else np.nan

    risk_3h = (
        df_3h_raw
        .groupby(["GEOID", "date", "hour_range_3h"], as_index=False)
        .agg(agg_3h)
    )

    risk_3h["risk_decile"] = pd.qcut(
        risk_3h["risk_score"].rank(method="first"),
        10, labels=False, duplicates="drop"
    ) + 1
    risk_3h["risk_level"] = risk_3h["risk_decile"].apply(assign_risk_level)
    risk_3h["risk_prob"] = risk_3h["risk_score"]
    risk_3h["expected_crimes"] = risk_3h["expected_count"]

    # ---------------------------------------------------------
    # RISK_DAILY (365 gün)
    # ---------------------------------------------------------
    agg_daily: dict = {"risk_score": "mean", "expected_count": "sum"}
    for c in EXPLAIN_NUM_FEATURES:
        if c in df_365.columns:
            agg_daily[c] = "mean"
    for c in EXPLAIN_BIN_FEATURES:
        if c in df_365.columns:
            agg_daily[c] = "max"
    for c in EXPLAIN_CAT_FEATURES:
        if c in df_365.columns:
            agg_daily[c] = lambda x: x.value_counts().index[0] if len(x.dropna()) else np.nan

    risk_daily = df_365.groupby(["GEOID", "date"], as_index=False).agg(agg_daily)

    risk_daily["risk_decile"] = pd.qcut(
        risk_daily["risk_score"].rank(method="first"),
        10, labels=False, duplicates="drop"
    ) + 1
    risk_daily["risk_level"] = risk_daily["risk_decile"].apply(assign_risk_level)
    risk_daily["risk_prob"] = risk_daily["risk_score"]
    risk_daily["expected_crimes"] = risk_daily["expected_count"]

    # ---------------------------------------------------------
    # TOP-N SUÇ TÜRÜ (GEÇMİŞTEN, KOŞULLU OLASILIK)
    # ---------------------------------------------------------
    if crime_cat_col:
        df_events = df_raw.copy()
        if "GEOID" not in df_events.columns and "geoid" in df_events.columns:
            df_events["GEOID"] = df_events["geoid"].astype(str)
        else:
            df_events["GEOID"] = df_events["GEOID"].astype(str)

        df_events["GEOID"] = _normalize_geoid_series(df_events["GEOID"], geoid_len)

        if "Y_label" in df_events.columns:
            df_events_target1 = df_events[df_events["Y_label"] == 1]
        else:
            df_events_target1 = df_events

        global_counts = df_events_target1[crime_cat_col].value_counts()
        global_probs = (global_counts / global_counts.sum()) if global_counts.sum() > 0 else None
        global_top = list(zip(global_probs.index, global_probs.values)) if global_probs is not None else []

        def build_topk_table_local(cat_stats, key_cols, cat_col, k):
            cat_stats = cat_stats.copy()
            cat_stats["total"] = cat_stats.groupby(key_cols)["count"].transform("sum")
            cat_stats["p_type_given_any"] = cat_stats["count"] / cat_stats["total"].replace(0, np.nan)

            def _pack(g):
                g2 = g.sort_values("p_type_given_any", ascending=False).head(k)
                out = {}
                for i, (_, row_) in enumerate(g2.iterrows(), start=1):
                    out[f"top{i}_category"] = row_[cat_col]
                    out[f"top{i}_share"] = float(row_["p_type_given_any"])
                return pd.Series(out)

            return cat_stats.groupby(key_cols).apply(_pack).reset_index()

        if "event_hour_x" in df_events_target1.columns:
            cat_stats_hr = (
                df_events_target1
                .groupby(["GEOID", "event_hour_x", crime_cat_col])
                .size().rename("count").reset_index()
            )
            top3_hr = build_topk_table_local(cat_stats_hr, ["GEOID", "event_hour_x"], crime_cat_col, 3)
            if "event_hour_x" in risk_hourly_24.columns:
                risk_hourly_24 = risk_hourly_24.merge(top3_hr, on=["GEOID", "event_hour_x"], how="left")

            cat_stats_3h = (
                df_events_target1
                .groupby(["GEOID", crime_cat_col])
                .size().rename("count").reset_index()
            )
            top3_geo = build_topk_table_local(cat_stats_3h, ["GEOID"], crime_cat_col, 3)
            risk_3h = risk_3h.merge(top3_geo, on="GEOID", how="left")

        if "GEOID" in df_events_target1.columns:
            cat_stats_day = (
                df_events_target1
                .groupby(["GEOID", crime_cat_col])
                .size().rename("count").reset_index()
            )
            top5_geo = build_topk_table_local(cat_stats_day, ["GEOID"], crime_cat_col, 5)
            risk_daily = risk_daily.merge(top5_geo, on="GEOID", how="left")

        def fill_topk_probs(df_out: pd.DataFrame, k: int) -> pd.DataFrame:
            if not global_top:
                return df_out
            for i in range(1, k + 1):
                cat_col_i = f"top{i}_category"
                share_col = f"top{i}_share"
                prob_col = f"top{i}_prob"
                exp_col  = f"top{i}_expected"

                if cat_col_i not in df_out.columns:
                    if len(global_top) >= i:
                        df_out[cat_col_i] = global_top[i - 1][0]
                        df_out[share_col] = float(global_top[i - 1][1])
                    else:
                        continue
                else:
                    if len(global_top) >= i:
                        df_out[cat_col_i] = df_out[cat_col_i].fillna(global_top[i - 1][0])
                        df_out[share_col] = df_out[share_col].fillna(float(global_top[i - 1][1]))
                    else:
                        df_out[share_col] = df_out[share_col].fillna(0.0)

                df_out[prob_col] = df_out["risk_prob"] * df_out[share_col].fillna(0.0)
                df_out[exp_col]  = df_out["expected_crimes"] * df_out[share_col].fillna(0.0)
            return df_out

        risk_hourly_24 = fill_topk_probs(risk_hourly_24, k=3)
        risk_3h       = fill_topk_probs(risk_3h, k=3)
        risk_daily    = fill_topk_probs(risk_daily, k=5)

    # ---------------------------------------------------------
    # KURAL-TABANLI AÇIKLAMA CÜMLELERİ
    # ---------------------------------------------------------
    print("\n🗣 Kural-tabanlı açıklama cümleleri üretiliyor...")

    explanation_df_24 = risk_hourly_24.apply(lambda row: build_explanations_for_row(row, num_medians), axis=1)
    risk_hourly_24 = pd.concat([risk_hourly_24, explanation_df_24], axis=1)

    explanation_df_3h = risk_3h.apply(lambda row: build_explanations_for_row(row, num_medians), axis=1)
    risk_3h = pd.concat([risk_3h, explanation_df_3h], axis=1)

    explanation_df_365 = risk_daily.apply(lambda row: build_explanations_for_row(row, num_medians), axis=1)
    risk_daily = pd.concat([risk_daily, explanation_df_365], axis=1)

    # ---------------------------------------------------------
    # ÇIKTILARI KAYDET
    # ---------------------------------------------------------
    risk_hourly_24.to_csv(RISK_HOURLY_24H_PATH, index=False)
    risk_3h.to_csv(RISK_3H_7D_PATH, index=False)
    risk_daily.to_csv(RISK_DAILY_365D_PATH, index=False)

    print("\n✅ Forecast tamamlandı. Çıktılar:")
    print("   24h saatlik  →", RISK_HOURLY_24H_PATH)
    print("   7gün 3-saat  →", RISK_3H_7D_PATH)
    print("   365gün günlük→", RISK_DAILY_365D_PATH)


if __name__ == "__main__":
    main()
