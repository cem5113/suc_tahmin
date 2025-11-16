#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_fr.py — SUTAM için forecast engine

Girdi:
  - FR_OUTPUT_DIR/fr_crime_09_clean.csv
  - FR_OUTPUT_DIR/model_stacking_fr09.pkl
  - FR_OUTPUT_DIR/features_fr09.json

Çıktılar (FR_OUTPUT_DIR altında):
  - risk_hourly_next24h_top3.csv
  - risk_3h_next7d_top3.csv
  - risk_daily_next365d_top5.csv

Özellikler:
  - Son 30 gün üzerinden GEOID bazlı profil
  - 24h (saatlik), 7gün (3-saat slot), 365gün (günlük) forecast
  - risk_prob, expected_crimes
  - top-N suç türü (hourly/3h: 3; daily: 5)
  - reason_1..5 + explanation_report (kural tabanlı açıklama)
"""

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta

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
    # Geçmiş suç yoğunluğu
    "neighbor_crime_7d",
    "neighbor_crime_24h",
    "neighbor_crime_72h",
    "past_7d_crimes",
    "crime_count_past_48h",
    "prev_crime_1h",
    "prev_crime_2h",
    "prev_crime_3h",

    # 911 / 311
    "911_request_count_daily_before_24_hours",
    "911_request_count_hour_range",
    "911_geo_hr_last3d",
    "911_geo_hr_last7d",
    "911_geo_last3d",
    "911_geo_last7d",
    "311_request_count",

    # POI
    "poi_risk_score",
    "poi_total_count",
    "poi_count_300m",
    "poi_risk_300m",
    "poi_count_600m",
    "poi_risk_600m",
    "poi_count_900m",
    "poi_risk_900m",

    # Ulaşım
    "bus_stop_count",
    "train_stop_count",
    "distance_to_bus",
    "distance_to_train",

    # Polis / kamu
    "distance_to_police",
    "distance_to_government_building",

    # Nüfus
    "population",
    "population_density",

    # Hava durumu
    "wx_tavg",
    "wx_tmin",
    "wx_tmax",
    "wx_prcp",
]

EXPLAIN_BIN_FEATURES = [
    "is_night",
    "is_weekend",
    "is_holiday",
    "is_business_hour",
    "is_school_hour",
    "is_near_police",
    "is_near_government",
    "wx_is_rainy",
    "wx_is_hot_day",
]

EXPLAIN_CAT_FEATURES = [
    "poi_dominant_type",
    "season_x",
    "day_of_week_x",
    "hour_range_x",
]

IGNORE_FEATURES = {
    "Y_label", "fr_snapshot_at", "fr_label_keys", "hr_key",
    "id", "date", "time", "datetime", "received_time",
}


# -------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------------------
def detect_crime_category_column(df_raw: pd.DataFrame) -> str | None:
    for c in CRIME_CAT_CANDIDATES:
        if c in df_raw.columns:
            return c
    return None


def get_feature_category(feat_name: str) -> str:
    name = feat_name.lower()

    if feat_name in IGNORE_FEATURES:
        return "ignore"

    if any(k in name for k in ["geoid", "latitude", "longitude"]):
        return "location"

    if any(
        k in name
        for k in [
            "day_of_week",
            "month",
            "season",
            "hour_range",
            "event_hour_x",
            "event_hour_y",
            "event_hour",
            "is_holiday",
            "is_weekend",
            "is_night",
            "is_school_hour",
            "is_business_hour",
        ]
    ):
        return "time"

    if name in ["category", "subcategory"]:
        return "base_crime_type"

    if "911" in name:
        return "911"
    if "311" in name:
        return "311"

    if any(k in name for k in ["population", "density"]):
        return "population"

    if any(k in name for k in ["bus_stop", "distance_to_bus"]):
        return "bus_stop"

    if any(k in name for k in ["train_stop", "distance_to_train"]):
        return "train_stop"

    if any(k in name for k in ["poi_", "poi"]):
        return "poi"

    if any(k in name for k in ["distance_to_police", "is_near_police"]):
        return "police_building"

    if any(k in name for k in ["distance_to_government", "is_near_government"]):
        return "government_building"

    if any(k in name for k in ["wx_t", "wx_prcp", "wx_", "is_hot", "is_rain"]):
        return "weather"

    if any(k in name for k in ["crime_count", "hr_cnt", "daily_cnt"]):
        return "crime_history"

    if any(k in name for k in ["neighbor_crime"]):
        return "neighbor_history"

    return "other"


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
    end = start + 3
    return f"{start:02d}-{end:02d}"


def build_topk_table(cat_stats: pd.DataFrame,
                     key_cols: list[str],
                     cat_col: str,
                     k: int = 3) -> pd.DataFrame:
    cat_stats = cat_stats.copy()
    cat_stats["total"] = cat_stats.groupby(key_cols)["count"].transform("sum")
    cat_stats["p_type_given_any"] = (
        cat_stats["count"] / cat_stats["total"].replace(0, np.nan)
    )

    def _pack(g: pd.DataFrame) -> pd.Series:
        g2 = g.sort_values("p_type_given_any", ascending=False).head(k)
        out = {}
        for i, (_, row) in enumerate(g2.iterrows(), start=1):
            out[f"top{i}_category"] = row[cat_col]
            out[f"top{i}_share"] = float(row["p_type_given_any"])
        return pd.Series(out)

    topk = cat_stats.groupby(key_cols).apply(_pack).reset_index()
    return topk


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

    # 1) Komşu ve geçmiş suç yoğunluğu
    _num_reason("neighbor_crime_7d",
                "Komşu GEOID'lerde son 7 gündeki suç yoğunluğu yüksek.", fmt="{:.1f}")
    _num_reason("neighbor_crime_24h",
                "Komşu GEOID'lerde son 24 saatte suç yoğunluğu artmış.", fmt="{:.1f}")
    _num_reason("neighbor_crime_72h",
                "Komşu GEOID'lerde son 72 saatte tekrar eden suçlar var.", fmt="{:.1f}")
    _num_reason("past_7d_crimes",
                "Bu GEOID'de son 7 gündeki suç sayısı yüksek.", fmt="{:.1f}")
    _num_reason("crime_count_past_48h",
                "Son 48 saatte bu bölgede suç tekrarları gözleniyor.", fmt="{:.1f}")
    _num_reason("prev_crime_1h",
                "Son 1 saatte bu bölgede suç olayı raporlanmış.", fmt="{:.0f}")
    _num_reason("prev_crime_2h",
                "Son 2 saatte bu bölgede suç olayı raporlanmış.", fmt="{:.0f}")
    _num_reason("prev_crime_3h",
                "Son 3 saatte bu bölgede suç olayı raporlanmış.", fmt="{:.0f}")

    # 2) 911 / 311
    _num_reason("911_request_count_daily_before_24_hours",
                "Önceki 24 saatte bu GEOID'de 911 çağrıları artmış.", fmt="{:.0f}")
    _num_reason("911_request_count_hour_range",
                "Bu saat aralığında 911 çağrıları yoğun.", fmt="{:.0f}")
    _num_reason("911_geo_hr_last3d",
                "Son 3 günde bu GEOID-saat kombinasyonunda 911 çağrıları yüksek.", fmt="{:.0f}")
    _num_reason("911_geo_hr_last7d",
                "Son 7 günde bu GEOID-saat kombinasyonunda 911 çağrıları yüksek.", fmt="{:.0f}")
    _num_reason("911_geo_last3d",
                "Son 3 günde bu GEOID için 911 çağrıları artmış.", fmt="{:.0f}")
    _num_reason("911_geo_last7d",
                "Son 7 günde bu GEOID için 911 çağrıları yüksek.", fmt="{:.0f}")
    _num_reason("311_request_count",
                "311 asayiş şikayetleri bu bölgede yoğun.", fmt="{:.0f}")

    # 3) POI
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

    # 4) Ulaşım
    _num_reason("bus_stop_count",
                "Otobüs durağı yoğunluğu yüksek; insan hareketliliği fazla.", fmt="{:.0f}")
    _num_reason("train_stop_count",
                "Tren/metro durağı yoğun; bölge aktarma noktası konumunda.", fmt="{:.0f}")

    # 5) Polis / kamu
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

    # 6) Nüfus
    _num_reason(
        "population_density",
        "Nüfus yoğunluğu yüksek; kişi sayısı arttıkça suç riski de artabiliyor.",
        fmt="{:.0f}",
    )

    # 7) Hava durumu
    if row.get("wx_is_rainy", 0) >= 0.5:
        reasons.append("Yağışlı hava koşulları mevcut; suç örüntüleri bu tip günlerde farklılaşabiliyor.")
    if row.get("wx_is_hot_day", 0) >= 0.5:
        reasons.append("Sıcak gün eşiği aşılmış; açık alan kullanımı ve hareketlilik artmış olabilir.")

    # 8) Zaman nitelikleri
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


def main() -> None:
    # ---------------------------------------------------------
    # PATHLER (ENV TABANLI)
    # ---------------------------------------------------------
    base_dir_env = os.environ.get("FR_OUTPUT_DIR") or os.environ.get("CRIME_DATA_DIR") or "."
    BASE_DIR = Path(base_dir_env).resolve()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    CSV_PATH = BASE_DIR / "fr_crime_09_clean.csv"
    MODEL_PATH = BASE_DIR / "model_stacking_fr09.pkl"
    FEAT_PATH = BASE_DIR / "features_fr09.json"

    OUT_DIR = BASE_DIR
    RISK_HOURLY_24H_PATH = OUT_DIR / "risk_hourly_next24h_top3.csv"
    RISK_3H_7D_PATH = OUT_DIR / "risk_3h_next7d_top3.csv"
    RISK_DAILY_365D_PATH = OUT_DIR / "risk_daily_next365d_top5.csv"

    print("📥 Veri:", CSV_PATH)
    print("📥 Model:", MODEL_PATH)
    print("📥 Features:", FEAT_PATH)
    print("📂 Çıktılar:", OUT_DIR)

    if not CSV_PATH.exists():
        raise RuntimeError(f"❌ fr_crime_09_clean.csv bulunamadı: {CSV_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"❌ MODEL bulunamadı: {MODEL_PATH}")
    if not FEAT_PATH.exists():
        raise RuntimeError(f"❌ features_fr09.json bulunamadı: {FEAT_PATH}")

    # ---------------------------------------------------------
    # MODEL + FEATURE LIST YÜKLE
    # ---------------------------------------------------------
    print("🤖 Kaydedilmiş STACKING modeli yükleniyor...")
    stack_model = joblib.load(MODEL_PATH)

    with open(FEAT_PATH, "r", encoding="utf-8") as f:
        feat_list = json.load(f)

    print(f"✅ Model yüklendi. Feature sayısı: {len(feat_list)}")

    # ---------------------------------------------------------
    # GEÇMİŞ VERİYİ YÜKLE (BASELINE)
    # ---------------------------------------------------------
    df_raw = pd.read_csv(CSV_PATH, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    if "date" not in df_raw.columns:
        raise RuntimeError("❌ 'date' kolonu bulunamadı!")
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    df = df_raw.copy()

    if "geoid" in df.columns:
        df["GEOID"] = df["geoid"].astype(str)
    elif "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    else:
        raise RuntimeError("❌ GEOID veya geoid kolonu yok!")

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
    # GEOID BAZLI ORTALAMA PROFİL
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

    base_num_cols = []
    base_cat_cols = []
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

    # Medyanlar (explanation için)
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
    # MODEL İÇİN FEATURE MATRİSLERİ
    # ---------------------------------------------------------
    print("\n🤖 Model tahmini başlıyor...")
    X_24h = ensure_columns(df_24h, feat_list)
    X_3h = ensure_columns(df_3h_raw, feat_list)
    X_365 = ensure_columns(df_365, feat_list)

    p_24h = stack_model.predict_proba(X_24h)[:, 1]
    p_3h = stack_model.predict_proba(X_3h)[:, 1]
    p_365 = stack_model.predict_proba(X_365)[:, 1]

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
        10,
        labels=False,
        duplicates="drop",
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

    agg_3h: dict = {
        "risk_score": "mean",
        "expected_count": "sum",
    }

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
        10,
        labels=False,
        duplicates="drop",
    ) + 1
    risk_3h["risk_level"] = risk_3h["risk_decile"].apply(assign_risk_level)
    risk_3h["risk_prob"] = risk_3h["risk_score"]
    risk_3h["expected_crimes"] = risk_3h["expected_count"]

    # ---------------------------------------------------------
    # RISK_DAILY (365 gün)
    # ---------------------------------------------------------
    agg_daily: dict = {
        "risk_score": "mean",
        "expected_count": "sum",
    }

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
        10,
        labels=False,
        duplicates="drop",
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

        df_events_target1 = df_events[df_events["Y_label"] == 1]

        global_counts = df_events_target1[crime_cat_col].value_counts()
        global_probs = (
            global_counts / global_counts.sum()
            if global_counts.sum() > 0
            else None
        )
        if global_probs is not None:
            global_top = list(zip(global_probs.index, global_probs.values))
        else:
            global_top = []

        def build_topk_table_local(
            cat_stats: pd.DataFrame,
            key_cols: list[str],
            cat_col: str,
            k: int,
        ) -> pd.DataFrame:
            cat_stats = cat_stats.copy()
            cat_stats["total"] = cat_stats.groupby(key_cols)["count"].transform("sum")
            cat_stats["p_type_given_any"] = (
                cat_stats["count"] / cat_stats["total"].replace(0, np.nan)
            )

            def _pack(g: pd.DataFrame) -> pd.Series:
                g2 = g.sort_values("p_type_given_any", ascending=False).head(k)
                out = {}
                for i, (_, row_) in enumerate(g2.iterrows(), start=1):
                    out[f"top{i}_category"] = row_[cat_col]
                    out[f"top{i}_share"] = float(row_["p_type_given_any"])
                return pd.Series(out)

            topk_ = cat_stats.groupby(key_cols).apply(_pack).reset_index()
            return topk_

        # 24h & 3h → top-3
        if "event_hour_x" in df_events_target1.columns:
            cat_stats_hr = (
                df_events_target1
                .groupby(["GEOID", "event_hour_x", crime_cat_col])
                .size()
                .rename("count")
                .reset_index()
            )

            top3_hr = build_topk_table_local(
                cat_stats_hr,
                key_cols=["GEOID", "event_hour_x"],
                cat_col=crime_cat_col,
                k=3,
            )

            if "event_hour_x" in risk_hourly_24.columns:
                risk_hourly_24 = risk_hourly_24.merge(
                    top3_hr,
                    on=["GEOID", "event_hour_x"],
                    how="left",
                )

            cat_stats_3h = (
                df_events_target1
                .groupby(["GEOID", crime_cat_col])
                .size()
                .rename("count")
                .reset_index()
            )
            top3_geo = build_topk_table_local(
                cat_stats_3h,
                key_cols=["GEOID"],
                cat_col=crime_cat_col,
                k=3,
            )
            risk_3h = risk_3h.merge(top3_geo, on="GEOID", how="left")

        # 365gün → top-5 (GEOID bazlı)
        if "GEOID" in df_events_target1.columns:
            cat_stats_day = (
                df_events_target1
                .groupby(["GEOID", crime_cat_col])
                .size()
                .rename("count")
                .reset_index()
            )
            top5_geo = build_topk_table_local(
                cat_stats_day,
                key_cols=["GEOID"],
                cat_col=crime_cat_col,
                k=5,
            )
            risk_daily = risk_daily.merge(top5_geo, on="GEOID", how="left")

        def fill_topk_probs(df_out: pd.DataFrame, k: int) -> pd.DataFrame:
            if not global_top:
                return df_out
            for i in range(1, k + 1):
                cat_col_i = f"top{i}_category"
                share_col = f"top{i}_share"
                prob_col = f"top{i}_prob"
                exp_col = f"top{i}_expected"

                if cat_col_i not in df_out.columns:
                    if len(global_top) >= i:
                        df_out[cat_col_i] = global_top[i - 1][0]
                        df_out[share_col] = float(global_top[i - 1][1])
                    else:
                        continue
                else:
                    if len(global_top) >= i:
                        df_out[cat_col_i] = df_out[cat_col_i].fillna(global_top[i - 1][0])
                        df_out[share_col] = df_out[share_col].fillna(
                            float(global_top[i - 1][1])
                        )
                    else:
                        df_out[share_col] = df_out[share_col].fillna(0.0)

                df_out[prob_col] = df_out["risk_prob"] * df_out[share_col].fillna(0.0)
                df_out[exp_col] = (
                    df_out["expected_crimes"] * df_out[share_col].fillna(0.0)
                )

            return df_out

        risk_hourly_24 = fill_topk_probs(risk_hourly_24, k=3)
        risk_3h = fill_topk_probs(risk_3h, k=3)
        risk_daily = fill_topk_probs(risk_daily, k=5)

    # ---------------------------------------------------------
    # KURAL-TABANLI AÇIKLAMA CÜMLELERİ
    # ---------------------------------------------------------
    print("\n🗣 Kural-tabanlı açıklama cümleleri üretiliyor...")

    explanation_df_24 = risk_hourly_24.apply(
        lambda row: build_explanations_for_row(row, num_medians),
        axis=1,
    )
    risk_hourly_24 = pd.concat([risk_hourly_24, explanation_df_24], axis=1)

    explanation_df_3h = risk_3h.apply(
        lambda row: build_explanations_for_row(row, num_medians),
        axis=1,
    )
    risk_3h = pd.concat([risk_3h, explanation_df_3h], axis=1)

    explanation_df_365 = risk_daily.apply(
        lambda row: build_explanations_for_row(row, num_medians),
        axis=1,
    )
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
