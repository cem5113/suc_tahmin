#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fr_reports_geo_time_crime.py — Suç, GEOID ve zaman bazlı özet raporlar

Girdi  :
  - FR_OUTPUT_DIR/fr_crime_09.csv

Çıktılar (FR_OUTPUT_DIR altında):
  - fr_report_by_geoid.csv
  - fr_report_by_time.csv
  - fr_report_by_crime.csv
  - fr_reports_summary.xlsx  (3 sheet)
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base_dir_env = os.environ.get("FR_OUTPUT_DIR") or os.environ.get("CRIME_DATA_DIR") or "."
    BASE_DIR = Path(base_dir_env).resolve()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = BASE_DIR / "fr_crime_09.csv"
    OUT_GEOID = BASE_DIR / "fr_report_by_geoid.csv"
    OUT_TIME = BASE_DIR / "fr_report_by_time.csv"
    OUT_CRIME = BASE_DIR / "fr_report_by_crime.csv"
    OUT_XLSX = BASE_DIR / "fr_reports_summary.xlsx"

    print("📥 Veri:", DATA_PATH)

    if not DATA_PATH.exists():
        raise RuntimeError(f"❌ fr_crime_09.csv bulunamadı: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if "Y_label" not in df.columns:
        raise RuntimeError("❌ 'Y_label' yok!")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # GEOID normalize
    if "GEOID" in df.columns:
        df["GEOID_norm"] = df["GEOID"].astype(str)
    elif "geoid" in df.columns:
        df["GEOID_norm"] = df["geoid"].astype(str)
    else:
        df["GEOID_norm"] = "UNKNOWN"

    # Saat / gün / ay
    if "event_hour_x" in df.columns:
        df["hour"] = df["event_hour_x"]
    elif "event_hour" in df.columns:
        df["hour"] = df["event_hour"]
    else:
        df["hour"] = np.nan

    if "day_of_week_x" in df.columns:
        df["dow"] = df["day_of_week_x"]
    elif "day_of_week" in df.columns:
        df["dow"] = df["day_of_week"]
    else:
        df["dow"] = np.nan

    if "month_x" in df.columns:
        df["month"] = df["month_x"]
    elif "month" in df.columns:
        df["month"] = df["month"]
    else:
        df["month"] = np.nan

    crime_col = None
    for c in [
        "crime_category",
        "Category",
        "category",
        "primary_type",
        "Primary_Type",
        "Offense",
        "offense",
    ]:
        if c in df.columns:
            crime_col = c
            break

    if crime_col is None:
        print("⚠️ Suç türü kolonu bulunamadı, suç bazlı raporda sadece Y_label kullanılacak.")
    else:
        print(f"🔎 Suç türü kolonu: {crime_col}")

    # ---------------------------------------------------------
    # 1) GEOID BAZLI RAPOR
    # ---------------------------------------------------------
    agg_geo: dict = {
        "Y_label": ["mean", "sum", "count"],
    }

    extra_num_cols = [
        "neighbor_crime_7d",
        "neighbor_crime_24h",
        "neighbor_crime_72h",
        "past_7d_crimes",
        "crime_count_past_48h",
        "911_request_count_hour_range",
        "911_request_count_daily_before_24_hours",
        "311_request_count",
        "population",
        "population_density",
        "poi_risk_score",
        "poi_total_count",
        "bus_stop_count",
        "train_stop_count",
    ]

    for c in extra_num_cols:
        if c in df.columns:
            agg_geo[c] = "mean"

    geo_report = df.groupby("GEOID_norm").agg(agg_geo)

    geo_report.columns = [
        "_".join([c for c in col if c]).strip("_")
        for col in geo_report.columns.to_flat_index()
    ]

    geo_report = geo_report.rename(
        columns={
            "Y_label_mean": "p_crime",
            "Y_label_sum": "crime_events",
            "Y_label_count": "total_records",
        }
    )

    geo_report["crime_rate_per_record"] = geo_report["crime_events"] / geo_report[
        "total_records"
    ].replace(0, np.nan)

    geo_report = geo_report.reset_index()
    geo_report.to_csv(OUT_GEOID, index=False)
    print("✅ GEOID bazlı rapor kaydedildi →", OUT_GEOID)

    # ---------------------------------------------------------
    # 2) ZAMAN BAZLI RAPOR
    # ---------------------------------------------------------
    time_rows: list[pd.DataFrame] = []

    if df["hour"].notna().any():
        t = df.groupby("hour")["Y_label"].agg(["mean", "sum", "count"]).reset_index()
        t["time_dim"] = "hour"
        t = t.rename(columns={"hour": "time_value"})
        time_rows.append(t)

    if df["dow"].notna().any():
        t = df.groupby("dow")["Y_label"].agg(["mean", "sum", "count"]).reset_index()
        t["time_dim"] = "dow"
        t = t.rename(columns={"dow": "time_value"})
        time_rows.append(t)

    if df["month"].notna().any():
        t = df.groupby("month")["Y_label"].agg(["mean", "sum", "count"]).reset_index()
        t["time_dim"] = "month"
        t = t.rename(columns={"month": "time_value"})
        time_rows.append(t)

    if time_rows:
        time_report = pd.concat(time_rows, axis=0, ignore_index=True)
        time_report = time_report.rename(
            columns={
                "mean": "p_crime",
                "sum": "crime_events",
                "count": "total_records",
            }
        )
        time_report["crime_rate_per_record"] = (
            time_report["crime_events"]
            / time_report["total_records"].replace(0, np.nan)
        )
        time_report.to_csv(OUT_TIME, index=False)
        print("✅ Zaman bazlı rapor kaydedildi →", OUT_TIME)
    else:
        time_report = pd.DataFrame()
        print("⚠️ Saat/gün/ay bilgisi bulunamadı, zaman raporu boş.")

    # ---------------------------------------------------------
    # 3) SUÇ TÜRÜ BAZLI RAPOR
    # ---------------------------------------------------------
    if crime_col is not None:
        crime_report = (
            df.groupby(crime_col)["Y_label"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "p_crime",
                    "sum": "crime_events",
                    "count": "total_records",
                }
            )
        )
        crime_report["crime_rate_per_record"] = (
            crime_report["crime_events"]
            / crime_report["total_records"].replace(0, np.nan)
        )
        crime_report.to_csv(OUT_CRIME, index=False)
        print("✅ Suç türü bazlı rapor kaydedildi →", OUT_CRIME)
    else:
        crime_report = pd.DataFrame()
        print("⚠️ Suç türü kolonu yok, suç bazlı rapor üretilemedi.")

    # ---------------------------------------------------------
    # 4) EXCEL ÖZET DOSYASI (3 sheet)
    # ---------------------------------------------------------
    with pd.ExcelWriter(OUT_XLSX) as writer:
        geo_report.to_excel(writer, sheet_name="by_geoid", index=False)
        if not time_report.empty:
            time_report.to_excel(writer, sheet_name="by_time", index=False)
        if not crime_report.empty:
            crime_report.to_excel(writer, sheet_name="by_crime", index=False)

    print("📊 Çok sayfalı Excel özet kaydedildi →", OUT_XLSX)
    print("✅ Raporlama tamamlandı.")


if __name__ == "__main__":
    main()
