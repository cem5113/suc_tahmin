# -----------------------------------------
# build_future_grid.py
# -----------------------------------------
import pandas as pd
import numpy as np
from datetime import timedelta

def make_future_calendar(start, horizon_hours):
    dates = [start + timedelta(hours=i) for i in range(horizon_hours)]
    df = pd.DataFrame({"datetime": dates})
    df["date"] = df["datetime"].dt.date
    df["event_hour"] = df["datetime"].dt.hour

    df["day_of_week"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month

    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["is_night"] = ((df["event_hour"]<=5) | (df["event_hour"]>=22)).astype(int)
    df["is_business_hour"] = df["event_hour"].between(8,18).astype(int)
    df["is_school_hour"] = df["event_hour"].between(8,15).astype(int)

    return df


def build_future_grid(train_df, geoid_list, horizon_hours=24):
    # 1) Son an
    tmax = pd.to_datetime(train_df["datetime"]).max()

    # 2) Gelecek takvim
    fut = make_future_calendar(tmax + timedelta(hours=1), horizon_hours)

    # 3) GEOID expansion
    fut = fut.assign(key=1)
    gid = pd.DataFrame({"GEOID": geoid_list, "key":1})
    fut = fut.merge(gid, on="key").drop("key", axis=1)

    # 4) STATİK özellikler → train_df’den unique değer çek
    static_cols = [
        c for c in train_df.columns 
        if c.startswith(("distance_to_","poi","population","bus_stop","train_stop")) 
    ]
    if static_cols:
        stat = train_df.groupby("GEOID")[static_cols].mean().reset_index()
        fut = fut.merge(stat, on="GEOID", how="left")

    # 5) LAG özellikleri – leakage SAFE
    train_df = train_df.sort_values(["GEOID","datetime"])

    lag_cols = ["Y_label","crime_count_past_48h","past_7d_crimes",
                "prev_crime_1h","prev_crime_3h","neighbor_crime_7d"]

    for col in lag_cols:
        if col not in train_df.columns: 
            continue

        tmp = train_df[["GEOID","datetime",col]].copy()
        tmp[col] = tmp[col].astype(float)

        # shift + rolling
        tmp["lag_series"] = tmp.groupby("GEOID")[col].shift(1)

        # Merge last known lag value into future rows
        last_vals = tmp.groupby("GEOID")["lag_series"].last().reset_index()
        fut = fut.merge(last_vals, on="GEOID", how="left", suffixes=("",""))

        # rename
        fut.rename(columns={"lag_series": f"{col}_future_lag"}, inplace=True)

    return fut
