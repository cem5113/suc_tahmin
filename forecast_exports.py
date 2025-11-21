# -----------------------------------------
# forecast_exports.py (TOP-3 / TOP-5 TYPES)
# -----------------------------------------
import pandas as pd

def top_k_types(forecast_df, k=3):
    type_cols = [c for c in forecast_df.columns if c.startswith("type_")]
    s = forecast_df[type_cols].sum().sort_values(ascending=False)
    return s.head(k)

def top3_next7d(forecasts):
    # forecasts[h] : df
    weekly = pd.concat([forecasts[h] for h in [1,2,3,4,5,6,7*24] if h in forecasts])
    return top_k_types(weekly, k=3)

def top5_next365d(forecasts):
    daily = pd.concat([forecasts[h] for h in forecasts])
    return top_k_types(daily, k=5)
