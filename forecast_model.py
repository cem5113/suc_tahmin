# -----------------------------------------
# forecast_model.py 
# -----------------------------------------
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# ===================================================
# (1) BINARY FORECAST (suç olur mu?)
# ===================================================
def train_binary_direct_models(train_df, feature_cols, horizon_list):
    """
    Direct multi-horizon forecasting.
    Ör: horizon_list=[1,2,3,24,48,168]
    """
    models = {}

    for h in horizon_list:
        df = train_df.copy()
        df["target_h"] = df.groupby("GEOID")["Y_label"].shift(-h)

        dtrain = df.dropna(subset=["target_h"])
        X = dtrain[feature_cols]
        y = dtrain["target_h"].astype(int)

        mdl = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("mdl", mdl)
        ])

        pipe.fit(X,y)
        models[h] = pipe
        print(f"[OK] Direct horizon model trained: t+{h}")

    return models


# ===================================================
# (2) CRIME TYPE FORECAST (conditional model)
# ===================================================
def train_crime_type_model(events_df, feature_cols, type_col="crime_type"):
    """
    Sadece Y=1 satırlarda çok sınıflı model.
    """
    df = events_df[events_df["Y_label"]==1].copy()
    df = df.dropna(subset=[type_col])

    X = df[feature_cols]
    y = df[type_col].astype(str)

    mdl = LGBMClassifier(
        objective="multiclass",
        num_class=len(y.unique()),
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("mdl", mdl)
    ])

    pipe.fit(X,y)
    print(f"[OK] Crime-type model trained ({len(y.unique())} classes).")

    return pipe


# ===================================================
# (3) Forecast çalıştırma
# ===================================================
def run_forecast(future_df, binary_models, type_model, feature_cols, horizon_list):
    """
    Her horizon için binary risk ve crime type conditional probability üretir.
    """
    out = {}

    X = future_df[feature_cols]

    # tür modelinin olasılıkları
    type_proba = None
    if type_model is not None:
        type_proba = type_model.predict_proba(X)
        type_labels = type_model["mdl"].classes_
    else:
        type_labels = []

    for h in horizon_list:
        # binary risk
        p = binary_models[h].predict_proba(X)[:,1]
        df = future_df.copy()
        df["risk_p"] = p

        # crime type conditional
        if type_model is not None:
            # P(type=c | X) = p * q
            df_type = pd.DataFrame(type_proba, columns=type_labels)
            for c in type_labels:
                df[f"type_{c}_risk"] = df_type[c] * p

        out[h] = df

    return out
