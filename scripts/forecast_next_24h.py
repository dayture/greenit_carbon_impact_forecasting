"""
forecast_next_24h.py

Loads the 24 saved per-horizon XGBoost models and produces a live
24-hour carbon intensity forecast from the most recently known hour
(t0), without retraining anything.

Feature construction reuses add_calendar_features / add_lag_features /
add_rolling_features from feature_engineering.py so that inference-time
features are computed with exactly the same logic used at training
time (avoiding train/serve skew).
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from feature_engineering_real import add_calendar_features, add_lag_features, add_rolling_features
from model import get_feature_columns

HORIZONS = list(range(1, 25))


def load_latest_known_data(path: str = "data/de_lu_carbon_intensity.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def build_current_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar/lag/rolling features up to and including the
    latest known hour. No future targets are built - the future is not
    yet known at inference time."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def get_latest_row(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """The most recent row represents t0 - 'now' - the decision point
    the 24-hour forecast is generated from."""
    return df.iloc[[-1]][feature_cols]


def load_all_models(horizons: list = HORIZONS, model_dir: str = "models") -> dict:
    models = {}
    for h in horizons:
        model = XGBRegressor()
        model.load_model(f"{model_dir}/model_h{h}.json")
        models[h] = model
    return models


def forecast_24h(latest_row: pd.DataFrame, models: dict) -> pd.DataFrame:
    t0 = latest_row.index[0]
    rows = []
    for h in models:
        pred_val = models[h].predict(latest_row)[0]
        rows.append({
            "timestamp": t0 + pd.Timedelta(hours=h),
            "horizon": f"{h}h",
            "predicted_CI": pred_val,
        })
    return pd.DataFrame(rows)


def plot_forecast(forecast_df: pd.DataFrame):
    plt.plot(forecast_df["timestamp"], forecast_df["predicted_CI"])
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted CI (gCO2/kWh)")
    plt.title("24-Hour Carbon Intensity Forecast")
    plt.savefig("data/forecast_next_24h.png")


if __name__ == "__main__":
    df_load = load_latest_known_data()
    df_features = build_current_features(df_load)
    feature_cols = get_feature_columns(df_features)
    latest_row = get_latest_row(df_features, feature_cols=feature_cols)

    models = load_all_models()
    forecast_df = forecast_24h(latest_row, models)
    print(forecast_df)
    plot_forecast(forecast_df)
