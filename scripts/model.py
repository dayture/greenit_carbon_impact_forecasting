"""
model.py

Trains 24 independent XGBoost models - one per forecast horizon
(1 to 24 hours ahead) - using the direct multi-horizon feature table
produced by feature_engineering.py, and evaluates each against a
seasonal persistence baseline.

The train/test split is strictly chronological (no shuffling): the
last TEST_RATIO fraction of the data is held out as the test set, and
training never sees data that occurs after it.

Hyperparameters are loaded per-horizon from data/best_hyperparams.json
(produced by hyperparameter_search.py).
"""

import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

HORIZONS = list(range(1, 25))
TEST_RATIO = 0.2
TARGET_COL = "carbon_intensity_gCO2_kWh"
RAW_GEN_COLS = [
    "coal_MW", "gas_MW", "hydro_MW", "wind_MW", "solar_MW",
    "biomass_MW", "waste_MW", "oil_MW", "other_MW",
    "other_renewable_MW", "geothermal_MW",
]


def load_multi_horizon_data(path: str = "data/de_lu_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Model input columns: exclude all target_ columns, the raw
    carbon_intensity target, and the raw generation columns."""
    return [
        col for col in df.columns
        if not col.startswith("target_") and col != TARGET_COL and col not in RAW_GEN_COLS
    ]


def chronological_split_index(df: pd.DataFrame, test_ratio: float = TEST_RATIO) -> int:
    return int(len(df) * (1 - test_ratio))


def seasonal_baseline_for_horizon(df: pd.DataFrame, h: int) -> pd.Series:
    """Simplified seasonal persistence baseline: forecast(t0+h) = CI(t0-24),
    i.e. the value 24 hours before the decision point, used as a fixed
    reference regardless of h."""
    return df["lag_24h"]


def train_and_evaluate_all_horizons(df: pd.DataFrame, feature_cols: list, split_index: int) -> pd.DataFrame:
    with open("data/best_hyperparams.json", "r") as f:
        best_params_all = json.load(f)

    results = []
    for h in HORIZONS:
        target_col = df[f"target_{h}h"]
        x_train = df[feature_cols].iloc[:split_index]
        x_test = df[feature_cols].iloc[split_index:]
        y_train = target_col.iloc[:split_index]
        y_test = target_col.iloc[split_index:]

        params = best_params_all[str(h)]
        model = XGBRegressor(**params, random_state=42)
        model.fit(x_train, y_train)

        os.makedirs("models", exist_ok=True)
        model.save_model(f"models/model_h{h}.json")

        y_pred_xgb = model.predict(x_test)
        y_pred_base = seasonal_baseline_for_horizon(df, h).iloc[split_index:]

        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
        rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
        mape_base = mean_absolute_percentage_error(y_test, y_pred_base) * 100

        results.append({
            "horizon": h,
            "RMSE_XGB": rmse_xgb,
            "MAPE_XGB": mape_xgb,
            "RMSE_Baseline": rmse_base,
            "MAPE_Baseline": mape_base,
        })

    return pd.DataFrame(results)


def plot_rmse_vs_horizon(results_df: pd.DataFrame):
    plt.plot(results_df["horizon"], results_df["RMSE_XGB"], label="XGBoost")
    plt.plot(results_df["horizon"], results_df["RMSE_Baseline"], label="Baseline")
    plt.xlabel("Horizon (hours)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("data/rmse_vs_horizon.png")


if __name__ == "__main__":
    df = load_multi_horizon_data()
    feature_cols = get_feature_columns(df)
    split_index = chronological_split_index(df)
    results = train_and_evaluate_all_horizons(df, feature_cols, split_index)
    print(results)
    results.to_csv("data/multi_horizon_results.csv", index=False)
    plot_rmse_vs_horizon(results)
