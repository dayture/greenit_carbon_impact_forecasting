"""
hyperparameter_search.py

Runs a per-horizon hyperparameter search for the 24 multi-horizon
XGBoost models (one search per forecast horizon, 1-24 hours).

Cross-validation uses TimeSeriesSplit rather than a standard/shuffled
KFold: shuffling would let a validation fold draw on data that occurs
after its training fold, which is a temporal leakage. The search is run
with GridSearchCV, and only on the training portion of the data
(everything at or after the chronological train/test split point is
withheld) - the test set is never touched during model selection.

Results are written to data/best_hyperparams.json, which model.py
reads to construct each horizon's XGBRegressor.
"""

import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from model import get_feature_columns, chronological_split_index

HORIZONS = list(range(1, 25))
N_CV_SPLITS = 3

PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.08],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def load_training_data(path: str = "data/de_lu_features.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df, get_feature_columns(df)


def search_best_params_for_horizon(
    df: pd.DataFrame,
    feature_cols: list,
    h: int,
    train_end_idx: int,
) -> dict:
    """Grid-search XGBoost hyperparameters for a single forecast horizon,
    using only rows before train_end_idx (the chronological split point)."""
    x = df[feature_cols].iloc[:train_end_idx]
    y = df[f"target_{h}h"].iloc[:train_end_idx]
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    search = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(x, y)
    return search.best_params_


def search_all_horizons(df: pd.DataFrame, feature_cols: list, train_end_idx: int) -> dict:
    best_params_dict = {}
    for h in HORIZONS:
        best_params = search_best_params_for_horizon(df, feature_cols, h, train_end_idx)
        best_params_dict[h] = best_params
        print(f"Horizon {h}/24 done - best params: {best_params}")
    return best_params_dict


def save_best_params(best_params_dict: dict, path: str = "data/best_hyperparams.json"):
    str_dict = {str(k): v for k, v in best_params_dict.items()}
    with open(path, "w") as f:
        json.dump(str_dict, f, indent=2)


if __name__ == "__main__":
    df, feature_cols = load_training_data()
    train_end_idx = chronological_split_index(df)
    best_params_dict = search_all_horizons(df, feature_cols, train_end_idx)
    save_best_params(best_params_dict)
    print("Saved data/best_hyperparams.json")
