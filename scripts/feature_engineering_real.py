"""
feature_engineering.py

Builds the supervised-learning feature table for direct multi-horizon
forecasting of grid carbon intensity.

Feature groups:
  1. Calendar features: hour, day_of_week, month, is_weekend, plus
     cyclical sin/cos encodings so that e.g. hour 23 and hour 0 are
     represented as adjacent rather than maximally distant.
  2. Lag features: lag_0h (the value at the decision point t0 itself,
     since it is known information) and lag_1h / lag_24h / lag_168h
     (1 hour, 1 day, 1 week back).
  3. Rolling features: 24-hour rolling mean/std, computed on values
     shifted by 1 hour so the current row's own target never leaks
     into its own feature.
  4. Targets: target_1h ... target_24h, i.e. CI(t0+h) for h = 1..24
     (direct multi-horizon targets, built with a negative shift).

Row loss after dropna(): the first 168 rows lack a full lag_168h
history, and the last 24 rows lack a full target_24h future - both
are dropped before the dataset is used for training.
"""

import pandas as pd
import numpy as np

TARGET_COL = "carbon_intensity_gCO2_kWh"
LAG_HOURS = [1, 24, 168]
ROLLING_WINDOW = 24
FORECAST_HORIZONS = 24


def load_carbon_intensity(path: str = "data/de_lu_carbon_intensity.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * df["hour"] * np.pi / 24)
    df["hour_cos"] = np.cos(2 * df["hour"] * np.pi / 24)
    df["week_sin"] = np.sin(2 * df["day_of_week"] * np.pi / 7)
    df["week_cos"] = np.cos(2 * df["day_of_week"] * np.pi / 7)
    df["month_sin"] = np.sin(2 * df["month"] * np.pi / 12)
    df["month_cos"] = np.cos(2 * df["month"] * np.pi / 12)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list = LAG_HOURS,
) -> pd.DataFrame:
    df["lag_0h"] = df[target_col]
    for h in lags:
        df[f"lag_{h}h"] = df[target_col].shift(h)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    df["rolling_mean_24h"] = df[target_col].shift(1).rolling(window).mean()
    df["rolling_std_24h"] = df[target_col].shift(1).rolling(window).std()
    return df


def add_multi_horizon_targets(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizons: int = FORECAST_HORIZONS,
) -> pd.DataFrame:
    for h in range(1, horizons + 1):
        df[f"target_{h}h"] = df[target_col].shift(-h)
    return df


def build_full_lag_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: list = LAG_HOURS,
    window: int = ROLLING_WINDOW,
    horizons: int = FORECAST_HORIZONS,
) -> pd.DataFrame:
    df = add_multi_horizon_targets(df, target_col=target_col, horizons=horizons)
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col=target_col, lags=lags)
    df = add_rolling_features(df, target_col=target_col, window=window)

    pre_count = len(df)
    df = df.dropna()
    print(f"Rows dropped (insufficient history/future): {pre_count - len(df)}")
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list:
    """Model input columns: everything except the raw generation columns
    (which are direct inputs to the CI formula and would leak the target)
    and every target_/raw target column."""
    raw_gen_cols = [
        "coal_MW", "gas_MW", "hydro_MW", "wind_MW", "solar_MW", "biomass_MW",
        "waste_MW", "oil_MW", "other_MW", "other_renewable_MW", "geothermal_MW",
    ]
    target_cols = [target_col] + [f"target_{h}h" for h in range(1, FORECAST_HORIZONS + 1)]
    exclude = set(target_cols) | set(raw_gen_cols)
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    df_load = load_carbon_intensity()
    df_full = build_full_lag_dataset(df_load)

    feature_cols = get_feature_columns(df_full)
    print(f"Model inputs ({len(feature_cols)}): {feature_cols}")

    df_full.to_csv("data/de_lu_features.csv")
    print(f"Saved {df_full.shape[0]} rows x {df_full.shape[1]} columns to data/de_lu_features.csv")
