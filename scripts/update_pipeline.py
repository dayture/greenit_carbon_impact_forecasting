"""
update_pipeline.py

Meant to run on a schedule (GitHub Actions cron). Each run:

1. Fetches the last `--days-back` days of DE-LU generation data from
   ENTSO-E (default 10 days - more than the 7 days needed for the
   lag_168h feature, plus a safety margin for late-arriving data).
2. Merges the new rows into a persistent master raw file
   (data/DE_LU_generation_master.csv), deduplicating by timestamp so
   re-running never creates duplicate rows.
3. Re-runs the existing pipeline stages (data_processing_real ->
   carbon_intensity_real -> feature_engineering_real) against the
   updated master file.
4. Produces a fresh 24-hour forecast from the saved models and writes
   it to data/latest_forecast.csv + data/forecast_next_24h.png.

Does NOT retrain or re-run hyperparameter search - that stays a
separate, manually triggered step (see hyperparameter_search.py /
model.py and the "Retrain models" GitHub Actions workflow).
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from entsoe import EntsoePandasClient
except ImportError:
    print("Error: entsoe-py is not installed. Run: pip install entsoe-py")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # allow "import data_processing_real" etc.

import data_processing_real
import carbon_intensity_real
import feature_engineering_real
import forecast_next_24h
from model import get_feature_columns

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR = PROJECT_ROOT / "data"
MASTER_RAW_PATH = DATA_DIR / "DE_LU_generation_master.csv"
# The original one-off 2-year export shipped with the repo - used only
# to seed the master file the very first time this script runs.
SEED_RAW_PATH = DATA_DIR / "DE_LU_generation_2024-08-01_2026-08-01.csv"


def get_api_key() -> str:
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key or api_key.strip() == "YOUR_API_KEY_HERE":
        print(f"\nENTSOE_API_KEY is not set. Add it to '{ENV_PATH}' "
              f"(locally) or as a repo secret (in CI).")
        sys.exit(1)
    return api_key.strip()


def fetch_recent_generation(days_back: int = 10) -> pd.DataFrame:
    """Pull the most recent `days_back` days of DE-LU generation data."""
    client = EntsoePandasClient(api_key=get_api_key())
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days_back)
    print(f"Fetching DE_LU generation: {start} -> {end}")
    return client.query_generation("DE_LU", start=start, end=end)


def load_master_raw() -> pd.DataFrame | None:
    if MASTER_RAW_PATH.exists():
        df = pd.read_csv(MASTER_RAW_PATH, header=[0, 1], index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    if SEED_RAW_PATH.exists():
        print(f"No master file yet - seeding from {SEED_RAW_PATH.name}")
        df = pd.read_csv(SEED_RAW_PATH, header=[0, 1], index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    return None


def merge_into_master(new_df: pd.DataFrame) -> pd.DataFrame:
    old_df = load_master_raw()
    if old_df is not None:
        combined = pd.concat([old_df, new_df])
    else:
        combined = new_df
    # Keep the most recently fetched value for any overlapping timestamp.
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(MASTER_RAW_PATH)
    print(f"Master raw file: {len(combined)} rows -> {MASTER_RAW_PATH}")
    return combined


def run_pipeline(days_back: int = 10) -> None:
    new_data = fetch_recent_generation(days_back=days_back)
    merge_into_master(new_data)

    # Re-run the existing, unmodified pipeline stages against the
    # updated master file.
    data_processing_real.run(path=str(MASTER_RAW_PATH))

    generation = carbon_intensity_real.load_hourly_generation()
    carbon_intensity_real.validate_emission_factors(carbon_intensity_real.EMISSION_FACTORS)
    ci_df = carbon_intensity_real.calculate_and_save(generation)
    carbon_intensity_real.sanity_check(ci_df)

    df_loaded = feature_engineering_real.load_carbon_intensity()
    df_full = feature_engineering_real.build_full_lag_dataset(df_loaded)
    df_full.to_csv(DATA_DIR / "de_lu_features.csv")

    # Fresh 24h forecast from the currently saved models (no retraining).
    df_for_forecast = forecast_next_24h.load_latest_known_data()
    df_features = forecast_next_24h.build_current_features(df_for_forecast)
    feature_cols = get_feature_columns(df_features)
    latest_row = forecast_next_24h.get_latest_row(df_features, feature_cols)
    models = forecast_next_24h.load_all_models()
    forecast_df = forecast_next_24h.forecast_24h(latest_row, models)
    forecast_df.to_csv(DATA_DIR / "latest_forecast.csv", index=False)
    forecast_next_24h.plot_forecast(forecast_df)
    print(forecast_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental ENTSO-E fetch + forecast refresh")
    parser.add_argument("--days-back", type=int, default=10,
                         help="How many trailing days of generation data to (re)fetch")
    args = parser.parse_args()
    run_pipeline(days_back=args.days_back)
