"""
data_processing.py

Processes raw ENTSO-E DE-LU electricity generation data: reads the
multi-index CSV export, filters to actual (aggregated) generation values,
maps ENTSO-E production types to a fixed set of source categories, and
resamples from 15-minute to hourly resolution.

Output schema (MW, one column per source):
    coal_MW, gas_MW, oil_MW, hydro_MW, wind_MW, solar_MW,
    geothermal_MW, biomass_MW, waste_MW, other_MW, other_renewable_MW

Note: nuclear_MW is intentionally absent - Germany decommissioned its
last nuclear plants in 2023.
"""

import pandas as pd

RAW_CSV_PATH = "data/DE_LU_generation_2024-08-01_2026-08-01.csv"

# Maps the 16 ENTSO-E PsrType categories present in the DE-LU export to
# the source categories used throughout this project.
PSR_MAP = {
    "Fossil Hard coal": "coal_MW",
    "Fossil Brown coal/Lignite": "coal_MW",
    "Fossil Gas": "gas_MW",
    "Fossil Coal-derived gas": "gas_MW",
    "Fossil Oil": "oil_MW",
    "Hydro Pumped Storage": "hydro_MW",
    "Wind Offshore": "wind_MW",
    "Wind Onshore": "wind_MW",
    "Geothermal": "geothermal_MW",
    "Hydro Run-of-river and poundage": "hydro_MW",
    "Hydro Water Reservoir": "hydro_MW",
    "Other": "other_MW",
    "Other renewable": "other_renewable_MW",
    "Solar": "solar_MW",
    "Waste": "waste_MW",
    "Biomass": "biomass_MW",
}


def load_raw_generation(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the raw ENTSO-E export (two-row header: PsrType / value type)."""
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def select_actual_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 'Actual Aggregated' (generation) values, dropping
    'Actual Consumption' (pumped-storage charging etc.), and flatten the
    column MultiIndex down to plain PsrType names."""
    return df.xs("Actual Aggregated", level=1, axis=1)


def map_to_categories(df: pd.DataFrame, psr_map: dict = PSR_MAP) -> pd.DataFrame:
    """Group the 16 PsrType columns into the project's source categories."""
    mapped_columns = df.columns.map(psr_map)
    return df.T.groupby(mapped_columns).sum().T


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample from 15-minute to hourly resolution (mean)."""
    return df.resample("1h").mean()


def run(path: str = RAW_CSV_PATH, out_path: str = "data/de_lu_hourly_generation.csv") -> pd.DataFrame:
    """Run the full pipeline and write the hourly generation dataset to disk."""
    df_raw = load_raw_generation(path)
    df_agg = select_actual_aggregated(df_raw)
    df_grouped = map_to_categories(df_agg)
    df_hourly = resample_hourly(df_grouped)
    df_hourly.to_csv(out_path)
    return df_hourly


if __name__ == "__main__":
    result = run()
    print(f"Saved {result.shape[0]} rows x {result.shape[1]} columns to data/de_lu_hourly_generation.csv")
