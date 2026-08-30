"""
carbon_intensity.py

Computes dynamic grid carbon intensity (gCO2/kWh) from hourly generation
data, using a generation-weighted average of source emission factors:

    CI(t) = sum(P_source(t) * EF_source) / sum(P_source(t))

Emission factors (gCO2eq/kWh) are sourced from IPCC AR5 Annex III and the
NREL Life Cycle Assessment Harmonization dataset. waste_MW and other_MW
have no primary-source value available; they use the nearest comparable
technology as a proxy (see README, Known limitations).
"""

import pandas as pd

EMISSION_FACTORS = {
    "coal_MW": 900.0,
    "gas_MW": 490.0,
    "hydro_MW": 24.0,
    "wind_MW": 11.0,
    "solar_MW": 45.0,
    "nuclear_MW": 12.0,
    "geothermal_MW": 38.0,
    "biomass_MW": 230.0,          # IPCC AR5 biopower lifecycle median
    "oil_MW": 840.0,               # NREL LCA harmonization median
    "other_renewable_MW": 8.0,     # NREL LCA harmonization median (ocean/tidal)
    "waste_MW": 900.0,             # proxy: coal (no primary source found)
    "other_MW": 490.0,             # proxy: gas (no primary source found)
}


def load_hourly_generation(path: str = "data/de_lu_hourly_generation.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def validate_emission_factors(emission_factors: dict) -> None:
    """Raise if any emission factor is missing (None), before it can
    silently distort the carbon intensity calculation."""
    missing = [k for k, v in emission_factors.items() if v is None]
    if missing:
        raise ValueError(f"Missing emission factor(s): {missing}")


def calculate_and_save(
    df: pd.DataFrame,
    emission_factors: dict = EMISSION_FACTORS,
    out_path: str = "data/de_lu_carbon_intensity.csv",
) -> pd.DataFrame:
    """Compute CI(t) and append it as a new column."""
    total_emissions = sum(
        df[col] * emission_factors[col] for col in df.columns if col in emission_factors
    )
    total_generation = df.sum(axis=1)
    df["carbon_intensity_gCO2_kWh"] = total_emissions / total_generation
    df.to_csv(out_path)
    return df


def sanity_check(df: pd.DataFrame) -> None:
    """Compare the computed mean CI against the published range for the
    German grid (~300-400 gCO2/kWh, electricityMaps/Ember annual reports)."""
    ci = df["carbon_intensity_gCO2_kWh"]
    print(ci.describe())
    mean_ci = ci.mean()
    print(f"Mean carbon intensity: {mean_ci:.1f} gCO2/kWh")
    if 300 <= mean_ci <= 400:
        print("Within the expected literature range.")
    else:
        print("Outside the expected literature range - review inputs.")


if __name__ == "__main__":
    generation = load_hourly_generation()
    validate_emission_factors(EMISSION_FACTORS)
    result = calculate_and_save(generation, EMISSION_FACTORS)
    sanity_check(result)
