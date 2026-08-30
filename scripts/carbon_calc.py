"""
carbon_calc.py

Simulates a 10 MW data center's hourly power draw, combines it with
grid carbon intensity, and computes operational carbon emissions:

    Emission(t) = P_DC(t)[MW] * 1000[kWh/MWh] * PUE * CI(t)[gCO2/kWh]

Also estimates the potential emissions savings from shifting a share of
the workload from the highest- to the lowest-carbon-intensity hours,
and compares emissions computed from actual CI against emissions
computed from the multi-horizon model's forecasted CI, to show how
forecast error at longer horizons propagates into emissions estimates.

No ML model is used in this module - the emissions formula is a fixed,
deterministic calculation.
"""

import sys
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Resolve base directories dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure script directory is in sys.path for local module imports
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import get_feature_columns

DC_CAPACITY_MW = 10.0
PUE = 1.2


def simulate_workload(
    index: pd.DatetimeIndex,
    capacity_mw: float = DC_CAPACITY_MW,
    seed: int = 42,
) -> pd.Series:
    """Simulate P_DC(t) [MW]: a base load (~65% of capacity) plus a mild
    daily cycle (business hours, +12%), a weekly cycle (weekends, -8%),
    and small random noise."""
    rng = np.random.default_rng(seed)
    n = len(index)
    hour = index.hour.values
    dow = index.dayofweek.values

    base = 0.65
    daily = 0.12 * np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, None)
    weekly = np.where(dow >= 5, -0.08, 0.0)
    noise = rng.normal(0, 0.025, n)

    load_fraction = base + daily + weekly + noise
    workload_mw = np.clip(capacity_mw * load_fraction, 0, capacity_mw)
    return pd.Series(workload_mw, index=index, name="workload_MW")


def load_carbon_intensity(path: Union[str, Path] = DATA_DIR / "de_lu_carbon_intensity.csv") -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df["carbon_intensity_gCO2_kWh"]


def calculate_emissions(
    workload_mw: pd.Series,
    carbon_intensity: pd.Series,
    pue: float = PUE,
) -> pd.DataFrame:
    df = pd.concat(
        [workload_mw.rename("workload_MW"), carbon_intensity.rename("carbon_intensity_gCO2_kWh")],
        axis=1,
        join="inner",
    )
    df["energy_kWh"] = df["workload_MW"] * 1000 * pue
    df["emission_gCO2"] = df["energy_kWh"] * df["carbon_intensity_gCO2_kWh"]
    df["emission_kgCO2"] = df["emission_gCO2"] / 1000
    return df


def summarize_emissions(df: pd.DataFrame) -> dict:
    total_ton = df["emission_kgCO2"].sum() / 1000
    avg_hourly_kg = df["emission_kgCO2"].mean()
    worst5 = df["emission_kgCO2"].nlargest(5)
    best5 = df["emission_kgCO2"].nsmallest(5)

    print(f"Total emissions: {total_ton:,.1f} tCO2")
    print(f"Average hourly emissions: {avg_hourly_kg:,.2f} kgCO2")
    print(f"\nHighest-emission hours:\n{worst5}")
    print(f"\nLowest-emission hours:\n{best5}")

    return {
        "total_tonCO2": total_ton,
        "avg_hourly_kgCO2": avg_hourly_kg,
        "top5_worst_hours": worst5,
        "top5_best_hours": best5,
    }


def estimate_shifting_savings(df: pd.DataFrame, shift_ratio: float = 0.2) -> dict:
    """Approximate the savings from shifting shift_ratio of the workload
    from the highest-CI hours to the lowest-CI hours: the CI gap between
    the worst and best shift_ratio slices, multiplied by the energy
    consumed in the worst slice."""
    n = len(df)
    k = max(int(n * shift_ratio), 1)

    sorted_by_ci = df.sort_values("carbon_intensity_gCO2_kWh")
    best_k = sorted_by_ci.iloc[:k]
    worst_k = sorted_by_ci.iloc[-k:]

    avg_ci_best = best_k["carbon_intensity_gCO2_kWh"].mean()
    avg_ci_worst = worst_k["carbon_intensity_gCO2_kWh"].mean()
    ci_diff = avg_ci_worst - avg_ci_best

    shifted_energy_kWh = worst_k["energy_kWh"].sum()
    potential_savings_kgCO2 = (shifted_energy_kWh * ci_diff) / 1000

    total_emission_kg = df["emission_kgCO2"].sum()
    savings_percent = (potential_savings_kgCO2 / total_emission_kg) * 100

    print(f"\nShifting scenario ({shift_ratio*100:.0f}% of workload)")
    print(f"Worst {shift_ratio*100:.0f}% hours - mean CI: {avg_ci_worst:,.1f} gCO2/kWh")
    print(f"Best  {shift_ratio*100:.0f}% hours - mean CI: {avg_ci_best:,.1f} gCO2/kWh")
    print(f"Estimated savings: {potential_savings_kgCO2:,.1f} kgCO2 ({savings_percent:.1f}%)")

    return {
        "potential_savings_kgCO2": potential_savings_kgCO2,
        "savings_percent": savings_percent,
    }


def plot_emissions_timeline(df: pd.DataFrame, sample_days: int = 7):
    sample = df.iloc[: sample_days * 24]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sample.index, sample["carbon_intensity_gCO2_kWh"], color="darkred")
    axes[0].set_ylabel("gCO2/kWh")
    axes[0].set_title(f"Grid Carbon Intensity (first {sample_days} days)")

    axes[1].plot(sample.index, sample["emission_kgCO2"], color="darkgreen")
    axes[1].set_ylabel("kg CO2")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Data Center Hourly Operational Emissions")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "emissions_timeline.png", dpi=110)
    plt.show()


def plot_emissions_full_period(df: pd.DataFrame):
    """Daily-resampled overview of the entire dataset (hourly data over
    ~2 years is too dense to read as a raw line plot)."""
    daily = df[["carbon_intensity_gCO2_kWh", "emission_kgCO2"]].resample("1D").mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(daily.index, daily["carbon_intensity_gCO2_kWh"], color="darkred", linewidth=0.8)
    axes[0].set_ylabel("gCO2/kWh")
    axes[0].set_title("Grid Carbon Intensity - Daily Mean, Full Period")

    axes[1].plot(daily.index, daily["emission_kgCO2"], color="darkgreen", linewidth=0.8)
    axes[1].set_ylabel("kg CO2")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Data Center Emissions - Daily Mean, Full Period")

    plt.tight_layout()
    plt.savefig(DATA_DIR / "emissions_timeline_full.png", dpi=110)
    plt.show()

# -----------------------------------------------------------------------------
# Forecast vs. actual comparison
# -----------------------------------------------------------------------------

def load_forecast_ci_series(
    h: int,
    features_path: Union[str, Path] = DATA_DIR / "de_lu_features.csv",
    model_dir: Union[str, Path] = MODELS_DIR,
) -> pd.Series:
    """Run the saved model for horizon h over the full feature dataset,
    aligning each prediction to the timestamp it forecasts (t0 + h)."""
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    feature_cols = get_feature_columns(df)

    model = XGBRegressor()
    model.load_model(str(Path(model_dir) / f"model_h{h}.json"))

    preds = model.predict(df[feature_cols])
    forecast_timestamps = df.index + pd.Timedelta(hours=h)

    series = pd.Series(preds, index=forecast_timestamps, name=f"forecast_CI_h{h}")
    return series[~series.index.duplicated(keep="first")].sort_index()


def compare_actual_vs_forecast(
    actual_ci: pd.Series,
    forecast_ci: pd.Series,
    workload_mw: pd.Series,
    pue: float = PUE,
) -> pd.DataFrame:
    common_index = actual_ci.index.intersection(forecast_ci.index).intersection(workload_mw.index)
    df = pd.DataFrame(index=common_index)
    df["workload_MW"] = workload_mw.loc[common_index]
    df["actual_CI"] = actual_ci.loc[common_index]
    df["forecast_CI"] = forecast_ci.loc[common_index]
    df["energy_kWh"] = df["workload_MW"] * 1000 * pue
    df["actual_emission_kgCO2"] = df["energy_kWh"] * df["actual_CI"] / 1000
    df["forecast_emission_kgCO2"] = df["energy_kWh"] * df["forecast_CI"] / 1000
    df["emission_error_kgCO2"] = df["forecast_emission_kgCO2"] - df["actual_emission_kgCO2"]

    mae = df["emission_error_kgCO2"].abs().mean()
    print(f"Mean absolute emissions error: {mae:,.2f} kgCO2/hour")
    return df


def plot_actual_vs_forecast(df: pd.DataFrame, horizon: int, sample_days: int = 7):
    sample = df.iloc[: sample_days * 24]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sample.index, sample["actual_CI"], label="Actual CI", color="darkred")
    axes[0].plot(sample.index, sample["forecast_CI"], label=f"Forecast CI (h={horizon})",
                 color="steelblue", linestyle="--")
    axes[0].set_ylabel("gCO2/kWh")
    axes[0].set_title(f"Actual vs. Forecast Carbon Intensity (horizon={horizon}h)")
    axes[0].legend()

    axes[1].plot(sample.index, sample["actual_emission_kgCO2"], label="Actual emissions", color="darkgreen")
    axes[1].plot(sample.index, sample["forecast_emission_kgCO2"], label="Forecast emissions",
                 color="orange", linestyle="--")
    axes[1].set_ylabel("kg CO2")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Actual vs. Forecast Operational Emissions")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / f"actual_vs_forecast_emissions_h{horizon}.png", dpi=110)
    plt.show()


def plot_actual_vs_forecast_full_period(df: pd.DataFrame, horizon: int):
    """Daily-resampled overview of actual vs. forecast CI/emissions over
    the entire comparison period."""
    daily = df.resample("1D").mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(daily.index, daily["actual_CI"], label="Actual CI", color="darkred", linewidth=0.8)
    axes[0].plot(daily.index, daily["forecast_CI"], label=f"Forecast CI (h={horizon})",
                 color="steelblue", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("gCO2/kWh")
    axes[0].set_title(f"Actual vs. Forecast Carbon Intensity - Daily Mean, Full Period (horizon={horizon}h)")
    axes[0].legend()

    axes[1].plot(daily.index, daily["actual_emission_kgCO2"], label="Actual emissions",
                 color="darkgreen", linewidth=0.8)
    axes[1].plot(daily.index, daily["forecast_emission_kgCO2"], label="Forecast emissions",
                 color="orange", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("kg CO2")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Actual vs. Forecast Operational Emissions - Daily Mean, Full Period")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / f"actual_vs_forecast_emissions_h{horizon}_full.png", dpi=110)
    plt.show()


def plot_ci_deviation(df: pd.DataFrame, horizon: int):
    """Plots the forecast error (forecast_CI - actual_CI) directly, as a
    single signed curve, instead of two overlapping lines. Positive =
    model overestimated CI at that hour; negative = underestimated.
    Daily-resampled so the full comparison period is readable."""
    daily = df.resample("1D").mean()
    deviation = daily["forecast_CI"] - daily["actual_CI"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(deviation.index, deviation, 0,
                     where=(deviation >= 0), color="tomato", alpha=0.6,
                     label="Model overestimates CI")
    ax.fill_between(deviation.index, deviation, 0,
                     where=(deviation < 0), color="steelblue", alpha=0.6,
                     label="Model underestimates CI")
    ax.set_ylabel("Forecast - Actual (gCO2/kWh)")
    ax.set_xlabel("Date")
    ax.set_title(f"Forecast Deviation, Daily Mean, Full Period (horizon={horizon}h)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / f"ci_deviation_h{horizon}.png", dpi=110)
    plt.show()

def plot_shifting_allocation(df: pd.DataFrame, shift_ratio: float = 0.2, sample_days: int = 14):
    """Marks, directly on the CI timeline, which hours are flagged for
    workload reduction (worst shift_ratio, highest CI - shown in red)
    and which hours are flagged as shift destinations (best shift_ratio,
    lowest CI - shown in green). Limited to sample_days for readability;
    the marking logic itself still uses the full dataset's ranking."""
    n = len(df)
    k = max(int(n * shift_ratio), 1)
    sorted_by_ci = df.sort_values("carbon_intensity_gCO2_kWh")
    worst_idx = sorted_by_ci.iloc[-k:].index
    best_idx = sorted_by_ci.iloc[:k].index

    sample = df.iloc[: sample_days * 24]
    worst_in_sample = sample.index.intersection(worst_idx)
    best_in_sample = sample.index.intersection(best_idx)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample.index, sample["carbon_intensity_gCO2_kWh"], color="gray", linewidth=1, zorder=1)
    ax.scatter(worst_in_sample, sample.loc[worst_in_sample, "carbon_intensity_gCO2_kWh"],
               color="red", s=25, zorder=2, label=f"Shift FROM (worst {shift_ratio*100:.0f}%)")
    ax.scatter(best_in_sample, sample.loc[best_in_sample, "carbon_intensity_gCO2_kWh"],
               color="green", s=25, zorder=2, label=f"Shift TO (best {shift_ratio*100:.0f}%)")
    ax.set_ylabel("gCO2/kWh")
    ax.set_xlabel("Time")
    ax.set_title(f"Workload Shifting Allocation (first {sample_days} days shown)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / "shifting_allocation.png", dpi=110)
    plt.show()

if __name__ == "__main__":
    ci = load_carbon_intensity()
    workload = simulate_workload(ci.index)
    emissions_df = calculate_emissions(workload, ci)

    summarize_emissions(emissions_df)
    estimate_shifting_savings(emissions_df, shift_ratio=0.2)
    plot_emissions_timeline(emissions_df)
    plot_emissions_full_period(emissions_df)
    plot_shifting_allocation(emissions_df, shift_ratio=0.2)
    emissions_df.to_csv(DATA_DIR / "emissions_full.csv")

    for h in [1, 24]:
        forecast_ci = load_forecast_ci_series(h)
        comparison_df = compare_actual_vs_forecast(ci, forecast_ci, workload)
        plot_actual_vs_forecast(comparison_df, horizon=h)
        plot_actual_vs_forecast_full_period(comparison_df, horizon=h)
        plot_ci_deviation(comparison_df, horizon=h)
        comparison_df.to_csv(DATA_DIR / f"actual_vs_forecast_h{h}.csv")
