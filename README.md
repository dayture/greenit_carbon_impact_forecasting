# Green IT & Carbon-Aware Computing: Multi-Horizon Grid Carbon Intensity Forecasting

An end-to-end pipeline that computes hourly grid carbon intensity (gCO2/kWh) for the
Germany-Luxembourg (DE-LU) bidding zone from real ENTSO-E generation data, forecasts it
up to 24 hours ahead using one XGBoost model per horizon (direct multi-horizon
forecasting), and applies the forecast to a simulated data center workload to estimate
operational carbon emissions and the savings available from carbon-aware scheduling.

## Contents
1. [Pipeline](#pipeline)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Setup and usage](#setup-and-usage)
5. [Project structure](#project-structure)
6. [Known limitations](#known-limitations)

## Pipeline

```
ENTSO-E Transparency Platform (DE-LU, 15-min resolution, 2 years)
        |  fetch_entsoe_data.py
        v
Raw generation data (16 PsrType categories, multi-row header)
        |  data_processing_real.py
        |  -> select Actual Aggregated values, map to 11 source categories, resample hourly
        v
de_lu_hourly_generation.csv  (17,520 rows x 11 sources, MW)
        |  carbon_intensity_real.py
        |  -> CI(t) = sum(P_source(t) * EF_source) / sum(P_source(t))
        v
de_lu_carbon_intensity.csv  (hourly CI, gCO2/kWh)
        |  feature_engineering_real.py
        |  -> calendar (sin/cos) + lag (0h, 1h, 24h, 168h) + rolling(24h) + 24 targets (target_1h..target_24h)
        v
de_lu_features.csv  (17,328 rows x 52 columns)
        |  hyperparameter_search.py  (GridSearchCV + TimeSeriesSplit, per horizon)
        |  model.py  (24 independent XGBoost models, chronological split, seasonal baseline)
        v
models/model_h1.json ... model_h24.json  +  multi_horizon_results.csv
        |  forecast_next_24h.py
        v
Live 24-hour carbon intensity forecast
        |  carbon_calc.py
        v
10 MW data center scenario -> operational emissions + workload-shifting savings
```

## Methodology

**Carbon intensity formula** - a generation-weighted average:
`CI(t) = sum(P_source(t) * EF_source) / sum(P_source(t))`. Emission factors come from
IPCC AR5 Annex III and the NREL Life Cycle Assessment Harmonization dataset
(`coal: 900, gas: 490, oil: 840, hydro: 24, wind: 11, solar: 45, geothermal: 38,
biomass: 230 gCO2/kWh`). No primary-source value was found for the `waste` and `other`
categories (together ~1.8% of generation); the nearest comparable technology is used as
a proxy (see [Known limitations](#known-limitations)).

**Direct multi-horizon forecasting** - each row represents a decision point `t0`. Its
features (calendar encodings plus lag/rolling values known at or before `t0`) stay fixed
while 24 separate target columns (`target_1h ... target_24h`, i.e. `CI(t0+h)`) are
predicted by 24 independently trained models. This avoids the error accumulation of
recursive (step-by-step) forecasting, where each prediction is fed back in as if it were
ground truth for the next step.

**Leakage prevention** - rolling statistics are computed on data shifted by one hour, so
a row's own target never enters its own features; the train/test split is strictly
chronological (`shuffle=False`); hyperparameter search uses `TimeSeriesSplit` for
cross-validation and only the training portion of the data - the test set is never
touched during model selection, including during CV.

**Hyperparameter tuning** - `GridSearchCV` with `TimeSeriesSplit(n_splits=3)`, run
independently for each of the 24 horizons, over a grid of `n_estimators: [100, 300, 500]`,
`max_depth: [4, 6, 8]`, `learning_rate: [0.03, 0.08]`, `subsample: [0.8, 1.0]`,
`colsample_bytree: [0.8, 1.0]` (72 combinations x 3 folds x 24 horizons).

**Baseline** - a simplified seasonal persistence baseline: `forecast(t0+h) = CI(t0-24)`
("the same hour, one day earlier"), used as a fixed reference regardless of horizon.

## Results

Chronological split: train `2024-08-08 -> 2026-03-08` (13,862 hours), test
`2026-03-08 -> 2026-07-30` (3,466 hours).

| Horizon | XGBoost RMSE | XGBoost MAPE | Baseline RMSE | Baseline MAPE |
|---|---|---|---|---|
| 1h | 11.71 | 3.31% | 107.26 | 29.97% |
| 6h | 55.39 | 17.05% | 176.93 | 61.85% |
| 12h | 78.73 | 24.72% | 217.73 | 80.19% |
| 24h | 86.32 | 27.29% | 121.22 | 34.47% |

XGBoost outperforms the baseline at every horizon; the margin narrows as the horizon
grows, which is expected - uncertainty increases and the predictive value of recent
history decreases further into the future. Feature importance shows the most recent
value (`lag_1h`, ~91%) dominates at `h=1`, while at `h=24` seasonal/cyclical features
(`lag_24h`, `lag_168h`, `rolling_mean_24h`, `month_sin/cos`) together contribute roughly
20% - the model relies on recent momentum at short horizons and on cyclical patterns at
longer ones.

Plots: `data/rmse_vs_horizon.png`, `data/forecast_next_24h.png`.

### Data center carbon footprint scenario (10 MW, PUE=1.2)

A simulated workload (base 65% of capacity, plus a daily/weekly cycle and noise,
averaging ~67% utilization) combined with 2 years of real CI data:

| Metric | Value |
|---|---|
| Total emissions (~2 years) | 46,492 tCO2 |
| Average hourly emissions | 2,653.7 kgCO2 |
| Mean CI, worst 20% of hours | 519 gCO2/kWh |
| Mean CI, best 20% of hours | 154 gCO2/kWh |
| **Savings from shifting 20% of workload** | **~9,936 tCO2 (21.4%)** |

Shifting a fifth of the workload from the highest- to the lowest-carbon-intensity hours
reduces total operational emissions by an estimated 21.4% - a direct, quantified case for
carbon-aware scheduling. Details: `data/emissions_timeline.png`, `data/emissions_full.csv`.

### Forecast accuracy vs. emissions estimate error

Comparing emissions computed from the actual CI series against emissions computed from
each horizon's forecasted CI series (`data/actual_vs_forecast_emissions_h{1,24}.png`):

| Horizon | Mean absolute emissions error |
|---|---|
| h=1 | 50.8 kgCO2/hour |
| h=24 | 531.9 kgCO2/hour (~10.5x) |

This is consistent with the RMSE ratio between the two horizons (11.7 vs 86.3, ~7.4x).
The h=24 plots also show a consistent pattern: the model underestimates the amplitude of
both the highest- and lowest-CI hours (regression toward the mean), a predictable
consequence of relying on seasonal/cyclical signal rather than recent observations at
longer horizons. Operational decisions based on the 24-hour forecast should account for
this margin at the extremes.

## Setup and usage

```bash
pip install pandas numpy xgboost scikit-learn matplotlib entsoe-py python-dotenv

# 1) Fetch raw data (requires an ENTSO-E API key; copy .env.example to .env and set ENTSOE_API_KEY)
python scripts/fetch_entsoe_data.py --start 2024-08-01 --end 2026-08-01 --country DE_LU --type generation

# 2) Run the pipeline
python scripts/data_processing_real.py
python scripts/carbon_intensity_real.py
python scripts/feature_engineering_real.py
python scripts/hyperparameter_search.py
python scripts/model.py
python scripts/forecast_next_24h.py
python scripts/carbon_calc.py
```

## Project structure

```
├── data/                        # processed data and plots (raw ENTSO-E export is gitignored)
├── models/                      # 24 saved XGBoost models (model_h1.json ... model_h24.json)
├── scripts/
│   ├── fetch_entsoe_data.py     # ENTSO-E data download
│   ├── data_processing_real.py  # PsrType mapping + hourly resample
│   ├── carbon_intensity_real.py # CI(t) calculation
│   ├── feature_engineering_real.py  # calendar + lag + rolling + multi-horizon targets
│   ├── hyperparameter_search.py # per-horizon GridSearchCV
│   ├── model.py                 # trains 24 models, evaluates against baseline
│   ├── forecast_next_24h.py     # live 24-hour forecast from saved models
│   └── carbon_calc.py           # data center workload + emissions + shifting savings
├── .env.example                 # required environment variable, no real key
└── README.md
```

## Known limitations

- `waste_MW` and `other_MW` emission factors (~1.8% of total generation) have no
  primary-source (IPCC/NREL) kWh-based value; the nearest technology is used as a proxy
  (`waste -> coal`, `other -> gas`). This is an assumption, not a primary-source figure.
- The seasonal baseline (`CI(t0-24)`) uses a fixed 24-hour-back reference for every
  horizon rather than an horizon-adjusted one (`CI(t0+h-24)`), which would be more
  precise.
- The 24-hour forecast underestimates the amplitude at CI extremes (see
  [Forecast accuracy vs. emissions estimate error](#forecast-accuracy-vs-emissions-estimate-error)).
- The dataset covers a single bidding zone (DE-LU); generalization to other grids has
  not been tested.
