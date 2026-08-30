"""
Live dashboard for the Green IT carbon intensity forecasting project.

Reads data directly from the raw GitHub URLs of the source repo, so it
always shows whatever the latest GitHub Actions run committed - no data
needs to be duplicated into this Space.

Deploy: create a new Space on huggingface.co (SDK: Streamlit), drop this
file in as app.py, add a requirements.txt with:
    streamlit
    pandas
    matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

GITHUB_USER = "dayture"
GITHUB_REPO = "greenit_carbon_impact_forecasting"
BRANCH = "main"

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}"
LATEST_FORECAST_URL = f"{RAW_BASE}/data/latest_forecast.csv"
MULTI_HORIZON_RESULTS_URL = f"{RAW_BASE}/data/multi_horizon_results.csv"

st.set_page_config(page_title="DE-LU Grid Carbon Intensity Forecast", layout="wide")
st.title("Data Center Carbon Intensity Forecasting (DE-LU)")
st.caption(
    "Live 24-hour ahead forecast of grid carbon intensity (gCO2/kWh), "
    "refreshed every 6 hours from ENTSO-E generation data. "
    f"Source: [{GITHUB_REPO}](https://github.com/{GITHUB_USER}/{GITHUB_REPO})"
)


@st.cache_data(ttl=3600)
def load_forecast():
    return pd.read_csv(LATEST_FORECAST_URL, parse_dates=["timestamp"])


@st.cache_data(ttl=3600)
def load_model_results():
    try:
        return pd.read_csv(MULTI_HORIZON_RESULTS_URL)
    except Exception:
        return None


try:
    forecast_df = load_forecast()
except Exception as e:
    st.error(
        "Couldn't load the latest forecast yet. This usually means the "
        "scheduled GitHub Actions run hasn't produced data/latest_forecast.csv "
        f"yet. Details: {e}"
    )
    st.stop()

t0 = forecast_df["timestamp"].min() - pd.Timedelta(hours=1)
st.subheader(f"Forecast generated from data as of {t0} UTC")

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(forecast_df["timestamp"], forecast_df["predicted_CI"], marker="o")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Predicted carbon intensity (gCO2/kWh)")
    ax.set_title("Next 24 hours")
    fig.autofmt_xdate()
    st.pyplot(fig)

with col2:
    st.metric("Next hour (h=1)", f"{forecast_df.iloc[0]['predicted_CI']:.0f} gCO2/kWh")
    st.metric("24h ahead (h=24)", f"{forecast_df.iloc[-1]['predicted_CI']:.0f} gCO2/kWh")
    best_hour = forecast_df.loc[forecast_df["predicted_CI"].idxmin()]
    st.metric(
        "Greenest hour in window",
        f"{best_hour['predicted_CI']:.0f} gCO2/kWh",
        help=f"at {best_hour['timestamp']}",
    )

st.dataframe(forecast_df, use_container_width=True, hide_index=True)

results_df = load_model_results()
if results_df is not None:
    st.subheader("Model accuracy by horizon (backtest)")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

st.caption(
    "Data source: ENTSO-E Transparency Platform (CC-BY 4.0). "
    "This is a portfolio/research project, not an operational forecasting service."
)
