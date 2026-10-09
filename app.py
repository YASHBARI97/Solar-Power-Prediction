
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Change these paths if your files are stored elsewhere.
HISTORICAL_FILE = Path(r"C:\Users\bariy\OneDrive\Desktop\Solar Power Prediction\data\solar_weather_dataset.csv")
PREDICTIONS_FILE = BASE_DIR / "solar_predictions.csv"


st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Power Prediction Dashboard")
st.caption("Historical solar data and future model predictions")


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


if not HISTORICAL_FILE.exists():
    st.error(f"Historical dataset not found: {HISTORICAL_FILE}")
    st.info("Update HISTORICAL_FILE to your actual dataset path.")
    st.stop()

if not PREDICTIONS_FILE.exists():
    st.warning("Prediction file not found. Generate it first.")
    st.code("python predict_solar.py")
    st.stop()


historical_df = load_csv(str(HISTORICAL_FILE))
forecast_df = load_csv(str(PREDICTIONS_FILE))


# --------------------------------------------------
# 3. Validate and prepare data
# --------------------------------------------------

# Update these candidates if your dataset uses different names.
date_candidates = ["timestamp", "datetime", "Date", "date", "time"]
solar_candidates = [
    "solar_power",
    "solar_output",
    "solar_generation",
    "power_output",
    "SolarPower",
]

date_column = next(
    (c for c in date_candidates if c in historical_df.columns),
    None,
)

solar_column = next(
    (c for c in solar_candidates if c in historical_df.columns),
    None,
)

if date_column is None or solar_column is None:
    st.error(
        "Could not identify the date or solar-output column "
        "in the historical dataset."
    )
    st.write("Available columns:", list(historical_df.columns))
    st.stop()

if not {"timestamp", "predicted_solar_output"}.issubset(
    forecast_df.columns
):
    st.error(
        "The prediction CSV must contain timestamp and "
        "predicted_solar_output columns."
    )
    st.stop()

historical_df[date_column] = pd.to_datetime(
    historical_df[date_column], errors="coerce"
)
historical_df[solar_column] = pd.to_numeric(
    historical_df[solar_column], errors="coerce"
)

forecast_df["timestamp"] = pd.to_datetime(
    forecast_df["timestamp"], errors="coerce"
)
forecast_df["predicted_solar_output"] = pd.to_numeric(
    forecast_df["predicted_solar_output"], errors="coerce"
)

historical_df = historical_df.dropna(
    subset=[date_column, solar_column]
).sort_values(date_column)

forecast_df = forecast_df.dropna(
    subset=["timestamp", "predicted_solar_output"]
).sort_values("timestamp")

if historical_df.empty or forecast_df.empty:
    st.error("Historical data or future predictions are empty.")
    st.stop()


# --------------------------------------------------
# 4. Summary metrics
# --------------------------------------------------

latest_historical = historical_df[solar_column].iloc[-1]
average_historical = historical_df[solar_column].mean()
average_forecast = forecast_df["predicted_solar_output"].mean()
forecast_hours = len(forecast_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Historical Output",
    f"{latest_historical:,.2f}",
)
col2.metric(
    "Historical Average",
    f"{average_historical:,.2f}",
)
col3.metric(
    "Forecast Average",
    f"{average_forecast:,.2f}",
)
col4.metric(
    "Forecast Hours",
    str(forecast_hours),
)


# --------------------------------------------------
# 5. Historical chart
# --------------------------------------------------

st.subheader("Historical Solar Power")

history_limit = st.slider(
    "Historical observations to display",
    min_value=24,
    max_value=max(24, min(500, len(historical_df))),
    value=min(168, max(24, len(historical_df))),
)

history_to_plot = historical_df.tail(history_limit)

history_fig = go.Figure()

history_fig.add_trace(
    go.Scatter(
        x=history_to_plot[date_column],
        y=history_to_plot[solar_column],
        mode="lines",
        name="Historical Solar Output",
        line=dict(width=2),
    )
)

history_fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Solar Power Output",
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(history_fig, use_container_width=True)


# --------------------------------------------------
# 6. Future prediction chart
# --------------------------------------------------

st.subheader("Future Solar Power Forecast")

forecast_fig = go.Figure()

forecast_fig.add_trace(
    go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_solar_output"],
        mode="lines+markers",
        name="Predicted Solar Output",
        line=dict(width=3),
    )
)

forecast_fig.update_layout(
    xaxis_title="Forecast Time",
    yaxis_title="Predicted Solar Power Output",
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(forecast_fig, use_container_width=True)


# --------------------------------------------------
# 7. Future forecast table
# --------------------------------------------------

st.subheader("Hourly Solar Forecast")

display_columns = [
    c for c in [
        "timestamp",
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "precipitation",
        "predicted_solar_output",
    ]
    if c in forecast_df.columns
]

st.dataframe(
    forecast_df[display_columns],
    use_container_width=True,
    hide_index=True,
)

# Download future forecast
csv_data = forecast_df[display_columns].to_csv(index=False)

st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name="solar_predictions.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 8. Forecast interpretation
# --------------------------------------------------

st.subheader("Forecast Summary")

max_row = forecast_df.loc[
    forecast_df["predicted_solar_output"].idxmax()
]
min_row = forecast_df.loc[
    forecast_df["predicted_solar_output"].idxmin()
]

left, right = st.columns(2)

left.write(
    f"**Highest predicted output:** "
    f"{max_row['predicted_solar_output']:,.2f}"
)
left.write(f"**Time:** {max_row['timestamp']}")

right.write(
    f"**Lowest predicted output:** "
    f"{min_row['predicted_solar_output']:,.2f}"
)
right.write(f"**Time:** {min_row['timestamp']}")

st.caption(
    "Predictions are generated by a model trained on synthetic data. "
    "Validate against actual solar generation measurements before "
    "using the forecasts for operational decisions."
)