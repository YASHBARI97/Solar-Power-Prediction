
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_api import fetch_weather_forecast


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "solar_model.joblib"
OUTPUT_PATH = BASE_DIR / "solar_predictions.csv"


def create_time_features(df):
    """Create the same time features used during model training."""

    df = df.copy()

    timestamps = pd.to_datetime(df["timestamp"])

    df["hour"] = timestamps.dt.hour
    df["day_of_week"] = timestamps.dt.dayofweek
    df["day_of_year"] = timestamps.dt.dayofyear
    df["month"] = timestamps.dt.month

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365
    )
    df["day_of_year_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365
    )

    return df


def predict_solar_power():
    """Predict solar output for the upcoming weather forecast."""

    # 1. Check that the trained model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run train_model.py first."
        )

    # 2. Load the trained model artifact
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    print(f"Loaded model: {artifact.get('model_name', 'Unknown')}")

    # 3. Fetch the latest hourly weather forecast
    weather_df = fetch_weather_forecast()

    if weather_df is None or weather_df.empty:
        raise RuntimeError(
            "Could not retrieve weather data. "
            "Check weather_api.py and your API configuration."
        )

    # 4. Generate time-based features
    prediction_df = create_time_features(weather_df)

    # 5. Verify all expected model features exist
    missing_features = [
        col for col in feature_columns
        if col not in prediction_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # 6. Select features in exactly the training order
    X_future = prediction_df[feature_columns].copy()

    # Check for invalid feature values
    if X_future.isna().any().any():
        raise ValueError(
            "Prediction features contain missing values."
        )

    # 7. Predict future solar output
    predictions = model.predict(X_future)

    # Solar output cannot be negative
    prediction_df["predicted_solar_output"] = np.maximum(
        predictions, 0
    )

    # 8. Save the results
    output_columns = [
        "timestamp",
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "precipitation",
        "predicted_solar_output",
    ]

    results = prediction_df[output_columns].copy()

    results.to_csv(OUTPUT_PATH, index=False)

    print("\nSolar predictions generated successfully!")
    print(f"Forecast rows: {len(results)}")
    print(f"Results saved to: {OUTPUT_PATH}")

    print("\nNext-hour solar predictions:")
    print(results.head(10).to_string(index=False))

    return results


if __name__ == "__main__":
    predict_solar_power()