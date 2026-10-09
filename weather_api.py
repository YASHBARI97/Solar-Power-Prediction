
import os
from datetime import timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.weatherapi.com/v1/forecast.json"

API_KEY = os.getenv("WEATHER_API_KEY")

LOCATION = "Pune"
FORECAST_HOURS = 48


def fetch_weather_forecast():
    """Fetch the next 48 hours of hourly weather for Pune."""

    if not API_KEY:
        raise ValueError(
            "WEATHER_API_KEY is missing. "
            "Add it to your .env file."
        )

    params = {
        "key": API_KEY,
        "q": LOCATION,
        "days": 3,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        hourly_records = []

        for forecast_day in data["forecast"]["forecastday"]:
            for hour in forecast_day["hour"]:
                hourly_records.append({
                    "timestamp": hour["time"],
                    "temperature": hour["temp_c"],
                    "humidity": hour["humidity"],
                    "cloud_cover": hour["cloud"],
                    "wind_speed": hour["wind_kph"] / 3.6,
                    "precipitation": hour["precip_mm"],
                })

        forecast = pd.DataFrame(hourly_records)

        if forecast.empty:
            raise ValueError("The API returned no hourly data.")

        forecast["timestamp"] = pd.to_datetime(
            forecast["timestamp"]
        )

        # WeatherAPI timestamps are local to the requested location.
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        end_time = now + timedelta(hours=FORECAST_HOURS)

        forecast = forecast[
            (forecast["timestamp"] > now)
            & (forecast["timestamp"] <= end_time)
        ].copy()

        forecast = forecast.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        if forecast.empty:
            raise ValueError(
                "No future hourly observations were returned."
            )

        numeric_columns = [
            "temperature",
            "humidity",
            "cloud_cover",
            "wind_speed",
            "precipitation",
        ]

        forecast[numeric_columns] = forecast[
            numeric_columns
        ].apply(pd.to_numeric, errors="coerce")

        if forecast[numeric_columns].isna().any().any():
            raise ValueError(
                "Weather forecast contains missing numeric values."
            )

        if not forecast["timestamp"].is_monotonic_increasing:
            raise ValueError("Forecast timestamps are not ordered.")

        print(f"Location: {LOCATION}")
        print(f"Forecast observations: {len(forecast)}")
        print(f"First forecast: {forecast['timestamp'].min()}")
        print(f"Last forecast: {forecast['timestamp'].max()}")

        return forecast

    except requests.RequestException as error:
        # Avoid printing the request URL, which contains the API key.
        status = (
            error.response.status_code
            if error.response is not None
            else "unavailable"
        )
        print(f"Weather API request failed. HTTP status: {status}")
        print("Check your API key, quota, connection, and plan.")
        return None

    except (KeyError, TypeError, ValueError) as error:
        print(f"Invalid weather forecast response: {error}")
        return None


if __name__ == "__main__":
    forecast_df = fetch_weather_forecast()

    if forecast_df is not None:
        print("\nFirst five forecast observations:")
        print(forecast_df.head().to_string(index=False))

        output_file = "weather_forecast.csv"
        forecast_df.to_csv(output_file, index=False)

        print(f"\nSaved successfully to {output_file}")