
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "solar_weather_dataset.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = []
start = datetime(2024, 1, 1)

for i in range(24 * 60):  # 60 days × 24 hours = 1440 rows
    ts = start + timedelta(hours=i)
    hour = ts.hour

    temperature = (
        18 + 10 * np.sin((hour / 24) * 2 * np.pi)
        + np.random.normal(0, 1)
    )

    humidity = np.clip(
        60 + np.random.normal(0, 10), 20, 100
    )

    cloud_cover = np.clip(
        np.random.normal(40, 25), 0, 100
    )

    wind_speed = np.clip(
        np.random.normal(5, 2), 0, 15
    )

    precipitation = max(0, np.random.normal(0.2, 0.5))

    daylight_factor = max(
        0, np.sin((hour - 6) / 12 * np.pi)
    )

    solar_output = max(
        0,
        daylight_factor
        * (1 - cloud_cover / 120)
        * (temperature / 30)
        * 1000
    )

    solar_output += np.random.normal(0, 20)
    solar_output = max(0, solar_output)

    rows.append([
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        temperature,
        humidity,
        cloud_cover,
        wind_speed,
        precipitation,
        solar_output
    ])

df = pd.DataFrame(rows, columns=[
    "timestamp",
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "precipitation",
    "solar_output"
])

df.to_csv(OUTPUT_PATH, index=False)

print("Dataset created successfully!")
print("Saved to:", OUTPUT_PATH)
print("Dataset shape:", df.shape)
print(df.head())