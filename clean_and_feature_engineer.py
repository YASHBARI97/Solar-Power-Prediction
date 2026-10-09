
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. File paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "solar_weather_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "solar_weather_cleaned.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "precipitation",
    "solar_output",
]

WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "precipitation",
]

# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df = df[REQUIRED_COLUMNS].copy()

print("Original shape:", df.shape)

# --------------------------------------------------
# 3. Convert timestamps and numeric columns
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

for column in WEATHER_COLUMNS + ["solar_output"]:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# Remove rows with invalid timestamps
df = df.dropna(subset=["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp")

# Keep the first occurrence of duplicate timestamps
df = df.drop_duplicates(
    subset=["timestamp"], keep="first"
)

df = df.set_index("timestamp")

print("Shape after timestamp validation:", df.shape)

# --------------------------------------------------
# 4. Validate physical ranges
# --------------------------------------------------

# Values outside plausible ranges are treated as invalid.
# Adjust these rules if the assignment specifies
# different sensor limits.

df.loc[
    ~df["humidity"].between(0, 100), "humidity"
] = np.nan

df.loc[
    ~df["cloud_cover"].between(0, 100), "cloud_cover"
] = np.nan

df.loc[
    df["wind_speed"] < 0, "wind_speed"
] = np.nan

df.loc[
    df["precipitation"] < 0, "precipitation"
] = np.nan

df.loc[
    df["solar_output"] < 0, "solar_output"
] = np.nan

# --------------------------------------------------
# 5. Restore the hourly timeline
# --------------------------------------------------

# Detect gaps before creating missing hourly rows.
expected_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="h"
)

missing_hours = expected_index.difference(df.index)

print("Missing hourly timestamps:", len(missing_hours))

# Add missing hourly timestamps, if any.
df = df.reindex(expected_index)
df.index.name = "timestamp"

# --------------------------------------------------
# 6. Fill missing weather features
# --------------------------------------------------

# Interpolate weather readings using time.
# Limit interpolation to short gaps of up to 3 hours.
df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].interpolate(
    method="time",
    limit=3,
    limit_area="inside"
)

# Fill any remaining weather gaps with column medians.
# This is a fallback for isolated or longer missing gaps.
for column in WEATHER_COLUMNS:
    df[column] = df[column].fillna(df[column].median())

# If a whole feature is missing, its median is also missing.
if df[WEATHER_COLUMNS].isna().any().any():
    raise ValueError(
        "Weather features still contain missing values. "
        "Inspect the source data before proceeding."
    )

# --------------------------------------------------
# 7. Handle the target variable
# --------------------------------------------------

# Do not invent solar-output labels by interpolation.
# Rows with missing or invalid targets cannot be used
# as supervised training examples.
df = df.dropna(subset=["solar_output"])

# --------------------------------------------------
# 8. Feature engineering
# --------------------------------------------------

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["day_of_year"] = df.index.dayofyear
df["month"] = df.index.month

# Cyclical encoding preserves the wraparound:
# hour 23 is close to hour 0.
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)
df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

# Annual cycle, useful for seasonal patterns.
df["day_of_year_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)
df["day_of_year_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)

# --------------------------------------------------
# 9. Final checks
# --------------------------------------------------

print("\n--- FINAL DATASET REPORT ---")
print("Final shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())

print("\nRemaining duplicate timestamps:", df.index.duplicated().sum())
print("\nDate range:", df.index.min(), "to", df.index.max())
print("\nProcessed data preview:")
print(df.head())

# --------------------------------------------------
# 10. Save processed data
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

df.to_csv(OUTPUT_PATH, index_label="timestamp")

print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)
print("Cleaning and feature engineering completed.")