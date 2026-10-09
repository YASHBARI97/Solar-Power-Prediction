
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Locate and load the dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "solar_weather_dataset.csv"

df = pd.read_csv(DATA_PATH)

# 2. Basic structure
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- LAST 5 ROWS ---")
print(df.tail())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

# 3. Dataset information
print("\n--- DATASET INFO ---")
df.info()

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 4. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- MISSING VALUE PERCENTAGE ---")
print((df.isnull().mean() * 100).round(2))

# 5. Duplicate records
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 6. Timestamp checks
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

print("\n--- INVALID TIMESTAMPS ---")
print(df["timestamp"].isna().sum())

print("\n--- TIMESTAMP RANGE ---")
print(df["timestamp"].min())
print(df["timestamp"].max())

print("\n--- DUPLICATE TIMESTAMPS ---")
print(df["timestamp"].duplicated().sum())

print("\n--- TIME INTERVALS ---")
print(df["timestamp"].sort_values().diff().value_counts().head())

# 7. Numerical feature distributions
numeric_columns = df.select_dtypes(
    include=np.number
).columns

df[numeric_columns].hist(
    figsize=(12, 8),
    bins=30,
    edgecolor="black"
)
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.show()

# 8. Correlation analysis
plt.figure(figsize=(9, 6))
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 9. Solar output over time
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["solar_output"])
plt.title("Solar Output Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Solar Output")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. Solar output by hour
df["hour"] = df["timestamp"].dt.hour

hourly_output = df.groupby("hour")[
    "solar_output"
].mean()

plt.figure(figsize=(10, 5))
hourly_output.plot(kind="bar")
plt.title("Average Solar Output by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Solar Output")
plt.tight_layout()
plt.show()