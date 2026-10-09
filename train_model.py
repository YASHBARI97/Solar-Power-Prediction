
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "solar_weather_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "solar_model.joblib"

TARGET = "solar_output"

# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

# --------------------------------------------------
# 3. Select features and target
# --------------------------------------------------

feature_columns = [
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "precipitation",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

missing_features = set(feature_columns + [TARGET]) - set(df.columns)

if missing_features:
    raise ValueError(
        f"Missing required columns: {sorted(missing_features)}"
    )

X = df[feature_columns].copy()
y = df[TARGET].copy()

if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing values remain in training data.")

# --------------------------------------------------
# 4. Chronological train/validation split
# --------------------------------------------------

# First 50 days for training, next 10 days for validation.
split_time = df["timestamp"].min() + pd.Timedelta(days=50)

train_mask = df["timestamp"] < split_time
validation_mask = df["timestamp"] >= split_time

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

if X_train.empty or X_val.empty:
    raise ValueError("Training or validation set is empty.")

print("\n--- CHRONOLOGICAL SPLIT ---")
print("Split timestamp:", split_time)
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Training ends:", df.loc[train_mask, "timestamp"].max())
print("Validation starts:", df.loc[validation_mask, "timestamp"].min())

# --------------------------------------------------
# 5. Define candidate regression models
# --------------------------------------------------

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),

}

# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    # Solar output cannot be negative.
    predictions = np.clip(predictions, 0, None)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    trained_models[model_name] = model

    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.4f}")

# --------------------------------------------------
# 7. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# --------------------------------------------------
# 8. Save selected model and metadata
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": best_model,
    "feature_columns": feature_columns,
    "target": TARGET,
    "model_name": best_model_name,
    "split_timestamp": str(split_time),
}

joblib.dump(artifact, MODEL_PATH)

print("\nBest model by validation RMSE:", best_model_name)
print("Saved model to:", MODEL_PATH)
print("\nTraining and evaluation completed.")