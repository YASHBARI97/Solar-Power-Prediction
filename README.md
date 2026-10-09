# Solar Power Prediction Application

## Overview

This project demonstrates an end-to-end solar power prediction pipeline using historical synthetic solar-weather data, machine learning, weather forecast API integration, and interactive visualization.

The application fetches hourly weather forecasts for Pune, Maharashtra, generates future model predictions, and displays the results through a Streamlit dashboard with Plotly charts.

**Note:** The model is trained on synthetic data. Its predictions are for demonstration and should not be treated as validated real-world solar generation estimates.

## Features

* Synthetic solar-weather dataset generation
* Exploratory data analysis and data cleaning
* Feature engineering for weather and time variables
* Chronological training and validation split
* Regression model training and comparison
* Model evaluation using MAE, RMSE, and R²
* WeatherAPI.com hourly forecast integration
* Future solar power predictions
* Interactive charts and forecast table
* CSV export of prediction results

## Technology Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Requests and python-dotenv
* Streamlit
* Plotly
* WeatherAPI.com

## Project Setup

### 1. Clone or download the repository

Download the project or clone the GitHub repository, then open the project folder in VS Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the weather API

Create a `.env` file in the project root:

```text
WEATHER_API_KEY=your_weatherapi_key
```

Replace the placeholder with your own key. Never publish your API key or commit the `.env` file.

### 5. Generate the dataset

```bash
python generate_dataset.py
```

Run this only when you need to create or recreate the dataset.

### 6. Train the model

```bash
python train_model.py
```

This step should save the selected model and report its validation metrics.

### 7. Fetch the weather forecast

```bash
python weather_api.py
```

### 8. Generate solar predictions

```bash
python predict_solar.py
```

### 9. Start the dashboard

```bash
streamlit run app.py
```

Open the local address displayed in the terminal.

## Model Evaluation

Record the actual validation results from your training script.

| Model                | MAE       | RMSE      | R2      |
| HistGradientBoosting | 15.266302 | 20.378471 | 0.985051|  => Best fit
| Random Forest        | 15.518257 | 20.815889 | 0.984403|

The selected model is HistGradientBoosting, based on the validation comparison performed in this project. Report the final model and metrics only after confirming them from your actual training run.

## Output Files

* `solar_weather_dataset.csv`: generated historical dataset
* `weather_forecast.csv`: hourly weather forecast, if saved by the API script
* `solar_predictions.csv`: future solar prediction results
* Saved model artifact: trained model used for inference

## Limitations

* The model uses synthetic training data.
* Forecast accuracy has not been established against actual measured solar generation.
* Weather forecast availability depends on the API provider, network connectivity, and account limits.
* Prediction features must match the training features.

## Future Improvements

* Train and evaluate against real solar generation data.
* Add automated model retraining and monitoring.
* Deploy the application to a cloud platform.
* Add API endpoints for prediction requests.
* Add model explainability and prediction uncertainty estimates.

## Candidature
Name: yash Bari