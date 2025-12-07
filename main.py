import os
import pandas as pd
import numpy as np
import fastf1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

# Don't need cache dir but was recommended for faster reloads
cache_dir = "f1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Get data from fastf1 api
session = fastf1.get_session(2025, "Bahrain", "R")  # For now using 2023 Bahrain race
session.load()

laps = session.laps.copy()

# Convert LapTime to seconds and add LapStart (in seconds) for merge
laps["LapTime"] = laps["LapTime"].dt.total_seconds()
laps["LapStart"] = laps["LapStartTime"].dt.total_seconds()

# Keep useful lap columns
laps = laps[[
    "LapTime",
    "TyreLife",
    "Compound",
    "Driver",
    "Stint",
    "LapStart",
    "LapNumber"
]].dropna()

# Get weather data
weather = session.weather_data.copy()
# the index is time so reset to get an actual "Time" column
weather = weather.reset_index()
weather["Time"] = weather["Time"].dt.total_seconds()

# Merge nearest weather sample to each lap (using merge_asof)
df = pd.merge_asof(
    laps.sort_values("LapStart"),
    weather.sort_values("Time"),
    left_on="LapStart",
    right_on="Time",
    direction="nearest"
)

# Add a fuel load col (but a simple estimate)
# Idea is more laps completed then less fuel
df["FuelLoad"] = df["LapNumber"].max() - df["LapNumber"]

# Set a simple categorical "weather condition" from rainfall
df["Weather"] = np.where(df["Rainfall"] > 0, "Wet", "Dry")

# For the ML model keep just the columns I want
df = df[[
    "LapTime",    # this is the target
    "FuelLoad",
    "TyreLife",
    "TrackTemp",
    "AirTemp",
    "Weather",
    "Driver",
    "LapNumber"  # going to be using this for per driver
]].dropna()

# Now encode categorical variables (prepare for ML)
"""
Our features:
- fuel load -> numeric
- tire wear / tyre life -> numeric
- track temperature -> numeric
- air temperatire -> numeric
- weather conditions -> categorical
- driver/team -> categorical
"""

X = df.drop(columns=["LapTime"])
y = df["LapTime"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Use a column transformer
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Split the cleaned dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Use random forest regressor (suppose to be good for mixed feature types)
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=20,
    ))
])

# Train the model (the model is learning)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

# Per driver lap time predictions
all_preds = model.predict(X)  # Now predict for all laps (not just test set)
df_results = df.copy()
df_results["PredLapTime"] = all_preds

# Table of lap predictions
prediction_table = df_results[[
    "Driver",
    "LapNumber",
    "LapTime",
    "PredLapTime",
    "TyreLife",
    "FuelLoad",
    "TrackTemp",
    "AirTemp",
    "Weather"
]].sort_values(["Driver", "LapNumber"])
print("\n===== Full Prediction Table (All Drivers) =====\n")
print(prediction_table.to_string(index=False))

# Per driver evaluation metrics (things like mean absolute error, etc)
driver_metrics = (
    df_results
    .groupby("Driver")
    .apply(
        lambda g: pd.Series({
            "LapCount": len(g),
            "MAE": mean_absolute_error(g["LapTime"], g["PredLapTime"]),
            "MeanActualLapTime": g["LapTime"].mean(),
            "MeanPredLapTime": g["PredLapTime"].mean(),
        })
    )
    .reset_index()
    .sort_values("MAE")
)
print("\nPer-driver prediction metrics (sorted by MAE):")
print(driver_metrics)
