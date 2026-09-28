import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

data_file = Path("data") / "freight_data.csv"

df = pd.read_csv(data_file)

print("==========================================")
print("INTELLIFREIGHT FREIGHT FORECASTING MODEL")
print("==========================================")

print(f"\nDataset loaded: {len(df)} records")


# ==========================================
# 2. PREPARE DATE FEATURES
# ==========================================

df["date"] = pd.to_datetime(df["date"])

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day_of_year"] = df["date"].dt.dayofyear


# ==========================================
# 3. CONVERT ROUTE INTO NUMERIC FEATURES
# ==========================================

df = pd.get_dummies(
    df,
    columns=["route"],
    dtype=int
)


# ==========================================
# 4. DEFINE FEATURES
# ==========================================

target = "freight_rate_usd_mt"

features = [
    column
    for column in df.columns
    if column not in [
        "date",
        target
    ]
]

X = df[features]
y = df[target]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ==========================================
# 6. CREATE MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 9. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"MAE  : {mae:.2f} USD/MT")
print(f"RMSE : {rmse:.2f} USD/MT")
print(f"R²   : {r2:.4f}")


# ==========================================
# 10. SHOW SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nSample predictions:")
print(results.head(10))


# ==========================================
# 11. SAVE MODEL
# ==========================================

import joblib

model_folder = Path("models")
model_folder.mkdir(exist_ok=True)

model_path = model_folder / "freight_forecasting_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)

print("\n==========================================")
print("FORECASTING MODEL READY!")
print("==========================================")