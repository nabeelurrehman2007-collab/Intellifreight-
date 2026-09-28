import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "real",
    "processed_bdi_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "real_bdi_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = (
    df
    .sort_values("date")
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "bci",
    "bpi",
    "bsi",
    "bhsi",
    "year",
    "month",
    "day",
    "day_of_week",
    "bdi_change_pct",
    "panamax_ratio",
    "capesize_ratio",
    "supramax_ratio",
    "handysize_ratio",
    "bdi_3day_avg",
]

target = "bdi"


X = df[features]

y = df[target]


# ============================================================
# TIME-BASED SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("INTELLIFREIGHT — BDI MODEL EVALUATION")
print("=" * 60)

print()

print(
    f"Total observations : {len(df)}"
)

print(
    f"Training records   : {len(X_train)}"
)

print(
    f"Testing records    : {len(X_test)}"
)

print()

print(
    f"MAE                : {mae:.2f} BDI points"
)

print(
    f"RMSE               : {rmse:.2f} BDI points"
)

print(
    f"R²                 : {r2:.4f}"
)

print()

print(
    "Model              : Random Forest Regressor"
)

print(
    "Validation          : Time-based 80/20 split"
)

print()

print("=" * 60)

print(
    "IMPORTANT:"
)

print(
    "These metrics are from a small prototype dataset."
)

print(
    "They should not be presented as guaranteed "
    "real-world forecasting accuracy."
)

print("=" * 60)