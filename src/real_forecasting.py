import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# INTELLIFREIGHT - REAL BDI FORECASTING
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "real",
    "bdi_historical_external.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "real")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("INTELLIFREIGHT - BDI FORECASTING MODEL")
print("=" * 70)


# ============================================================
# 1. CHECK DATA FILE
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nBDI dataset not found:\n{DATA_PATH}\n"
        "\nMake sure bdi_historical_external.csv exists in:\n"
        "D:\\IntelliFreight\\data\\real\\"
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset")
print("-" * 70)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 3. CLEAN DATE
# ============================================================

# IMPORTANT:
# Dataset uses DD-MM-YYYY format.
#
# Example:
# 04-09-2026 = 4 September 2026
# 03-09-2026 = 3 September 2026

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


# Check invalid dates

invalid_dates = df["Date"].isna().sum()

if invalid_dates > 0:
    print("\nWarning:")
    print("Invalid dates removed:", invalid_dates)

df = df.dropna(subset=["Date"]).copy()


# ============================================================
# 4. CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Price",
    "Open",
    "High",
    "Low",
    "Vol.",
    "Change %"
]

for column in numeric_columns:

    if column not in df.columns:
        continue

    df[column] = (
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace(
            {
                "nan": np.nan,
                "None": np.nan,
                "": np.nan
            }
        )
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 5. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Date",
    ascending=True
).reset_index(drop=True)


# Remove duplicate dates

duplicate_count = df.duplicated(
    subset=["Date"]
).sum()

if duplicate_count > 0:

    print(
        "\nDuplicate dates removed:",
        duplicate_count
    )

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    ).reset_index(drop=True)


print("\nCleaned dataset")
print("-" * 70)

print("Rows:", len(df))

print(
    "Date range:",
    df["Date"].min().strftime("%d-%m-%Y"),
    "to",
    df["Date"].max().strftime("%d-%m-%Y")
)


# ============================================================
# 6. CREATE BDI TARGET
# ============================================================

# In this dataset, Price represents the Baltic Dry Index value.

df["BDI"] = df["Price"]


# Remove rows without BDI

df = df.dropna(
    subset=["BDI"]
).reset_index(drop=True)


# ============================================================
# 7. CREATE TIME-SERIES FEATURES
# ============================================================

print("\nCreating time-series features...")


# ------------------------------------------------------------
# Lag features
# ------------------------------------------------------------

df["bdi_lag_1"] = df["BDI"].shift(1)
df["bdi_lag_2"] = df["BDI"].shift(2)
df["bdi_lag_3"] = df["BDI"].shift(3)
df["bdi_lag_5"] = df["BDI"].shift(5)
df["bdi_lag_7"] = df["BDI"].shift(7)
df["bdi_lag_14"] = df["BDI"].shift(14)


# ------------------------------------------------------------
# Moving averages
# ------------------------------------------------------------

df["bdi_ma_3"] = (
    df["BDI"]
    .rolling(window=3)
    .mean()
)

df["bdi_ma_5"] = (
    df["BDI"]
    .rolling(window=5)
    .mean()
)

df["bdi_ma_7"] = (
    df["BDI"]
    .rolling(window=7)
    .mean()
)

df["bdi_ma_14"] = (
    df["BDI"]
    .rolling(window=14)
    .mean()
)

df["bdi_ma_30"] = (
    df["BDI"]
    .rolling(window=30)
    .mean()
)


# ------------------------------------------------------------
# BDI percentage changes
# ------------------------------------------------------------

df["bdi_change_1d"] = (
    df["BDI"]
    .pct_change(1)
    * 100
)

df["bdi_change_3d"] = (
    df["BDI"]
    .pct_change(3)
    * 100
)

df["bdi_change_7d"] = (
    df["BDI"]
    .pct_change(7)
    * 100
)

df["bdi_change_14d"] = (
    df["BDI"]
    .pct_change(14)
    * 100
)


# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

df["bdi_volatility_7d"] = (
    df["bdi_change_1d"]
    .rolling(window=7)
    .std()
)

df["bdi_volatility_14d"] = (
    df["bdi_change_1d"]
    .rolling(window=14)
    .std()
)


# ------------------------------------------------------------
# Calendar features
# ------------------------------------------------------------

df["day_of_week"] = (
    df["Date"].dt.dayofweek
)

df["month"] = (
    df["Date"].dt.month
)

df["day_of_month"] = (
    df["Date"].dt.day
)


# ============================================================
# 8. FEATURE LIST
# ============================================================

FEATURES = [

    # Lag features
    "bdi_lag_1",
    "bdi_lag_2",
    "bdi_lag_3",
    "bdi_lag_5",
    "bdi_lag_7",
    "bdi_lag_14",

    # Moving averages
    "bdi_ma_3",
    "bdi_ma_5",
    "bdi_ma_7",
    "bdi_ma_14",
    "bdi_ma_30",

    # Changes
    "bdi_change_1d",
    "bdi_change_3d",
    "bdi_change_7d",
    "bdi_change_14d",

    # Volatility
    "bdi_volatility_7d",
    "bdi_volatility_14d",

    # Calendar
    "day_of_week",
    "month",
    "day_of_month"
]


# ============================================================
# 9. PREPARE MODEL DATA
# ============================================================

model_df = df.dropna(
    subset=FEATURES + ["BDI"]
).copy()

model_df = model_df.reset_index(
    drop=True
)


print("\nFeature engineering complete")
print("-" * 70)
print("Original rows:", len(df))
print("Usable rows:", len(model_df))


# ============================================================
# 10. CHECK DATA SIZE
# ============================================================

if len(model_df) < 100:

    raise ValueError(
        "\nNot enough usable observations for training.\n"
        f"Only {len(model_df)} rows are available."
    )


# ============================================================
# 11. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# We do NOT randomly shuffle time-series data.

split_index = int(
    len(model_df) * 0.80
)

train_df = model_df.iloc[
    :split_index
].copy()

test_df = model_df.iloc[
    split_index:
].copy()


X_train = train_df[FEATURES]
y_train = train_df["BDI"]

X_test = test_df[FEATURES]
y_test = test_df["BDI"]


print("\nTrain/Test split")
print("-" * 70)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training period:",
    train_df["Date"].min().strftime("%d-%m-%Y"),
    "to",
    train_df["Date"].max().strftime("%d-%m-%Y")
)

print(
    "Testing period:",
    test_df["Date"].min().strftime("%d-%m-%Y"),
    "to",
    test_df["Date"].max().strftime("%d-%m-%Y")
)


# ============================================================
# 12. NAIVE BASELINE
# ============================================================

print("\nTraining baseline...")


# Baseline assumption:
# Tomorrow's BDI = today's BDI

baseline_predictions = (
    test_df["bdi_lag_1"]
    .values
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

print("Training Random Forest...")

rf_model = RandomForestRegressor(

    n_estimators=300,

    max_depth=10,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = rf_model.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ============================================================
# 14. XGBOOST
# ============================================================

print("Training XGBoost...")

xgb_model = XGBRegressor(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_predictions = xgb_model.predict(
    X_test
)


xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Naive Previous Day",
        "Random Forest",
        "XGBoost"
    ],

    "MAE": [
        baseline_mae,
        rf_mae,
        xgb_mae
    ],

    "RMSE": [
        baseline_rmse,
        rf_rmse,
        xgb_rmse
    ],

    "R2": [
        baseline_r2,
        rf_r2,
        xgb_r2
    ]
})


print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 16. SELECT BEST ML MODEL
# ============================================================

# Primary selection metric:
# MAE — lower is better.

if xgb_mae <= rf_mae:

    best_model = xgb_model
    best_model_name = "XGBoost"
    best_predictions = xgb_predictions
    best_mae = xgb_mae
    best_rmse = xgb_rmse
    best_r2 = xgb_r2

else:

    best_model = rf_model
    best_model_name = "Random Forest"
    best_predictions = rf_predictions
    best_mae = rf_mae
    best_rmse = rf_rmse
    best_r2 = rf_r2


# ============================================================
# 17. SAVE BEST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "real_bdi_model.pkl"
)


model_package = {

    "model": best_model,

    "model_name": best_model_name,

    "features": FEATURES,

    "mae": best_mae,

    "rmse": best_rmse,

    "r2": best_r2,

    "training_rows": len(train_df),

    "testing_rows": len(test_df),

    "data_start": str(
        df["Date"].min().date()
    ),

    "data_end": str(
        df["Date"].max().date()
    )
}


joblib.dump(
    model_package,
    MODEL_PATH
)


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model:", best_model_name)
print("MAE:", round(best_mae, 2))
print("RMSE:", round(best_rmse, 2))
print("R²:", round(best_r2, 4))

print("\nModel saved:")
print(MODEL_PATH)


# ============================================================
# 18. SAVE FEATURE LIST
# ============================================================

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "real_bdi_features.pkl"
)


joblib.dump(
    FEATURES,
    FEATURE_PATH
)


print("\nFeature list saved:")
print(FEATURE_PATH)


# ============================================================
# 19. SAVE PERFORMANCE RESULTS
# ============================================================

RESULTS_PATH = os.path.join(
    DATA_OUTPUT_DIR,
    "bdi_model_performance.csv"
)


results.to_csv(
    RESULTS_PATH,
    index=False
)


print("\nPerformance results saved:")
print(RESULTS_PATH)


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Date": test_df["Date"].values,

    "Actual_BDI": y_test.values,

    "Naive_Prediction": baseline_predictions,

    "RandomForest_Prediction": rf_predictions,

    "XGBoost_Prediction": xgb_predictions,

    "Selected_Model_Prediction": best_predictions
})


PREDICTION_PATH = os.path.join(
    DATA_OUTPUT_DIR,
    "bdi_test_predictions.csv"
)


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


print("\nTest predictions saved:")
print(PREDICTION_PATH)


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance":
            best_model.feature_importances_
    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    IMPORTANCE_PATH = os.path.join(
        DATA_OUTPUT_DIR,
        "bdi_feature_importance.csv"
    )


    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False
    )


    print("\nTop 10 features:")
    print("-" * 70)

    print(
        importance_df.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    print(
        "\nFeature importance saved:"
    )

    print(IMPORTANCE_PATH)


# ============================================================
# 22. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE TEST PREDICTIONS")
print("=" * 70)

sample_predictions = prediction_df.tail(10).copy()

sample_predictions["Date"] = (
    sample_predictions["Date"]
    .dt.strftime("%d-%m-%Y")
)


print(
    sample_predictions.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Dataset rows:", len(df))
print("Usable rows:", len(model_df))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "\nDate range:",
    df["Date"].min().strftime("%d-%m-%Y"),
    "to",
    df["Date"].max().strftime("%d-%m-%Y")
)

print("\nModel comparison:")

print(
    f"Naive baseline MAE: {baseline_mae:.2f}"
)

print(
    f"Random Forest MAE: {rf_mae:.2f}"
)

print(
    f"XGBoost MAE: {xgb_mae:.2f}"
)

print(
    f"\nBest ML model: {best_model_name}"
)

print(
    f"Best ML MAE: {best_mae:.2f}"
)

print(
    f"Best ML RMSE: {best_rmse:.2f}"
)

print(
    f"Best ML R²: {best_r2:.4f}"
)

print("\nModel file:")
print(MODEL_PATH)

print("\nPerformance file:")
print(RESULTS_PATH)

print("\nPrediction file:")
print(PREDICTION_PATH)

print("\n" + "=" * 70)
print("READY FOR DASHBOARD INTEGRATION")
print("=" * 70)