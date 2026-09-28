import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ==========================================
# 1. LOAD DATA AND MODEL
# ==========================================

data_file = Path("data") / "freight_data.csv"
model_file = Path("models") / "freight_forecasting_model.pkl"

df = pd.read_csv(data_file)

df["date"] = pd.to_datetime(df["date"])

model = joblib.load(model_file)


# ==========================================
# 2. USER REQUIREMENTS
# ==========================================

route = "Newcastle_Australia_Visakhapatnam"

cargo_quantity = 70000

contract_duration = "Short-term"


# ==========================================
# 3. GET CURRENT MARKET INFORMATION
# ==========================================

route_data = df[df["route"] == route].copy()

route_data = route_data.sort_values("date")

latest = route_data.iloc[-1]

current_rate = latest["freight_rate_usd_mt"]

last_date = df["date"].max()


# ==========================================
# 4. GENERATE FUTURE DATES
# ==========================================

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)


# ==========================================
# 5. CREATE FUTURE DATA
# ==========================================

future_data = []

for date in future_dates:

    future_data.append({
        "date": date,
        "route": route,
        "fuel_price": latest["fuel_price"],
        "demand_index": latest["demand_index"],
        "congestion_index": latest["congestion_index"]
    })


future_df = pd.DataFrame(future_data)


# ==========================================
# 6. DATE FEATURES
# ==========================================

future_df["day"] = future_df["date"].dt.day
future_df["month"] = future_df["date"].dt.month
future_df["year"] = future_df["date"].dt.year
future_df["day_of_year"] = future_df["date"].dt.dayofyear


# ==========================================
# 7. ENCODE ROUTE
# ==========================================

future_df = pd.get_dummies(
    future_df,
    columns=["route"],
    dtype=int
)


# ==========================================
# 8. MATCH MODEL FEATURES
# ==========================================

model_features = model.feature_names_in_

for feature in model_features:

    if feature not in future_df.columns:

        future_df[feature] = 0


future_X = future_df[model_features]


# ==========================================
# 9. FORECAST
# ==========================================

predictions = model.predict(future_X)


# ==========================================
# 10. CALCULATE FORECAST METRICS
# ==========================================

forecast_average = np.mean(predictions)

forecast_min = np.min(predictions)

forecast_max = np.max(predictions)

expected_change = forecast_average - current_rate

expected_change_percent = (
    expected_change / current_rate
) * 100


# ==========================================
# 11. CHARTER DECISION
# ==========================================

if expected_change_percent > 3:

    decision = "CONSIDER EARLIER CHARTERING"

    explanation = (
        "Forecast indicates a meaningful upward "
        "freight movement."
    )

elif expected_change_percent < -3:

    decision = "MONITOR / CONSIDER WAITING"

    explanation = (
        "Forecast indicates a meaningful downward "
        "freight movement."
    )

else:

    decision = "MONITOR MARKET"

    explanation = (
        "Forecast indicates relatively stable "
        "freight conditions."
    )


# ==========================================
# 12. ESTIMATE FREIGHT COST
# ==========================================

current_cost = current_rate * cargo_quantity

forecast_cost = forecast_average * cargo_quantity

potential_difference = (
    forecast_cost - current_cost
)


# ==========================================
# 13. DISPLAY RESULT
# ==========================================

print("==========================================")
print("INTELLIFREIGHT CHARTER TIMING ENGINE")
print("==========================================")

print(f"\nRoute: {route}")

print(f"Cargo quantity: {cargo_quantity:,} MT")

print(f"Contract duration: {contract_duration}")

print("\n------------------------------------------")

print(
    f"Current freight rate: "
    f"${current_rate:.2f}/MT"
)

print(
    f"7-day average forecast: "
    f"${forecast_average:.2f}/MT"
)

print(
    f"Forecast range: "
    f"${forecast_min:.2f} - "
    f"${forecast_max:.2f}/MT"
)

print(
    f"Expected change: "
    f"{expected_change_percent:.2f}%"
)

print("\n------------------------------------------")

print("CHARTER DECISION")

print(f"\n{decision}")

print(f"\nReason:")
print(explanation)

print("\n------------------------------------------")

print(
    f"Current estimated freight cost: "
    f"${current_cost:,.2f}"
)

print(
    f"Forecast estimated freight cost: "
    f"${forecast_cost:,.2f}"
)

print(
    f"Potential cost difference: "
    f"${potential_difference:,.2f}"
)

print("\n==========================================")