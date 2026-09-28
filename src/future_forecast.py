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
# 2. SELECT ROUTE
# ==========================================

route = "Newcastle_Australia_Visakhapatnam"


# ==========================================
# 3. GET LAST AVAILABLE DATA
# ==========================================

route_data = df[df["route"] == route].copy()

last_date = df["date"].max()

latest = route_data.iloc[-1]


# ==========================================
# 4. GENERATE FUTURE DATES
# ==========================================

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)


# ==========================================
# 5. CREATE FUTURE INPUT DATA
# ==========================================

future_data = []

for date in future_dates:

    fuel_price = latest["fuel_price"]
    demand_index = latest["demand_index"]
    congestion_index = latest["congestion_index"]

    future_data.append({
        "date": date,
        "route": route,
        "fuel_price": fuel_price,
        "demand_index": demand_index,
        "congestion_index": congestion_index
    })


future_df = pd.DataFrame(future_data)


# ==========================================
# 6. CREATE DATE FEATURES
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
# 9. PREDICT
# ==========================================

predictions = model.predict(future_X)


# ==========================================
# 10. DISPLAY FORECAST
# ==========================================

forecast = pd.DataFrame({
    "date": future_dates,
    "predicted_freight_usd_mt": predictions
})


print("==========================================")
print("INTELLIFREIGHT 7-DAY FREIGHT FORECAST")
print("==========================================")

print(f"\nRoute:")
print(route)

print("\nForecast:")

for _, row in forecast.iterrows():

    print(
        f"{row['date'].date()} → "
        f"${row['predicted_freight_usd_mt']:.2f}/MT"
    )


# ==========================================
# 11. MARKET TREND
# ==========================================

first_prediction = predictions[0]
last_prediction = predictions[-1]

change = last_prediction - first_prediction

print("\n==========================================")
print("MARKET TREND")
print("==========================================")

if change > 1:
    print("Trend: RISING 📈")

elif change < -1:
    print("Trend: FALLING 📉")

else:
    print("Trend: STABLE ➡️")


print(f"Expected change: ${change:.2f}/MT")