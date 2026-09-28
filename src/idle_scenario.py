import pandas as pd
import joblib

print("INTELLIFREIGHT IDLE SCENARIO MANAGEMENT")
print("=" * 50)

# --------------------------------------------------
# 1. Load dataset and trained forecasting model
# --------------------------------------------------
data = pd.read_csv("data/freight_data.csv")
model = joblib.load("models/freight_forecasting_model.pkl")

# --------------------------------------------------
# 2. User inputs
# --------------------------------------------------
route = "Newcastle_Australia_Visakhapatnam"
vessel_type = "Panamax"
cargo_quantity = 70000

# --------------------------------------------------
# 3. Get latest data for selected route
# --------------------------------------------------
route_data = data[data["route"] == route].copy()

if route_data.empty:
    print("ERROR: Selected route was not found in the dataset.")
    exit()

latest = route_data.iloc[-1]

current_demand = latest["demand_index"]
current_congestion = latest["congestion_index"]
current_freight = latest["freight_rate_usd_mt"]

# --------------------------------------------------
# 4. Create next 7 days
# --------------------------------------------------
future_dates = pd.date_range(
    start=pd.to_datetime(latest["date"]) + pd.Timedelta(days=1),
    periods=7
)

future_data = pd.DataFrame({
    "date": future_dates,
    "route": route,
    "fuel_price": latest["fuel_price"],
    "demand_index": latest["demand_index"],
    "congestion_index": latest["congestion_index"]
})

# --------------------------------------------------
# 5. Create date features
# --------------------------------------------------
future_data["day"] = future_data["date"].dt.day
future_data["month"] = future_data["date"].dt.month
future_data["year"] = future_data["date"].dt.year
future_data["day_of_year"] = future_data["date"].dt.dayofyear

# --------------------------------------------------
# 6. Create ALL route columns used during training
# --------------------------------------------------
future_data["route_Australia_Kakinada"] = 0
future_data["route_Indonesia_Chennai"] = 0
future_data["route_Newcastle_Australia_Visakhapatnam"] = 1
future_data["route_Richards_Bay_South_Africa_Paradip"] = 0

# --------------------------------------------------
# 7. Select features in the SAME order as training
# --------------------------------------------------
feature_columns = [
    "fuel_price",
    "demand_index",
    "congestion_index",
    "day",
    "month",
    "year",
    "day_of_year",
    "route_Australia_Kakinada",
    "route_Indonesia_Chennai",
    "route_Newcastle_Australia_Visakhapatnam",
    "route_Richards_Bay_South_Africa_Paradip"
]

# --------------------------------------------------
# 8. Generate freight forecast
# --------------------------------------------------
predictions = model.predict(future_data[feature_columns])

average_forecast = predictions.mean()

# --------------------------------------------------
# 9. Demand classification
# --------------------------------------------------
if current_demand < 0.8:
    demand_status = "LOW"
elif current_demand < 1.2:
    demand_status = "NORMAL"
else:
    demand_status = "HIGH"

# --------------------------------------------------
# 10. Calculate idle risk
# --------------------------------------------------
idle_risk = 0

# Low demand increases idle risk
if current_demand < 0.8:
    idle_risk += 40

# Very low demand adds additional risk
if current_demand < 0.6:
    idle_risk += 20

# High congestion can increase waiting/idle time
if current_congestion > 1.2:
    idle_risk += 20

# Falling freight demand/rates can indicate weaker employment
if average_forecast < current_freight:
    idle_risk += 20

# Keep score between 0 and 100
idle_risk = min(idle_risk, 100)

# --------------------------------------------------
# 11. Generate recommendation
# --------------------------------------------------
if idle_risk >= 70:

    recommendation = "HIGH IDLE RISK"

    action = (
        "Consider alternative employment or vessel repositioning."
    )

elif idle_risk >= 40:

    recommendation = "MODERATE IDLE RISK"

    action = (
        "Monitor cargo demand and prepare an alternative voyage."
    )

else:

    recommendation = "LOW IDLE RISK"

    action = (
        "Continue planned employment."
    )

# --------------------------------------------------
# 12. Display results
# --------------------------------------------------
print(f"\nRoute: {route}")
print(f"Vessel type: {vessel_type}")
print(f"Cargo quantity: {cargo_quantity:,} MT")

print("\n------------------------------------------")

print(f"Current demand index: {current_demand:.2f}")
print(f"Demand status: {demand_status}")
print(f"Current congestion index: {current_congestion:.2f}")
print(f"Current freight rate: ${current_freight:.2f}/MT")
print(f"7-day average forecast: ${average_forecast:.2f}/MT")

print("\n------------------------------------------")
print("IDLE RISK ANALYSIS")

print(f"Idle risk score: {idle_risk}/100")
print(f"Status: {recommendation}")

print("\nRecommended action:")
print(action)

print("\n==========================================")