import pandas as pd
import joblib

print("INTELLIFREIGHT MASTER DECISION ENGINE")
print("=" * 60)

# ==========================================================
# 1. LOAD DATA AND MODEL
# ==========================================================

data = pd.read_csv("data/freight_data.csv")
model = joblib.load("models/freight_forecasting_model.pkl")

# ==========================================================
# 2. USER INPUTS
# ==========================================================

route = "Newcastle_Australia_Visakhapatnam"
cargo_quantity = 70000
contract_duration = "Short-term"

# ==========================================================
# 3. VESSEL DATABASE
# ==========================================================

vessels = {
    "Handysize": {
        "min_cargo": 10000,
        "max_cargo": 40000,
        "draft": 10.0,
        "loa": 180
    },

    "Supramax": {
        "min_cargo": 40000,
        "max_cargo": 60000,
        "draft": 12.0,
        "loa": 200
    },

    "Panamax": {
        "min_cargo": 60000,
        "max_cargo": 85000,
        "draft": 14.0,
        "loa": 230
    },

    "Capesize": {
        "min_cargo": 100000,
        "max_cargo": 200000,
        "draft": 18.0,
        "loa": 290
    }
}

# ==========================================================
# 4. PORT DATABASE
# ==========================================================

ports = {
    "Visakhapatnam": {
        "max_draft": 16.0,
        "max_loa": 300
    },

    "Paradip": {
        "max_draft": 16.5,
        "max_loa": 300
    },

    "Chennai": {
        "max_draft": 13.0,
        "max_loa": 250
    },

    "Kakinada": {
        "max_draft": 12.0,
        "max_loa": 220
    }
}

# ==========================================================
# 5. EXTRACT DESTINATION PORT
# ==========================================================

if "Visakhapatnam" in route:
    destination_port = "Visakhapatnam"

elif "Paradip" in route:
    destination_port = "Paradip"

elif "Chennai" in route:
    destination_port = "Chennai"

elif "Kakinada" in route:
    destination_port = "Kakinada"

else:
    print("ERROR: Destination port not found.")
    exit()

port = ports[destination_port]

# ==========================================================
# 6. GET LATEST ROUTE DATA
# ==========================================================

route_data = data[data["route"] == route].copy()

if route_data.empty:
    print("ERROR: Route not found in dataset.")
    exit()

route_data["date"] = pd.to_datetime(route_data["date"])
route_data = route_data.sort_values("date")

latest = route_data.iloc[-1]

current_freight = latest["freight_rate_usd_mt"]
current_fuel = latest["fuel_price"]
current_demand = latest["demand_index"]
current_congestion = latest["congestion_index"]

# ==========================================================
# 7. CREATE 7-DAY FORECAST DATA
# ==========================================================

future_dates = pd.date_range(
    start=latest["date"] + pd.Timedelta(days=1),
    periods=7
)

future_data = pd.DataFrame({
    "date": future_dates,
    "route": route,
    "fuel_price": current_fuel,
    "demand_index": current_demand,
    "congestion_index": current_congestion
})

future_data["day"] = future_data["date"].dt.day
future_data["month"] = future_data["date"].dt.month
future_data["year"] = future_data["date"].dt.year
future_data["day_of_year"] = future_data["date"].dt.dayofyear

# ==========================================================
# 8. ROUTE ONE-HOT ENCODING
# ==========================================================

future_data["route_Australia_Kakinada"] = 0
future_data["route_Indonesia_Chennai"] = 0
future_data["route_Newcastle_Australia_Visakhapatnam"] = 1
future_data["route_Richards_Bay_South_Africa_Paradip"] = 0

# ==========================================================
# 9. MODEL FEATURES
# ==========================================================

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

# ==========================================================
# 10. FREIGHT FORECAST
# ==========================================================

predictions = model.predict(
    future_data[feature_columns]
)

average_forecast = predictions.mean()
minimum_forecast = predictions.min()
maximum_forecast = predictions.max()

expected_change = average_forecast - current_freight

expected_change_percent = (
    expected_change / current_freight
) * 100

# ==========================================================
# 11. CHARTER TIMING
# ==========================================================

if expected_change_percent > 3:

    charter_decision = "CONSIDER EARLIER CHARTERING"

    charter_reason = (
        "Forecast indicates freight rates may increase."
    )

elif expected_change_percent < -3:

    charter_decision = "CONSIDER WAITING"

    charter_reason = (
        "Forecast indicates freight rates may decrease."
    )

else:

    charter_decision = "MONITOR MARKET"

    charter_reason = (
        "Forecast indicates relatively stable freight conditions."
    )

# ==========================================================
# 12. FREIGHT COST COMPARISON
# ==========================================================

current_cost = current_freight * cargo_quantity
forecast_cost = average_forecast * cargo_quantity

cost_difference = forecast_cost - current_cost

# ==========================================================
# 13. VESSEL OPTIMIZATION
# ==========================================================

recommended_vessel = None

for vessel_name, vessel in vessels.items():

    cargo_ok = (
        vessel["min_cargo"]
        <= cargo_quantity
        <= vessel["max_cargo"]
    )

    draft_ok = vessel["draft"] <= port["max_draft"]

    loa_ok = vessel["loa"] <= port["max_loa"]

    if cargo_ok and draft_ok and loa_ok:

        recommended_vessel = vessel_name
        break

if recommended_vessel is None:
    recommended_vessel = "NO SUITABLE VESSEL"

# ==========================================================
# 14. IDLE RISK
# ==========================================================

idle_risk = 0

# Dataset demand is approximately 0–100
if current_demand < 30:
    idle_risk += 40

elif current_demand < 60:
    idle_risk += 20

# High congestion increases waiting risk
if current_congestion > 70:
    idle_risk += 20

elif current_congestion > 40:
    idle_risk += 10

# Falling freight rate
if average_forecast < current_freight:
    idle_risk += 20

idle_risk = min(idle_risk, 100)

if idle_risk >= 70:

    idle_status = "HIGH IDLE RISK"

elif idle_risk >= 40:

    idle_status = "MODERATE IDLE RISK"

else:

    idle_status = "LOW IDLE RISK"

# ==========================================================
# 15. MARKET RISK
# ==========================================================

recent_rates = route_data["freight_rate_usd_mt"].tail(30)

freight_volatility = recent_rates.std()

recent_fuel = route_data["fuel_price"].tail(30)

fuel_volatility = recent_fuel.std()

risk_score = 0

# Freight volatility
if freight_volatility > 4:
    risk_score += 30

elif freight_volatility > 2:
    risk_score += 20

else:
    risk_score += 10

# Fuel volatility
if fuel_volatility > 5:
    risk_score += 25

elif fuel_volatility > 2:
    risk_score += 15

else:
    risk_score += 5

# Demand
if current_demand < 30:
    risk_score += 20

elif current_demand < 60:
    risk_score += 10

else:
    risk_score += 5

# Congestion
if current_congestion > 70:
    risk_score += 25

elif current_congestion > 40:
    risk_score += 15

else:
    risk_score += 5

risk_score = min(risk_score, 100)

if risk_score >= 70:

    risk_level = "HIGH RISK"

elif risk_score >= 40:

    risk_level = "MODERATE RISK"

else:

    risk_level = "LOW RISK"

# ==========================================================
# 16. MASTER RECOMMENDATION
# ==========================================================

if recommended_vessel == "NO SUITABLE VESSEL":

    master_action = (
        "Review vessel selection because no suitable vessel "
        "matches the cargo and port constraints."
    )

elif risk_score >= 70:

    master_action = (
        "High market risk detected. Keep alternative charter "
        "options available and review contract flexibility."
    )

elif expected_change_percent < -3:

    master_action = (
        "Freight rates are forecast to decline. "
        "Consider monitoring the market before fixing the vessel."
    )

elif expected_change_percent > 3:

    master_action = (
        "Freight rates are forecast to increase. "
        "Consider securing charter capacity earlier."
    )

else:

    master_action = (
        "Market conditions are relatively stable. "
        "Monitor the market and maintain alternative options."
    )

# ==========================================================
# 17. DISPLAY MASTER REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("INTELLIFREIGHT MASTER DECISION REPORT")
print("=" * 60)

print("\nPROJECT INPUTS")
print("-" * 60)

print(f"Route: {route}")
print(f"Destination port: {destination_port}")
print(f"Cargo quantity: {cargo_quantity:,} MT")
print(f"Contract duration: {contract_duration}")

print("\nFREIGHT FORECAST")
print("-" * 60)

print(f"Current freight rate: ${current_freight:.2f}/MT")
print(f"7-day average forecast: ${average_forecast:.2f}/MT")
print(
    f"Forecast range: "
    f"${minimum_forecast:.2f} - ${maximum_forecast:.2f}/MT"
)
print(f"Expected change: {expected_change_percent:.2f}%")

print("\nCHARTER TIMING")
print("-" * 60)

print(f"Decision: {charter_decision}")
print(f"Reason: {charter_reason}")

print("\nVESSEL OPTIMIZATION")
print("-" * 60)

print(f"Recommended vessel: {recommended_vessel}")

print("\nIDLE SCENARIO")
print("-" * 60)

print(f"Idle risk score: {idle_risk}/100")
print(f"Idle risk status: {idle_status}")

print("\nMARKET RISK")
print("-" * 60)

print(f"Freight volatility: {freight_volatility:.2f}")
print(f"Fuel volatility: {fuel_volatility:.2f}")
print(f"Risk score: {risk_score}/100")
print(f"Risk level: {risk_level}")

print("\nCOST ANALYSIS")
print("-" * 60)

print(f"Current estimated cost: ${current_cost:,.2f}")
print(f"Forecast estimated cost: ${forecast_cost:,.2f}")

if cost_difference < 0:

    print(
        f"Potential difference: "
        f"${abs(cost_difference):,.2f} lower"
    )

elif cost_difference > 0:

    print(
        f"Potential difference: "
        f"${cost_difference:,.2f} higher"
    )

else:

    print("Potential difference: $0.00")

print("\nMASTER RECOMMENDATION")
print("-" * 60)

print(master_action)

print("\n")
print("=" * 60)
print("END OF INTELLIFREIGHT DECISION REPORT")
print("=" * 60)