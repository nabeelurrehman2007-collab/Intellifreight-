import pandas as pd
import numpy as np

print("INTELLIFREIGHT RISK & DISRUPTION ENGINE")
print("=" * 50)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data = pd.read_csv("data/freight_data.csv")

# --------------------------------------------------
# 2. User inputs
# --------------------------------------------------
route = "Newcastle_Australia_Visakhapatnam"

# --------------------------------------------------
# 3. Get route data
# --------------------------------------------------
route_data = data[data["route"] == route].copy()

if route_data.empty:
    print("ERROR: Selected route was not found.")
    exit()

# Make sure data is sorted by date
route_data["date"] = pd.to_datetime(route_data["date"])
route_data = route_data.sort_values("date")

latest = route_data.iloc[-1]

# --------------------------------------------------
# 4. Current market indicators
# --------------------------------------------------
current_freight = latest["freight_rate_usd_mt"]
current_fuel = latest["fuel_price"]
current_demand = latest["demand_index"]
current_congestion = latest["congestion_index"]

# --------------------------------------------------
# 5. Calculate recent freight volatility
# --------------------------------------------------
recent_rates = route_data["freight_rate_usd_mt"].tail(30)

freight_volatility = recent_rates.std()

# --------------------------------------------------
# 6. Calculate fuel volatility
# --------------------------------------------------
recent_fuel = route_data["fuel_price"].tail(30)

fuel_volatility = recent_fuel.std()

# --------------------------------------------------
# 7. Calculate risk components
# --------------------------------------------------

risk_score = 0

# ------------------------------------------
# Freight volatility risk
# ------------------------------------------
if freight_volatility > 4:
    freight_risk = 30
    freight_status = "HIGH"

elif freight_volatility > 2:
    freight_risk = 20
    freight_status = "MODERATE"

else:
    freight_risk = 10
    freight_status = "LOW"

risk_score += freight_risk

# ------------------------------------------
# Fuel price risk
# ------------------------------------------
if fuel_volatility > 5:
    fuel_risk = 25
    fuel_status = "HIGH"

elif fuel_volatility > 2:
    fuel_risk = 15
    fuel_status = "MODERATE"

else:
    fuel_risk = 5
    fuel_status = "LOW"

risk_score += fuel_risk

# ------------------------------------------
# Demand risk
# ------------------------------------------
if current_demand < 30:
    demand_risk = 20
    demand_status = "HIGH"

elif current_demand < 60:
    demand_risk = 10
    demand_status = "MODERATE"

else:
    demand_risk = 5
    demand_status = "LOW"

risk_score += demand_risk

# ------------------------------------------
# Congestion risk
# ------------------------------------------
if current_congestion > 70:
    congestion_risk = 25
    congestion_status = "HIGH"

elif current_congestion > 40:
    congestion_risk = 15
    congestion_status = "MODERATE"

else:
    congestion_risk = 5
    congestion_status = "LOW"

risk_score += congestion_risk

# --------------------------------------------------
# 8. Limit score
# --------------------------------------------------
risk_score = min(risk_score, 100)

# --------------------------------------------------
# 9. Overall risk classification
# --------------------------------------------------
if risk_score >= 70:
    overall_risk = "HIGH RISK"
    recommendation = (
        "Consider risk mitigation measures and avoid "
        "committing to inflexible contracts."
    )

elif risk_score >= 40:
    overall_risk = "MODERATE RISK"
    recommendation = (
        "Monitor market conditions closely and "
        "keep alternative charter options available."
    )

else:
    overall_risk = "LOW RISK"
    recommendation = (
        "Current market indicators show relatively "
        "stable conditions."
    )

# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print(f"\nRoute: {route}")

print("\n------------------------------------------")
print("CURRENT MARKET CONDITIONS")

print(f"Freight rate: ${current_freight:.2f}/MT")
print(f"Fuel price: ${current_fuel:.2f}")
print(f"Demand index: {current_demand:.2f}")
print(f"Congestion index: {current_congestion:.2f}")

print("\n------------------------------------------")
print("RISK COMPONENTS")

print(
    f"Freight volatility: "
    f"{freight_volatility:.2f} → {freight_status}"
)

print(
    f"Fuel volatility: "
    f"{fuel_volatility:.2f} → {fuel_status}"
)

print(
    f"Demand risk: "
    f"{demand_status}"
)

print(
    f"Congestion risk: "
    f"{congestion_status}"
)

print("\n------------------------------------------")
print("OVERALL RISK")

print(f"Risk score: {risk_score}/100")
print(f"Risk level: {overall_risk}")

print("\nRecommended action:")
print(recommendation)

print("\n==========================================")