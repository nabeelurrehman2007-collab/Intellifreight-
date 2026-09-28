import pandas as pd


# ==========================================
# VESSEL DATABASE
# ==========================================

vessels = [
    {
        "vessel_type": "Handysize",
        "min_capacity": 10000,
        "max_capacity": 40000,
        "draft": 10.0,
        "loa": 180,
        "handling_rate": 20000
    },
    {
        "vessel_type": "Supramax",
        "min_capacity": 40000,
        "max_capacity": 60000,
        "draft": 12.0,
        "loa": 200,
        "handling_rate": 25000
    },
    {
        "vessel_type": "Panamax",
        "min_capacity": 60000,
        "max_capacity": 85000,
        "draft": 14.0,
        "loa": 230,
        "handling_rate": 30000
    },
    {
        "vessel_type": "Capesize",
        "min_capacity": 100000,
        "max_capacity": 200000,
        "draft": 18.0,
        "loa": 290,
        "handling_rate": 40000
    }
]


# ==========================================
# PORT DATABASE
# ==========================================

ports = {
    "Visakhapatnam": {
        "max_draft": 16.0,
        "max_loa": 300,
        "handling_capacity": 40000
    },

    "Paradip": {
        "max_draft": 16.5,
        "max_loa": 300,
        "handling_capacity": 45000
    },

    "Chennai": {
        "max_draft": 13.0,
        "max_loa": 250,
        "handling_capacity": 30000
    },

    "Kakinada": {
        "max_draft": 12.0,
        "max_loa": 220,
        "handling_capacity": 25000
    }
}


# ==========================================
# USER INPUT
# ==========================================

cargo_quantity = 70000

destination_port = "Visakhapatnam"


# ==========================================
# GET PORT INFORMATION
# ==========================================

port = ports[destination_port]

print("==========================================")
print("INTELLIFREIGHT VESSEL OPTIMIZATION")
print("==========================================")

print(f"\nCargo quantity: {cargo_quantity:,} MT")
print(f"Destination: {destination_port}")

print("\nPort constraints:")
print(f"Maximum draft: {port['max_draft']} m")
print(f"Maximum LOA: {port['max_loa']} m")
print(f"Handling capacity: {port['handling_capacity']:,} MT/day")


# ==========================================
# CHECK VESSELS
# ==========================================

results = []

for vessel in vessels:

    capacity_ok = (
        vessel["min_capacity"]
        <= cargo_quantity
        <= vessel["max_capacity"]
    )

    draft_ok = (
        vessel["draft"]
        <= port["max_draft"]
    )

    loa_ok = (
        vessel["loa"]
        <= port["max_loa"]
    )

    handling_ok = (
        vessel["handling_rate"]
        <= port["handling_capacity"]
    )

    feasible = (
        capacity_ok
        and draft_ok
        and loa_ok
        and handling_ok
    )

    results.append({
        "vessel_type": vessel["vessel_type"],
        "capacity_ok": capacity_ok,
        "draft_ok": draft_ok,
        "loa_ok": loa_ok,
        "handling_ok": handling_ok,
        "feasible": feasible
    })


# ==========================================
# DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("VESSEL COMPATIBILITY")
print("==========================================")

print(results_df.to_string(index=False))


# ==========================================
# RECOMMENDATION
# ==========================================

feasible_vessels = results_df[
    results_df["feasible"] == True
]

print("\n==========================================")
print("RECOMMENDATION")
print("==========================================")

if len(feasible_vessels) == 0:

    print("⚠ No suitable vessel found.")

else:

    # Select the smallest feasible vessel
    vessel_order = {
        "Handysize": 1,
        "Supramax": 2,
        "Panamax": 3,
        "Capesize": 4
    }

    feasible_vessels = feasible_vessels.copy()

    feasible_vessels["order"] = (
        feasible_vessels["vessel_type"]
        .map(vessel_order)
    )

    recommendation = feasible_vessels.sort_values(
        "order"
    ).iloc[0]

    print(
        f"Recommended vessel: "
        f"{recommendation['vessel_type']}"
    )

    print(
        "Reason: Suitable for the cargo quantity "
        "and port constraints."
    )