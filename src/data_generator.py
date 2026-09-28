import pandas as pd
import numpy as np
from pathlib import Path

# Make the generated data reproducible
np.random.seed(42)

# Number of historical records
num_records = 1000

# Generate daily dates
dates = pd.date_range(
    start="2023-01-01",
    periods=num_records,
    freq="D"
)

# Trade routes
routes = [
    "Newcastle_Australia_Visakhapatnam",
    "Richards_Bay_South_Africa_Paradip",
    "Indonesia_Chennai",
    "Australia_Kakinada"
]

# Base freight rates for each route
base_rates = {
    "Newcastle_Australia_Visakhapatnam": 28,
    "Richards_Bay_South_Africa_Paradip": 25,
    "Indonesia_Chennai": 22,
    "Australia_Kakinada": 27
}

data = []

for date in dates:

    # Select a random route
    route = np.random.choice(routes)

    # Simulated market factors
    fuel_price = np.random.normal(650, 40)
    demand_index = np.random.normal(70, 10)
    congestion_index = np.random.normal(30, 12)

    # Seasonal effect
    seasonal_effect = 3 * np.sin(
        2 * np.pi * date.dayofyear / 365
    )

    # Simulated freight rate
    freight_rate = (
        base_rates[route]
        + 0.02 * (fuel_price - 650)
        + 0.08 * (demand_index - 70)
        + 0.05 * (congestion_index - 30)
        + seasonal_effect
        + np.random.normal(0, 2)
    )

    data.append([
        date,
        route,
        round(freight_rate, 2),
        round(fuel_price, 2),
        round(demand_index, 2),
        round(congestion_index, 2)
    ])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "date",
        "route",
        "freight_rate_usd_mt",
        "fuel_price",
        "demand_index",
        "congestion_index"
    ]
)

# Make sure the data folder exists
output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

# Save CSV
output_file = output_folder / "freight_data.csv"

df.to_csv(output_file, index=False)

print("==========================================")
print("INTELLIFREIGHT DATA GENERATOR")
print("==========================================")
print(f"Dataset created successfully!")
print(f"Number of records: {len(df)}")
print(f"Saved to: {output_file}")
print("\nFirst 5 records:")
print(df.head())
print("\nDataset columns:")
print(df.columns.tolist())