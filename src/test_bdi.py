import pandas as pd

url = "https://www.balticdryindex.com/data/latest.json"

try:
    df = pd.read_json(url)

    print("BDI data downloaded successfully!")
    print()
    print(df.head())
    print()
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

except Exception as e:
    print("Could not download the data.")
    print("Error:", e)