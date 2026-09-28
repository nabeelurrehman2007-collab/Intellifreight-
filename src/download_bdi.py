from pathlib import Path
import pandas as pd
import yfinance as yf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DIR = BASE_DIR / "data" / "real"

REAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = REAL_DIR / "bdi_history.csv"


# ============================================================
# DOWNLOAD BDI DATA
# ============================================================

print("Downloading extended BDI historical data...")
print("----------------------------------------")

ticker = "^BDIY"

try:
    data = yf.download(
        ticker,
        start="2010-01-01",
        end="2026-09-20",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError("Yahoo Finance returned no data.")

    # Handle possible MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Identify the date column
    if "Date" in data.columns:
        data["date"] = pd.to_datetime(data["Date"])
    elif "Datetime" in data.columns:
        data["date"] = pd.to_datetime(data["Datetime"])
    else:
        raise ValueError("Date column was not found.")

    # Select the available BDI value
    if "Close" in data.columns:
        data["bdi"] = pd.to_numeric(data["Close"], errors="coerce")
    elif "Adj Close" in data.columns:
        data["bdi"] = pd.to_numeric(
            data["Adj Close"],
            errors="coerce",
        )
    else:
        raise ValueError("No Close or Adj Close column found.")

    result = data[["date", "bdi"]].copy()

    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["bdi"] = pd.to_numeric(result["bdi"], errors="coerce")

    result = result.dropna()
    result = result.drop_duplicates(subset=["date"])
    result = result.sort_values("date")

    if len(result) == 0:
        raise ValueError("No valid BDI records remained after cleaning.")

    result.to_csv(OUTPUT_CSV, index=False)

    print(f"Rows: {len(result)}")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print()
    print("Latest records:")
    print(result.tail())
    print()
    print(f"Saved file:")
    print(OUTPUT_CSV)

except Exception as error:
    print()
    print("ERROR: Could not download extended BDI data.")
    print(error)
    print()
    print(
        "Yahoo Finance may not provide this index through the current "
        "public endpoint. Do not delete your existing dataset."
    )