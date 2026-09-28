import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "real"
    / "bdi_history.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "real"
    / "processed_bdi_data.csv"
)


def load_data():

    if not INPUT_PATH.exists():

        print("BDI dataset not found.")
        print(f"Expected file: {INPUT_PATH}")

        return None

    df = pd.read_csv(INPUT_PATH)

    return df


def clean_data(df):

    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["date"]
    )

    # Remove duplicate dates
    df = df.drop_duplicates(
        subset=["date"]
    )

    # Sort chronologically
    df = df.sort_values(
        "date"
    )

    return df


def create_features(df):

    df = df.copy()

    # Date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    # BDI percentage change
    df["bdi_change_pct"] = (
        df["bdi"]
        .pct_change()
        * 100
    )

    # Vessel index relationships
    df["panamax_ratio"] = (
        df["bpi"] / df["bdi"]
    )

    df["capesize_ratio"] = (
        df["bci"] / df["bdi"]
    )

    df["supramax_ratio"] = (
        df["bsi"] / df["bdi"]
    )

    df["handysize_ratio"] = (
        df["bhsi"] / df["bdi"]
    )

    # Rolling market indicator
    df["bdi_3day_avg"] = (
        df["bdi"]
        .rolling(3)
        .mean()
    )

    return df


def validate_data(df):

    print("\nDataset information")
    print("-------------------")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")

    for column in df.columns:
        print(f" - {column}")

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )


def save_data(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nProcessed dataset saved to:")

    print(OUTPUT_PATH)


def main():

    print(
        "IntelliFreight Data Preparation"
    )

    print(
        "================================"
    )

    df = load_data()

    if df is None:
        return

    print("\nBDI dataset loaded successfully.")

    df = clean_data(df)

    df = create_features(df)

    validate_data(df)

    save_data(df)

    print(
        "\nData preparation completed."
    )


if __name__ == "__main__":
    main()