import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "real"
    / "bdi_history.csv"
)


def main():

    print("IntelliFreight Vessel Market Analysis")
    print("=====================================")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    latest = df.iloc[-1]

    previous = df.iloc[-2]

    vessels = {
        "Capesize": latest["bci"],
        "Panamax": latest["bpi"],
        "Supramax": latest["bsi"],
        "Handysize": latest["bhsi"]
    }

    print(f"\nLatest market date: {latest['date'].date()}")

    print("\nCurrent Vessel Indices")
    print("----------------------")

    for vessel, value in vessels.items():

        print(
            f"{vessel:10} : "
            f"{value:.0f}"
        )

    print("\nDaily Changes")
    print("-------------")

    for vessel, column in {
        "Capesize": "bci",
        "Panamax": "bpi",
        "Supramax": "bsi",
        "Handysize": "bhsi"
    }.items():

        change = (
            (latest[column] - previous[column])
            / previous[column]
        ) * 100

        print(
            f"{vessel:10} : "
            f"{change:+.2f}%"
        )

    strongest = max(
        vessels,
        key=vessels.get
    )

    weakest = min(
        vessels,
        key=vessels.get
    )

    print("\nMarket Observation")
    print("------------------")

    print(
        f"Highest index: {strongest} "
        f"({vessels[strongest]:.0f})"
    )

    print(
        f"Lowest index: {weakest} "
        f"({vessels[weakest]:.0f})"
    )


if __name__ == "__main__":
    main()