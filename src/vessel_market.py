import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "real"
    / "bdi_history.csv"
)


def get_latest_vessel_market():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    vessel_data = {
        "Capesize": {
            "index": float(latest["bci"]),
            "change_pct": float(
                (latest["bci"] - previous["bci"])
                / previous["bci"]
                * 100
            )
        },

        "Panamax": {
            "index": float(latest["bpi"]),
            "change_pct": float(
                (latest["bpi"] - previous["bpi"])
                / previous["bpi"]
                * 100
            )
        },

        "Supramax": {
            "index": float(latest["bsi"]),
            "change_pct": float(
                (latest["bsi"] - previous["bsi"])
                / previous["bsi"]
                * 100
            )
        },

        "Handysize": {
            "index": float(latest["bhsi"]),
            "change_pct": float(
                (latest["bhsi"] - previous["bhsi"])
                / previous["bhsi"]
                * 100
            )
        }
    }

    return {
        "date": latest["date"].date(),
        "vessels": vessel_data
    }


if __name__ == "__main__":

    market = get_latest_vessel_market()

    print("Latest Vessel Market Data")
    print("=========================")

    print(f"Date: {market['date']}")

    for vessel, data in market["vessels"].items():

        print(
            f"{vessel}: "
            f"Index={data['index']:.0f}, "
            f"Change={data['change_pct']:+.2f}%"
        )