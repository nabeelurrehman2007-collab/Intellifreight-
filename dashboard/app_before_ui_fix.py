import os
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IntelliFreight",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "real")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FREIGHT_DATA_PATH = os.path.join(
    DATA_DIR,
    "freight_data.csv"
)

BDI_DATA_PATH = os.path.join(
    REAL_DIR,
    "bdi_history.csv"
)

PROCESSED_BDI_PATH = os.path.join(
    REAL_DIR,
    "processed_bdi_data.csv"
)

FREIGHT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "freight_forecasting_model.pkl"
)

BDI_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "real_bdi_model.pkl"
)

EXTERNAL_BDI_PATH = os.path.join(
    REAL_DIR,
    "bdi_historical_external.csv"
)

BDI_PERFORMANCE_PATH = os.path.join(
    REAL_DIR,
    "bdi_model_performance.csv"
)

BDI_PREDICTIONS_PATH = os.path.join(
    REAL_DIR,
    "bdi_test_predictions.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
    }

    .decision-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #dbe4ee;
        background: #f8fafc;
        margin: 0.5rem 0 1rem 0;
    }

    .small-note {
        color: #64748b;
        font-size: 0.88rem;
    }

    .source-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: #f1f5f9;
        border-left: 4px solid #64748b;
        margin: 0.6rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_read_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_freight_data():

    df = safe_read_csv(FREIGHT_DATA_PATH)

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    numeric_columns = [
        "freight_rate_usd_mt",
        "fuel_price",
        "demand_index",
        "congestion_index",
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return (
        df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data
def load_bdi_data():

    df = safe_read_csv(BDI_DATA_PATH)

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    for column in [
        "bdi",
        "bci",
        "bpi",
        "bsi",
        "bhsi",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return (
        df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data
def load_processed_bdi():

    df = safe_read_csv(PROCESSED_BDI_PATH)

    if df.empty:
        return df

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    return (
        df
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data
def load_external_bdi():

    df = safe_read_csv(EXTERNAL_BDI_PATH)

    if df.empty:
        return df

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d-%m-%Y",
            errors="coerce",
        )

    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(
            df["Price"],
            errors="coerce",
        )

    return (
        df
        .dropna(subset=["Date", "Price"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


@st.cache_data
def load_bdi_performance():

    df = safe_read_csv(BDI_PERFORMANCE_PATH)

    if df.empty:
        return df

    return df


@st.cache_data
def load_bdi_predictions():

    df = safe_read_csv(BDI_PREDICTIONS_PATH)

    if df.empty:
        return df

    for column in [
        "Date",
        "date",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


@st.cache_resource
def load_model(path):

    if not os.path.exists(path):
        return None

    try:
        return joblib.load(path)

    except Exception:
        return None


def money(value):

    return f"${value:,.0f}"


# ============================================================
# VESSEL DATABASE
# ============================================================

def vessel_database():

    return {

        "Handysize": {
            "min_cargo": 10000,
            "max_cargo": 40000,
            "draft": 10.0,
            "loa": 180,
            "handling": 20000,
        },

        "Supramax": {
            "min_cargo": 40000,
            "max_cargo": 60000,
            "draft": 12.0,
            "loa": 200,
            "handling": 25000,
        },

        "Panamax": {
            "min_cargo": 60000,
            "max_cargo": 85000,
            "draft": 14.0,
            "loa": 230,
            "handling": 30000,
        },

        "Capesize": {
            "min_cargo": 100000,
            "max_cargo": 200000,
            "draft": 18.0,
            "loa": 290,
            "handling": 40000,
        },
    }


# ============================================================
# PORT DATABASE
# ============================================================

def port_database():

    return {

        "Visakhapatnam": {
            "draft": 16.0,
            "loa": 300,
            "handling": 40000,
        },

        "Paradip": {
            "draft": 16.5,
            "loa": 300,
            "handling": 45000,
        },

        "Chennai": {
            "draft": 13.0,
            "loa": 250,
            "handling": 30000,
        },

        "Kakinada": {
            "draft": 12.0,
            "loa": 220,
            "handling": 25000,
        },
    }


# ============================================================
# VESSEL OPTIMIZATION
# ============================================================

def recommend_vessel(
    cargo,
    destination
):
    """
    Prototype vessel-fit engine.

    Checks:
    1. Cargo capacity
    2. Destination draft
    3. Destination LOA
    4. Port handling capacity

    The vessel/port values are prototype assumptions used for the SIH MVP.
    """
    vessels = vessel_database()
    ports = port_database()

    if destination not in ports:
        return None, [], {}

    port = ports[destination]
    suitable = []
    assessment = {}

    for name, specs in vessels.items():

        checks = {
            "Cargo capacity": (
                specs["min_cargo"] <= cargo <= specs["max_cargo"],
                f"Cargo {cargo:,.0f} MT is within "
                f"{specs['min_cargo']:,.0f}–{specs['max_cargo']:,.0f} MT range."
                if specs["min_cargo"] <= cargo <= specs["max_cargo"]
                else (
                    f"Cargo {cargo:,.0f} MT is outside the "
                    f"{specs['min_cargo']:,.0f}–{specs['max_cargo']:,.0f} MT range."
                ),
            ),
            "Draft": (
                specs["draft"] <= port["draft"],
                f"Draft {specs['draft']:.1f} m <= port limit {port['draft']:.1f} m."
                if specs["draft"] <= port["draft"]
                else f"Draft {specs['draft']:.1f} m exceeds port limit {port['draft']:.1f} m.",
            ),
            "LOA": (
                specs["loa"] <= port["loa"],
                f"LOA {specs['loa']} m <= port limit {port['loa']} m."
                if specs["loa"] <= port["loa"]
                else f"LOA {specs['loa']} m exceeds port limit {port['loa']} m.",
            ),
            "Handling": (
                specs["handling"] <= port["handling"],
                f"Handling requirement {specs['handling']:,.0f} MT <= "
                f"port capacity {port['handling']:,.0f} MT."
                if specs["handling"] <= port["handling"]
                else (
                    f"Handling requirement {specs['handling']:,.0f} MT exceeds "
                    f"port capacity {port['handling']:,.0f} MT."
                ),
            ),
        }

        passed = sum(ok for ok, _ in checks.values())
        is_suitable = passed == len(checks)

        if is_suitable:
            suitable.append(name)

        assessment[name] = {
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "suitable": is_suitable,
        }

    # For the prototype, choose the smallest suitable vessel class
    # that can carry the cargo. This avoids unnecessarily selecting
    # a larger class when several classes pass.
    selected = suitable[0] if suitable else None

    return selected, suitable, assessment


# ============================================================
# REAL VESSEL MARKET SNAPSHOT
# ============================================================

def get_latest_vessel_market(bdi_df):

    if bdi_df.empty:
        return pd.DataFrame()

    latest = bdi_df.iloc[-1]

    previous = (
        bdi_df.iloc[-2]
        if len(bdi_df) >= 2
        else None
    )

    rows = []

    for name, column in [
        ("Capesize", "bci"),
        ("Panamax", "bpi"),
        ("Supramax", "bsi"),
        ("Handysize", "bhsi"),
    ]:

        if column not in latest.index:

            continue

        value = float(latest[column])

        if (
            previous is not None
            and column in previous.index
            and pd.notna(previous[column])
            and previous[column] != 0
        ):

            change = (
                (value - float(previous[column]))
                / float(previous[column])
            ) * 100

        else:

            change = np.nan

        rows.append(
            {
                "Vessel Type": name,
                "Index": value,
                "Daily Change %": change,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BDI FORECAST
# ============================================================

def _build_bdi_feature_row(history, target_date):

    series = pd.Series(
        history,
        dtype="float64"
    )

    def safe_change(period):
        if len(series) <= period:
            return 0.0
        previous = float(series.iloc[-1 - period])
        if previous == 0:
            return 0.0
        return (
            (float(series.iloc[-1]) - previous)
            / previous
        ) * 100.0

    row = {
        "bdi_lag_1": float(series.iloc[-1]),
        "bdi_lag_2": float(series.iloc[-2]) if len(series) >= 2 else float(series.iloc[-1]),
        "bdi_lag_3": float(series.iloc[-3]) if len(series) >= 3 else float(series.iloc[-1]),
        "bdi_lag_5": float(series.iloc[-5]) if len(series) >= 5 else float(series.iloc[-1]),
        "bdi_lag_7": float(series.iloc[-7]) if len(series) >= 7 else float(series.iloc[-1]),
        "bdi_lag_14": float(series.iloc[-14]) if len(series) >= 14 else float(series.iloc[-1]),
        "bdi_ma_3": float(series.tail(3).mean()),
        "bdi_ma_5": float(series.tail(5).mean()),
        "bdi_ma_7": float(series.tail(7).mean()),
        "bdi_ma_14": float(series.tail(14).mean()),
        "bdi_ma_30": float(series.tail(30).mean()),
        "bdi_change_1d": safe_change(1),
        "bdi_change_3d": safe_change(3),
        "bdi_change_7d": safe_change(7),
        "bdi_change_14d": safe_change(14),
        "bdi_volatility_7d": float(series.tail(7).std(ddof=1)) if len(series.tail(7)) >= 2 else 0.0,
        "bdi_volatility_14d": float(series.tail(14).std(ddof=1)) if len(series.tail(14)) >= 2 else 0.0,
        "day_of_week": int(target_date.dayofweek),
        "month": int(target_date.month),
        "day_of_month": int(target_date.day),
    }

    return row


def forecast_bdi(
    bdi_model,
    external_df,
    horizon=7,
    feature_names=None,
):

    if (
        bdi_model is None
        or external_df.empty
        or "Price" not in external_df.columns
        or "Date" not in external_df.columns
    ):
        return pd.DataFrame()

    history_df = external_df[
        ["Date", "Price"]
    ].dropna().sort_values("Date")

    if len(history_df) < 30:
        return pd.DataFrame()

    history = history_df["Price"].astype(float).tolist()
    last_date = pd.to_datetime(
        history_df["Date"].iloc[-1]
    )

    rows = []

    for step in range(1, horizon + 1):

        target_date = (
            last_date
            + timedelta(days=step)
        )

        features = _build_bdi_feature_row(
            history,
            target_date,
        )

        X = pd.DataFrame([features])

        if feature_names is not None:
            X = X.reindex(
                columns=list(feature_names),
                fill_value=0,
            )

        try:
            prediction = float(
                bdi_model.predict(X)[0]
            )
        except Exception:
            return pd.DataFrame()

        prediction = max(
            prediction,
            0.0
        )

        rows.append(
            {
                "date": target_date,
                "predicted_bdi": prediction,
            }
        )

        history.append(prediction)

    return pd.DataFrame(rows)



# ============================================================
# ROUTE FREIGHT FORECAST
# ============================================================

def forecast_route(
    route_model,
    freight_df,
    route,
    horizon=7
):

    if (
        route_model is None
        or freight_df.empty
    ):

        return pd.DataFrame()

    route_df = (
        freight_df[
            freight_df["route"] == route
        ]
        .sort_values("date")
    )

    if route_df.empty:
        return pd.DataFrame()

    last = route_df.iloc[-1]

    last_date = pd.to_datetime(
        last["date"]
    )

    feature_names = getattr(
        route_model,
        "feature_names_in_",
        None
    )

    rows = []

    for i in range(
        1,
        horizon + 1
    ):

        date = (
            last_date
            + timedelta(days=i)
        )

        row = {

            "day":
                date.day,

            "month":
                date.month,

            "year":
                date.year,

            "day_of_year":
                date.dayofyear,

            "fuel_price":
                float(
                    last.get(
                        "fuel_price",
                        0
                    )
                ),

            "demand_index":
                float(
                    last.get(
                        "demand_index",
                        0
                    )
                ),

            "congestion_index":
                float(
                    last.get(
                        "congestion_index",
                        0
                    )
                ),
        }

        if feature_names is not None:

            for feature in feature_names:

                if feature.startswith(
                    "route_"
                ):

                    row[feature] = int(
                        feature
                        == f"route_{route}"
                    )

        rows.append(row)

    X = pd.DataFrame(rows)

    if feature_names is not None:

        X = X.reindex(
            columns=list(feature_names),
            fill_value=0
        )

    try:

        predictions = (
            route_model.predict(X)
        )

    except Exception:

        return pd.DataFrame()

    return pd.DataFrame(
        {
            "date": [
                last_date
                + timedelta(days=i)
                for i in range(
                    1,
                    horizon + 1
                )
            ],

            "predicted_rate":
                predictions,
        }
    )


def apply_bdi_market_context(route_forecast, current_rate, bdi_forecast, latest_bdi, weight=0.25):
    """
    Prototype market-context adjustment.

    The route model remains the primary forecast. The real BDI forecast is
    used only as a transparent secondary market signal. `weight=0.25` means
    25% of the BDI percentage move is passed through to the route forecast.
    This is a prototype heuristic, not a statistically trained fusion model.
    """
    if route_forecast.empty or latest_bdi <= 0 or bdi_forecast.empty:
        return route_forecast.copy()

    out = route_forecast.copy()
    bdi_avg = float(bdi_forecast["predicted_bdi"].mean())
    bdi_change_pct = ((bdi_avg - latest_bdi) / latest_bdi) * 100.0

    adjustment_pct = np.clip(bdi_change_pct * weight, -3.0, 3.0)
    out["bdi_change_pct"] = bdi_change_pct
    out["bdi_adjustment_pct"] = adjustment_pct
    out["market_adjusted_rate"] = (
        out["predicted_rate"] * (1.0 + adjustment_pct / 100.0)
    )
    return out


# ============================================================
# RISK ENGINE
# ============================================================

def risk_engine(freight_df):

    if freight_df.empty:

        return {
            "score": 0,
            "level": "Unknown",
            "freight_volatility": 0,
            "fuel_volatility": 0,
            "demand_risk": "Unknown",
            "congestion_risk": "Unknown",
        }

    recent = freight_df.tail(30)

    freight_vol = float(
        recent[
            "freight_rate_usd_mt"
        ].std()
    )

    fuel_vol = float(
        recent[
            "fuel_price"
        ].std()
    )

    demand = float(
        recent.iloc[-1][
            "demand_index"
        ]
    )

    congestion = float(
        recent.iloc[-1][
            "congestion_index"
        ]
    )

    score = 0

    if freight_vol > 4:
        score += 30

    elif freight_vol > 2:
        score += 20

    else:
        score += 10

    if fuel_vol > 5:
        score += 25

    elif fuel_vol > 2:
        score += 15

    else:
        score += 5

    if demand < 30:

        score += 20
        demand_risk = "High"

    elif demand < 60:

        score += 10
        demand_risk = "Moderate"

    else:

        score += 5
        demand_risk = "Low"

    if congestion > 70:

        score += 25
        congestion_risk = "High"

    elif congestion > 40:

        score += 15
        congestion_risk = "Moderate"

    else:

        score += 5
        congestion_risk = "Low"

    score = min(
        score,
        100
    )

    if score >= 70:

        level = "High"

    elif score >= 40:

        level = "Moderate"

    else:

        level = "Low"

    return {

        "score": score,

        "level": level,

        "freight_volatility":
            freight_vol,

        "fuel_volatility":
            fuel_vol,

        "demand_risk":
            demand_risk,

        "congestion_risk":
            congestion_risk,
    }


# ============================================================
# CONTRACT STRATEGY
# ============================================================

def contract_strategy(
    current_rate,
    forecast_rate,
    voyages,
    risk_score
):

    if current_rate <= 0:

        return {
            "strategy": "Unavailable",
            "reason": "No valid freight rate is available.",
            "risk_note": "",
            "expected_change": 0,
        }

    expected_change = (
        (
            forecast_rate
            - current_rate
        )
        / current_rate
    ) * 100

    if expected_change > 3:

        if voyages >= 3:

            strategy = "SHORT-TERM"

            reason = (
                "Forecast indicates potential upward freight "
                "pressure. A short-term multiple-voyage "
                "arrangement may reduce exposure to future "
                "rate increases."
            )

        else:

            strategy = (
                "SPOT / EARLY REVIEW"
            )

            reason = (
                "Forecast indicates potential upward freight "
                "pressure. Review charter timing before "
                "committing."
            )

    elif expected_change < -3:

        strategy = "SPOT / WAIT"

        reason = (
            "Forecast indicates potential downward freight "
            "pressure. Maintaining flexibility may be "
            "preferable to longer commitment."
        )

    else:

        if voyages >= 6:

            strategy = "MEDIUM-TERM"

            reason = (
                "Freight conditions appear relatively stable "
                "and voyage volume is high. A medium-term "
                "arrangement can improve planning stability."
            )

        elif voyages >= 3:

            strategy = "SHORT-TERM"

            reason = (
                "Freight conditions appear relatively stable "
                "and multiple voyages are planned. A "
                "short-term arrangement balances flexibility "
                "and planning stability."
            )

        else:

            strategy = "SPOT"

            reason = (
                "Only a small number of voyages are planned. "
                "Spot contracting preserves flexibility."
            )

    if risk_score >= 70:

        risk_note = (
            "Market risk is elevated. "
            "Maintain alternative charter options."
        )

    elif risk_score >= 40:

        risk_note = (
            "Market risk is moderate. "
            "Monitor conditions before final commitment."
        )

    else:

        risk_note = (
            "Market risk indicators are relatively low."
        )

    return {

        "strategy":
            strategy,

        "reason":
            reason,

        "risk_note":
            risk_note,

        "expected_change":
            expected_change,
    }


# ============================================================
# CONTRACT COSTS
# ============================================================

def contract_costs(
    current_rate,
    forecast_rate,
    cargo,
    voyages
):

    # Prototype assumptions.
    spot_rate = current_rate

    short_rate = (
        current_rate * 0.99
        + forecast_rate * 0.01
    )

    medium_rate = (
        current_rate * 0.97
        + forecast_rate * 0.03
    )

    return pd.DataFrame(
        {

            "Strategy": [
                "SPOT",
                "SHORT-TERM",
                "MEDIUM-TERM",
            ],

            "Prototype Rate ($/MT)": [
                spot_rate,
                short_rate,
                medium_rate,
            ],

            "Total Cost ($)": [
                spot_rate
                * cargo
                * voyages,

                short_rate
                * cargo
                * voyages,

                medium_rate
                * cargo
                * voyages,
            ],
        }
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

freight_df = load_freight_data()

bdi_df = load_bdi_data()

processed_bdi_df = (
    load_processed_bdi()
)

external_bdi_df = load_external_bdi()

bdi_performance_df = load_bdi_performance()

bdi_predictions_df = load_bdi_predictions()

freight_model = load_model(
    FREIGHT_MODEL_PATH
)

bdi_model_package = load_model(
    BDI_MODEL_PATH
)

if isinstance(bdi_model_package, dict):
    bdi_model = bdi_model_package.get("model")
    bdi_feature_names = bdi_model_package.get("features")
else:
    bdi_model = bdi_model_package
    bdi_feature_names = getattr(
        bdi_model,
        "feature_names_in_",
        None,
    )


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title(
    "⚙️ Shipment Inputs"
)

cargo_quantity = st.sidebar.number_input(
    "Cargo Quantity (MT)",
    min_value=1000,
    max_value=200000,
    value=70000,
    step=1000,
)

origin = st.sidebar.selectbox(
    "Origin",
    [
        "Newcastle, Australia",
        "Richards Bay, South Africa",
        "Indonesia",
        "Australia",
    ],
    index=0,
)

destination = st.sidebar.selectbox(
    "Destination",
    [
        "Visakhapatnam",
        "Paradip",
        "Chennai",
        "Kakinada",
    ],
)

duration_months = (
    st.sidebar.number_input(
        "Contract Duration (months)",
        min_value=1,
        max_value=24,
        value=3,
        step=1,
    )
)

number_of_voyages = (
    st.sidebar.number_input(
        "Number of Voyages",
        min_value=1,
        max_value=24,
        value=3,
        step=1,
    )
)

st.sidebar.markdown("---")

# Explicit analysis button for the user.
if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = False

if st.sidebar.button(
    "🔍 Analyze Shipment",
    type="primary",
    use_container_width=True,
):
    st.session_state.analysis_requested = True
    st.rerun()

if st.session_state.analysis_requested:
    st.sidebar.success("Shipment analysis is active.")
else:
    st.sidebar.info(
        "Enter shipment details and click Analyze Shipment."
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "IntelliFreight is a decision-support prototype. "
    "Forecasts and contract calculations should not be "
    "treated as binding market quotations."
)


# ============================================================
# ANALYSIS GATE
# ============================================================

if not st.session_state.analysis_requested:
    st.markdown(
        """
        <div class="decision-box">
            <h2>🚢 Ready to Analyze Your Shipment</h2>
            <p>Enter the shipment details in the sidebar, then click <b>🔍 Analyze Shipment</b> to generate the freight forecast, vessel fit, risk assessment and contract analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 Select your shipment details in the sidebar and click **Analyze Shipment** to start the analysis.")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚢 IntelliFreight</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered dry-bulk freight forecasting, vessel "
    "optimization and contract decision support"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# DETERMINE ROUTE
# ============================================================

route_candidates = (
    sorted(
        freight_df["route"]
        .dropna()
        .unique()
        .tolist()
    )
    if (
        not freight_df.empty
        and "route" in freight_df.columns
    )
    else []
)

# Match the selected origin + destination to the
# corresponding prototype route in freight_data.csv.
route_map = {
    (
        "Newcastle, Australia",
        "Visakhapatnam",
    ): "Newcastle_Australia_Visakhapatnam",

    (
        "Richards Bay, South Africa",
        "Paradip",
    ): "Richards_Bay_South_Africa_Paradip",

    (
        "Indonesia",
        "Chennai",
    ): "Indonesia_Chennai",

    (
        "Australia",
        "Kakinada",
    ): "Australia_Kakinada",
}

selected_route = route_map.get(
    (origin, destination)
)

if selected_route:
    preferred_routes = [selected_route]
else:
    preferred_routes = []


route_key = next(
    (
        route
        for route in preferred_routes
        if route in route_candidates
    ),
    "",
)


# ============================================================
# ROUTE FORECAST
# ============================================================

current_rate = np.nan

forecast_rate = np.nan

route_forecast = (
    pd.DataFrame()
)

if (
    route_key
    and not freight_df.empty
):

    route_history = (
        freight_df[
            freight_df["route"]
            == route_key
        ]
        .sort_values("date")
    )

    if not route_history.empty:

        current_rate = float(
            route_history.iloc[-1][
                "freight_rate_usd_mt"
            ]
        )

        route_forecast = (
            forecast_route(
                freight_model,
                freight_df,
                route_key,
                horizon=7,
            )
        )

        if not route_forecast.empty:

            forecast_rate = float(
                route_forecast[
                    "predicted_rate"
                ].mean()
            )

        else:

            forecast_rate = (
                current_rate
            )


# ============================================================
# BDI MARKET CONTEXT FOR ROUTE FORECAST
# ============================================================

bdi_forecast = forecast_bdi(
    bdi_model,
    external_bdi_df,
    horizon=7,
    feature_names=bdi_feature_names,
)

if (
    not route_forecast.empty
    and not external_bdi_df.empty
    and not bdi_forecast.empty
):
    latest_bdi = float(external_bdi_df.iloc[-1]["Price"])
    route_forecast = apply_bdi_market_context(
        route_forecast,
        current_rate,
        bdi_forecast,
        latest_bdi,
        weight=0.25,
    )
    forecast_rate = float(
        route_forecast["market_adjusted_rate"].mean()
    )
    bdi_context_available = True
else:
    bdi_context_available = False


# ============================================================
# VESSEL / RISK / STRATEGY
# ============================================================

selected_vessel, suitable_vessels, vessel_assessment = (
    recommend_vessel(
        cargo_quantity,
        destination,
    )
)

risk = risk_engine(
    freight_df
)

if (
    np.isfinite(current_rate)
    and np.isfinite(forecast_rate)
):

    strategy = contract_strategy(
        current_rate,
        forecast_rate,
        number_of_voyages,
        risk["score"],
    )

else:

    strategy = {

        "strategy":
            "Unavailable",

        "reason":
            "Freight model/data unavailable.",

        "risk_note":
            "",

        "expected_change":
            0,
    }


# ============================================================
# EXECUTIVE DECISION SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🎯 Executive Decision Summary"
    "</div>",
    unsafe_allow_html=True,
)

change = strategy[
    "expected_change"
]

if change > 2:

    trend = "📈 Increasing"

elif change < -2:

    trend = "📉 Decreasing"

else:

    trend = "➡️ Stable"


vessel_display = (
    selected_vessel
    if selected_vessel
    else "No fit"
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "🚢 Vessel",
        vessel_display,
    )


with k2:

    st.metric(
        "📈 Market Trend",
        trend,
    )


with k3:

    st.metric(
        "⚠️ Risk",
        f"{risk['level']} "
        f"({risk['score']}/100)",
    )


with k4:

    st.metric(
        "📋 Contract",
        strategy["strategy"],
    )


if np.isfinite(
    current_rate
):

    current_cost = (
        current_rate
        * cargo_quantity
        * number_of_voyages
    )

    forecast_cost = (
        forecast_rate
        * cargo_quantity
        * number_of_voyages
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "💰 Current Estimated Cost",
            money(current_cost),
        )

    with c2:

        st.metric(
            "🔮 Forecast Estimated Cost",
            money(forecast_cost),
        )

    with c3:

        st.metric(
            "📦 Cargo / Voyages",
            f"{cargo_quantity:,} MT / "
            f"{number_of_voyages}",
        )

    st.info(
        f"### 🧠 Why this recommendation?\n\n"
        f"{strategy['reason']}\n\n"
        f"**Route:** "
        f"{route_key or 'Unavailable'}  \n"
        f"**Origin:** {origin}  \n"
        f"**Destination:** {destination}  \n"
        f"**Contract duration:** "
        f"{duration_months} month(s)  \n"
        f"**Forecast change:** "
        f"{change:+.2f}%  \n"
        f"**Risk note:** "
        f"{strategy['risk_note']}"
    )

else:

    st.warning(
        "Freight route data/model is unavailable. "
        "BDI market information can still be displayed."
    )


st.markdown("---")


# ============================================================
# REAL DRY-BULK MARKET
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🌍 Real Dry-Bulk Market Snapshot"
    "</div>",
    unsafe_allow_html=True,
)

market_df = (
    get_latest_vessel_market(
        bdi_df
    )
)

if not market_df.empty:

    latest_market_date = (
        pd.to_datetime(
            bdi_df["date"].max()
        ).date()
    )

    st.caption(
        f"Latest available market observation: "
        f"{latest_market_date}. "
        "These are BDI-related market indices, "
        "not route-specific freight quotations."
    )

    cols = st.columns(4)

    for i, row in market_df.iterrows():

        with cols[i]:

            daily_change = (
                row["Daily Change %"]
            )

            change_text = (
                "n/a"
                if pd.isna(
                    daily_change
                )
                else f"{daily_change:+.2f}%"
            )

            st.metric(
                row["Vessel Type"],
                f"{row['Index']:,.0f}",
                change_text,
            )

    st.dataframe(
        market_df.style.format(
            {
                "Index":
                    "{:,.0f}",

                "Daily Change %":
                    "{:+.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.warning(
        "Real BDI data not found."
    )


# ============================================================
# BDI TREND + FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📊 BDI Market Trend & Forecast"
    "</div>",
    unsafe_allow_html=True,
)

if not external_bdi_df.empty:

    bdi_forecast = forecast_bdi(
        bdi_model,
        external_bdi_df,
        horizon=7,
        feature_names=bdi_feature_names,
    )

    fig = go.Figure()

    history_plot = external_bdi_df.tail(180)

    fig.add_trace(
        go.Scatter(
            x=history_plot["Date"],
            y=history_plot["Price"],
            mode="lines",
            name="Historical BDI",
        )
    )

    if not bdi_forecast.empty:

        fig.add_trace(
            go.Scatter(
                x=bdi_forecast["date"],
                y=bdi_forecast["predicted_bdi"],
                mode="lines+markers",
                name="Random Forest Forecast",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=(
            "Baltic Dry Index — "
            "Historical + 7-Day ML Forecast"
        ),
        xaxis_title="Date",
        yaxis_title="BDI",
        hovermode="x unified",
        height=430,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if not bdi_forecast.empty:

        b1, b2, b3 = st.columns(3)

        last_bdi = float(
            external_bdi_df.iloc[-1]["Price"]
        )

        avg_forecast_bdi = float(
            bdi_forecast["predicted_bdi"].mean()
        )

        change_bdi = (
            (avg_forecast_bdi - last_bdi)
            / last_bdi
        ) * 100

        with b1:
            st.metric(
                "Latest Training BDI",
                f"{last_bdi:,.0f}",
            )

        with b2:
            st.metric(
                "7-Day RF Forecast",
                f"{avg_forecast_bdi:,.0f}",
            )

        with b3:
            st.metric(
                "Expected Change",
                f"{change_bdi:+.2f}%",
            )

    if isinstance(bdi_model_package, dict):

        st.markdown("#### 🤖 Real BDI Model Performance")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Model",
                str(
                    bdi_model_package.get(
                        "model_name",
                        "Random Forest",
                    )
                ),
            )

        with p2:
            st.metric(
                "MAE",
                f"{float(bdi_model_package.get('mae', np.nan)):.2f}",
            )

        with p3:
            st.metric(
                "RMSE",
                f"{float(bdi_model_package.get('rmse', np.nan)):.2f}",
            )

        with p4:
            st.metric(
                "R²",
                f"{float(bdi_model_package.get('r2', np.nan)):.4f}",
            )

        st.caption(
            "Test-set metrics from a chronological holdout. "
            "R² is not an accuracy percentage."
        )

        st.caption(
            "Training data: "
            f"{bdi_model_package.get('data_start', 'N/A')} "
            "to "
            f"{bdi_model_package.get('data_end', 'N/A')} "
            f"({bdi_model_package.get('training_rows', 'N/A')} "
            "training rows, "
            f"{bdi_model_package.get('testing_rows', 'N/A')} "
            "test rows)."
        )

else:

    st.warning(
        "Historical BDI model dataset is unavailable."
    )


# ============================================================
# ROUTE FREIGHT FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🔮 Route Freight Forecast"
    "</div>",
    unsafe_allow_html=True,
)

if (
    route_key
    and not route_forecast.empty
):

    fig = go.Figure()

    route_history = (
        freight_df[
            freight_df["route"]
            == route_key
        ]
        .sort_values("date")
    )

    fig.add_trace(
        go.Scatter(
            x=route_history[
                "date"
            ].tail(60),

            y=route_history[
                "freight_rate_usd_mt"
            ].tail(60),

            mode="lines",

            name="Historical "
                 "Route Rate",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=route_forecast[
                "date"
            ],

            y=(
                route_forecast["market_adjusted_rate"]
                if "market_adjusted_rate" in route_forecast.columns
                else route_forecast["predicted_rate"]
            ),

            mode="lines+markers",

            name=(
                "7-Day Market-Adjusted Forecast"
                if "market_adjusted_rate" in route_forecast.columns
                else "7-Day Forecast"
            ),

            line=dict(
                dash="dash"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"Freight Rate Forecast — "
            f"{route_key}"
        ),
        xaxis_title="Date",
        yaxis_title="USD / MT",
        hovermode="x unified",
        height=430,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    r1, r2, r3 = (
        st.columns(3)
    )

    with r1:

        st.metric(
            "Current Rate",
            f"${current_rate:.2f}/MT",
        )

    with r2:

        st.metric(
            "7-Day Average",
            f"${forecast_rate:.2f}/MT",
        )

    with r3:

        st.metric(
            "Expected Change",
            f"{change:+.2f}%",
        )

    if bdi_context_available:
        bdi_change = float(route_forecast["bdi_change_pct"].iloc[0])
        bdi_adjustment = float(route_forecast["bdi_adjustment_pct"].iloc[0])
        st.info(
            f"🌐 BDI market context is influencing the route forecast: "
            f"the 7-day BDI signal is {bdi_change:+.2f}%, and a capped "
            f"25% weighting contributes {bdi_adjustment:+.2f}% to the route-rate forecast. "
            "This is a transparent prototype heuristic, not a trained fusion model."
        )

else:

    st.warning(
        "Route forecast unavailable. "
        "Check the route dataset and model."
    )


# ============================================================
# CHARTER TIMING
# ============================================================

st.markdown(
    '<div class="section-title">'
    "⏱️ Optimal Market Entry Timing"
    "</div>",
    unsafe_allow_html=True,
)

if (
    np.isfinite(current_rate)
    and np.isfinite(forecast_rate)
):

    timing_cols = st.columns(3)

    with timing_cols[0]:

        st.metric(
            "Current Rate",
            f"${current_rate:.2f}/MT",
        )

    with timing_cols[1]:

        st.metric(
            "Forecast Rate",
            f"${forecast_rate:.2f}/MT",
        )

    with timing_cols[2]:

        st.metric(
            "Difference",
            f"${forecast_rate-current_rate:+.2f}/MT",
        )

    if change > 3:

        st.warning(
            "Prototype signal: review charter timing "
            "earlier because the forecast indicates "
            "potential upward rate pressure."
        )

    elif change < -3:

        st.success(
            "Prototype signal: maintaining flexibility "
            "may be useful because the forecast indicates "
            "potential downward rate pressure."
        )

    else:

        st.info(
            "Prototype signal: market conditions are "
            "relatively stable. Continue monitoring "
            "before final commitment."
        )


# ============================================================
# VESSEL OPTIMIZATION

st.markdown(
    '<div class="section-title">'
    "⚓ Vessel Type Optimization"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Prototype constraint engine: cargo capacity, destination draft, "
    "LOA and port handling capacity are checked for every vessel class."
)

vessel_columns = st.columns(4)

for i, (name, specs) in enumerate(vessel_database().items()):

    with vessel_columns[i]:

        details = vessel_assessment.get(name, {})
        suitable = details.get("suitable", False)
        checks = details.get("checks", {})

        st.markdown(f"### {name}")

        st.write(
            f"Cargo: {specs['min_cargo']:,}–{specs['max_cargo']:,} MT"
        )
        st.write(f"Draft: {specs['draft']:.1f} m")
        st.write(f"LOA: {specs['loa']} m")
        st.write(f"Handling: {specs['handling']:,} MT")

        if suitable:
            st.success(
                f"✓ Suitable ({details.get('passed', 0)}/"
                f"{details.get('total', 0)} checks)"
            )
        else:
            st.error(
                f"✗ Not suitable ({details.get('passed', 0)}/"
                f"{details.get('total', 0)} checks)"
            )

        with st.expander("Why?"):
            for check_name, (passed, reason) in checks.items():
                if passed:
                    st.markdown(f"✅ **{check_name}:** {reason}")
                else:
                    st.markdown(f"❌ **{check_name}:** {reason}")

st.markdown("### 🧭 Recommendation Explanation")

if selected_vessel:

    selected_details = vessel_assessment[selected_vessel]

    st.success(
        f"### Recommended prototype vessel: {selected_vessel}"
    )

    st.write(
        f"For **{cargo_quantity:,} MT** to **{destination}**, "
        f"**{selected_vessel}** passes all {selected_details['total']} "
        "prototype compatibility checks."
    )

    explanation_cols = st.columns(4)

    for col, (check_name, (passed, reason)) in zip(
        explanation_cols,
        selected_details["checks"].items()
    ):
        with col:
            st.metric(
                check_name,
                "PASS" if passed else "FAIL"
            )
            st.caption(reason)

    st.info(
        "Selection rule: when multiple vessel classes satisfy all "
        "constraints, the prototype selects the smallest suitable "
        "class that can carry the requested cargo."
    )

else:

    st.error(
        "No vessel in the prototype database satisfies all selected constraints."
    )

    st.warning(
        "Consider changing the cargo quantity or destination, or review "
        "port/vessel constraints with an actual maritime data source."
    )

# IDLE SCENARIO
# ============================================================

st.markdown(
    '<div class="section-title">'
    "💤 Idle Scenario Management"
    "</div>",
    unsafe_allow_html=True,
)

if not freight_df.empty:

    if route_key:

        route_rows = freight_df[
            freight_df["route"]
            == route_key
        ]

    else:

        route_rows = pd.DataFrame()

    if not route_rows.empty:

        latest_route = (
            route_rows
            .sort_values("date")
            .iloc[-1]
        )

    else:

        latest_route = (
            freight_df
            .sort_values("date")
            .iloc[-1]
        )

    demand = float(
        latest_route[
            "demand_index"
        ]
    )

    congestion = float(
        latest_route[
            "congestion_index"
        ]
    )

    idle_risk = 0

    if demand < 30:

        idle_risk += 50

    elif demand < 60:

        idle_risk += 25

    if congestion > 70:

        idle_risk += 30

    elif congestion > 40:

        idle_risk += 15

    idle_risk = min(
        idle_risk,
        100
    )

    if idle_risk >= 70:

        idle_level = "High"

        idle_action = (
            "Consider alternative employment, "
            "repositioning or reducing idle exposure."
        )

    elif idle_risk >= 40:

        idle_level = "Moderate"

        idle_action = (
            "Monitor demand and congestion and "
            "maintain alternative voyage options."
        )

    else:

        idle_level = "Low"

        idle_action = (
            "Current prototype indicators do not "
            "suggest significant idle exposure."
        )

    i1, i2, i3 = (
        st.columns(3)
    )

    with i1:

        st.metric(
            "Demand Index",
            f"{demand:.1f}",
        )

    with i2:

        st.metric(
            "Congestion Index",
            f"{congestion:.1f}",
        )

    with i3:

        st.metric(
            "Idle Risk",
            f"{idle_level} "
            f"({idle_risk}/100)",
        )

    st.info(
        f"**Action:** {idle_action}"
    )


# ============================================================
# FINAL DECISION REPORT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🧾 Final Decision Report"
    "</div>",
    unsafe_allow_html=True,
)

# Build a single judge-friendly summary from the modules above.
if np.isfinite(current_rate) and np.isfinite(forecast_rate):
    estimated_savings = (
        current_rate - forecast_rate
    ) * cargo_quantity * number_of_voyages

    if estimated_savings > 0:
        cost_signal = (
            f"Potential modeled cost reduction of "
            f"{money(estimated_savings)} versus the current rate."
        )
    elif estimated_savings < 0:
        cost_signal = (
            f"Potential modeled additional cost of "
            f"{money(abs(estimated_savings))} versus the current rate."
        )
    else:
        cost_signal = "Modeled cost difference is approximately zero."
else:
    cost_signal = "Cost comparison unavailable."

report_cols = st.columns(2)

with report_cols[0]:
    st.markdown(
        f"""
        **Shipment**  
        {origin} → {destination}  
        **Cargo:** {cargo_quantity:,} MT  
        **Voyages:** {number_of_voyages}  
        **Duration:** {duration_months} month(s)

        **Recommended vessel:** {selected_vessel or "No compatible vessel found"}  
        **Contract strategy:** {strategy["strategy"]}  
        **Market trend:** {trend}
        """
    )

with report_cols[1]:
    bdi_signal_text = "BDI market context unavailable."
    if bdi_context_available and not bdi_forecast.empty:
        latest_bdi = float(external_bdi_df.iloc[-1]["Price"])
        forecast_bdi_value = float(bdi_forecast["predicted_bdi"].iloc[-1])
        bdi_change = ((forecast_bdi_value - latest_bdi) / latest_bdi) * 100
        bdi_signal_text = (
            f"BDI signal: {bdi_change:+.2f}% over the forecast horizon."
        )

    st.markdown(
        f"""
        **Market signal**  
        Current route freight: **{money(current_rate)} / MT**  
        Market-adjusted forecast: **{money(forecast_rate)} / MT**  
        {bdi_signal_text}

        **Risk:** {risk["level"]} ({risk["score"]}/100)  
        **Idle risk:** {idle_level} ({idle_risk}/100)  
        **Cost signal:** {cost_signal}
        """
    )

st.info(
    f"**Decision rationale:** {strategy['reason']} "
    f"{strategy['risk_note']} "
    f"Idle-risk action: {idle_action}"
)

st.caption(
    "Prototype decision-support output: route freight, fuel, demand and "
    "congestion inputs are synthetic unless explicitly marked as real market data. "
    "The system provides modeled guidance, not a binding charter quotation."
)


# ============================================================
# MARKET RISK
# ============================================================

st.markdown(
    '<div class="section-title">'
    "⚠️ Market Risk Engine"
    "</div>",
    unsafe_allow_html=True,
)

risk_cols = st.columns(4)

with risk_cols[0]:

    st.metric(
        "Risk Score",
        f"{risk['score']}/100",
    )

with risk_cols[1]:

    st.metric(
        "Freight Volatility",
        f"{risk['freight_volatility']:.2f}",
    )

with risk_cols[2]:

    st.metric(
        "Fuel Volatility",
        f"{risk['fuel_volatility']:.2f}",
    )

with risk_cols[3]:

    st.metric(
        "Overall Risk",
        risk["level"],
    )


risk_table = pd.DataFrame(
    {

        "Indicator": [
            "Freight volatility",
            "Fuel volatility",
            "Demand risk",
            "Congestion risk",
        ],

        "Status": [

            f"{risk['freight_volatility']:.2f}",

            f"{risk['fuel_volatility']:.2f}",

            risk["demand_risk"],

            risk["congestion_risk"],
        ],
    }
)

st.dataframe(
    risk_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CONTRACT STRATEGY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📋 Contract Strategy"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="decision-box">

    <h3>{strategy['strategy']}</h3>

    <p>{strategy['reason']}</p>

    <p>
    <b>Risk note:</b>
    {strategy['risk_note']}
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONTRACT COST COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">'
    "💰 Contract Cost Comparison"
    "</div>",
    unsafe_allow_html=True,
)

if (
    np.isfinite(current_rate)
    and np.isfinite(forecast_rate)
):

    cost_df = contract_costs(
        current_rate,
        forecast_rate,
        cargo_quantity,
        number_of_voyages,
    )

    st.dataframe(
        cost_df.style.format(
            {

                "Prototype Rate ($/MT)":
                    "${:.2f}",

                "Total Cost ($)":
                    "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure(

        data=[

            go.Bar(

                x=cost_df[
                    "Strategy"
                ],

                y=cost_df[
                    "Total Cost ($)"
                ],

                text=[

                    f"${value:,.0f}"

                    for value
                    in cost_df[
                        "Total Cost ($)"
                    ]
                ],

                textposition="auto",
            )
        ]
    )

    fig.update_layout(

        title=(
            "Estimated Total "
            "Contract Cost"
        ),

        xaxis_title=(
            "Contract Strategy"
        ),

        yaxis_title="USD",

        height=400,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Prototype assumption: contract rates are "
        "estimated using weighted current/forecast "
        "rates. These are not actual negotiated "
        "charter quotations."
    )


# ============================================================
# MULTIPLE VOYAGE PLANNING
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🔁 Multiple-Voyage Planning"
    "</div>",
    unsafe_allow_html=True,
)

voyage_count = int(
    number_of_voyages
)

voyage_table = pd.DataFrame(
    {

        "Voyage":
            list(
                range(
                    1,
                    voyage_count + 1
                )
            ),

        "Cargo (MT)":
            [
                cargo_quantity
            ]
            * voyage_count,

        "Route":
            [
                route_key
                or "Unavailable"
            ]
            * voyage_count,

        "Current Rate ($/MT)":
            [
                current_rate
                if np.isfinite(
                    current_rate
                )
                else np.nan
            ]
            * voyage_count,

        "Forecast Rate ($/MT)":
            [
                forecast_rate
                if np.isfinite(
                    forecast_rate
                )
                else np.nan
            ]
            * voyage_count,
    }
)

st.dataframe(
    voyage_table.style.format(
        {

            "Cargo (MT)":
                "{:,.0f}",

            "Current Rate ($/MT)":
                "${:.2f}",

            "Forecast Rate ($/MT)":
                "${:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# MARKET INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📡 Market Indicators"
    "</div>",
    unsafe_allow_html=True,
)

if not freight_df.empty:

    latest_freight = (
        freight_df
        .sort_values("date")
        .iloc[-1]
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    with m1:

        st.metric(
            "Fuel Price",
            f"${float(latest_freight['fuel_price']):.2f}",
        )

    with m2:

        st.metric(
            "Demand Index",
            f"{float(latest_freight['demand_index']):.1f}",
        )

    with m3:

        st.metric(
            "Congestion Index",
            f"{float(latest_freight['congestion_index']):.1f}",
        )

    with m4:

        st.metric(
            "Route",
            route_key
            or "Unavailable",
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🧠 How IntelliFreight Makes the Decision"
    "</div>",
    unsafe_allow_html=True,
)

ex1, ex2, ex3, ex4 = (
    st.columns(4)
)

with ex1:

    st.markdown(
        "### 1️⃣ Forecast"
    )

    st.write(
        "The ML forecasting layer estimates "
        "future freight conditions using "
        "available market and route-level "
        "features."
    )

with ex2:

    st.markdown(
        "### 2️⃣ Vessel Fit"
    )

    st.write(
        "Cargo quantity is checked against "
        "vessel capacity and prototype port "
        "draft, LOA and handling constraints."
    )

with ex3:

    st.markdown(
        "### 3️⃣ Risk"
    )

    st.write(
        "Freight volatility, fuel volatility, "
        "demand and congestion signals are "
        "combined into a transparent "
        "prototype risk score."
    )

with ex4:

    st.markdown(
        "### 4️⃣ Contract"
    )

    st.write(
        "Forecast direction, voyage volume "
        "and risk are converted into a "
        "transparent contract-strategy "
        "recommendation."
    )


# ============================================================
# SIH ALIGNMENT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🏆 SIH Problem Statement Alignment"
    "</div>",
    unsafe_allow_html=True,
)

alignment_df = pd.DataFrame(
    {

        "SIH Requirement": [

            "Optimal Market Entry Timing",

            "Vessel Type Optimization",

            "Idle Scenario Management",

            "Risk Mitigation",

            "User-Friendly Decision Dashboard",

            "Multiple-Voyage Contract Support",
        ],

        "IntelliFreight Module": [

            "Charter Timing + Freight Forecast",

            "Vessel/Port Constraint Engine",

            "Idle Risk + Alternative Employment",

            "Market Risk Engine",

            "Streamlit Dashboard",

            "Contract Cost + Voyage Planning",
        ],

        "Status": [

            "Prototype",

            "Prototype",

            "Prototype",

            "Prototype",

            "Implemented",

            "Prototype",
        ],
    }
)

st.dataframe(
    alignment_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DATA + MODEL TRANSPARENCY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📚 Data & Model Transparency"
    "</div>",
    unsafe_allow_html=True,
)

t1, t2 = st.columns(2)

with t1:

    st.markdown(
        "### Real market data"
    )

    st.write(
        "The dashboard uses locally stored "
        "BDI history for market context and "
        "the BDI baseline forecasting "
        "experiment."
    )

    st.write(
        f"BDI records: "
        f"**{len(bdi_df):,}**"
    )

    st.write(
        f"Processed BDI records: "
        f"**{len(processed_bdi_df):,}**"
    )

    st.write(
        f"Historical ML BDI records: "
        f"**{len(external_bdi_df):,}**"
    )

with t2:

    st.markdown(
        "### Prototype route data"
    )

    st.write(
        "The route-level freight, fuel, "
        "demand and congestion dataset is "
        "synthetic prototype data created "
        "for demonstrating the end-to-end "
        "IntelliFreight pipeline."
    )

    st.write(
        f"Route records: "
        f"**{len(freight_df):,}**"
    )


st.markdown(
    """
    <div class="source-box">

    <b>Important:</b>

    Real BDI/vessel indices are market
    indicators. They are not the same as
    a verified Newcastle-to-Visakhapatnam
    freight quotation.

    Route-level prototype values should be
    replaced with licensed/verified commercial
    freight, bunker, port and operational data
    before real-world deployment.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🛠️ System Status"
    "</div>",
    unsafe_allow_html=True,
)

status_df = pd.DataFrame(
    {

        "Component": [

            "Route freight dataset",

            "Route freight ML model",

            "BDI history",

            "Processed BDI dataset",

            "BDI ML model",
        ],

        "Status": [

            (
                "✅ Available"
                if not freight_df.empty
                else "❌ Missing"
            ),

            (
                "✅ Loaded"
                if freight_model is not None
                else "❌ Missing"
            ),

            (
                "✅ Available"
                if not bdi_df.empty
                else "❌ Missing"
            ),

            (
                "✅ Available"
                if not processed_bdi_df.empty
                else "❌ Missing"
            ),

            (
                "✅ Loaded"
                if bdi_model is not None
                else "❌ Missing"
            ),
        ],
    }
)

st.dataframe(
    status_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "IntelliFreight — SIH prototype | "
    "AI-assisted dry-bulk freight forecasting, "
    "vessel optimization, risk analysis and "
    "contract decision support."
)