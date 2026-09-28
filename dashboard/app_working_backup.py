import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
REAL_DATA_DIR = DATA_DIR / "real"
MODEL_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

sys.path.append(str(SRC_DIR))


# ============================================================
# IMPORT REAL VESSEL MARKET MODULE
# ============================================================

try:
    from vessel_market import get_latest_vessel_market
    VESSEL_MARKET_AVAILABLE = True
except Exception:
    VESSEL_MARKET_AVAILABLE = False


# ============================================================
# FILE PATHS
# ============================================================

SYNTHETIC_DATA_PATH = DATA_DIR / "freight_data.csv"

REAL_BDI_PATH = (
    REAL_DATA_DIR / "bdi_history.csv"
)

PROCESSED_BDI_PATH = (
    REAL_DATA_DIR / "processed_bdi_data.csv"
)

MODEL_PATH = (
    MODEL_DIR / "freight_forecasting_model.pkl"
)

REAL_BDI_MODEL_PATH = (
    MODEL_DIR / "real_bdi_model.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliFreight",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .recommendation-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        background-color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .small-note {
        font-size: 13px;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def load_freight_data():

    if SYNTHETIC_DATA_PATH.exists():

        df = pd.read_csv(
            SYNTHETIC_DATA_PATH
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["date"]
        )

        return df.sort_values("date")

    return pd.DataFrame()


def load_real_bdi():

    if not REAL_BDI_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        REAL_BDI_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    return df.sort_values("date")


@st.cache_resource
def load_forecasting_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(
        MODEL_PATH
    )


@st.cache_resource
def load_real_bdi_model():

    if not REAL_BDI_MODEL_PATH.exists():
        return None

    return joblib.load(
        REAL_BDI_MODEL_PATH
    )


# ============================================================
# ROUTE MAPPING
# ============================================================

ROUTE_MAP = {

    "Newcastle, Australia|Visakhapatnam":
        "Newcastle_Australia_Visakhapatnam",

    "Richards Bay, South Africa|Paradip":
        "Richards_Bay_South_Africa_Paradip",

    "Indonesia|Chennai":
        "Indonesia_Chennai",

    "Australia|Kakinada":
        "Australia_Kakinada"
}


# ============================================================
# VESSEL DATABASE
# PROTOTYPE VALUES
# ============================================================

VESSELS = {

    "Handysize": {
        "min_cargo": 10000,
        "max_cargo": 40000,
        "draft": 10.0,
        "loa": 180,
        "handling": 20000
    },

    "Supramax": {
        "min_cargo": 40000,
        "max_cargo": 60000,
        "draft": 12.0,
        "loa": 200,
        "handling": 25000
    },

    "Panamax": {
        "min_cargo": 60000,
        "max_cargo": 85000,
        "draft": 14.0,
        "loa": 230,
        "handling": 30000
    },

    "Capesize": {
        "min_cargo": 100000,
        "max_cargo": 200000,
        "draft": 18.0,
        "loa": 290,
        "handling": 40000
    }
}


# ============================================================
# PORT DATABASE
# PROTOTYPE VALUES
# ============================================================

PORTS = {

    "Visakhapatnam": {
        "max_draft": 16.0,
        "max_loa": 300,
        "handling": 40000
    },

    "Paradip": {
        "max_draft": 16.5,
        "max_loa": 300,
        "handling": 45000
    },

    "Chennai": {
        "max_draft": 13.0,
        "max_loa": 250,
        "handling": 30000
    },

    "Kakinada": {
        "max_draft": 12.0,
        "max_loa": 220,
        "handling": 25000
    }
}


# ============================================================
# VESSEL OPTIMIZATION
# ============================================================

def recommend_vessel(
    cargo_quantity,
    destination
):

    port = PORTS[destination]

    candidates = []

    for vessel_name, vessel in VESSELS.items():

        cargo_ok = (
            vessel["min_cargo"]
            <= cargo_quantity
            <= vessel["max_cargo"]
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
            vessel["handling"]
            <= port["handling"]
        )

        if (
            cargo_ok
            and draft_ok
            and loa_ok
            and handling_ok
        ):
            candidates.append(
                vessel_name
            )

    if not candidates:

        # Fallback based on cargo
        if cargo_quantity <= 40000:
            recommendation = "Handysize"

        elif cargo_quantity <= 60000:
            recommendation = "Supramax"

        elif cargo_quantity <= 85000:
            recommendation = "Panamax"

        else:
            recommendation = "Capesize"

        return recommendation

    # Choose the smallest suitable vessel
    vessel_order = [
        "Handysize",
        "Supramax",
        "Panamax",
        "Capesize"
    ]

    for vessel_name in vessel_order:

        if vessel_name in candidates:
            return vessel_name

    return candidates[0]


# ============================================================
# CHARTER TIMING
# ============================================================

def calculate_charter_timing(
    current_rate,
    forecast_rate
):

    expected_change = (
        (
            forecast_rate
            - current_rate
        )
        / current_rate
    ) * 100

    if expected_change > 3:

        decision = (
            "CONSIDER EARLIER CHARTERING"
        )

        reason = (
            "Forecast indicates a potential "
            "increase in freight rates."
        )

    elif expected_change < -3:

        decision = (
            "MONITOR / CONSIDER WAITING"
        )

        reason = (
            "Forecast indicates a potential "
            "decrease in freight rates."
        )

    else:

        decision = "MONITOR MARKET"

        reason = (
            "Forecast indicates relatively "
            "stable freight conditions."
        )

    return {
        "expected_change": expected_change,
        "decision": decision,
        "reason": reason
    }


# ============================================================
# IDLE RISK
# ============================================================

def calculate_idle_risk(
    demand_index,
    congestion_index
):

    score = 0

    if demand_index < 30:
        score += 40

    elif demand_index < 60:
        score += 20

    else:
        score += 5

    if congestion_index > 70:
        score += 40

    elif congestion_index > 40:
        score += 20

    else:
        score += 5

    score = min(
        score,
        100
    )

    if score >= 70:

        status = "HIGH IDLE RISK"

        action = (
            "Consider alternative cargo, "
            "voyage repositioning or "
            "alternative employment."
        )

    elif score >= 40:

        status = "MODERATE IDLE RISK"

        action = (
            "Monitor cargo demand and "
            "prepare an alternative voyage."
        )

    else:

        status = "LOW IDLE RISK"

        action = (
            "Current demand and congestion "
            "conditions indicate limited "
            "idle exposure."
        )

    return {
        "score": score,
        "status": status,
        "action": action
    }


# ============================================================
# MARKET RISK
# ============================================================

def calculate_market_risk(
    freight_series,
    fuel_price,
    demand_index,
    congestion_index
):

    freight_volatility = (
        freight_series.std()
        if len(freight_series) > 1
        else 0
    )

    if freight_volatility > 4:
        freight_risk = 30

    elif freight_volatility > 2:
        freight_risk = 20

    else:
        freight_risk = 10

    # Fuel volatility is unavailable from the
    # current synthetic snapshot, so use a
    # conservative prototype indicator.

    fuel_risk = 15

    if demand_index < 30:
        demand_risk = 20

    elif demand_index < 60:
        demand_risk = 10

    else:
        demand_risk = 5

    if congestion_index > 70:
        congestion_risk = 25

    elif congestion_index > 40:
        congestion_risk = 15

    else:
        congestion_risk = 5

    score = min(
        freight_risk
        + fuel_risk
        + demand_risk
        + congestion_risk,
        100
    )

    if score >= 70:
        level = "HIGH RISK"

        recommendation = (
            "Maintain alternative charter "
            "options and monitor market "
            "conditions closely."
        )

    elif score >= 40:
        level = "MODERATE RISK"

        recommendation = (
            "Monitor market conditions before "
            "final commitment."
        )

    else:
        level = "LOW RISK"

        recommendation = (
            "Current risk indicators are "
            "relatively low."
        )

    return {
        "freight_volatility": freight_volatility,
        "fuel_price": fuel_price,
        "score": score,
        "level": level,
        "recommendation": recommendation
    }


# ============================================================
# CONTRACT STRATEGY
# ============================================================

def recommend_contract_strategy(
    current_rate,
    forecast_rate,
    number_of_voyages,
    market_risk_score
):

    expected_change = (
        (
            forecast_rate
            - current_rate
        )
        / current_rate
    ) * 100

    if expected_change > 3:

        if number_of_voyages >= 3:

            strategy = "SHORT-TERM"

            reason = (
                "Forecast indicates a potential "
                "increase in freight rates. A "
                "short-term multiple-voyage contract "
                "may reduce exposure to future "
                "rate increases."
            )

        else:

            strategy = (
                "SPOT / EARLY CHARTER REVIEW"
            )

            reason = (
                "Forecast indicates potential "
                "rate increases. Review charter "
                "timing before rates potentially "
                "increase."
            )

    elif expected_change < -3:

        strategy = "SPOT / WAIT"

        reason = (
            "Forecast indicates potential "
            "freight-rate decreases. Maintaining "
            "flexibility may be preferable to "
            "committing to a longer contract."
        )

    else:

        if number_of_voyages >= 6:

            strategy = "MEDIUM-TERM"

            reason = (
                "Freight conditions appear "
                "relatively stable and the planned "
                "voyage volume is high. A medium-term "
                "arrangement can provide greater "
                "planning stability."
            )

        elif number_of_voyages >= 3:

            strategy = "SHORT-TERM"

            reason = (
                "Freight conditions appear "
                "relatively stable and multiple "
                "voyages are planned. A short-term "
                "contract provides a balance between "
                "flexibility and planning stability."
            )

        else:

            strategy = "SPOT"

            reason = (
                "Only a small number of voyages "
                "are planned and freight conditions "
                "are relatively stable. Spot "
                "contracting preserves flexibility."
            )

    if market_risk_score >= 70:

        risk_note = (
            "Market risk is elevated. Maintain "
            "alternative charter options."
        )

    elif market_risk_score >= 40:

        risk_note = (
            "Market risk is moderate. Monitor "
            "market conditions before final "
            "commitment."
        )

    else:

        risk_note = (
            "Market risk indicators are "
            "relatively low."
        )

    return {
        "strategy": strategy,
        "reason": reason,
        "risk_note": risk_note,
        "expected_change": expected_change
    }


# ============================================================
# REAL BDI MARKET DATA
# ============================================================

def get_real_market_data():

    df = load_real_bdi()

    if df.empty:

        return None

    latest = df.iloc[-1]

    previous = (
        df.iloc[-2]
        if len(df) >= 2
        else latest
    )

    market = {

        "date": latest["date"],

        "bdi": safe_float(
            latest["bdi"]
        ),

        "capesize": safe_float(
            latest["bci"]
        ),

        "panamax": safe_float(
            latest["bpi"]
        ),

        "supramax": safe_float(
            latest["bsi"]
        ),

        "handysize": safe_float(
            latest["bhsi"]
        ),

        "capesize_change": (
            (
                latest["bci"]
                - previous["bci"]
            )
            / previous["bci"]
        ) * 100,

        "panamax_change": (
            (
                latest["bpi"]
                - previous["bpi"]
            )
            / previous["bpi"]
        ) * 100,

        "supramax_change": (
            (
                latest["bsi"]
                - previous["bsi"]
            )
            / previous["bsi"]
        ) * 100,

        "handysize_change": (
            (
                latest["bhsi"]
                - previous["bhsi"]
            )
            / previous["bhsi"]
        ) * 100
    }

    return market


# ============================================================
# REAL BDI FORECAST
# ============================================================

def generate_real_bdi_forecast():

    df = load_real_bdi()

    model = load_real_bdi_model()

    if df.empty or model is None:

        return None

    df = df.sort_values("date")

    latest = df.iloc[-1]

    latest_date = latest["date"]

    future_dates = pd.date_range(
        start=latest_date + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

    rows = []

    for date in future_dates:

        row = {

            "bci": latest["bci"],
            "bpi": latest["bpi"],
            "bsi": latest["bsi"],
            "bhsi": latest["bhsi"],

            "year": date.year,
            "month": date.month,
            "day": date.day,
            "day_of_week": date.dayofweek,

            "bdi_change_pct": 0,

            "panamax_ratio": (
                latest["bpi"]
                / latest["bdi"]
            ),

            "capesize_ratio": (
                latest["bci"]
                / latest["bdi"]
            ),

            "supramax_ratio": (
                latest["bsi"]
                / latest["bdi"]
            ),

            "handysize_ratio": (
                latest["bhsi"]
                / latest["bdi"]
            ),

            "bdi_3day_avg": (
                df["bdi"]
                .tail(3)
                .mean()
            )
        }

        rows.append(row)

    future_df = pd.DataFrame(
        rows
    )

    predictions = model.predict(
        future_df
    )

    result = pd.DataFrame({

        "date": future_dates,

        "bdi_forecast": predictions
    })

    return result


# ============================================================
# SYNTHETIC FREIGHT FORECAST
# ============================================================

def generate_freight_forecast(
    route,
    freight_df,
    model
):

    route_df = freight_df[
        freight_df["route"] == route
    ].copy()

    if route_df.empty:

        return None

    route_df = route_df.sort_values(
        "date"
    )

    latest = route_df.iloc[-1]

    current_rate = safe_float(
        latest["freight_rate_usd_mt"]
    )

    fuel_price = safe_float(
        latest["fuel_price"]
    )

    demand_index = safe_float(
        latest["demand_index"]
    )

    congestion_index = safe_float(
        latest["congestion_index"]
    )

    last_date = latest["date"]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

    routes = [
        "Newcastle_Australia_Visakhapatnam",
        "Richards_Bay_South_Africa_Paradip",
        "Indonesia_Chennai",
        "Australia_Kakinada"
    ]

    rows = []

    for date in future_dates:

        row = {

            "day": date.day,
            "month": date.month,
            "year": date.year,

            "day_of_year":
                date.dayofyear,

            "fuel_price":
                fuel_price,

            "demand_index":
                demand_index,

            "congestion_index":
                congestion_index
        }

        for route_name in routes:

            column = (
                "route_"
                + route_name
            )

            row[column] = (
                1
                if route_name == route
                else 0
            )

        rows.append(row)

    future_df = pd.DataFrame(
        rows
    )

    # Match model feature order
    if hasattr(
        model,
        "feature_names_in_"
    ):

        expected_features = list(
            model.feature_names_in_
        )

        for feature in expected_features:

            if feature not in future_df.columns:
                future_df[feature] = 0

        future_df = future_df[
            expected_features
        ]

    predictions = model.predict(
        future_df
    )

    result = pd.DataFrame({

        "date": future_dates,

        "forecast_rate":
            predictions
    })

    return {
        "forecast": result,
        "current_rate": current_rate,
        "fuel_price": fuel_price,
        "demand_index": demand_index,
        "congestion_index": congestion_index
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🚢 IntelliFreight"
)

st.sidebar.caption(
    "Intelligent Freight Forecasting "
    "and Vessel Chartering Platform"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Voyage Inputs"
)

cargo_quantity = st.sidebar.number_input(
    "Cargo Quantity (MT)",
    min_value=1000,
    max_value=200000,
    value=70000,
    step=1000
)

origin = st.sidebar.selectbox(
    "Origin",
    [
        "Newcastle, Australia",
        "Richards Bay, South Africa",
        "Indonesia",
        "Australia"
    ]
)

destination = st.sidebar.selectbox(
    "Destination",
    [
        "Visakhapatnam",
        "Paradip",
        "Chennai",
        "Kakinada"
    ]
)

contract_duration = st.sidebar.selectbox(
    "Contract Duration",
    [
        "Spot",
        "Short-term",
        "Medium-term"
    ]
)

default_voyages = {

    "Spot": 1,

    "Short-term": 3,

    "Medium-term": 6
}

number_of_voyages = st.sidebar.number_input(
    "Number of Voyages",
    min_value=1,
    max_value=20,
    value=default_voyages[
        contract_duration
    ],
    step=1
)

analyze = st.sidebar.button(
    "🔍 Analyze Voyage",
    use_container_width=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚢 IntelliFreight'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered freight forecasting, '
    'vessel optimization and charter '
    'decision support'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

freight_df = load_freight_data()

forecast_model = load_forecasting_model()

real_market = get_real_market_data()


# ============================================================
# REAL MARKET SNAPSHOT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🌍 Real Dry-Bulk Market Snapshot'
    '</div>',
    unsafe_allow_html=True
)

if real_market is not None:

    st.caption(
        f"Latest available real BDI observation: "
        f"{real_market['date'].date()}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "BDI",
            f"{real_market['bdi']:.0f}"
        )

    with c2:

        st.metric(
            "Capesize",
            f"{real_market['capesize']:.0f}",
            f"{real_market['capesize_change']:+.2f}%"
        )

    with c3:

        st.metric(
            "Panamax",
            f"{real_market['panamax']:.0f}",
            f"{real_market['panamax_change']:+.2f}%"
        )

    with c4:

        st.metric(
            "Supramax",
            f"{real_market['supramax']:.0f}",
            f"{real_market['supramax_change']:+.2f}%"
        )

    with c5:

        st.metric(
            "Handysize",
            f"{real_market['handysize']:.0f}",
            f"{real_market['handysize_change']:+.2f}%"
        )

    st.info(
        "These are real dry-bulk market indicators. "
        "They are market proxies and are not themselves "
        "route-specific freight rates for East Coast India."
    )

else:

    st.warning(
        "Real BDI data is currently unavailable."
    )


# ============================================================
# BDI FORECAST
# ============================================================

real_bdi_forecast = (
    generate_real_bdi_forecast()
)

if real_bdi_forecast is not None:

    st.markdown(
        '<div class="section-title">'
        '📈 Real BDI 7-Day Forecast'
        '</div>',
        unsafe_allow_html=True
    )

    fig_bdi = px.line(
        real_bdi_forecast,
        x="date",
        y="bdi_forecast",
        markers=True,
        title="Forecasted Baltic Dry Index"
    )

    fig_bdi.update_layout(
        xaxis_title="Date",
        yaxis_title="BDI"
    )

    st.plotly_chart(
        fig_bdi,
        use_container_width=True
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    route_key = (
        f"{origin}|{destination}"
    )

    route = ROUTE_MAP[
        route_key
    ]

    if freight_df.empty:

        st.error(
            "Synthetic freight dataset could "
            "not be found."
        )

        st.stop()

    if forecast_model is None:

        st.error(
            "Freight forecasting model could "
            "not be found."
        )

        st.stop()

    forecast_result = (
        generate_freight_forecast(
            route,
            freight_df,
            forecast_model
        )
    )

    if forecast_result is None:

        st.error(
            "No data found for the selected route."
        )

        st.stop()

    forecast_df = (
        forecast_result["forecast"]
    )

    current_rate = (
        forecast_result["current_rate"]
    )

    fuel_price = (
        forecast_result["fuel_price"]
    )

    demand_index = (
        forecast_result["demand_index"]
    )

    congestion_index = (
        forecast_result["congestion_index"]
    )

    average_forecast = (
        forecast_df["forecast_rate"]
        .mean()
    )

    minimum_forecast = (
        forecast_df["forecast_rate"]
        .min()
    )

    maximum_forecast = (
        forecast_df["forecast_rate"]
        .max()
    )

    # ========================================================
    # CHARTER TIMING
    # ========================================================

    charter = calculate_charter_timing(
        current_rate,
        average_forecast
    )

    # ========================================================
    # VESSEL
    # ========================================================

    recommended_vessel = recommend_vessel(
        cargo_quantity,
        destination
    )

    # ========================================================
    # IDLE RISK
    # ========================================================

    idle = calculate_idle_risk(
        demand_index,
        congestion_index
    )

    # ========================================================
    # MARKET RISK
    # ========================================================

    market_risk = calculate_market_risk(
        freight_df[
            freight_df["route"] == route
        ]["freight_rate_usd_mt"],
        fuel_price,
        demand_index,
        congestion_index
    )

    # ========================================================
    # CONTRACT STRATEGY
    # ========================================================

    strategy = recommend_contract_strategy(
        current_rate,
        average_forecast,
        number_of_voyages,
        market_risk["score"]
    )

    # ========================================================
    # COST CALCULATIONS
    # ========================================================

    spot_rate = current_rate

    short_term_rate = (
        current_rate * 0.99
        + average_forecast * 0.01
    )

    medium_term_rate = (
        current_rate * 0.97
        + average_forecast * 0.03
    )

    spot_total = (
        spot_rate
        * cargo_quantity
        * number_of_voyages
    )

    short_total = (
        short_term_rate
        * cargo_quantity
        * number_of_voyages
    )

    medium_total = (
        medium_term_rate
        * cargo_quantity
        * number_of_voyages
    )

    # ========================================================
    # VOYAGE HEADER
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📋 Voyage Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**Route:** {origin} → {destination}"
    )

    summary1, summary2, summary3, summary4, summary5 = (
        st.columns(5)
    )

    with summary1:

        st.metric(
            "Cargo",
            f"{cargo_quantity:,} MT"
        )

    with summary2:

        st.metric(
            "Current Rate",
            f"${current_rate:.2f}/MT"
        )

    with summary3:

        st.metric(
            "7-Day Forecast",
            f"${average_forecast:.2f}/MT"
        )

    with summary4:

        st.metric(
            "Recommended Vessel",
            recommended_vessel
        )

    with summary5:

        st.metric(
            "Voyages",
            number_of_voyages
        )

    # ========================================================
    # FREIGHT FORECAST
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Route Freight Forecast'
        '</div>',
        unsafe_allow_html=True
    )

    fig_forecast = px.line(
        forecast_df,
        x="date",
        y="forecast_rate",
        markers=True,
        title=(
            f"7-Day Freight Forecast — "
            f"{origin} → {destination}"
        )
    )

    fig_forecast.update_layout(
        xaxis_title="Date",
        yaxis_title="Freight Rate (USD/MT)"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    # ========================================================
    # CHARTER TIMING
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⏱️ Optimal Market Entry Timing'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Expected Change",
            f"{charter['expected_change']:+.2f}%"
        )

    with c2:

        st.metric(
            "Forecast Range",
            f"${minimum_forecast:.2f} – "
            f"${maximum_forecast:.2f}"
        )

    with c3:

        st.metric(
            "Decision",
            charter["decision"]
        )

    st.info(
        charter["reason"]
    )

    # ========================================================
    # VESSEL OPTIMIZATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🚢 Vessel Type Optimization'
        '</div>',
        unsafe_allow_html=True
    )

    vessel = VESSELS[
        recommended_vessel
    ]

    v1, v2, v3, v4 = st.columns(4)

    with v1:

        st.metric(
            "Recommended Vessel",
            recommended_vessel
        )

    with v2:

        st.metric(
            "Draft",
            f"{vessel['draft']:.1f} m"
        )

    with v3:

        st.metric(
            "LOA",
            f"{vessel['loa']} m"
        )

    with v4:

        st.metric(
            "Handling",
            f"{vessel['handling']:,} MT"
        )

    st.info(
        "Vessel recommendation is based on "
        "cargo quantity and prototype port "
        "draft, LOA and handling constraints."
    )

    # ========================================================
    # REAL VESSEL MARKET
    # ========================================================

    if real_market is not None:

        st.subheader(
            "Real Vessel Market Indicators"
        )

        vessel_chart_df = pd.DataFrame({

            "Vessel": [
                "Capesize",
                "Panamax",
                "Supramax",
                "Handysize"
            ],

            "Index": [
                real_market["capesize"],
                real_market["panamax"],
                real_market["supramax"],
                real_market["handysize"]
            ]
        })

        fig_vessel = px.bar(
            vessel_chart_df,
            x="Vessel",
            y="Index",
            title="Latest Real Vessel Indices"
        )

        st.plotly_chart(
            fig_vessel,
            use_container_width=True
        )

        st.caption(
            "Market indices support market-context analysis; "
            "they do not independently determine the "
            "recommended vessel."
        )

    # ========================================================
    # IDLE SCENARIO
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⚓ Idle Scenario Management'
        '</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "Idle Risk Score",
            f"{idle['score']}/100"
        )

    with i2:

        st.metric(
            "Demand Index",
            f"{demand_index:.2f}"
        )

    with i3:

        st.metric(
            "Congestion Index",
            f"{congestion_index:.2f}"
        )

    st.info(
        f"**{idle['status']}** — "
        f"{idle['action']}"
    )

    # ========================================================
    # MARKET RISK
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⚠️ Market Risk'
        '</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Risk Score",
            f"{market_risk['score']}/100"
        )

    with r2:

        st.metric(
            "Risk Level",
            market_risk["level"]
        )

    with r3:

        st.metric(
            "Freight Volatility",
            f"{market_risk['freight_volatility']:.2f}"
        )

    st.warning(
        market_risk["recommendation"]
    )

    # ========================================================
    # MARKET INDICATORS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Market Indicators'
        '</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Fuel Price",
            f"${fuel_price:.2f}"
        )

    with m2:

        st.metric(
            "Demand",
            f"{demand_index:.2f}"
        )

    with m3:

        st.metric(
            "Congestion",
            f"{congestion_index:.2f}"
        )

    # ========================================================
    # CONTRACT STRATEGY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📑 Contract Strategy Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    contract_df = pd.DataFrame({

        "Strategy": [
            "Spot",
            "Short-term",
            "Medium-term"
        ],

        "Rate (USD/MT)": [
            spot_rate,
            short_term_rate,
            medium_term_rate
        ],

        "Total Cost (USD)": [
            spot_total,
            short_total,
            medium_total
        ]
    })

    st.dataframe(
        contract_df.style.format({
            "Rate (USD/MT)": "${:.2f}",
            "Total Cost (USD)": "${:,.2f}"
        }),
        use_container_width=True
    )

    fig_contract = px.bar(
        contract_df,
        x="Strategy",
        y="Total Cost (USD)",
        title="Estimated Contract Cost"
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )

    # ========================================================
    # MULTIPLE VOYAGE PLANNING
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔁 Multiple-Voyage Planning'
        '</div>',
        unsafe_allow_html=True
    )

    voyage_df = pd.DataFrame({

        "Contract": [
            "Spot",
            "Short-term",
            "Medium-term"
        ],

        "Voyages": [
            number_of_voyages,
            number_of_voyages,
            number_of_voyages
        ],

        "Estimated Rate (USD/MT)": [
            spot_rate,
            short_term_rate,
            medium_term_rate
        ],

        "Total Estimated Cost (USD)": [
            spot_total,
            short_total,
            medium_total
        ],

        "Difference vs Spot (USD)": [
            0,
            spot_total - short_total,
            spot_total - medium_total
        ]
    })

    st.dataframe(
        voyage_df.style.format({
            "Estimated Rate (USD/MT)": "${:.2f}",
            "Total Estimated Cost (USD)": "${:,.2f}",
            "Difference vs Spot (USD)": "${:,.2f}"
        }),
        use_container_width=True
    )

    fig_voyages = px.bar(
        voyage_df,
        x="Contract",
        y="Total Estimated Cost (USD)",
        title=(
            f"Cost Comparison for "
            f"{number_of_voyages} Voyages"
        )
    )

    st.plotly_chart(
        fig_voyages,
        use_container_width=True
    )

    # ========================================================
    # RECOMMENDED CONTRACT STRATEGY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🧠 Intelligent Contract Strategy'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="recommendation-box">',
        unsafe_allow_html=True
    )

    st.subheader(
        strategy["strategy"]
    )

    st.write(
        strategy["reason"]
    )

    st.write(
        strategy["risk_note"]
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # POTENTIAL COST DIFFERENCE
    # ========================================================

    selected_strategy = strategy["strategy"]

    if "MEDIUM" in selected_strategy:

        selected_cost = medium_total

    elif "SHORT" in selected_strategy:

        selected_cost = short_total

    else:

        selected_cost = spot_total

    potential_difference = (
        spot_total
        - selected_cost
    )

    st.metric(
        "Potential Cost Difference vs Spot",
        f"${potential_difference:,.2f}"
    )

    st.caption(
        "Contract-rate calculations shown here "
        "are prototype assumptions for demonstration. "
        "They are not actual negotiated market contract prices."
    )

    # ========================================================
    # MASTER RECOMMENDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 IntelliFreight Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    master_recommendation = (
        f"For the {cargo_quantity:,} MT cargo from "
        f"{origin} to {destination}, the system "
        f"recommends a {recommended_vessel} vessel. "
        f"The current freight rate is "
        f"${current_rate:.2f}/MT, while the 7-day "
        f"forecast average is "
        f"${average_forecast:.2f}/MT. "
        f"The suggested contract strategy is "
        f"{strategy['strategy']}. "
        f"Current market risk is "
        f"{market_risk['level'].lower()}."
    )

    st.success(
        master_recommendation
    )

    # ========================================================
    # SIH OBJECTIVE ALIGNMENT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 SIH Problem Statement Alignment'
        '</div>',
        unsafe_allow_html=True
    )

    alignment = pd.DataFrame({

        "SIH Objective": [
            "Optimal Market Entry Timing",
            "Vessel Type Optimization",
            "Idle Scenario Management",
            "Risk Mitigation",
            "Multiple-Voyage Contract Planning"
        ],

        "IntelliFreight Module": [
            "7-day freight forecasting + charter timing",
            "Cargo + port constraint vessel optimizer",
            "Demand/congestion based idle-risk engine",
            "Freight, fuel, demand and congestion risk",
            "Spot vs short-term vs medium-term analysis"
        ]
    })

    st.dataframe(
        alignment,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.markdown(
        """
        ### Welcome to IntelliFreight 🚢

        Use the controls in the sidebar to enter a
        proposed cargo movement and click
        **Analyze Voyage**.

        The system will evaluate:

        - 📈 Freight-rate forecast
        - ⏱️ Charter timing
        - 🚢 Vessel type
        - ⚓ Idle risk
        - ⚠️ Market risk
        - 📑 Contract strategy
        - 🔁 Multiple-voyage economics
        - 🌍 Real dry-bulk market indicators
        """
    )

    st.info(
        "The current prototype combines synthetic route-level "
        "freight data with real publicly available BDI/vessel "
        "market indicators. The next development stage is to "
        "replace the prototype route/fuel/congestion inputs "
        "with longer historical and route-specific datasets."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IntelliFreight — SIH MVP | "
    "Prototype decision-support system"
)

st.caption(
    "Important: prototype assumptions and synthetic "
    "route-level data are clearly separated from the "
    "real BDI/vessel-market data. Results should not "
    "be treated as actual commercial charter quotes."
)