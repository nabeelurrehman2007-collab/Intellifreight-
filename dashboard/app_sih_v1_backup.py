# ============================================================
# INTELLIFREIGHT
# Intelligent Freight Forecasting & Vessel Chartering System
# SIH MVP
# ============================================================

import os
import sys
import warnings

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
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
REAL_DATA_DIR = os.path.join(DATA_DIR, "real")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# ============================================================
# FILE PATHS
# ============================================================

FREIGHT_DATA_PATH = os.path.join(
    DATA_DIR, "freight_data.csv"
)

BDI_DATA_PATH = os.path.join(
    REAL_DATA_DIR, "bdi_history.csv"
)

PROCESSED_BDI_PATH = os.path.join(
    REAL_DATA_DIR, "processed_bdi_data.csv"
)

FREIGHT_MODEL_PATH = os.path.join(
    MODEL_DIR, "freight_forecasting_model.pkl"
)

BDI_MODEL_PATH = os.path.join(
    MODEL_DIR, "real_bdi_model.pkl"
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
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .small-note {
        font-size: 13px;
        color: #777;
    }

    .recommendation-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚢 IntelliFreight</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Intelligent Freight Forecasting & Vessel Chartering
    Decision Support System
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_freight_data():

    if not os.path.exists(FREIGHT_DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(FREIGHT_DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_bdi_data():

    if not os.path.exists(BDI_DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(BDI_DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_processed_bdi():

    if not os.path.exists(PROCESSED_BDI_PATH):
        return pd.DataFrame()

    df = pd.read_csv(PROCESSED_BDI_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_resource
def load_freight_model():

    if not os.path.exists(FREIGHT_MODEL_PATH):
        return None

    return joblib.load(FREIGHT_MODEL_PATH)


@st.cache_resource
def load_bdi_model():

    if not os.path.exists(BDI_MODEL_PATH):
        return None

    return joblib.load(BDI_MODEL_PATH)


freight_df = load_freight_data()
bdi_df = load_bdi_data()
processed_bdi_df = load_processed_bdi()

freight_model = load_freight_model()
bdi_model = load_bdi_model()


# ============================================================
# REAL VESSEL MARKET FUNCTION
# ============================================================

def get_latest_vessel_market():

    if bdi_df.empty:
        return {}

    latest = bdi_df.iloc[-1]

    result = {
        "date": latest.get("date"),
        "Capesize": {
            "index": float(latest.get("bci", 0)),
        },
        "Panamax": {
            "index": float(latest.get("bpi", 0)),
        },
        "Supramax": {
            "index": float(latest.get("bsi", 0)),
        },
        "Handysize": {
            "index": float(latest.get("bhsi", 0)),
        },
    }

    if len(bdi_df) >= 2:

        previous = bdi_df.iloc[-2]

        for vessel, column in [
            ("Capesize", "bci"),
            ("Panamax", "bpi"),
            ("Supramax", "bsi"),
            ("Handysize", "bhsi"),
        ]:

            current = float(latest.get(column, 0))
            prev = float(previous.get(column, 0))

            if prev != 0:
                change = ((current - prev) / prev) * 100
            else:
                change = 0

            result[vessel]["change"] = change

    return result


vessel_market = get_latest_vessel_market()


# ============================================================
# ROUTE MAP
# ============================================================

ROUTE_MAP = {
    "Newcastle, Australia → Visakhapatnam, India":
        "Newcastle_Australia_Visakhapatnam",

    "Richards Bay, South Africa → Paradip, India":
        "Richards_Bay_South_Africa_Paradip",

    "Indonesia → Chennai, India":
        "Indonesia_Chennai",

    "Australia → Kakinada, India":
        "Australia_Kakinada",
}


# ============================================================
# PROTOTYPE VESSEL DATABASE
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
# PROTOTYPE PORT DATABASE
# ============================================================

PORTS = {

    "Visakhapatnam, India": {
        "draft": 16.0,
        "loa": 300,
        "handling": 40000
    },

    "Paradip, India": {
        "draft": 16.5,
        "loa": 300,
        "handling": 45000
    },

    "Chennai, India": {
        "draft": 13.0,
        "loa": 250,
        "handling": 30000
    },

    "Kakinada, India": {
        "draft": 12.0,
        "loa": 220,
        "handling": 25000
    }
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📦 Voyage Inputs")

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
        "Visakhapatnam, India",
        "Paradip, India",
        "Chennai, India",
        "Kakinada, India"
    ]
)

contract_duration = st.sidebar.selectbox(
    "Contract Duration",
    [
        "Spot",
        "1 month",
        "3 months",
        "6 months",
        "12 months"
    ]
)

number_of_voyages = st.sidebar.number_input(
    "Number of Voyages",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)


# ============================================================
# SELECT ROUTE
# ============================================================

route_key = f"{origin} → {destination}"

if route_key in ROUTE_MAP:
    selected_route = ROUTE_MAP[route_key]
else:
    selected_route = None


# ============================================================
# DATA & MODEL TRANSPARENCY
# ============================================================

st.markdown(
    '<div class="section-title">📊 Data & Model Transparency</div>',
    unsafe_allow_html=True
)

st.info(
    "IntelliFreight combines real dry-bulk market indicators "
    "with prototype route-level data. Real and prototype inputs "
    "are clearly separated."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### 🌍 Real Market Data")

    st.success("BDI / BCI / BPI / BSI / BHSI")

    st.caption(
        "Historical dry-bulk market indices used for "
        "market-condition analysis and BDI forecasting."
    )

with col2:

    st.markdown("### 🧪 Prototype Data")

    st.warning("Route-level freight dataset")

    st.caption(
        "Synthetic/prototype data currently used for "
        "route-level forecasting and demonstration."
    )

with col3:

    st.markdown("### 🤖 ML Model")

    st.success("Random Forest")

    st.caption(
        "Baseline machine-learning model used for "
        "market and freight forecasting."
    )


st.markdown("### 📈 BDI Model Evaluation")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "MAE",
        "29.43",
        help="Mean Absolute Error in BDI points"
    )

with metric2:
    st.metric(
        "RMSE",
        "38.80",
        help="Root Mean Squared Error in BDI points"
    )

with metric3:
    st.metric(
        "R²",
        "0.846",
        help="Coefficient of determination"
    )

st.caption(
    "Baseline evaluation on a small historical test set. "
    "These metrics should not be interpreted as a guarantee "
    "of future forecasting accuracy."
)


# ============================================================
# REAL DRY-BULK MARKET SNAPSHOT
# ============================================================

st.markdown(
    '<div class="section-title">🌍 Real Dry-Bulk Market Snapshot</div>',
    unsafe_allow_html=True
)

if vessel_market:

    latest_date = vessel_market.get("date")

    if pd.notna(latest_date):
        st.caption(
            f"Latest available market observation: "
            f"{pd.to_datetime(latest_date).strftime('%d %b %Y')}"
        )

    c1, c2, c3, c4 = st.columns(4)

    market_items = [
        ("Capesize", c1),
        ("Panamax", c2),
        ("Supramax", c3),
        ("Handysize", c4)
    ]

    for vessel, column in market_items:

        data = vessel_market[vessel]

        with column:

            st.metric(
                vessel,
                f"{data['index']:,.0f}",
                f"{data.get('change', 0):+.2f}%"
            )

else:

    st.warning("Real vessel-market data unavailable.")


# ============================================================
# BDI FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">🔮 BDI Market Forecast</div>',
    unsafe_allow_html=True
)

if not processed_bdi_df.empty:

    forecast_df = processed_bdi_df.copy()

    forecast_df = forecast_df.sort_values("date")

    latest_bdi = float(
        forecast_df.iloc[-1]["bdi"]
    )

    latest_date = forecast_df.iloc[-1]["date"]

    bdi_forecasts = []

    if bdi_model is not None:

        feature_columns = [
            "bci",
            "bpi",
            "bsi",
            "bhsi",
            "year",
            "month",
            "day",
            "day_of_week",
            "bdi_change_pct",
            "panamax_ratio",
            "capesize_ratio",
            "supramax_ratio",
            "handysize_ratio",
            "bdi_3day_avg"
        ]

        temp = forecast_df.copy()

        temp = temp.dropna(
            subset=feature_columns
        )

        if len(temp) > 0:

            latest_row = temp.iloc[-1]

            for i in range(1, 8):

                future_date = (
                    latest_date +
                    pd.Timedelta(days=i)
                )

                row = latest_row.copy()

                row["year"] = future_date.year
                row["month"] = future_date.month
                row["day"] = future_date.day
                row["day_of_week"] = future_date.dayofweek

                X_future = pd.DataFrame(
                    [[row[col] for col in feature_columns]],
                    columns=feature_columns
                )

                try:

                    prediction = float(
                        bdi_model.predict(X_future)[0]
                    )

                except Exception:

                    prediction = latest_bdi

                bdi_forecasts.append(
                    {
                        "date": future_date,
                        "bdi": prediction
                    }
                )

    if bdi_forecasts:

        future_df = pd.DataFrame(
            bdi_forecasts
        )

        avg_forecast = future_df["bdi"].mean()

        expected_change = (
            (avg_forecast - latest_bdi)
            / latest_bdi
        ) * 100

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Current BDI",
                f"{latest_bdi:,.0f}"
            )

        with c2:
            st.metric(
                "7-Day Forecast Avg.",
                f"{avg_forecast:,.0f}",
                f"{expected_change:+.2f}%"
            )

        with c3:

            if expected_change > 2:
                trend = "Increasing"
            elif expected_change < -2:
                trend = "Decreasing"
            else:
                trend = "Stable"

            st.metric(
                "Forecast Trend",
                trend
            )

        chart_df = pd.concat(
            [
                forecast_df[["date", "bdi"]].tail(20),
                future_df
            ],
            ignore_index=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df["bdi"],
                mode="lines+markers",
                name="BDI"
            )
        )

        fig.add_vline(
            x=latest_date,
            line_dash="dash"
        )

        fig.update_layout(
            title="Historical BDI + 7-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Baltic Dry Index",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Unable to generate BDI forecast."
        )

else:

    st.warning(
        "Processed BDI data unavailable."
    )


# ============================================================
# ROUTE FREIGHT FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">🚢 Route Freight Forecast</div>',
    unsafe_allow_html=True
)

if selected_route and not freight_df.empty:

    route_data = freight_df[
        freight_df["route"] == selected_route
    ].copy()

    if not route_data.empty:

        route_data = route_data.sort_values(
            "date"
        )

        latest_route = route_data.iloc[-1]

        current_rate = float(
            latest_route["freight_rate_usd_mt"]
        )

        latest_fuel = float(
            latest_route["fuel_price"]
        )

        latest_demand = float(
            latest_route["demand_index"]
        )

        latest_congestion = float(
            latest_route["congestion_index"]
        )

        predictions = []

        if freight_model is not None:

            for i in range(1, 8):

                future_date = (
                    latest_route["date"]
                    + pd.Timedelta(days=i)
                )

                X = pd.DataFrame(
                    [{
                        "day": future_date.day,
                        "month": future_date.month,
                        "year": future_date.year,
                        "day_of_year": future_date.dayofyear,
                        "route": selected_route,
                        "fuel_price": latest_fuel,
                        "demand_index": latest_demand,
                        "congestion_index": latest_congestion
                    }]
                )

                try:

                    prediction = float(
                        freight_model.predict(X)[0]
                    )

                except Exception:

                    prediction = current_rate

                predictions.append(
                    {
                        "date": future_date,
                        "freight_rate": prediction
                    }
                )

        if predictions:

            prediction_df = pd.DataFrame(
                predictions
            )

            forecast_rate = prediction_df[
                "freight_rate"
            ].mean()

            expected_change = (
                (forecast_rate - current_rate)
                / current_rate
            ) * 100

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Current Freight",
                    f"${current_rate:.2f}/MT"
                )

            with c2:
                st.metric(
                    "7-Day Forecast",
                    f"${forecast_rate:.2f}/MT",
                    f"{expected_change:+.2f}%"
                )

            with c3:

                if expected_change > 2:
                    trend = "Increasing"
                elif expected_change < -2:
                    trend = "Decreasing"
                else:
                    trend = "Stable"

                st.metric(
                    "Route Trend",
                    trend
                )

            history_plot = route_data[
                ["date", "freight_rate_usd_mt"]
            ].tail(30).copy()

            history_plot = history_plot.rename(
                columns={
                    "freight_rate_usd_mt":
                    "freight_rate"
                }
            )

            plot_df = pd.concat(
                [
                    history_plot,
                    prediction_df
                ],
                ignore_index=True
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=plot_df["date"],
                    y=plot_df["freight_rate"],
                    mode="lines+markers",
                    name="Freight Rate"
                )
            )

            fig.add_vline(
                x=latest_route["date"],
                line_dash="dash"
            )

            fig.update_layout(
                title=f"Freight Forecast — {route_key}",
                xaxis_title="Date",
                yaxis_title="USD / MT",
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            forecast_rate = current_rate

    else:

        st.warning(
            "No route data available for this route."
        )

        current_rate = 0
        forecast_rate = 0

else:

    st.warning(
        "This route combination is currently outside "
        "the prototype route dataset."
    )

    current_rate = 0
    forecast_rate = 0


# ============================================================
# CHARTER TIMING
# ============================================================

st.markdown(
    '<div class="section-title">⏱️ Optimal Market Entry Timing</div>',
    unsafe_allow_html=True
)

if current_rate > 0:

    expected_change = (
        (forecast_rate - current_rate)
        / current_rate
    ) * 100

    if expected_change > 3:

        charter_decision = (
            "CONSIDER EARLIER CHARTERING"
        )

        charter_reason = (
            "The prototype forecast indicates a potential "
            "increase in freight rates."
        )

    elif expected_change < -3:

        charter_decision = (
            "MONITOR / CONSIDER WAITING"
        )

        charter_reason = (
            "The prototype forecast indicates a potential "
            "decrease in freight rates."
        )

    else:

        charter_decision = "MONITOR MARKET"

        charter_reason = (
            "Forecasted freight-rate movement is relatively "
            "small."
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Current Rate",
            f"${current_rate:.2f}/MT"
        )

    with c2:
        st.metric(
            "Forecast Rate",
            f"${forecast_rate:.2f}/MT"
        )

    with c3:
        st.metric(
            "Expected Change",
            f"{expected_change:+.2f}%"
        )

    st.info(
        f"**{charter_decision}** — {charter_reason}"
    )

    current_cost = (
        current_rate *
        cargo_quantity
    )

    forecast_cost = (
        forecast_rate *
        cargo_quantity
    )

    potential_difference = (
        forecast_cost - current_cost
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Current Voyage Cost",
            f"${current_cost:,.2f}"
        )

    with c2:

        st.metric(
            "Forecast Voyage Cost",
            f"${forecast_cost:,.2f}",
            f"${potential_difference:,.2f}"
        )


# ============================================================
# VESSEL OPTIMIZATION
# ============================================================

st.markdown(
    '<div class="section-title">⚓ Vessel Type Optimization</div>',
    unsafe_allow_html=True
)

selected_port = PORTS.get(destination)

recommended_vessels = []

if selected_port:

    for vessel_name, specs in VESSELS.items():

        cargo_ok = (
            specs["min_cargo"]
            <= cargo_quantity
            <= specs["max_cargo"]
        )

        draft_ok = (
            specs["draft"]
            <= selected_port["draft"]
        )

        loa_ok = (
            specs["loa"]
            <= selected_port["loa"]
        )

        handling_ok = (
            specs["handling"]
            <= selected_port["handling"]
        )

        if (
            cargo_ok
            and draft_ok
            and loa_ok
            and handling_ok
        ):

            recommended_vessels.append(
                vessel_name
            )

if recommended_vessels:

    primary_vessel = recommended_vessels[-1]

    st.success(
        f"Recommended vessel type: **{primary_vessel}**"
    )

    vessel_specs = VESSELS[
        primary_vessel
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Cargo Range",
            f"{vessel_specs['min_cargo']:,}"
            f"–{vessel_specs['max_cargo']:,} MT"
        )

    with c2:
        st.metric(
            "Draft",
            f"{vessel_specs['draft']} m"
        )

    with c3:
        st.metric(
            "LOA",
            f"{vessel_specs['loa']} m"
        )

    with c4:
        st.metric(
            "Handling",
            f"{vessel_specs['handling']:,} MT"
        )

    st.caption(
        "Vessel and port constraints shown here are "
        "prototype assumptions for the SIH MVP and "
        "should be replaced/validated with authoritative "
        "port and vessel specifications."
    )

else:

    primary_vessel = "No suitable vessel"

    st.warning(
        "No vessel satisfies all prototype cargo and "
        "port constraints."
    )


# ============================================================
# IDLE SCENARIO MANAGEMENT
# ============================================================

st.markdown(
    '<div class="section-title">💤 Idle Scenario Management</div>',
    unsafe_allow_html=True
)

if selected_route and not freight_df.empty:

    route_data = freight_df[
        freight_df["route"] == selected_route
    ].copy()

    if not route_data.empty:

        latest = route_data.iloc[-1]

        demand = float(
            latest["demand_index"]
        )

        congestion = float(
            latest["congestion_index"]
        )

        idle_risk = 20

        if demand < 30:
            idle_risk += 40
        elif demand < 60:
            idle_risk += 20

        if congestion > 70:
            idle_risk += 30
        elif congestion > 40:
            idle_risk += 15

        idle_risk = min(
            idle_risk,
            100
        )

        if idle_risk >= 70:

            idle_level = "HIGH"

            idle_action = (
                "Consider alternative employment, "
                "voyage repositioning and demand monitoring."
            )

        elif idle_risk >= 40:

            idle_level = "MODERATE"

            idle_action = (
                "Monitor demand and maintain alternative "
                "voyage options."
            )

        else:

            idle_level = "LOW"

            idle_action = (
                "Current prototype indicators show "
                "relatively low idle risk."
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Demand Index",
                f"{demand:.2f}"
            )

        with c2:
            st.metric(
                "Congestion Index",
                f"{congestion:.2f}"
            )

        with c3:
            st.metric(
                "Idle Risk",
                f"{idle_risk}/100",
                idle_level
            )

        st.info(
            f"**Action:** {idle_action}"
        )


# ============================================================
# MARKET RISK ENGINE
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Market Risk Engine</div>',
    unsafe_allow_html=True
)

if not freight_df.empty:

    latest_market = freight_df.iloc[-1]

    freight_volatility = float(
        freight_df["freight_rate_usd_mt"]
        .tail(30)
        .std()
    )

    fuel_volatility = float(
        freight_df["fuel_price"]
        .tail(30)
        .std()
    )

    demand_value = float(
        latest_market["demand_index"]
    )

    congestion_value = float(
        latest_market["congestion_index"]
    )

    risk_score = 0

    if freight_volatility > 4:
        risk_score += 30
    elif freight_volatility > 2:
        risk_score += 20
    else:
        risk_score += 10

    if fuel_volatility > 5:
        risk_score += 25
    elif fuel_volatility > 2:
        risk_score += 15
    else:
        risk_score += 5

    if demand_value < 30:
        risk_score += 20
    elif demand_value < 60:
        risk_score += 10
    else:
        risk_score += 5

    if congestion_value > 70:
        risk_score += 25
    elif congestion_value > 40:
        risk_score += 15
    else:
        risk_score += 5

    risk_score = min(
        risk_score,
        100
    )

    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    with c2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with c3:
        st.metric(
            "Freight Volatility",
            f"{freight_volatility:.2f}"
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Fuel Volatility",
            f"{fuel_volatility:.2f}"
        )

    with c2:
        st.metric(
            "Demand Risk Input",
            f"{demand_value:.2f}"
        )

    with c3:
        st.metric(
            "Congestion Input",
            f"{congestion_value:.2f}"
        )

    if risk_score >= 70:

        st.error(
            "Market risk is elevated. Maintain alternative "
            "charter and routing options."
        )

    elif risk_score >= 40:

        st.warning(
            "Market risk is moderate. Monitor market "
            "conditions before final commitment."
        )

    else:

        st.success(
            "Market risk indicators are relatively low."
        )


# ============================================================
# MARKET INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Market Indicators</div>',
    unsafe_allow_html=True
)

if not freight_df.empty:

    latest = freight_df.iloc[-1]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Fuel Price",
            f"${latest['fuel_price']:.2f}"
        )

    with c2:
        st.metric(
            "Demand Index",
            f"{latest['demand_index']:.2f}"
        )

    with c3:
        st.metric(
            "Congestion Index",
            f"{latest['congestion_index']:.2f}"
        )


# ============================================================
# CONTRACT STRATEGY
# ============================================================

st.markdown(
    '<div class="section-title">📋 Contract Strategy</div>',
    unsafe_allow_html=True
)

if current_rate > 0:

    expected_change = (
        (forecast_rate - current_rate)
        / current_rate
    ) * 100

    # Use the previously calculated risk score
    try:
        contract_risk = risk_score
    except NameError:
        contract_risk = 50

    if expected_change > 3:

        if number_of_voyages >= 3:

            strategy = "SHORT-TERM"

            reason = (
                "Forecast indicates a potential increase "
                "in freight rates. A short-term multiple-voyage "
                "contract may reduce exposure to future "
                "rate increases."
            )

        else:

            strategy = "SPOT / EARLY CHARTER REVIEW"

            reason = (
                "Forecast indicates potential rate increases. "
                "Review charter timing before rates potentially "
                "increase."
            )

    elif expected_change < -3:

        strategy = "SPOT / WAIT"

        reason = (
            "Forecast indicates potential freight-rate "
            "decreases. Maintaining flexibility may be "
            "preferable to committing to a longer contract."
        )

    else:

        if number_of_voyages >= 6:

            strategy = "MEDIUM-TERM"

            reason = (
                "Freight conditions appear relatively stable "
                "and planned voyage volume is high. A "
                "medium-term arrangement can provide greater "
                "planning stability."
            )

        elif number_of_voyages >= 3:

            strategy = "SHORT-TERM"

            reason = (
                "Freight conditions appear relatively stable "
                "and multiple voyages are planned. A short-term "
                "contract balances flexibility and stability."
            )

        else:

            strategy = "SPOT"

            reason = (
                "Only a small number of voyages are planned "
                "and freight conditions are relatively stable. "
                "Spot contracting preserves flexibility."
            )

    if contract_risk >= 70:

        risk_note = (
            "Market risk is elevated. Maintain alternative "
            "charter options."
        )

    elif contract_risk >= 40:

        risk_note = (
            "Market risk is moderate. Monitor market conditions "
            "before final commitment."
        )

    else:

        risk_note = (
            "Market risk indicators are relatively low."
        )

    st.success(
        f"Recommended prototype strategy: **{strategy}**"
    )

    st.write(reason)

    st.warning(
        f"Risk note: {risk_note}"
    )


# ============================================================
# CONTRACT STRATEGY COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">💰 Contract Cost Comparison</div>',
    unsafe_allow_html=True
)

if current_rate > 0:

    spot_rate = current_rate

    short_term_rate = (
        current_rate * 0.99
        + forecast_rate * 0.01
    )

    medium_term_rate = (
        current_rate * 0.97
        + forecast_rate * 0.03
    )

    spot_cost = (
        spot_rate
        * cargo_quantity
        * number_of_voyages
    )

    short_term_cost = (
        short_term_rate
        * cargo_quantity
        * number_of_voyages
    )

    medium_term_cost = (
        medium_term_rate
        * cargo_quantity
        * number_of_voyages
    )

    comparison_df = pd.DataFrame(
        {
            "Strategy": [
                "Spot",
                "Short-Term",
                "Medium-Term"
            ],

            "Prototype Rate (USD/MT)": [
                spot_rate,
                short_term_rate,
                medium_term_rate
            ],

            "Estimated Total Cost (USD)": [
                spot_cost,
                short_term_cost,
                medium_term_cost
            ]
        }
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Prototype Rate (USD/MT)": "${:.2f}",
                "Estimated Total Cost (USD)": "${:,.2f}"
            }
        ),
        use_container_width=True
    )

    st.caption(
        "Important: contract-rate formulas are prototype "
        "assumptions for demonstrating the optimization "
        "logic. They are not actual negotiated charter rates."
    )


# ============================================================
# MULTIPLE VOYAGE PLANNING
# ============================================================

st.markdown(
    '<div class="section-title">🔄 Multiple-Voyage Planning</div>',
    unsafe_allow_html=True
)

if current_rate > 0:

    total_spot_cost = (
        current_rate
        * cargo_quantity
        * number_of_voyages
    )

    total_forecast_cost = (
        forecast_rate
        * cargo_quantity
        * number_of_voyages
    )

    difference = (
        total_forecast_cost
        - total_spot_cost
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Planned Voyages",
            number_of_voyages
        )

    with c2:

        st.metric(
            "Current Estimated Cost",
            f"${total_spot_cost:,.2f}"
        )

    with c3:

        st.metric(
            "Forecast Estimated Cost",
            f"${total_forecast_cost:,.2f}",
            f"${difference:,.2f}"
        )


# ============================================================
# INTELLIGENT CONTRACT STRATEGY
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Intelligent Contract Strategy</div>',
    unsafe_allow_html=True
)

if current_rate > 0:

    if (
        expected_change > 3
        and number_of_voyages >= 3
    ):

        final_strategy = (
            "SHORT-TERM MULTIPLE-VOYAGE CONTRACT"
        )

    elif expected_change < -3:

        final_strategy = (
            "MAINTAIN FLEXIBILITY / SPOT"
        )

    elif number_of_voyages >= 6:

        final_strategy = (
            "MEDIUM-TERM CONTRACT"
        )

    elif number_of_voyages >= 3:

        final_strategy = (
            "SHORT-TERM CONTRACT"
        )

    else:

        final_strategy = (
            "SPOT CONTRACT"
        )

    st.markdown(
        f"""
        <div class="recommendation-box">

        <h3>Recommended Strategy</h3>

        <h2>{final_strategy}</h2>

        <p>
        Decision is based on forecasted freight movement,
        planned voyage volume and market-risk indicators.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MASTER RECOMMENDATION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 IntelliFreight Master Recommendation</div>',
    unsafe_allow_html=True
)

if current_rate > 0:

    recommendations = []

    recommendations.append(
        f"Route: {route_key}"
    )

    recommendations.append(
        f"Cargo: {cargo_quantity:,} MT"
    )

    recommendations.append(
        f"Recommended vessel: {primary_vessel}"
    )

    recommendations.append(
        f"Contract strategy: {strategy}"
    )

    recommendations.append(
        f"Expected freight movement: "
        f"{expected_change:+.2f}%"
    )

    try:
        recommendations.append(
            f"Market risk: {risk_level} "
            f"({risk_score}/100)"
        )
    except NameError:
        pass

    if expected_change > 3:

        market_message = (
            "Forecast indicates upward freight pressure. "
            "Review chartering earlier and consider "
            "multiple-voyage coverage."
        )

    elif expected_change < -3:

        market_message = (
            "Forecast indicates downward freight pressure. "
            "Maintaining flexibility may be useful."
        )

    else:

        market_message = (
            "Market conditions are relatively stable. "
            "Monitor the market and maintain alternative "
            "options."
        )

    st.info(
        f"### 🧠 Decision Summary\n\n"
        f"{market_message}"
    )

    for item in recommendations:
        st.write(f"• {item}")


# ============================================================
# SIH OBJECTIVE ALIGNMENT
# ============================================================

st.markdown(
    '<div class="section-title">🏆 SIH Problem Statement Alignment</div>',
    unsafe_allow_html=True
)

alignment = pd.DataFrame(
    {
        "SIH Requirement": [
            "Freight Forecasting",
            "Optimal Market Entry Timing",
            "Vessel Type Optimization",
            "Idle Scenario Management",
            "Risk Mitigation",
            "Multiple-Voyage Contract Planning"
        ],

        "IntelliFreight Module": [
            "BDI + route freight forecasting",
            "Charter timing engine",
            "Cargo + port constraint matching",
            "Idle risk engine",
            "Market risk engine",
            "Contract strategy optimizer"
        ]
    }
)

st.dataframe(
    alignment,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🔎 System Data Status</div>',
    unsafe_allow_html=True
)

status_data = pd.DataFrame(
    {
        "Component": [
            "Historical BDI data",
            "Processed BDI data",
            "Real BDI ML model",
            "Route freight dataset",
            "Route freight ML model"
        ],

        "Status": [
            "✅ Available"
            if os.path.exists(BDI_DATA_PATH)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(PROCESSED_BDI_PATH)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(BDI_MODEL_PATH)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(FREIGHT_DATA_PATH)
            else "❌ Missing",

            "✅ Available"
            if os.path.exists(FREIGHT_MODEL_PATH)
            else "❌ Missing"
        ]
    }
)

st.dataframe(
    status_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    """
    **Prototype / SIH Demonstration Notice**

    IntelliFreight is currently a decision-support prototype.
    Historical BDI and vessel-index observations are used as
    real market indicators. Route-level freight, fuel,
    demand and congestion inputs are currently prototype/
    synthetic data.

    Vessel and port constraints are prototype assumptions and
    should be validated against authoritative maritime and
    port data before operational use.

    Forecasts are model outputs and should not be treated as
    guaranteed future prices or financial/chartering advice.
    """
)

st.caption(
    "IntelliFreight — Intelligent Freight Forecasting Model "
    "for Optimized Vessel Chartering and Bulk Cargo Procurement"
)