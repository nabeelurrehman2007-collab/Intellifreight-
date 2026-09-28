import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliFreight",
    page_icon="🚢",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "freight_data.csv"
MODEL_PATH = BASE_DIR / "models" / "freight_forecasting_model.pkl"


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


df = load_data()
model = load_model()


# ============================================================
# ROUTE MAPPING
# ============================================================

route_mapping = {

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

vessels = {

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

ports = {

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

def recommend_vessel(cargo_quantity, destination):

    port = ports[destination]

    suitable_vessels = []

    for vessel_name, vessel in vessels.items():

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
            suitable_vessels.append(vessel_name)

    if suitable_vessels:

        # Select the smallest suitable vessel
        return suitable_vessels[0]

    # If no exact vessel is suitable,
    # select based on cargo range.
    for vessel_name, vessel in vessels.items():

        if cargo_quantity <= vessel["max_cargo"]:

            if vessel["draft"] <= port["max_draft"]:

                return vessel_name

    return "No suitable vessel"


# ============================================================
# CHARTER TIMING
# ============================================================

def calculate_charter_timing(
    current_rate,
    forecast_average,
    cargo_quantity
):

    expected_change = (
        (forecast_average - current_rate)
        / current_rate
    ) * 100

    if expected_change > 3:

        decision = "CONSIDER EARLIER CHARTERING"

        reason = (
            "Forecast indicates that freight rates "
            "may increase."
        )

    elif expected_change < -3:

        decision = "CONSIDER WAITING"

        reason = (
            "Forecast indicates that freight rates "
            "may decrease."
        )

    else:

        decision = "MONITOR MARKET"

        reason = (
            "Forecast indicates relatively stable "
            "freight conditions."
        )

    current_cost = (
        current_rate
        * cargo_quantity
    )

    forecast_cost = (
        forecast_average
        * cargo_quantity
    )

    cost_difference = (
        current_cost
        - forecast_cost
    )

    return {
        "expected_change": expected_change,
        "decision": decision,
        "reason": reason,
        "current_cost": current_cost,
        "forecast_cost": forecast_cost,
        "cost_difference": cost_difference
    }


# ============================================================
# IDLE SCENARIO
# ============================================================

def calculate_idle_risk(
    demand_index,
    congestion_index,
    forecast_average,
    current_rate
):

    score = 0

    # Demand risk
    if demand_index < 30:

        score += 40

    elif demand_index < 60:

        score += 20

    else:

        score += 5

    # Congestion risk
    if congestion_index > 70:

        score += 30

    elif congestion_index > 40:

        score += 15

    else:

        score += 5

    # Freight forecast component
    if forecast_average < current_rate:

        score += 20

    else:

        score += 5

    score = min(score, 100)

    if score >= 70:

        status = "HIGH IDLE RISK"

        action = (
            "Consider alternative employment, "
            "cargo positioning, or route options."
        )

    elif score >= 40:

        status = "MODERATE IDLE RISK"

        action = (
            "Monitor cargo demand and prepare "
            "an alternative voyage."
        )

    else:

        status = "LOW IDLE RISK"

        action = (
            "Current demand conditions appear "
            "relatively favorable."
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
    current_rate,
    fuel_price,
    demand_index,
    congestion_index,
    historical_route_data
):

    score = 0

    # Freight volatility
    freight_volatility = (
        historical_route_data["freight_rate_usd_mt"]
        .std()
    )

    # Fuel volatility
    fuel_volatility = (
        historical_route_data["fuel_price"]
        .std()
    )

    # Freight risk
    if freight_volatility > 4:

        score += 30
        freight_risk = "HIGH"

    elif freight_volatility > 2:

        score += 20
        freight_risk = "MODERATE"

    else:

        score += 10
        freight_risk = "LOW"

    # Fuel risk
    if fuel_volatility > 5:

        score += 25
        fuel_risk = "HIGH"

    elif fuel_volatility > 2:

        score += 15
        fuel_risk = "MODERATE"

    else:

        score += 5
        fuel_risk = "LOW"

    # Demand risk
    if demand_index < 30:

        score += 20
        demand_risk = "HIGH"

    elif demand_index < 60:

        score += 10
        demand_risk = "MODERATE"

    else:

        score += 5
        demand_risk = "LOW"

    # Congestion risk
    if congestion_index > 70:

        score += 25
        congestion_risk = "HIGH"

    elif congestion_index > 40:

        score += 15
        congestion_risk = "MODERATE"

    else:

        score += 5
        congestion_risk = "LOW"

    score = min(score, 100)

    if score >= 70:

        risk_level = "HIGH RISK"

        recommendation = (
            "Consider strong contingency planning "
            "and alternative charter options."
        )

    elif score >= 40:

        risk_level = "MODERATE RISK"

        recommendation = (
            "Monitor market conditions closely "
            "and keep alternative charter options available."
        )

    else:

        risk_level = "LOW RISK"

        recommendation = (
            "Current market indicators appear relatively stable."
        )

    return {
        "freight_volatility": freight_volatility,
        "fuel_volatility": fuel_volatility,
        "freight_risk": freight_risk,
        "fuel_risk": fuel_risk,
        "demand_risk": demand_risk,
        "congestion_risk": congestion_risk,
        "score": score,
        "risk_level": risk_level,
        "recommendation": recommendation
    }


# ============================================================
# CONTRACT COMPARISON
# ============================================================

def calculate_contract_comparison(
    current_rate,
    forecast_rate,
    cargo_quantity
):

    spot_rate = current_rate

    short_term_rate = (
        current_rate * 0.99
        + forecast_rate * 0.01
    )

    medium_term_rate = (
        current_rate * 0.97
        + forecast_rate * 0.03
    )

    data = {

        "Contract Type": [
            "Spot",
            "Short-term",
            "Medium-term"
        ],

        "Estimated Rate (USD/MT)": [
            spot_rate,
            short_term_rate,
            medium_term_rate
        ],

        "Estimated Voyage Cost (USD)": [
            spot_rate * cargo_quantity,
            short_term_rate * cargo_quantity,
            medium_term_rate * cargo_quantity
        ],

        "Flexibility": [
            "Very High",
            "Medium",
            "Lower"
        ],

        "Rate Stability": [
            "Low",
            "Medium",
            "Higher"
        ]
    }

    return pd.DataFrame(data)


# ============================================================
# MULTI-VOYAGE COST ANALYSIS
# ============================================================

def calculate_multi_voyage_costs(
    current_rate,
    forecast_rate,
    cargo_quantity,
    number_of_voyages
):

    spot_rate = current_rate

    short_term_rate = (
        current_rate * 0.99
        + forecast_rate * 0.01
    )

    medium_term_rate = (
        current_rate * 0.97
        + forecast_rate * 0.03
    )

    spot_total = (
        spot_rate
        * cargo_quantity
        * number_of_voyages
    )

    short_term_total = (
        short_term_rate
        * cargo_quantity
        * number_of_voyages
    )

    medium_term_total = (
        medium_term_rate
        * cargo_quantity
        * number_of_voyages
    )

    short_term_savings = (
        spot_total
        - short_term_total
    )

    medium_term_savings = (
        spot_total
        - medium_term_total
    )

    return {

        "spot_total": spot_total,

        "short_term_total":
            short_term_total,

        "medium_term_total":
            medium_term_total,

        "short_term_savings":
            short_term_savings,

        "medium_term_savings":
            medium_term_savings
    }


# ============================================================
# MASTER RECOMMENDATION
# ============================================================

def generate_master_recommendation(
    charter_decision,
    vessel,
    idle_status,
    risk_level
):

    if risk_level == "HIGH RISK":

        return (
            "Market risk is elevated. "
            "Review alternative charter options, "
            "monitor freight and port conditions, "
            "and maintain contingency plans."
        )

    if idle_status == "HIGH IDLE RISK":

        return (
            "Idle risk is elevated. "
            "Consider alternative cargo employment "
            "or vessel positioning."
        )

    if "EARLIER" in charter_decision:

        return (
            "Forecast indicates potential freight-rate "
            "increases. Review earlier chartering options "
            "while monitoring market conditions."
        )

    if "WAITING" in charter_decision:

        return (
            "Forecast indicates potential freight-rate "
            "decreases. Monitor the market before "
            "committing where operationally feasible."
        )

    return (
        f"Market conditions are relatively stable. "
        f"Monitor the market and maintain alternative "
        f"options. Recommended vessel: {vessel}."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🚢 IntelliFreight")

st.subheader(
    "Intelligent Freight Forecasting & "
    "Vessel Chartering Decision Platform"
)

st.write(
    "AI-assisted freight forecasting, vessel optimization, "
    "risk analysis and multi-voyage contract planning."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Voyage Configuration")

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


# Default number of voyages
if contract_duration == "Spot":

    default_voyages = 1

elif contract_duration == "Short-term":

    default_voyages = 3

else:

    default_voyages = 6


number_of_voyages = st.sidebar.number_input(
    "Number of Voyages",
    min_value=1,
    max_value=20,
    value=default_voyages,
    step=1
)


analyze = st.sidebar.button(
    "🔍 Analyze Voyage",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    route_key = f"{origin}|{destination}"

    route_name = route_mapping[route_key]

    route_data = df[
        df["route"] == route_name
    ].copy()

    route_data = route_data.sort_values("date")

    latest_row = route_data.iloc[-1]

    current_rate = float(
        latest_row["freight_rate_usd_mt"]
    )

    fuel_price = float(
        latest_row["fuel_price"]
    )

    demand_index = float(
        latest_row["demand_index"]
    )

    congestion_index = float(
        latest_row["congestion_index"]
    )


    # ========================================================
    # FUTURE FORECAST
    # ========================================================

    last_date = latest_row["date"]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )


    future_df = pd.DataFrame({

        "date": future_dates,

        "fuel_price": [
            fuel_price
        ] * 7,

        "demand_index": [
            demand_index
        ] * 7,

        "congestion_index": [
            congestion_index
        ] * 7
    })


    future_df["day"] = (
        future_df["date"].dt.day
    )

    future_df["month"] = (
        future_df["date"].dt.month
    )

    future_df["year"] = (
        future_df["date"].dt.year
    )

    future_df["day_of_year"] = (
        future_df["date"].dt.dayofyear
    )


    # Create all route columns
    all_routes = [
        "Newcastle_Australia_Visakhapatnam",
        "Richards_Bay_South_Africa_Paradip",
        "Indonesia_Chennai",
        "Australia_Kakinada"
    ]


    for route in all_routes:

        column_name = f"route_{route}"

        future_df[column_name] = 0


    future_df[
        f"route_{route_name}"
    ] = 1


    # ========================================================
    # MATCH MODEL FEATURES
    # ========================================================

    feature_columns = list(
        model.feature_names_in_
    )


    for column in feature_columns:

        if column not in future_df.columns:

            future_df[column] = 0


    X_future = future_df[
        feature_columns
    ]


    predictions = model.predict(
        X_future
    )


    future_df[
        "forecast_rate"
    ] = predictions


    forecast_average = float(
        np.mean(predictions)
    )

    forecast_min = float(
        np.min(predictions)
    )

    forecast_max = float(
        np.max(predictions)
    )


    expected_change = (
        (forecast_average - current_rate)
        / current_rate
    ) * 100


    # ========================================================
    # CHARTER TIMING
    # ========================================================

    charter = calculate_charter_timing(
        current_rate,
        forecast_average,
        cargo_quantity
    )


    # ========================================================
    # VESSEL OPTIMIZATION
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
        congestion_index,
        forecast_average,
        current_rate
    )


    # ========================================================
    # MARKET RISK
    # ========================================================

    market_risk = calculate_market_risk(
        current_rate,
        fuel_price,
        demand_index,
        congestion_index,
        route_data
    )


    # ========================================================
    # CONTRACT COMPARISON
    # ========================================================

    contract_table = calculate_contract_comparison(
        current_rate,
        forecast_average,
        cargo_quantity
    )


    # ========================================================
    # MULTI-VOYAGE ANALYSIS
    # ========================================================

    multi_voyage = calculate_multi_voyage_costs(
        current_rate,
        forecast_average,
        cargo_quantity,
        number_of_voyages
    )


    # ========================================================
    # MASTER RECOMMENDATION
    # ========================================================

    master_recommendation = generate_master_recommendation(
        charter["decision"],
        recommended_vessel,
        idle["status"],
        market_risk["risk_level"]
    )


    # ========================================================
    # SAVE EVERYTHING TO SESSION STATE
    # ========================================================

    st.session_state["analysis_done"] = True

    st.session_state["route_name"] = route_name

    st.session_state["origin"] = origin

    st.session_state["destination"] = destination

    st.session_state["cargo_quantity"] = cargo_quantity

    st.session_state["contract_duration"] = (
        contract_duration
    )

    st.session_state["number_of_voyages"] = (
        number_of_voyages
    )

    st.session_state["current_rate"] = current_rate

    st.session_state["forecast_average"] = (
        forecast_average
    )

    st.session_state["forecast_min"] = (
        forecast_min
    )

    st.session_state["forecast_max"] = (
        forecast_max
    )

    st.session_state["expected_change"] = (
        expected_change
    )

    st.session_state["fuel_price"] = fuel_price

    st.session_state["demand_index"] = demand_index

    st.session_state["congestion_index"] = (
        congestion_index
    )

    st.session_state["future_df"] = future_df

    st.session_state["charter"] = charter

    st.session_state["recommended_vessel"] = (
        recommended_vessel
    )

    st.session_state["idle"] = idle

    st.session_state["market_risk"] = market_risk

    st.session_state["contract_table"] = (
        contract_table
    )

    st.session_state["multi_voyage"] = (
        multi_voyage
    )

    st.session_state["master_recommendation"] = (
        master_recommendation
    )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.get(
    "analysis_done",
    False
):

    route_name = st.session_state[
        "route_name"
    ]

    origin = st.session_state[
        "origin"
    ]

    destination = st.session_state[
        "destination"
    ]

    cargo_quantity = st.session_state[
        "cargo_quantity"
    ]

    contract_duration = st.session_state[
        "contract_duration"
    ]

    number_of_voyages = st.session_state[
        "number_of_voyages"
    ]

    current_rate = st.session_state[
        "current_rate"
    ]

    forecast_average = st.session_state[
        "forecast_average"
    ]

    forecast_min = st.session_state[
        "forecast_min"
    ]

    forecast_max = st.session_state[
        "forecast_max"
    ]

    expected_change = st.session_state[
        "expected_change"
    ]

    fuel_price = st.session_state[
        "fuel_price"
    ]

    demand_index = st.session_state[
        "demand_index"
    ]

    congestion_index = st.session_state[
        "congestion_index"
    ]

    future_df = st.session_state[
        "future_df"
    ]

    charter = st.session_state[
        "charter"
    ]

    recommended_vessel = st.session_state[
        "recommended_vessel"
    ]

    idle = st.session_state[
        "idle"
    ]

    market_risk = st.session_state[
        "market_risk"
    ]

    contract_table = st.session_state[
        "contract_table"
    ]

    multi_voyage = st.session_state[
        "multi_voyage"
    ]

    master_recommendation = (
        st.session_state[
            "master_recommendation"
        ]
    )


    # ========================================================
    # ROUTE HEADER
    # ========================================================

    st.success(
        f"Analysis completed: "
        f"{origin} → {destination}"
    )

    st.header(
        f"🚢 {origin} → {destination}"
    )

    st.caption(
        f"Route ID: {route_name}"
    )


    # ========================================================
    # VOYAGE SUMMARY
    # ========================================================

    st.subheader("Voyage Summary")

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Cargo",
            f"{cargo_quantity:,} MT"
        )


    with col2:

        st.metric(
            "Contract",
            contract_duration
        )


    with col3:

        st.metric(
            "Destination",
            destination
        )


    with col4:

        st.metric(
            "Vessel",
            recommended_vessel
        )


    with col5:

        st.metric(
            "Voyages",
            number_of_voyages
        )


    st.divider()


    # ========================================================
    # FREIGHT FORECAST
    # ========================================================

    st.subheader(
        "📈 Freight Rate Forecast"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Current Rate",
            f"${current_rate:.2f}/MT"
        )


    with col2:

        st.metric(
            "7-Day Average",
            f"${forecast_average:.2f}/MT"
        )


    with col3:

        st.metric(
            "Expected Change",
            f"{expected_change:.2f}%"
        )


    with col4:

        st.metric(
            "Forecast Range",
            f"${forecast_min:.2f} - "
            f"${forecast_max:.2f}"
        )


    chart_df = future_df[
        ["date", "forecast_rate"]
    ].copy()


    chart_df.columns = [
        "Date",
        "Forecast Rate"
    ]


    fig = px.line(
        chart_df,
        x="Date",
        y="Forecast Rate",
        markers=True,
        title="7-Day Freight Rate Forecast"
    )


    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Freight Rate (USD/MT)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CHARTER TIMING
    # ========================================================

    st.subheader(
        "⏱️ Optimal Market Entry Timing"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Charter Decision",
            charter["decision"]
        )


    with col2:

        st.metric(
            "Expected Rate Change",
            f"{charter['expected_change']:.2f}%"
        )


    st.info(
        charter["reason"]
    )


    # ========================================================
    # VESSEL OPTIMIZATION
    # ========================================================

    st.subheader(
        "🚢 Vessel Type Optimization"
    )


    vessel_info = vessels[
        recommended_vessel
    ]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Recommended Vessel",
            recommended_vessel
        )


    with col2:

        st.metric(
            "Draft",
            f"{vessel_info['draft']:.1f} m"
        )


    with col3:

        st.metric(
            "LOA",
            f"{vessel_info['loa']} m"
        )


    with col4:

        st.metric(
            "Handling",
            f"{vessel_info['handling']:,} MT"
        )


    st.info(
        f"The prototype recommends a "
        f"{recommended_vessel} for the selected "
        f"cargo and destination constraints."
    )


    # ========================================================
    # IDLE RISK
    # ========================================================

    st.subheader(
        "⚓ Idle Scenario Management"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Idle Risk Score",
            f"{idle['score']}/100"
        )


    with col2:

        st.metric(
            "Idle Status",
            idle["status"]
        )


    with col3:

        st.metric(
            "Demand Index",
            f"{demand_index:.2f}"
        )


    st.info(
        idle["action"]
    )


    # ========================================================
    # MARKET RISK
    # ========================================================

    st.subheader(
        "⚠️ Market Risk Analysis"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Risk Score",
            f"{market_risk['score']}/100"
        )


    with col2:

        st.metric(
            "Risk Level",
            market_risk["risk_level"]
        )


    with col3:

        st.metric(
            "Freight Volatility",
            f"{market_risk['freight_volatility']:.2f}"
        )


    with col4:

        st.metric(
            "Fuel Volatility",
            f"{market_risk['fuel_volatility']:.2f}"
        )


    risk_table = pd.DataFrame({

        "Risk Factor": [
            "Freight",
            "Fuel",
            "Demand",
            "Port Congestion"
        ],

        "Status": [
            market_risk["freight_risk"],
            market_risk["fuel_risk"],
            market_risk["demand_risk"],
            market_risk["congestion_risk"]
        ]
    })


    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        market_risk["recommendation"]
    )


    # ========================================================
    # MARKET INDICATORS
    # ========================================================

    st.subheader(
        "📊 Current Market Indicators"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Fuel Price",
            f"${fuel_price:.2f}"
        )


    with col2:

        st.metric(
            "Demand Index",
            f"{demand_index:.2f}"
        )


    with col3:

        st.metric(
            "Congestion Index",
            f"{congestion_index:.2f}"
        )


    with col4:

        st.metric(
            "Current Freight",
            f"${current_rate:.2f}/MT"
        )


    # ========================================================
    # CONTRACT COMPARISON
    # ========================================================

    st.subheader(
        "📋 Contract Strategy Comparison"
    )


    st.dataframe(
        contract_table,
        use_container_width=True,
        hide_index=True
    )


    contract_chart = contract_table[
        [
            "Contract Type",
            "Estimated Voyage Cost (USD)"
        ]
    ]


    fig_contract = px.bar(
        contract_chart,
        x="Contract Type",
        y="Estimated Voyage Cost (USD)",
        title="Estimated Cost per Voyage"
    )


    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )


    st.info(
        "The contract rates shown here are "
        "prototype assumptions for demonstration. "
        "They are not actual market contract prices."
    )


    # ========================================================
    # MULTIPLE-VOYAGE PLANNING
    # ========================================================

    st.subheader(
        "🔄 Multiple-Voyage Contract Planning"
    )


    st.write(
        f"Planning **{number_of_voyages} voyage(s)** "
        f"of **{cargo_quantity:,} MT** each."
    )


    total_cargo = (
        cargo_quantity
        * number_of_voyages
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Cargo Volume",
            f"{total_cargo:,} MT"
        )


    with col2:

        st.metric(
            "Number of Voyages",
            number_of_voyages
        )


    with col3:

        st.metric(
            "Current Spot Cost/Voyage",
            f"${current_rate * cargo_quantity:,.0f}"
        )


    st.write("### Estimated Multi-Voyage Costs")


    multi_voyage_table = pd.DataFrame({

        "Strategy": [
            "Spot",
            "Short-term",
            "Medium-term"
        ],

        "Total Contract Cost (USD)": [
            multi_voyage["spot_total"],
            multi_voyage["short_term_total"],
            multi_voyage["medium_term_total"]
        ],

        "Savings vs Spot (USD)": [
            0,
            multi_voyage["short_term_savings"],
            multi_voyage["medium_term_savings"]
        ]
    })


    st.dataframe(
        multi_voyage_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MULTI-VOYAGE COST CHART
    # ========================================================

    fig_multi = px.bar(
        multi_voyage_table,
        x="Strategy",
        y="Total Contract Cost (USD)",
        title=(
            f"Estimated Total Cost for "
            f"{number_of_voyages} Voyages"
        )
    )


    st.plotly_chart(
        fig_multi,
        use_container_width=True
    )


    # ========================================================
    # SAVINGS
    # ========================================================

    st.write(
        "### Potential Cost Difference vs Spot"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Short-term Difference",
            f"${multi_voyage['short_term_savings']:,.2f}"
        )


    with col2:

        st.metric(
            "Medium-term Difference",
            f"${multi_voyage['medium_term_savings']:,.2f}"
        )


    st.caption(
        "Positive values indicate lower estimated "
        "cost compared with the spot-contract "
        "prototype baseline."
    )


    # ========================================================
    # MASTER RECOMMENDATION
    # ========================================================

    st.subheader(
        "🧠 IntelliFreight Recommendation"
    )


    st.success(
        master_recommendation
    )


    # ========================================================
    # PROJECT OBJECTIVE
    # ========================================================

    st.subheader(
        "🎯 Decision Objective"
    )


    st.write(
        """
        IntelliFreight is designed to help logistics
        managers move from repeated spot-market
        decisions toward data-driven short-term and
        medium-term multiple-voyage planning.

        The system combines freight forecasting,
        vessel suitability, idle-risk analysis,
        market-risk monitoring and contract-cost
        estimation into one decision-support workflow.
        """
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ Prototype Notice: The current freight data, "
        "vessel specifications, port constraints, "
        "contract discounts and risk thresholds are "
        "demonstration assumptions. They must be replaced "
        "with verified real-world datasets and industry "
        "parameters before operational use."
    )


# ============================================================
# INITIAL PAGE
# ============================================================

else:

    st.info(
        "👈 Configure your voyage parameters in the "
        "sidebar and click **Analyze Voyage**."
    )


    st.subheader(
        "What IntelliFreight Does"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            "### 📈\nFreight Forecasting"
        )

        st.write(
            "Forecast short-term freight-rate movements."
        )


    with col2:

        st.markdown(
            "### 🚢\nVessel Optimization"
        )

        st.write(
            "Match cargo requirements with suitable vessels."
        )


    with col3:

        st.markdown(
            "### ⚠️\nRisk Detection"
        )

        st.write(
            "Monitor market, fuel, demand and congestion risks."
        )


    with col4:

        st.markdown(
            "### 🔄\nContract Planning"
        )

        st.write(
            "Compare spot and multiple-voyage strategies."
        )


    st.divider()


    st.subheader(
        "IntelliFreight Decision Pipeline"
    )


    st.write(
        """
        **User Input**
        → **Freight Data**
        → **Feature Engineering**
        → **ML Forecast**
        → **Vessel Optimization**
        → **Risk Engine**
        → **Contract Analysis**
        → **Decision Recommendation**
        """
    )


    st.caption(
        "Prototype developed for the Smart India Hackathon "
        "problem statement."
    )