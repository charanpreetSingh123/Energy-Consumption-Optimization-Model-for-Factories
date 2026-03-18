import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Eco-Smart Factory Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DARK UI STYLE ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background gradient */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    
            color: white;
}

/* Sidebar styling */
section[data-testid="stSidebar"]{
    
            background: linear-gradient(180deg,#141E30,#243B55);
    border-right: 1px solid #333;
}

/* Main titles */
h1, h2, h3 {
    color: #00F5D4;
}

/* Metric cards (REAL hover effect) */
.metric-card {
    background: rgba(255,255,255,0.06);
    
            border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    transition: all 0.35s ease;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 16px 40px rgba(0,245,212,0.18);
    border: 1px solid rgba(0,245,212,0.35);
}
/* Metric text styles */
.metric-title {
    font-size: 16px;
    font-weight: 500;
    color: #B8FFF5;
    margin-bottom: 12px;
    letter-spacing: 0.3px;}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
    margin-bottom: 8px;}

.metric-sub {
    font-size: 13px;
    color: #B0BEC5;
    opacity: 0.9;}

/* Buttons */
.stButton > button {
    background: linear-gradient(45deg,#00F5D4,#4FACFE);
    color: black;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;}

.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 8px 20px rgba(79,172,254,0.35);
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg,#00F5D4,#4FACFE);
}

</style>
""", unsafe_allow_html=True)
# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():

    df_steel = pd.read_csv("Steel_industry_data.csv")
    df_steel.columns = df_steel.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    df_iiot = pd.read_csv("Energy_dataset.csv")
    df_iiot["timestamp"] = pd.to_datetime(df_iiot["timestamp"])

    machine_coords = {
        "MCH_1":(1,1),
        "MCH_2":(1,3),
        "MCH_3":(2,2),
        "MCH_4":(3,1),
        "MCH_5":(3,3)
    }

    df_iiot["grid_x"] = df_iiot["machine_id"].map(lambda m: machine_coords.get(m,(0,0))[0])
    df_iiot["grid_y"] = df_iiot["machine_id"].map(lambda m: machine_coords.get(m,(0,0))[1])

    return df_steel, df_iiot


# ---------------- FORECAST MODEL ----------------

@st.cache_resource
def train_forecasting_model(df):

    le_day = LabelEncoder()
    le_load = LabelEncoder()

    X = pd.DataFrame({
        "Day":le_day.fit_transform(df["Day_of_week"]),
        "Load":le_load.fit_transform(df["Load_Type"])
    })

    y_energy = df["Usage_kWh"]
    y_co2 = df["CO2tCO2"]

    rf_energy = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_co2 = RandomForestRegressor(n_estimators=50, random_state=42)

    rf_energy.fit(X,y_energy)
    rf_co2.fit(X,y_co2)

    return rf_energy, rf_co2, le_day, le_load


# ---------------- ANOMALY MODEL ----------------

@st.cache_resource
def train_anomaly_model(df):

    features = ["energy_kWh","current_A","machine_utilization_%"]

    X = df[features].fillna(0)

    iso = IsolationForest(contamination=0.03, random_state=42)

    iso.fit(X)

    return iso, features


# ---------------- NLP MODEL ----------------

@st.cache_resource
def train_nlp_model():

    logs = [
        "Motor overheating on MCH_1",
        "Routine oil change",
        "Someone left lights on",
        "Vibration detected critical fault",
        "Weekly calibration complete"
    ]

    labels = [
        "High Priority",
        "Routine",
        "Spam",
        "High Priority",
        "Routine"
    ]

    vec = CountVectorizer()
    X = vec.fit_transform(logs)

    clf = MultinomialNB()
    clf.fit(X,labels)

    return vec, clf


# ---------------- LOAD SYSTEM ----------------

with st.spinner("Loading AI Systems..."):

    df_steel, df_iiot = load_data()

    rf_energy, rf_co2, le_day, le_load = train_forecasting_model(df_steel)

    iso_model, iiot_features = train_anomaly_model(df_iiot)

    vec, clf = train_nlp_model()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("⚡ Eco Smart Factory")

    st.markdown("---")

    navigation = st.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "AI Forecast",
            "Digital Twin Simulator",
            "Spatial Heatmap",
            "NLP Alert System"
        ]
    )

    st.markdown("---")
    st.caption("AI Powered Factory OS")

# ---------------- DASHBOARD ----------------

if navigation == "Executive Dashboard":

    st.title("Enterprise Energy Overview")

    latest_energy = df_steel['Usage_kWh'].iloc[-1]
    latest_co2 = df_steel['CO2tCO2'].iloc[-1]
    latest_cost = latest_energy * 8

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Energy Usage</div>
            <div class="metric-value">{latest_energy:.1f} kWh</div>
            <div class="metric-sub">Latest factory energy draw</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌍 CO₂ Emission</div>
            <div class="metric-value">{latest_co2:.2f} tCO2</div>
            <div class="metric-sub">Current carbon footprint</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Estimated Cost</div>
            <div class="metric-value">₹{latest_cost:.2f}</div>
            <div class="metric-sub">Approx energy expense</div>
        </div>
        """, unsafe_allow_html=True)

    trend = df_steel.tail(200)

    fig = px.line(
        trend,
        y="Usage_kWh",
        template="plotly_dark",
        title="Energy Consumption Trend"
    )

    fig.update_traces(line=dict(width=3, color="#00F5D4"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECAST ----------------

elif navigation == "AI Forecast":

    st.title("AI Energy Forecast")

    day = st.selectbox("Day", le_day.classes_)
    load = st.selectbox("Load Type", le_load.classes_)

    if st.button("Generate Forecast"):

        with st.spinner("Running AI Model..."):
            time.sleep(1)

        X_pred = pd.DataFrame({
            "Day":[le_day.transform([day])[0]],
            "Load":[le_load.transform([load])[0]]
        })

        energy = rf_energy.predict(X_pred)[0]
        co2 = rf_co2.predict(X_pred)[0]

        c1,c2,c3 = st.columns(3)

        c1.metric("Predicted Energy",f"{energy:.2f} kWh")
        c2.metric("Predicted CO2",f"{co2:.3f} tCO2")
        c3.metric("Estimated Cost",f"₹{energy*8:.2f}")


# ---------------- DIGITAL TWIN ----------------

elif navigation == "Digital Twin Simulator":

    
    st.title("Factory Digital Twin Simulator")

    col1,col2 = st.columns([1,2])

    with col1:

        load_level = st.slider("Factory Load (%)",50,100,80)

        shift_hours = st.slider("Shift Hours",4,12,8)

        machines = st.slider("Active Machines",1,5,3)

        run = st.button("Run Simulation")

    with col2:

        if run:

            with st.spinner("Running Simulation..."):
                time.sleep(1)

            base_energy = rf_energy.predict(
                pd.DataFrame({"Day":[0],"Load":[0]})
            )[0]

            sim_energy = base_energy*(load_level/100)*(shift_hours/8)*(machines/3)

            cost = sim_energy*8

            co2 = sim_energy*0.000233

            c1,c2,c3 = st.columns(3)

            c1.metric("Simulated Energy",f"{sim_energy:.2f} kWh")

            c2.metric("Estimated Cost",f"₹{cost:.2f}")

            c3.metric("CO2 Output",f"{co2:.3f} tCO2")

            sim_data = pd.DataFrame({
                "Metric":["Energy","Cost","CO2"],
                "Value":[sim_energy,cost,co2]
            })

            fig = px.bar(
                sim_data,
                x="Metric",
                y="Value",
                template="plotly_dark",
                color="Metric"
            )

            st.plotly_chart(fig,use_container_width=True)

            efficiency = max(0,100-(sim_energy/base_energy*100))

            st.progress(int(min(100,efficiency)))

            st.write(f"Energy Efficiency Score: {efficiency:.1f}/100")

# ---------------- HEATMAP ----------------

elif navigation == "Spatial Heatmap":

    st.title("Factory Anomaly Detection")

    df = df_iiot.copy()

    df["Anomaly"] = iso_model.predict(df[iiot_features])

    fig = px.density_heatmap(
        df,
        x="grid_x",
        y="grid_y",
        z="energy_kWh",
        template="plotly_dark",
        nbinsx=5,
        nbinsy=5,
        color_continuous_scale="Turbo"
    )

    fig.update_layout(title="Factory Energy Heatmap")

    st.plotly_chart(fig,use_container_width=True)


# ---------------- NLP ----------------

elif navigation == "NLP Alert System":

    st.title("Maintenance Log Classifier")

    text = st.text_input("Enter maintenance log","Machine vibration detected")

    if st.button("Classify"):

        vec_input = vec.transform([text])

        pred = clf.predict(vec_input)[0]

        st.write(f"Classification: {pred}")

        if pred == "High Priority":
            st.error("Urgent maintenance required!")

        elif pred == "Routine":
            st.success("Routine maintenance logged.")

        else:
            st.warning("Ignored message.")
