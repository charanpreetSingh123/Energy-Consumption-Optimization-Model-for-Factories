import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Eco-Smart Factory Hub", page_icon="⚡", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df_steel = pd.read_csv('Steel_industry_data.csv')
    df_steel.columns = df_steel.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    df_iiot = pd.read_csv('Energy_dataset.csv')
    df_iiot['timestamp'] = pd.to_datetime(df_iiot['timestamp'])

    machine_coords = {
        'MCH_1': (1,1),
        'MCH_2': (1,3),
        'MCH_3': (2,2),
        'MCH_4': (3,1),
        'MCH_5': (3,3)
    }

    df_iiot['grid_x'] = df_iiot['machine_id'].map(lambda m: machine_coords.get(m,(0,0))[0])
    df_iiot['grid_y'] = df_iiot['machine_id'].map(lambda m: machine_coords.get(m,(0,0))[1])

    return df_steel, df_iiot


# --- FORECAST MODEL ---
@st.cache_resource
def train_forecasting_model(df):

    le_day = LabelEncoder()
    le_load = LabelEncoder()

    X = pd.DataFrame({
        'Day': le_day.fit_transform(df['Day_of_week']),
        'Load': le_load.fit_transform(df['Load_Type'])
    })

    y_energy = df['Usage_kWh']
    y_co2 = df['CO2tCO2']

    rf_energy = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_co2 = RandomForestRegressor(n_estimators=50, random_state=42)

    rf_energy.fit(X, y_energy)
    rf_co2.fit(X, y_co2)

    return rf_energy, rf_co2, le_day, le_load


# --- ANOMALY MODEL ---
@st.cache_resource
def train_anomaly_model(df):

    features = ['energy_kWh','current_A','machine_utilization_%']
    X = df[features].fillna(0)

    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(X)

    return iso, features


# --- NLP MODEL ---
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
    clf.fit(X, labels)

    return vec, clf


with st.spinner("Loading AI Systems..."):
    df_steel, df_iiot = load_data()
    rf_energy, rf_co2, le_day, le_load = train_forecasting_model(df_steel)
    iso_model, iiot_features = train_anomaly_model(df_iiot)
    vec, clf = train_nlp_model()

# --- SIDEBAR ---
with st.sidebar:

    st.title("Factory OS")

    navigation = st.radio(
        "System Modules",
        [
            "Executive Dashboard",
            "Prescriptive AI & Forecast",
            "Factory Digital Twin Simulator",
            "Spatial Heatmap & Diagnostics",
            "NLP Alert Dispatcher"
        ]
    )


# --- EXECUTIVE DASHBOARD ---
if navigation == "Executive Dashboard":

    st.title("Enterprise Energy Overview")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Latest Energy Draw",
        f"{df_steel['Usage_kWh'].iloc[-1]:.1f} kWh"
    )

    col2.metric(
        "CO2 Output",
        f"{df_steel['CO2tCO2'].iloc[-1]:.2f} tCO2"
    )

    col3.metric(
        "Estimated Cost",
        f"₹{df_steel['Usage_kWh'].iloc[-1]*8:.2f}"
    )

    trend = df_steel.tail(200)

    fig = px.line(
        trend,
        y="Usage_kWh",
        title="Energy Consumption Trend"
    )

    st.plotly_chart(fig, use_container_width=True)


# --- FORECAST MODULE ---
elif navigation == "Prescriptive AI & Forecast":

    st.title("AI Energy Forecast")

    day = st.selectbox("Day", le_day.classes_)
    load = st.selectbox("Load Type", le_load.classes_)

    if st.button("Generate Forecast"):

        X_pred = pd.DataFrame({
            'Day':[le_day.transform([day])[0]],
            'Load':[le_load.transform([load])[0]]
        })

        energy = rf_energy.predict(X_pred)[0]
        co2 = rf_co2.predict(X_pred)[0]

        st.metric("Predicted Energy", f"{energy:.2f} kWh")
        st.metric("Predicted CO2", f"{co2:.3f} tCO2")
        st.metric("Estimated Cost", f"₹{energy*8:.2f}")


# --- DIGITAL TWIN SIMULATOR ---
elif navigation == "Factory Digital Twin Simulator":

    st.title("Factory Digital Twin Energy Simulator")

    col1,col2 = st.columns([1,2])

    with col1:

        load_level = st.slider(
            "Factory Load (%)",
            50,
            100,
            80
        )

        shift_hours = st.slider(
            "Shift Hours",
            4,
            12,
            8
        )

        machines = st.slider(
            "Active Machines",
            1,
            5,
            3
        )

        run = st.button("Run Simulation")

    with col2:

        if run:

            base_energy = rf_energy.predict(
                pd.DataFrame({
                    'Day':[0],
                    'Load':[0]
                })
            )[0]

            sim_energy = base_energy*(load_level/100)*(shift_hours/8)*(machines/3)

            cost = sim_energy*8
            co2 = sim_energy*0.000233

            c1,c2,c3 = st.columns(3)

            c1.metric("Simulated Energy",f"{sim_energy:.2f} kWh")
            c2.metric("Estimated Cost",f"₹{cost:.2f}")
            c3.metric("CO2 Output",f"{co2:.3f} tCO2")

            efficiency = max(0,100-(sim_energy/base_energy*100))

            st.progress(int(min(100,efficiency)))

            st.write(f"Energy Efficiency Score: {efficiency:.1f}/100")

            if load_level > 90:
                st.warning("Reduce machine load to improve efficiency.")

            elif machines > 4:
                st.info("Too many machines active. Stagger production.")

            else:
                st.success("Configuration appears efficient.")


# --- HEATMAP MODULE ---
elif navigation == "Spatial Heatmap & Diagnostics":

    st.title("Factory Anomaly Detection")

    df = df_iiot.copy()

    df['Anomaly'] = iso_model.predict(df[iiot_features])

    latest = df[df['timestamp']==df['timestamp'].max()]

    fig = px.scatter(
        latest,
        x="machine_utilization_%",
        y="energy_kWh",
        color="Anomaly",
        text="machine_id"
    )

    st.plotly_chart(fig,use_container_width=True)


# --- NLP MODULE ---
elif navigation == "NLP Alert Dispatcher":

    st.title("Maintenance Log Classifier")

    text = st.text_input(
        "Enter maintenance log",
        "Machine vibration detected"
    )

    if st.button("Classify"):

        vec_input = vec.transform([text])
        pred = clf.predict(vec_input)[0]

        st.write(f"Classification: {pred}")

        if pred == "High Priority":
            st.error("Urgent maintenance required!")

        elif pred == "Routine":
            st.info("Routine maintenance logged.")

        else:
            st.warning("Ignored message.")