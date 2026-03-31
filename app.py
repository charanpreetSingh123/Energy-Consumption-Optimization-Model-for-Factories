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

/* Hero card */
.hero-card {
    background: linear-gradient(90deg, rgba(0,245,212,0.12), rgba(79,172,254,0.12));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 24px 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    transition: all 0.35s ease;
    min-height: 160px;
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
    letter-spacing: 0.3px;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
    margin-bottom: 8px;
}

.metric-sub {
    font-size: 13px;
    color: #B0BEC5;
    opacity: 0.95;
    line-height: 1.5;
}

/* Insight card */
.insight-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,245,212,0.18);
    border-left: 4px solid #00F5D4;
    border-radius: 16px;
    padding: 18px;
    margin: 14px 0;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 8px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Mini card */
.mini-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 16px;
    min-height: 110px;
}

/* Section divider */
.section-divider {
    margin: 18px 0 10px 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,245,212,0.4), rgba(79,172,254,0.05));
    border-radius: 999px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(45deg,#00F5D4,#4FACFE);
    color: black;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

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
        "MCH_1": (1, 1),
        "MCH_2": (1, 3),
        "MCH_3": (2, 2),
        "MCH_4": (3, 1),
        "MCH_5": (3, 3)
    }

    df_iiot["grid_x"] = df_iiot["machine_id"].map(lambda m: machine_coords.get(m, (0, 0))[0])
    df_iiot["grid_y"] = df_iiot["machine_id"].map(lambda m: machine_coords.get(m, (0, 0))[1])

    return df_steel, df_iiot


# ---------------- FORECAST MODEL ----------------

@st.cache_resource
def train_forecasting_model(df):

    le_day = LabelEncoder()
    le_load = LabelEncoder()

    X = pd.DataFrame({
        "Day": le_day.fit_transform(df["Day_of_week"]),
        "Load": le_load.fit_transform(df["Load_Type"])
    })

    y_energy = df["Usage_kWh"]
    y_co2 = df["CO2tCO2"]

    rf_energy = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_co2 = RandomForestRegressor(n_estimators=50, random_state=42)

    rf_energy.fit(X, y_energy)
    rf_co2.fit(X, y_co2)

    return rf_energy, rf_co2, le_day, le_load


# ---------------- ANOMALY MODEL ----------------

@st.cache_resource
def train_anomaly_model(df):

    features = ["energy_kWh", "current_A", "machine_utilization_%"]

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
    clf.fit(X, labels)

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

    st.title("Executive Dashboard")

    st.markdown("""
    <div class="hero-card">
        <h2 style="margin:0; color:#00F5D4;">⚡ Eco-Smart Factory Hub</h2>
        <p style="margin:8px 0 0 0; color:#CFE8EF;">
            AI-powered energy optimization, predictive monitoring, and digital twin intelligence for smart manufacturing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Enterprise Energy Overview")

    latest_energy = df_steel['Usage_kWh'].iloc[-1]
    latest_co2 = df_steel['CO2tCO2'].iloc[-1]
    latest_cost = latest_energy * 8

    prev_energy = df_steel['Usage_kWh'].iloc[-2]
    prev_co2 = df_steel['CO2tCO2'].iloc[-2]
    prev_cost = prev_energy * 8

    energy_delta = latest_energy - prev_energy
    co2_delta = latest_co2 - prev_co2
    cost_delta = latest_cost - prev_cost

    # Status Logic
    if energy_delta < -5:
        status_label = "🟢 Efficient Operation"
        status_color = "#00E676"
        status_bg = "rgba(0,230,118,0.10)"
    elif energy_delta > 5:
        status_label = "🔴 Rising Consumption"
        status_color = "#FF6B6B"
        status_bg = "rgba(255,107,107,0.10)"
    else:
        status_label = "🟡 Stable Load"
        status_color = "#FFD166"
        status_bg = "rgba(255,209,102,0.10)"

    if energy_delta > 0:
        energy_change_text = f'<span style="color:#FF6B6B;">⬆ Increase: {energy_delta:+.1f} kWh from previous reading</span>'
    elif energy_delta < 0:
        energy_change_text = f'<span style="color:#00E676;">⬇ Decrease: {energy_delta:+.1f} kWh from previous reading</span>'
    else:
        energy_change_text = '<span style="color:#B0BEC5;">➡ No change from previous reading</span>'

    if co2_delta > 0:
        co2_change_text = f'<span style="color:#FF6B6B;">⬆ Increase: {co2_delta:+.2f} tCO2 from previous reading</span>'
    elif co2_delta < 0:
        co2_change_text = f'<span style="color:#00E676;">⬇ Decrease: {co2_delta:+.2f} tCO2 from previous reading</span>'
    else:
        co2_change_text = '<span style="color:#B0BEC5;">➡ No change from previous reading</span>'

    if cost_delta > 0:
        cost_change_text = f'<span style="color:#FF6B6B;">⬆ Increase: ₹{cost_delta:+.2f} from previous reading</span>'
    elif cost_delta < 0:
        cost_change_text = f'<span style="color:#00E676;">⬇ Decrease: ₹{cost_delta:+.2f} from previous reading</span>'
    else:
        cost_change_text = '<span style="color:#B0BEC5;">➡ No change from previous reading</span>'

    # AI Insight
    insight_text = (
        f"AI Insight: Current energy demand is {latest_energy:.1f} kWh with a movement of "
        f"{energy_delta:+.1f} kWh versus the previous reading. The factory is currently operating in a "
        f"{status_label.replace('🟢 ', '').replace('🔴 ', '').replace('🟡 ', '').lower()} state."
    )

    st.markdown(f"""
    <div class="insight-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <div style="font-size:14px; color:#B8FFF5; font-weight:600;">📡 Live Operational Status</div>
                <div style="font-size:13px; color:#CFE8EF; margin-top:6px;">Real-time executive layer summarizing current factory behavior.</div>
            </div>
            <div class="status-badge" style="background:{status_bg}; color:{status_color}; border-color:{status_color};">
                {status_label}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Energy Usage</div>
            <div class="metric-value">{latest_energy:.1f} kWh</div>
            <div class="metric-sub">
                {energy_change_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌍 CO₂ Emission</div>
            <div class="metric-value">{latest_co2:.2f} tCO2</div>
            <div class="metric-sub">
                {co2_change_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Estimated Cost</div>
            <div class="metric-value">₹{latest_cost:.2f}</div>
            <div class="metric-sub">
                {cost_change_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-card">
        <div style="font-size:15px; font-weight:600; color:#00F5D4;">🧠 Executive AI Insight</div>
        <div style="margin-top:8px; font-size:14px; color:#D8F3FF;">
            {insight_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    trend = df_steel.tail(200).copy()
    trend["Rolling_Avg_7"] = trend["Usage_kWh"].rolling(7).mean()

    fig = px.line(
        trend,
        y=["Usage_kWh", "Rolling_Avg_7"],
        template="plotly_dark",
        title="Energy Consumption Trend vs 7-Point Rolling Average"
    )

    fig.update_traces(line=dict(width=3))
    fig.data[0].line.color = "#00F5D4"
    if len(fig.data) > 1:
        fig.data[1].line.color = "#FFD166"
        fig.data[1].line.width = 2
        fig.data[1].line.dash = "dash"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(fig, use_container_width=True)

    # Mini summary cards
    avg_energy_20 = df_steel["Usage_kWh"].tail(20).mean()
    peak_energy_50 = df_steel["Usage_kWh"].tail(50).max()
    min_energy_50 = df_steel["Usage_kWh"].tail(50).min()

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="mini-card">
            <div style="font-size:14px; color:#B8FFF5;">📊 Avg (Last 20 Readings)</div>
            <div style="font-size:26px; font-weight:700; margin-top:8px; color:#FFFFFF;">{avg_energy_20:.1f} kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="mini-card">
            <div style="font-size:14px; color:#B8FFF5;">🚀 Peak (Last 50 Readings)</div>
            <div style="font-size:26px; font-weight:700; margin-top:8px; color:#FFFFFF;">{peak_energy_50:.1f} kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="mini-card">
            <div style="font-size:14px; color:#B8FFF5;">🌙 Lowest (Last 50 Readings)</div>
            <div style="font-size:26px; font-weight:700; margin-top:8px; color:#FFFFFF;">{min_energy_50:.1f} kWh</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FORECAST ----------------

elif navigation == "AI Forecast":

    st.title("AI Energy Forecast")

    st.markdown("""
    <div class="hero-card">
        <h2 style="margin:0; color:#00F5D4;">🔮 Forecast Intelligence</h2>
        <p style="margin:8px 0 0 0; color:#CFE8EF;">
            Generate predictive energy and CO₂ estimates using historical operational patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    day = st.selectbox("Day", le_day.classes_)
    load = st.selectbox("Load Type", le_load.classes_)

    if st.button("Generate Forecast"):

        with st.spinner("Running AI Model..."):
            time.sleep(1)

        X_pred = pd.DataFrame({
            "Day": [le_day.transform([day])[0]],
            "Load": [le_load.transform([load])[0]]
        })

        energy = rf_energy.predict(X_pred)[0]
        co2 = rf_co2.predict(X_pred)[0]
        cost = energy * 8

        energy_low = energy * 0.92
        energy_high = energy * 1.08

        co2_low = co2 * 0.92
        co2_high = co2 * 1.08

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">⚡ Predicted Energy</div>
                <div class="metric-value">{energy:.2f} kWh</div>
                <div class="metric-sub">Model-generated energy estimate for selected operating profile.</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🌍 Predicted CO₂</div>
                <div class="metric-value">{co2:.3f} tCO2</div>
                <div class="metric-sub">Projected carbon output under selected conditions.</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">💰 Estimated Cost</div>
                <div class="metric-value">₹{cost:.2f}</div>
                <div class="metric-sub">Approximate operational energy expense.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card">
            <div style="font-size:15px; font-weight:600; color:#00F5D4;">📈 Forecast Confidence Range</div>
            <div style="margin-top:8px; font-size:14px; color:#D8F3FF;">
                Energy: <b>{energy_low:.2f} - {energy_high:.2f} kWh</b><br>
                CO₂: <b>{co2_low:.3f} - {co2_high:.3f} tCO2</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if energy > df_steel["Usage_kWh"].mean():
            forecast_msg = "Predicted energy demand is above average historical usage. Consider scheduling optimization or load balancing."
        else:
            forecast_msg = "Predicted energy demand is within an efficient operating band based on historical usage."

        st.markdown(f"""
        <div class="insight-card">
            <div style="font-size:15px; font-weight:600; color:#00F5D4;">🧠 Forecast Interpretation</div>
            <div style="margin-top:8px; font-size:14px; color:#D8F3FF;">
                {forecast_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- DIGITAL TWIN ----------------

elif navigation == "Digital Twin Simulator":

    st.title("Factory Digital Twin Simulator")

    st.markdown("""
    <div class="hero-card">
        <h2 style="margin:0; color:#00F5D4;">🏭 Digital Twin Control Layer</h2>
        <p style="margin:8px 0 0 0; color:#CFE8EF;">
            Simulate operational scenarios to understand energy, cost, and carbon outcomes before deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:

        load_level = st.slider("Factory Load (%)", 50, 100, 80)

        shift_hours = st.slider("Shift Hours", 4, 12, 8)

        machines = st.slider("Active Machines", 1, 5, 3)

        run = st.button("Run Simulation")

    with col2:

        if run:

            with st.spinner("Running Simulation..."):
                time.sleep(1)

            base_energy = rf_energy.predict(
                pd.DataFrame({"Day": [0], "Load": [0]})
            )[0]

            sim_energy = base_energy * (load_level / 100) * (shift_hours / 8) * (machines / 3)

            cost = sim_energy * 8

            co2 = sim_energy * 0.000233

            c1, c2, c3 = st.columns(3)

            c1.metric("Simulated Energy", f"{sim_energy:.2f} kWh")
            c2.metric("Estimated Cost", f"₹{cost:.2f}")
            c3.metric("CO2 Output", f"{co2:.3f} tCO2")

            sim_data = pd.DataFrame({
                "Metric": ["Energy", "Cost", "CO2"],
                "Value": [sim_energy, cost, co2]
            })

            fig = px.bar(
                sim_data,
                x="Metric",
                y="Value",
                template="plotly_dark",
                color="Metric"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

            efficiency = max(0, 100 - (sim_energy / base_energy * 100))

            st.progress(int(min(100, efficiency)))

            if efficiency >= 20:
                sim_status = "🟢 High Efficiency Scenario"
                sim_color = "#00E676"
            elif efficiency >= 5:
                sim_status = "🟡 Moderate Efficiency Scenario"
                sim_color = "#FFD166"
            else:
                sim_status = "🔴 Heavy Consumption Scenario"
                sim_color = "#FF6B6B"

            st.markdown(f"""
            <div class="insight-card">
                <div style="font-size:15px; font-weight:600; color:#00F5D4;">🧠 Simulation Insight</div>
                <div style="margin-top:8px; font-size:14px; color:#D8F3FF;">
                    Current simulated operating condition indicates: 
                    <span style="color:{sim_color}; font-weight:700;">{sim_status}</span><br><br>
                    Energy Efficiency Score: <b>{efficiency:.1f}/100</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------- HEATMAP ----------------

elif navigation == "Spatial Heatmap":

    st.title("Factory Anomaly Detection")

    st.markdown("""
    <div class="hero-card">
        <h2 style="margin:0; color:#00F5D4;">🌡️ Spatial Energy & Anomaly Layer</h2>
        <p style="margin:8px 0 0 0; color:#CFE8EF;">
            Detect abnormal machine behavior and visualize energy concentration across the factory floor.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = df_iiot.copy()

    df["Anomaly"] = iso_model.predict(df[iiot_features])

    anomaly_count = (df["Anomaly"] == -1).sum()
    total_records = len(df)
    anomaly_pct = (anomaly_count / total_records) * 100

    st.markdown(f"""
    <div class="insight-card">
        <div style="font-size:15px; font-weight:600; color:#00F5D4;">🚨 Anomaly Summary</div>
        <div style="margin-top:8px; font-size:14px; color:#D8F3FF;">
            Detected <b>{anomaly_count}</b> anomalous records out of <b>{total_records}</b> observations
            ({anomaly_pct:.2f}% anomaly rate).
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    fig.update_layout(
        title="Factory Energy Heatmap",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------- NLP ----------------

elif navigation == "NLP Alert System":

    st.title("Maintenance Log Classifier")

    st.markdown("""
    <div class="hero-card">
        <h2 style="margin:0; color:#00F5D4;">📝 Maintenance Intelligence Console</h2>
        <p style="margin:8px 0 0 0; color:#CFE8EF;">
            Classify operator maintenance logs into priority levels using lightweight NLP.
        </p>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_input("Enter maintenance log", "Machine vibration detected")

    if st.button("Classify"):

        vec_input = vec.transform([text])

        pred = clf.predict(vec_input)[0]

        if pred == "High Priority":
            badge = '<span class="status-badge" style="background:rgba(255,107,107,0.10); color:#FF6B6B; border-color:#FF6B6B;">🔴 High Priority</span>'
            message = "Urgent maintenance required! Immediate inspection is recommended."
        elif pred == "Routine":
            badge = '<span class="status-badge" style="background:rgba(255,209,102,0.10); color:#FFD166; border-color:#FFD166;">🟡 Routine</span>'
            message = "Routine maintenance logged successfully."
        else:
            badge = '<span class="status-badge" style="background:rgba(176,190,197,0.10); color:#B0BEC5; border-color:#B0BEC5;">⚪ Spam / Ignore</span>'
            message = "This message appears non-actionable and can be ignored."

        st.markdown(f"""
        <div class="insight-card">
            <div style="font-size:15px; font-weight:600; color:#00F5D4;">📌 Classification Result</div>
            <div style="margin-top:10px;">{badge}</div>
            <div style="margin-top:12px; font-size:14px; color:#D8F3FF;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if pred == "High Priority":
            st.error("Urgent maintenance required!")

        elif pred == "Routine":
            st.success("Routine maintenance logged.")

        else:
            st.warning("Ignored message.")
