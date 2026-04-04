# ⚡ Eco-Smart Factory Hub  
## AI-Based Energy Consumption Optimization Model for Factories

An **AI-powered smart manufacturing dashboard** designed to analyze, predict, and optimize industrial energy consumption.  
This project combines **machine learning**, **anomaly detection**, **digital twin simulation**, and **NLP-based maintenance intelligence** to help factories improve operational efficiency, reduce energy waste, and lower carbon emissions.

---

## 📌 Project Overview

Industrial facilities consume a significant amount of electrical energy during day-to-day production.  
In many factories, inefficient machine usage, poor scheduling, and undetected equipment abnormalities can lead to:

- Excessive energy consumption  
- Increased electricity costs  
- Higher carbon emissions  
- Reduced operational efficiency  
- Unexpected machine downtime  

To address these challenges, this project introduces the **Eco-Smart Factory Hub**, an intelligent analytics platform built with **Streamlit** and **Machine Learning**.

The system provides a centralized dashboard that allows factory operators and decision-makers to:

- Monitor enterprise-level energy performance
- Forecast future energy demand and CO₂ emissions
- Simulate factory operations using a digital twin model
- Detect machine-level anomalies from IoT sensor data
- Classify maintenance logs based on urgency using NLP

This solution demonstrates how **AI can support sustainable and data-driven manufacturing operations**.

---

## 🚀 Key Features

### 1. Executive Energy Dashboard
A premium dashboard for monitoring factory-wide operational metrics in real time.

**Includes:**
- Current energy usage (kWh)
- CO₂ emissions (tCO₂)
- Estimated electricity cost
- KPI delta tracking (increase/decrease vs previous reading)
- Color-coded trend indicators
- Energy consumption trend visualization

**Purpose:**  
Provides management-level visibility into energy performance and operational cost trends.

---

### 2. AI Energy Forecasting
Uses a **Random Forest Regressor** to predict future energy-related metrics based on operational conditions.

**Predicts:**
- Energy consumption
- CO₂ emissions
- Estimated cost

**Inputs:**
- Day of the week
- Load type

**Additional Enhancement:**
- Forecast confidence range for predicted Energy and CO₂

**Purpose:**  
Supports proactive energy planning and better resource allocation.

---

### 3. Factory Digital Twin Simulator
A simplified **digital twin simulation module** that estimates factory performance under different operating scenarios.

**Simulation Parameters:**
- Factory load (%)
- Shift duration (hours)
- Number of active machines

**Outputs:**
- Simulated energy consumption
- Estimated operating cost
- Simulated CO₂ output
- Energy efficiency score

**Purpose:**  
Allows engineers to test different production conditions before applying them in real operations.

---

### 4. Machine Anomaly Detection
Detects abnormal machine behavior using **Isolation Forest** on IoT sensor data.

**Monitored Features:**
- Energy consumption (kWh)
- Current (A)
- Machine utilization (%)

**Purpose:**  
Helps identify unusual machine activity that may indicate:
- Energy inefficiency
- Faults
- Overload conditions
- Preventive maintenance needs

---

### 5. NLP Maintenance Log Classifier
A lightweight **Natural Language Processing (NLP)** model that classifies maintenance logs into priority levels.

**Categories:**
- High Priority
- Routine
- Spam / Ignored messages

**Model Used:**
- CountVectorizer
- Multinomial Naive Bayes

**Purpose:**  
Helps maintenance teams quickly identify urgent operational issues.

---

## 🧠 Machine Learning Models Used

This project integrates multiple AI/ML models for different industrial intelligence tasks:

### 1. Random Forest Regressor
Used for:
- Energy consumption forecasting
- CO₂ emission prediction

### 2. Isolation Forest
Used for:
- Unsupervised anomaly detection in machine-level IoT sensor data

### 3. Multinomial Naive Bayes
Used for:
- Maintenance log classification (NLP-based alert prioritization)

---

## 🛠️ Technologies Used

### Programming & Framework
- **Python**
- **Streamlit**

### Data Processing
- **Pandas**
- **NumPy**

### Machine Learning
- **Scikit-learn**

### Data Visualization
- **Plotly Express**

---

## 📂 Datasets Used

This project uses two datasets:

1. **Steel Industry Energy Consumption Dataset**  
   Used for:
   - Energy forecasting
   - CO₂ prediction
   - Dashboard KPIs

2. **Energy Dataset**  
   Used for:
   - IoT sensor-based anomaly detection
   - Machine spatial heatmap analysis

---

## 📊 System Modules

The application is organized into the following major modules:

- **Executive Dashboard**
- **AI Forecast**
- **Digital Twin Simulator**
- **Spatial Heatmap / Anomaly Detection**
- **NLP Alert System**

---

## 🎯 Real-World Applications

This project can be applied in:

- Smart manufacturing plants
- Industrial energy management systems
- Sustainable production monitoring
- Factory operations analytics
- Predictive maintenance support systems
- ESG / carbon reduction monitoring platforms

---

## 🌱 Project Impact

The **Eco-Smart Factory Hub** supports the future of **Industry 4.0** and **sustainable manufacturing** by helping industries:

- Reduce unnecessary energy consumption
- Improve machine-level operational efficiency
- Lower electricity expenditure
- Minimize carbon emissions
- Detect abnormal equipment behavior early
- Make data-driven production decisions

---

## 💻 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
