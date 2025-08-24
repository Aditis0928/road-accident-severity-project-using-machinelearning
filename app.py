import streamlit as st
import joblib, json
import pandas as pd
from pathlib import Path

# ----------------- PAGE SETTINGS -----------------
st.set_page_config(
    page_title="🚦 Road Accident Severity Prediction",
    page_icon="🚗",
    layout="centered"
)

# ----------------- LOAD MODEL -----------------
MODEL_PATH = Path("casualty_model.pkl")
COLS_PATH = Path("model_columns.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_columns():
    try:
        cols = list(load_model().feature_names_in_)
        if cols:
            return cols
    except Exception:
        pass
    return json.loads(COLS_PATH.read_text())

model = load_model()
model_columns = load_columns()

# ----------------- TITLE & DESCRIPTION -----------------
st.title("🚦 Road Accident Severity Prediction")
st.markdown(
    """
    Enter accident details below and click **Predict**  
    to see the severity level of the accident.
    """
)

# ----------------- INPUT FORM -----------------
with st.form("accident_form"):
    st.subheader("Accident Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Driver Age", min_value=16, max_value=100, value=30)
        vehicle_type = st.selectbox("🚘 Vehicle Type", ["Car", "Bike", "Bus", "Truck"])
    
    with col2:
        weather = st.selectbox("☁️ Weather Condition", ["Clear", "Rainy", "Foggy", "Snowy"])
        road_type = st.selectbox("🛣 Road Type", ["Highway", "City Road", "Rural Road"])

    submitted = st.form_submit_button("🔮 Predict Severity")

# ----------------- PREDICTION -----------------
if submitted:
    # Raw dataframe
    raw = pd.DataFrame(
        [[age, weather, vehicle_type, road_type]],
        columns=["Age", "Weather", "Vehicle_Type", "Road_Type"]
    )

    # Encoding
    enc = pd.get_dummies(raw)
    aligned = enc.reindex(columns=model_columns, fill_value=0)

    # Prediction
    pred = model.predict(aligned)[0]

    # Map numeric prediction to text
    severity_mapping = {
        0: "🔴 Fatal Injury",
        1: "🟡 Serious Injury",
        2: "🟢 Slight Injury"
    }
    severity_text = severity_mapping.get(int(pred), "Unknown")

    st.success(f"**Predicted Accident Severity: {severity_text}**")

    with st.expander("See processed features"):
        st.write(aligned)
