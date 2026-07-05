import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="centered"
)

# ------------------------------------------------------------
# Load model & pipeline components (cached so it only loads once)
# ------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model = joblib.load("xgb_air_quality_model.pkl")
    le_city = joblib.load("label_encoder_city.pkl")
    le_country = joblib.load("label_encoder_country.pkl")
    le_target = joblib.load("label_encoder_target.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    unique_cities = joblib.load("unique_cities.pkl")
    unique_countries = joblib.load("unique_countries.pkl")
    return model, le_city, le_country, le_target, feature_columns, unique_cities, unique_countries

model, le_city, le_country, le_target, feature_columns, unique_cities, unique_countries = load_pipeline()

# City -> Country mapping, built directly from what the encoders were fitted on
# (dataset confirms this is a strict 1:1 mapping)
CITY_COUNTRY_MAP = {
    "Beijing": "China",
    "Cairo": "Egypt",
    "Delhi": "India",
    "London": "UK",
    "Los Angeles": "USA",
    "New York": "USA",
    "Paris": "France",
    "Sydney": "Australia",
    "São Paulo": "Brazil",
    "Tokyo": "Japan",
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🌍 Air Quality Predictor")
st.markdown(
    "Predict whether air quality is **Safe** or **Unsafe** based on pollutant "
    "concentrations and weather conditions, using a trained XGBoost model."
)
st.divider()

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.subheader("Input Conditions")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", options=unique_cities, index=0)
    country = CITY_COUNTRY_MAP.get(city, "Unknown")
    st.text_input("Country", value=country, disabled=True)

with col2:
    month_name = st.selectbox("Month", options=MONTH_NAMES, index=0)
    month = MONTH_NAMES.index(month_name) + 1

st.markdown("**Pollutant Levels**")
p1, p2, p3 = st.columns(3)
with p1:
    pm25 = st.slider("PM2.5 (µg/m³)", min_value=5.0, max_value=250.0, value=126.0, step=0.1)
    so2 = st.slider("SO2 (ppb)", min_value=2.0, max_value=50.0, value=26.0, step=0.1)
with p2:
    pm10 = st.slider("PM10 (µg/m³)", min_value=10.0, max_value=300.0, value=155.0, step=0.1)
    co = st.slider("CO (ppm)", min_value=0.1, max_value=10.0, value=5.0, step=0.01)
with p3:
    no2 = st.slider("NO2 (ppb)", min_value=5.0, max_value=100.0, value=53.0, step=0.1)
    o3 = st.slider("O3 (ppb)", min_value=10.0, max_value=200.0, value=105.0, step=0.1)

st.markdown("**Weather Conditions**")
w1, w2, w3 = st.columns(3)
with w1:
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=40.0, value=15.0, step=0.1)
with w2:
    humidity = st.slider("Humidity (%)", min_value=10, max_value=90, value=51)
with w3:
    wind_speed = st.slider("Wind Speed (m/s)", min_value=0.5, max_value=15.0, value=7.8, step=0.1)

st.divider()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Predict Air Quality", type="primary", use_container_width=True):

    # Encode City and Country using the fitted label encoders
    city_encoded = le_city.transform([city])[0]
    country_encoded = le_country.transform([country])[0]

    # Assemble the feature row in the EXACT order used during training
    input_dict = {
        "City": city_encoded,
        "Country": country_encoded,
        "PM25": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "Temperature": temperature,
        "Humidity": humidity,
        "WindSpeed": wind_speed,
        "Month": month,
    }
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_label = le_target.inverse_transform([prediction])[0]

    safe_idx = list(le_target.classes_).index("Safe")
    unsafe_idx = list(le_target.classes_).index("Unsafe")
    prob_safe = probabilities[safe_idx]
    prob_unsafe = probabilities[unsafe_idx]

    st.subheader("Prediction Result")

    if predicted_label == "Safe":
        st.success(f"✅ Predicted Air Quality: **Safe**")
    else:
        st.error(f"⚠️ Predicted Air Quality: **Unsafe**")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Probability: Safe", f"{prob_safe * 100:.1f}%")
    with res_col2:
        st.metric("Probability: Unsafe", f"{prob_unsafe * 100:.1f}%")

    st.progress(float(prob_unsafe))
    st.caption("Progress bar shows the model's confidence toward 'Unsafe'.")

    with st.expander("View input sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption("Model: XGBoost Classifier · Deployment stage demo")
