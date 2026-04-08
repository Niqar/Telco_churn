# app.py (minimal version for selected_original features)

import streamlit as st
import pandas as pd
import joblib

# ── Page config ─────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)

st.title("Telco Customer Churn Predictor")

# ── Load trained small pipeline ─────────
@st.cache_resource
def load_model():
    return joblib.load("best_pipeline_small.pkl")  # Small pipeline

pipeline = load_model()

# ── Layout ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile & Services")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

with col2:
    st.subheader("Billing & Contract")

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.slider(
        "Total Charges ($)", 0.0, 9000.0, float(monthly_charges * tenure)
    )
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

# ── Predict button ───────────────────────
if st.button("Predict Churn Risk"):

    # Only the selected 8 columns
    input_data = pd.DataFrame([{
        "Tenure": tenure,
        "TotalCharges": total_charges,
        "InternetService": internet,
        "StreamingTV": streaming_tv,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "MultipleLines": multiple_lines,
        "StreamingMovies": streaming_movies
    }])

    # Prediction
    prob = pipeline.predict_proba(input_data)[0][1]
    pct = int(prob * 100)

    if prob >= 0.5:
        st.error(f"🔴 High churn risk ({pct}%)")
    else:
        st.success(f"🟢 Low churn risk ({pct}%)")