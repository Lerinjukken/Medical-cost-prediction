
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import accuracy_score, r2_score

# Load your trained model
scaler = pickle.load(open('models/scalar1_.sav1', 'rb'))
data = pickle.load(open('models/XGBREGERSSOR_with_metrics.pkl', 'rb'))
model = data["model"]
r2 = data["r2"]

encoder_city=pickle.load(open('models/city_.sav', 'rb'))
encoder_gender=pickle.load(open('models/gender_.sav', 'rb'))
encoder_insurance=pickle.load(open('models/insurance1_.sav', 'rb'))
encoder_smoker=pickle.load(open('models/smoker_.sav', 'rb'))

st.set_page_config(
    page_title="MEDCAL | Medical Cost Prediction",
    page_icon="🩺",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1.5rem;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
}
.metric-card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.footer {
    text-align: center;
    color: #8b949e;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("## 🩺 **MEDCAL**")
st.caption("Medical Cost Prediction")

st.markdown("---")

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.2, 2])

# ---------------- LEFT INFO PANEL ----------------
with left:
    st.markdown("""
    <div class="card">
        <h4>Welcome to MEDCAL</h4>
        <p>
        MEDCAL is your intuitive tool for predicting medical costs based on key
        demographic and health indicators. Leveraging advanced machine learning
        models, we provide estimates to help you understand potential healthcare
        expenses.
        </p>
        <p>
        Simply input your details into the form, click <b>Predict Cost</b>,
        and receive an estimated cost.
        </p>
    </div>
    """, unsafe_allow_html=True)
    img = Image.open('assets/medical_insurance-removebg-preview.png')
    st.image(img, width=1000)

img=Image.open('assets/medical_insurance-removebg-preview.png')
# Input fields
with right:
    st.header("🧑‍⚕️ Input Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 0, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.text_input("BMI","")
        smoker = st.selectbox("Smoker", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    with col2:
        stress_level = st.slider("Stress Level (1=Low, 10=High)", 1, 10, 5)
        doctor_visits = st.slider("Doctor Visits per Year", 0, 50, 2)
        hospital_admissions = st.slider("Hospital Admissions", 0, 10, 0)
        medication_count = st.slider("Number of Medications", 0, 20, 1)
        insurance_coverage_pct=st.text_input("Enter the insurance coverage pack inn $","")
        insurance_type = st.selectbox("Insurance Type", ["Government", "Private"])
        city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
        previous_year_cost = st.text_input("Previous Year Medical Cost in $", "")
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Cost", use_container_width=False)
#tranform categorical input
city=encoder_city.transform([city_type])[0]
gen=encoder_gender.transform([gender])[0]
insurance=encoder_insurance.transform([insurance_type])[0]
smok=encoder_smoker.transform([smoker])[0]
features = [
    age,
    gen,
    bmi,
    smok,
    1 if diabetes == "Yes" else 0,
    1 if hypertension == "Yes" else 0,
    1 if heart_disease == "Yes" else 0,
    stress_level,
    doctor_visits,
    hospital_admissions,
    medication_count,
    insurance,
    insurance_coverage_pct,
    city,
    previous_year_cost
]
# Predict button
with right:
    if predict_btn:
        res = model.predict(scaler.transform([features]))
        st.subheader("💰 Prediction Results")
        st.metric("Predicted Cost", f"${res[0]}")



        with left:
            st.subheader("📊 Prediction score")
            import plotly.graph_objects as go


            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r2,
                gauge={'axis': {'range': [0, 1]},'bar': {'color': "blue"}},

            ))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔗 [About](#) | [Documentation](#) | [Contact](#)")



