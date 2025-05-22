import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and feature names
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('features.pkl')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("🔍 Customer Churn Prediction App")
st.write("Fill in the customer details below to predict churn likelihood.")

# User input section
def user_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 75.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    data = {
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        f'gender_{gender}': 1,
        f'InternetService_{internet}': 1,
        f'OnlineSecurity_{online_security}': 1,
        f'OnlineBackup_{online_backup}': 1,
        f'DeviceProtection_{device_protection}': 1,
        f'TechSupport_{tech_support}': 1,
        f'StreamingTV_{streaming_tv}': 1,
        f'StreamingMovies_{streaming_movies}': 1,
        f'Contract_{contract}': 1,
        f'PaymentMethod_{payment}': 1,
        f'MultipleLines_{multiple_lines}': 1
    }

    input_df = pd.DataFrame([data])

    # Add missing columns
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match training
    input_df = input_df[feature_names]

    return input_df

# Main prediction section
input_df = user_input()

if st.button("Predict Churn"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Prediction Result")
    st.write("Churn Prediction:", "**Yes**" if prediction == 1 else "**No**")
    st.write(f"Churn Probability: **{prediction_proba:.2%}**")
