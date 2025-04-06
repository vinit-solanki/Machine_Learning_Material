import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("\U0001F4CA Customer Churn Prediction App")
st.markdown("""
This application predicts **customer churn** using a machine learning model.
""")

# Input form
with st.form("churn_form"):
    st.header("Customer Details")

    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    total_purchase = st.number_input("Total Purchase Amount", min_value=0.0, value=10000.0)
    account_manager = st.selectbox("Has Account Manager?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    years = st.slider("Years as Customer", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    num_sites = st.number_input("Number of Sites Used", min_value=0, value=8)

    submitted = st.form_submit_button("Predict Churn")

# Prediction logic
if submitted:
    input_data = pd.DataFrame({
        'Age': [age],
        'Total_Purchase': [total_purchase],
        'Account_Manager': [account_manager],
        'Years': [years],
        'Num_Sites': [num_sites]
    })

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    if prediction == 1 or prediction == 2:  # assuming 2 was the original positive class
        st.error(f"\U0001F6AB The customer is likely to churn! (Confidence: {prob:.2f})")
    else:
        st.success(f"\U0001F44D The customer is likely to stay. (Confidence: {1 - prob:.2f})")
