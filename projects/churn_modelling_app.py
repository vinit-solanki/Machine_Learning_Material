import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("best_random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title and description
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Prediction App")
st.markdown("Enter customer details to predict if they are likely to churn.")

# User inputs
customer_id = st.number_input("Customer ID", 10000000, 20000000, value=15634602)
credit_score = st.slider("Credit Score", 300, 850, value=650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, value=35)
tenure = st.slider("Tenure (years with bank)", 0, 10, value=3)
balance = st.number_input("Balance", 0.0, 300000.0, value=50000.0, step=1000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.radio("Has Credit Card?", [0, 1])
is_active_member = st.radio("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 10000.0, 200000.0, value=50000.0, step=1000.0)

# Mapping inputs
geo_map = {"France": 0, "Spain": 1, "Germany": 2}
gender_map = {"Male": 0, "Female": 1}

# Final input for model
input_data = pd.DataFrame([[
    customer_id,
    credit_score,
    geo_map[geography],
    gender_map[gender],
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary
]], columns=[
    'CustomerId', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
])

# Scale inputs (standard scaling — assuming model was trained on scaled inputs)
scaler = StandardScaler()
# NOTE: Fit scaler on dummy data with same structure (or load real scaler if saved)
scaler.fit(pd.DataFrame([[1]*11], columns=input_data.columns))  # dummy fit
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("🚀 Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction > 0.45 :
        st.error(f"⚠️ Customer is likely to churn. (Confidence: {probability:.2%})")
        st.markdown("""
        **Suggestions**:
        - Offer personalized retention strategies.
        - Analyze account dissatisfaction triggers.
        """)
    else:
        st.success(f"✅ Customer is likely to stay. (Churn Probability: {probability:.2%})")
        st.markdown("""
        **Recommendations**:
        - Explore cross-selling opportunities.
        - Continue monitoring engagement.
        """)

# Optional: Show input and scaled data
with st.expander("📊 Show Input Data"):
    st.write(input_data)

with st.expander("📈 Scaled Data (for model)"):
    st.write(pd.DataFrame(scaled_input, columns=input_data.columns))
