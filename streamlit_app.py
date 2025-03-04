import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model (including preprocessor)
with open('model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Define the Streamlit app
def main():
    st.title("Telco Customer Churn Prediction")
    st.write("Enter customer information to predict churn probability.")

    # Collect user input for the key features with Yes/No options
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure_binned = st.selectbox("Tenure Binned", ["0-12 months", "13-24 months", "25-48 months", "49-60 months", "61-72 months"])
    monthly_charges = st.slider(
    'Select Monthly Charges',
    min_value=0.0,
    max_value=120.0,
    value=50.0,   # Default value
    step=0.5)
    st.write(f"Selected Monthly Charges: ${monthly_charges}")

    # Convert input to DataFrame with Yes/No to Binary Conversion
    input_data = pd.DataFrame({
        'gender': [1 if gender == "Male" else 0],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],  
        'Partner': [1 if partner == "Yes" else 0],
        'Dependents': [1 if dependents == "Yes" else 0],
        'PhoneService': [1 if phone_service == "Yes" else 0],
        'MultipleLines': [1 if multiple_lines == "Yes" else 0],
        'InternetService': [internet_service],
        'OnlineSecurity': [1 if online_security == "Yes" else 0],
        'OnlineBackup': [1 if online_backup == "Yes" else 0],
        'DeviceProtection': [1 if device_protection == "Yes" else 0],
        'TechSupport': [1 if tech_support == "Yes" else 0],
        'StreamingTV': [1 if streaming_tv == "Yes" else 0],
        'StreamingMovies': [1 if streaming_movies == "Yes" else 0],
        'Contract': [contract],
        'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
        'PaymentMethod': [payment_method],
        'tenure_binned': [tenure_binned],
        'MonthlyCharges': [monthly_charges]
    })

    # Make prediction directly with the model (includes preprocessor)
    churn_proba = best_model.predict_proba(input_data)[:, 1][0]
    churn_prediction = "Yes" if churn_proba >= 0.39 else "No"

    # Display results
    st.write(f"### Churn Prediction: **{churn_prediction}**")
    st.write(f"Churn Probability: **{churn_proba:.2%}**")

if __name__ == "__main__":
    main()
