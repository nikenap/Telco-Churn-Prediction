import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import plotly.express as px
import plotly.graph_objects as go

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
    monthly_charges = st.slider('Select Monthly Charges', min_value=0.0, max_value=120.0, value=50.0, step=0.5)
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

    # Model performance metrics
    st.subheader("Model Performance Metrics")
    # Assuming y_test and X_test are available for evaluation
    if st.button("Show Performance Metrics"):
        y_test = pd.read_csv('y_test.csv')  # Load y_test
        X_test = pd.read_csv('X_test.csv')  # Load X_test
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2_score = fbeta_score(y_test, y_pred, beta=2)
        
        st.write(f"**F2 Score:** {f2_score:.4f}")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")

    # Feature importance visualization (for tree-based models)
    if st.button("Show Feature Importance"):
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            feature_names = X_test.columns
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

            fig = px.bar(feature_df, x='Importance', y='Feature', title='Feature Importance', orientation='h')
            st.plotly_chart(fig)
        else:
            st.write("Feature importance is not available for this model.")

    # Customizable themes
    st.sidebar.header("Customize Theme")
    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark", "Streamlit Default"])

    if theme == "Light":
        st.markdown(
            """
            <style>
            body { background-color: #f0f0f0; color: #000; }
            </style>
            """, unsafe_allow_html=True
        )
    elif theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #2e2e2e; color: #fff; }
            </style>
            """, unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
