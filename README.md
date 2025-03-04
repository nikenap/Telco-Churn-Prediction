# Telco Customer Churn Prediction

## Project Overview
This Capstone Project explores the application of machine learning algorithms to predict customer churn using the Telco Customer dataset. The project focuses on understanding how various factors, including demographic details (e.g., gender, dependents), customer account information (e.g., tenure, monthly charges), and the services each customer has signed up for (e.g., phone service, internet service), influence churn behavior. Various models were employed, including Logistic Regression, LightGBM, K-Nearest Neighbors, XGBoost, Random Forest, and Decision Tree. The analysis emphasizes optimizing the F2-score to minimize missed churners, as losing high-risk customers can be costly. This project highlights the potential of machine learning in driving data-driven retention strategies to reduce customer churn and maximize revenue.

## Context
Customer churn, or attrition, is the loss of clients, which is a critical concern for industries reliant on long-term customer relationships, such as telecommunications. High churn rates lead to revenue losses and increased costs for acquiring new customers. Therefore, understanding churn factors is vital. By analyzing and predicting at-risk customers, companies can implement proactive measures like personalized offers, better customer service, or loyalty programs to reduce churn and maximize long-term customer value.

## Project goals
1. Analyze the dataset to understand the key factors affecting costumer churn behaviour
2. Develop predictive models to predict costumer churn with high accuracy
3. Evaluate and compare the performance of different machine learning models

## Workflow
1. **Data cleaning**: 
    - Addressed missing values, inconsistent formats, duplicate data, and outliers to ensure data quality.
2. **Data Exploration**:
    - Analyzed data distribution, correlations, and multicollinearity to understand relationships and patterns.
3. **Feature Engineering**: 
    - Created new features (e.g., tenure_binned) to enhance the model’s predictive power.
    - Changed categorical to binary values (binary encoding)
4. **Model Testing and Optimization**:
    - Tested multiple classification models and optimized the best-performing model using GridSearchCV for hyperparameter tuning.
5. **Evaluation and Recommendations**:
    - Evaluated the models using various metrics, with the F2-score guiding the final decision to prioritize minimizing missed churners.

## Key Results
1. **Model Performance**
    - Best model: XGboost
    - F2-score: 0.71
    - Recall: 0.87
    - Precision: 0.41
    - Accuracy: 0.64
2. **Feature Importance Highlights**
    - `Contract_Two year`: “Contract_Two year” is the most impactful feature affecting churn prediction, suggesting that longer contract durations strongly reduce the likelihood of customer churn.
    _ `tenure_binned`: Various tenure categories (e.g., “61-72 months,” “25-48 months”) significantly influence the model’s output, indicating that customers with longer tenures are less likely to churn.
    - `InternetService_Fiber optic`: Customers with “Fiber optic” internet service are more likely to churn compared to other types of internet service.

## Business Recommendation
1. Focus on early retention efforts for new customers by offering welcome incentives and personalized support during the first 12 months.
2. ffer targeted discounts or tiered pricing for customers with high monthly charges to address price sensitivity.
3. romote long-term contracts with attractive incentives, as customers on longer-term plans are less likely to churn.
4. Upsell value-added services like online security, online backup, and tech support, as these services contribute to higher retention.
5. Address common complaints or concerns related to fiber optic internet service to reduce churn in that customer segment.
6. Leverage entertainment services like streaming TV and movies as part of retention strategies by bundling them into attractive packages.

## Files
- Notebook: final_project.ipynb
- Best Model: model.pkl
- Streamlit: stream.py

## Prediction Dashboard and Apps
- Visit the dashboard here
- Visit the app here

