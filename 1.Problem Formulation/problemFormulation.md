# Customer Churn Prediction – Problem Formulation



---



## 1. Business Problem

Customer churn (i.e., customers leaving the service) is a critical issue for subscription-based businesses such as telecom companies. Retaining existing customers is more cost-effective than acquiring new ones. The business problem is to predict whether a customer will churn based on demographic, service usage, and billing information, enabling proactive retention strategies.



---



## 2. Key Business Objectives

- Predict churn probability for each customer.

- Identify churn drivers (e.g., contract type, monthly charges, tenure).

- Enable targeted retention campaigns by segmenting high-risk customers.

- Improve customer lifetime value (CLV) through reduced churn rates.



---



## 3. Data Sources and Attributes

**Customer Demographics**: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`.



**Services Subscribed**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.



**Account Information**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`.



**Charges**: `MonthlyCharges`, `TotalCharges`.



**Target Variable**: `Churn` (Yes/No).



Data sources can be:

- Internal CRM systems (customer details, services).

- Billing systems (charges, payment method).

- Customer support systems (service usage, support tickets).



---



## 4. Expected Outputs from the Pipeline

**Clean Datasets for EDA**:

- Handle missing values (e.g., `TotalCharges` inconsistencies).

- Encode categorical variables.

- Normalize numerical features.



**Transformed Features for ML**:

- One-hot encoding for categorical variables (e.g., `PaymentMethod`, `Contract`).

- Feature scaling for numerical attributes (`MonthlyCharges`, `tenure`, `TotalCharges`).

- Derived features (e.g., average charges per month = `TotalCharges` / `tenure`).



**Deployable Model**:

- Train ML models (Logistic Regression, Random Forest, XGBoost, etc.).

- Select best-performing model.

- Save model for deployment (e.g., via Flask/FastAPI for real-time churn prediction).



---



## 5. Evaluation Metrics

To assess model performance:

- **Accuracy** – overall prediction correctness.

- **Precision** – proportion of predicted churners who actually churn.

- **Recall (Sensitivity)** – ability to correctly identify churners.

- **F1-score** – balance between precision and recall.

- **ROC-AUC** – ability to discriminate churn vs non-churn.



*Business focus*: Recall and AUC are more important since missing a potential churner is more costly.



---



## 6. Deliverables

**Documentation (PDF/Markdown)**:

- Business problem, objectives, and dataset description.

- Data cleaning and feature engineering process.

- Model development pipeline and results.

- Evaluation metrics and insights.



**Code (Jupyter Notebook / Python scripts)**:

- Data ingestion, cleaning, transformation.

- Model training and evaluation.



**Final Model Deployment**:

- Export trained ML model.

- Deploy as API/endpoint for churn prediction.





