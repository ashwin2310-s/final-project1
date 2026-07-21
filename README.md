# final-project1

# 🏦 Bank Term Deposit Subscription Prediction

A Machine Learning project that predicts whether a bank customer will subscribe to a **term deposit** based on demographic, financial, and previous marketing campaign data. The project includes data preprocessing, feature engineering, model comparison, hyperparameter tuning, model stacking, explainable AI (SHAP), and deployment using Streamlit.

---

## 📌 Project Overview

Banks conduct marketing campaigns to promote term deposits, but contacting every customer is expensive and inefficient. This project uses machine learning to identify customers who are more likely to subscribe, helping banks improve campaign effectiveness and reduce marketing costs.

---

## 🎯 Objective

The objective of this project is to build a machine learning model that predicts whether a customer will subscribe to a bank term deposit. The project focuses on:

- Data preprocessing and cleaning
- Feature engineering
- Multiple model comparison
- Hyperparameter tuning
- Ensemble learning
- Explainable AI using SHAP
- Streamlit deployment

---

## 📂 Domain

- Financial Services
- Banking
- Predictive Analytics
- Machine Learning

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- SHAP
- Joblib

---

## 📁 Dataset

Dataset: Bank Marketing Dataset

The dataset contains customer demographic details, financial information, and previous marketing campaign details.

Target Variable:

```
y

Yes → Customer subscribed
No  → Customer did not subscribe
```

---

## 📊 Features

Some important features include:

- Age
- Job
- Marital Status
- Education
- Balance
- Housing Loan
- Personal Loan
- Contact Type
- Campaign
- Previous
- Duration
- Poutcome

Target:

```
y (Subscription)
```

---

# 🔄 Project Workflow

## Step 1 – Load Dataset

- Upload bank-full.csv
- Read data using Pandas

---

## Step 2 – Data Cleaning

- Remove duplicate records
- Handle missing values
- Convert target variable
- Identify numerical and categorical columns

---

## Step 3 – Feature Engineering

New features created:

- Age × Balance
- Duration × Balance
- Age × Duration

These features improve model learning by capturing relationships between customer characteristics.

---

## Step 4 – Data Preprocessing

### Numerical Features

- Median Imputation
- StandardScaler

### Categorical Features

Education

- Ordinal Encoding

Remaining Categorical Features

- One-Hot Encoding

Pipeline used:

- ColumnTransformer
- Pipeline

---

## Step 5 – Train-Test Split

Dataset is divided into

- 80% Training Data
- 20% Testing Data

using Stratified Split.

---

## Step 6 – Machine Learning Models

Models compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Naive Bayes
- Gradient Boosting
- Support Vector Machine (SVM)

---

## Step 7 – Hyperparameter Tuning

Random Forest is optimized using

RandomizedSearchCV

Best parameters are selected using Cross Validation.

---

## Step 8 – Stacking

Base Models

- Decision Tree
- Random Forest

Meta Model

- Logistic Regression

Purpose:

Improve prediction performance.

---

## Step 9 – Model Evaluation

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

## Step 10 – Explainable AI

SHAP (SHapley Additive exPlanations)

Used to:

- Explain model predictions
- Identify important features
- Improve model transparency

---

## Step 11 – Save Model

Best trained model is saved as

```
best_model.pkl
```

using Joblib.

---

## Step 12 – Streamlit Deployment

Features available in the application:

- Upload dataset
- Preview dataset
- Compare multiple models
- Hyperparameter tuning
- Train stacking classifier
- Generate SHAP explanations
- Download trained model
- Predict new customer subscription

---

# 📈 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Classification Report
- Confusion Matrix
- ROC Curve

---

# 📊 Feature Engineering

Three engineered features were added:

- Age × Balance
- Duration × Balance
- Age × Duration

These help the model learn hidden relationships in the banking data.

---

# 🔤 Encoding Techniques

## Ordinal Encoding

Used for:

- Education

Reason:

Education has a natural order.

```
Primary
↓

Secondary
↓

Tertiary
```

---

## One-Hot Encoding

Used for:

- Job
- Marital Status
- Housing
- Loan
- Contact
- Month
- Poutcome

Reason:

These features do not have a natural order.

---

# 🚀 Machine Learning Pipeline

The project uses Scikit-learn Pipeline to automate:

- Missing value handling
- Encoding
- Scaling
- Model Training
- Prediction

Benefits:

- Cleaner code
- Reusable workflow
- Prevents data leakage
- Easy deployment

---

# 📌 Business Benefits

- Improve customer targeting
- Reduce marketing costs
- Increase campaign success rate
- Improve customer conversion
- Data-driven decision making
- Explainable AI predictions

---

# 📷 Streamlit Interface

The application includes:

✔ Upload Dataset

✔ Dataset Preview

✔ Model Selection

✔ Hyperparameter Tuning

✔ SHAP Visualization

✔ Model Comparison

✔ Download Best Model

✔ Predict New Customer



shwins1274@gmail.com

---
