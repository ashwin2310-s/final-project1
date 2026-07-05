import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# -----------------------------
# Load & Clean Dataset
# -----------------------------

DATA_PATH = "data/bank.csv"

# FIX: Added sep=";" to handle the semicolon-separated format
# Pass the file path as the first argument, and the semicolon separator as the second
df = pd.read_csv("C:/Users/ashwi/Downloads/bank+marketing (1)/bank/bank-full.csv", sep=";")

# Strip any accidental whitespace from column names
df.columns = df.columns.str.strip()

df = df.drop_duplicates()

TARGET = "y"

# Safety check
if TARGET not in df.columns:
    raise KeyError(
        f"Could not find the target column '{TARGET}' in your dataset. "
        f"The actual columns available are: {list(df.columns)}"
    )

if df[TARGET].dtype == 'O':
    df[TARGET] = df[TARGET].map({'yes': 1, 'no': 0})

# -----------------------------
# Feature Engineering
# -----------------------------
if {"age", "balance"}.issubset(df.columns):
    df["age_balance"] = df["age"] * df["balance"]

if {"duration", "balance"}.issubset(df.columns):
    df["duration_balance"] = df["duration"] * df["balance"]

if {"age", "duration"}.issubset(df.columns):
    df["age_duration"] = df["age"] * df["duration"]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------------
# Column Processing & Pipelines
# -----------------------------
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

education_order = [['unknown', 'primary', 'secondary', 'tertiary']]
ordinal_cols = [c for c in categorical_cols if c.lower() == "education"]
nominal_cols = [c for c in categorical_cols if c.lower() != "education"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=education_order,
                               handle_unknown="use_encoded_value",
                               unknown_value=-1))
])

# Fixed: set sparse_output=False so dense-only algorithms like GaussianNB work smoothly
nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("ord", ordinal_pipeline, ordinal_cols),
    ("nom", nominal_pipeline, nominal_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Model Evaluation Setup
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Optional third-party models
for lib_name, cls_name, tag in [("xgboost", "XGBClassifier", "XGBoost"), 
                                 ("lightgbm", "LGBMClassifier", "LightGBM"), 
                                 ("catboost", "CatBoostClassifier", "CatBoost")]:
    try:
        mod = __import__(lib_name)
        cls = getattr(mod, cls_name)
        if tag == "XGBoost":
            models[tag] = cls(eval_metric="logloss", random_state=42)
        elif tag == "LightGBM":
            models[tag] = cls(random_state=42, verbose=-1)
        else:
            models[tag] = cls(verbose=0, random_state=42)
    except ImportError:
        pass

results = []
for name, model in models.items():
    # Fixed: Unified pipeline execution (Naive Bayes no longer requires custom logic)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
    
    results.append([name, acc, pre, rec, f1, auc])

comparison = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
print("\n--- Model Baseline Comparison ---")
print(comparison.sort_values("Accuracy", ascending=False).to_string(index=False))

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10]
}

search = RandomizedSearchCV(
    rf_pipe,
    param_distributions=params,
    n_iter=5,
    cv=skf,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)

print("\n--- Hyperparameter Tuning Results ---")
print("Best Parameters:", search.best_params_)
print("Best CV Score:", search.best_score_)

# -----------------------------
# Stacking Classifier
# -----------------------------
estimators = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("dt", DecisionTreeClassifier(random_state=42))
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
stack_pipe = Pipeline([("prep", preprocessor), ("model", stack)])
stack_pipe.fit(X_train, y_train)

stack_pred = stack_pipe.predict(X_test)
print("\n--- Stacking Classifier Metrics ---")
print(classification_report(y_test, stack_pred))

# Visualizations
ConfusionMatrixDisplay.from_predictions(y_test, stack_pred)
plt.title("Stacking Classifier Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(stack_pipe, X_test, y_test)
plt.title("Stacking Classifier ROC Curve")
plt.show()

# -----------------------------
# Export & Interpretability
# -----------------------------
best_pipeline = search.best_estimator_
joblib.dump(best_pipeline, "best_model.pkl")
print("\nModel saved successfully as 'best_model.pkl'")

if SHAP_AVAILABLE:
    print("\nGenerating SHAP explanations...")
    rf_model = best_pipeline.named_steps["model"]
    
    # Process dataset through the preprocessor stage for SHAP input
    X_test_transformed = best_pipeline.named_steps["prep"].transform(X_test)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    # Fixed: Select class index [1] for binary classification outputs 
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:  
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, X_test_transformed)
else:
    print("\nSHAP is not installed. Skipping interpretability plots.")

# -----------------------------
# Prediction Infrastructure
# -----------------------------
def predict_new_customer(customer_dataframe):
    prediction = best_pipeline.predict(customer_dataframe)[0]
    probability = best_pipeline.predict_proba(customer_dataframe)[0][1] if hasattr(best_pipeline, "predict_proba") else None
    
    print(f"Prediction: {'Subscribed' if prediction == 1 else 'Not Subscribed'}")
    if probability is not None:
        print(f"Subscription Probability: {probability:.2%}")
    return prediction, probability

print("\nPipeline execution complete.")