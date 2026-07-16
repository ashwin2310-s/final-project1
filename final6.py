import os
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, classification_report,
                              ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

TARGET = "y"

st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")
st.title("Bank Marketing Term Deposit Predictor")
st.caption("Upload bank-full.csv, compare models, tune the best one, and predict on a new customer.")


st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload bank-full.csv", type=["csv"])
default_path = st.sidebar.text_input("...or a path on disk", value="C:/Users/ashwi/Downloads/bank+marketing (1)/bank/bank-full.csv")


@st.cache_data(show_spinner=False)
def load_from_upload(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes), sep=";")


@st.cache_data(show_spinner=False)
def load_from_path(path):
    return pd.read_csv(path, sep=";")


df = None
if uploaded_file is not None:
    df = load_from_upload(uploaded_file.getvalue())
elif default_path and os.path.exists(default_path):
    df = load_from_path(default_path)

if df is None:
    st.info("Upload a CSV file or enter a valid path in the sidebar to get started.")
    st.stop()

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found. Columns available: {list(df.columns)}")
    st.stop()

if df[TARGET].dtype == "O":
    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), width='stretch')
    st.caption(f"{df.shape[0]:,} rows x {df.shape[1]} columns")

# Keep the raw (pre-engineering) feature columns for the prediction form
raw_feature_cols = [c for c in df.columns if c != TARGET]
raw_numeric_cols = df[raw_feature_cols].select_dtypes(include=np.number).columns.tolist()
raw_categorical_cols = [c for c in raw_feature_cols if c not in raw_numeric_cols]


def add_engineered_features(data):
    data = data.copy()
    if {"age", "balance"}.issubset(data.columns):
        data["age_balance"] = data["age"] * data["balance"]
    if {"duration", "balance"}.issubset(data.columns):
        data["duration_balance"] = data["duration"] * data["balance"]
    if {"age", "duration"}.issubset(data.columns):
        data["age_duration"] = data["age"] * data["duration"]
    return data


df_eng = add_engineered_features(df)
X = df_eng.drop(columns=[TARGET])
y = df_eng[TARGET]

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

education_order = [["unknown", "primary", "secondary", "tertiary"]]
ordinal_cols = [c for c in categorical_cols if c.lower() == "education"]
nominal_cols = [c for c in categorical_cols if c.lower() != "education"]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=education_order,
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1)),
    ])
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("ord", ordinal_pipeline, ordinal_cols),
        ("nom", nominal_pipeline, nominal_cols),
    ])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


st.sidebar.header("2. Models")

MODEL_MAP = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (slow)": SVC(probability=True),
}

if LIGHTGBM_AVAILABLE:
    MODEL_MAP["LightGBM (Ultra Fast)"] = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

chosen_models = st.sidebar.multiselect(
    "Models to compare",
    list(MODEL_MAP.keys()),
    default=["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"],
)
run_tuning = st.sidebar.checkbox("Tune Random Forest (Fast RandomizedSearchCV)", value=False)
run_stacking = st.sidebar.checkbox("Train Fast Stacking Classifier", value=False)
run_shap = st.sidebar.checkbox("Generate SHAP explanations", value=False) and SHAP_AVAILABLE
if run_shap and not SHAP_AVAILABLE:
    st.sidebar.caption("shap is not installed, so this option is disabled.")

train_clicked = st.sidebar.button("Train models", type="primary", width='stretch')

@st.cache_resource(show_spinner=False)
def train_everything(_X_train, _y_train, _X_test, _y_test, model_names, do_tuning, do_stacking):
    results = []
    trained_pipes = {}
    for name in model_names:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", MODEL_MAP[name])])
        pipe.fit(_X_train, _y_train)
        trained_pipes[name] = pipe

        pred = pipe.predict(_X_test)
        proba = pipe.predict_proba(_X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(_y_test, pred),
            "Precision": precision_score(_y_test, pred, zero_division=0),
            "Recall": recall_score(_y_test, pred, zero_division=0),
            "F1": f1_score(_y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(_y_test, proba) if proba is not None else np.nan,
        })

    comparison_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    tuned_pipeline, tuned_params, tuned_score = None, None, None
    if do_tuning:
        # Optimized to 3 folds and internal n_jobs framework to minimize overhead lag
        skf_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        rf_pipe = Pipeline([("prep", build_preprocessor()), ("model", RandomForestClassifier(random_state=42, n_jobs=-1))])
        params = {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [10, 15, 20],
            "model__min_samples_split": [5, 10],
        }
        search = RandomizedSearchCV(rf_pipe, param_distributions=params, n_iter=3, cv=skf_fast,
                                     scoring="accuracy", random_state=42, n_jobs=-1)
        search.fit(_X_train, _y_train)
        tuned_pipeline = search.best_estimator_
        tuned_params = search.best_params_
        tuned_score = search.best_score_

    stack_pipe, stack_pred = None, None
    if do_stacking:
        estimators = [
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=12)),
            ("dt", DecisionTreeClassifier(random_state=42, max_depth=10)),
        ]
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500), cv=3, n_jobs=-1)
        stack_pipe = Pipeline([("prep", build_preprocessor()), ("model", stack)])
        stack_pipe.fit(_X_train, _y_train)
        stack_pred = stack_pipe.predict(_X_test)

    return comparison_df, trained_pipes, tuned_pipeline, tuned_params, tuned_score, stack_pipe, stack_pred


if train_clicked:
    if not chosen_models:
        st.sidebar.error("Pick at least one model to compare.")
    else:
        with st.spinner("Training... this can take a while for larger datasets or slower models."):
            (comparison_df, trained_pipes, tuned_pipeline, tuned_params,
             tuned_score, stack_pipe, stack_pred) = train_everything(
                X_train, y_train, X_test, y_test,
                tuple(chosen_models), run_tuning, run_stacking
            )
        st.session_state["comparison_df"] = comparison_df
        st.session_state["trained_pipes"] = trained_pipes
        st.session_state["tuned_pipeline"] = tuned_pipeline
        st.session_state["tuned_params"] = tuned_params
        st.session_state["tuned_score"] = tuned_score
        st.session_state["stack_pipe"] = stack_pipe
        st.session_state["stack_pred"] = stack_pred
        st.session_state["y_test"] = y_test
        st.session_state["X_test"] = X_test

if "comparison_df" not in st.session_state:
    st.info("Choose your models in the sidebar and click **Train models** to get started.")
    st.stop()

comparison_df = st.session_state["comparison_df"]
trained_pipes = st.session_state["trained_pipes"]
tuned_pipeline = st.session_state["tuned_pipeline"]
tuned_params = st.session_state["tuned_params"]
tuned_score = st.session_state["tuned_score"]
stack_pipe = st.session_state["stack_pipe"]
stack_pred = st.session_state["stack_pred"]
y_test = st.session_state["y_test"]
X_test = st.session_state["X_test"]

st.subheader("Model comparison")
st.dataframe(
    comparison_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}",
                                 "Recall": "{:.3f}", "F1": "{:.3f}", "ROC_AUC": "{:.3f}"}),
    width='stretch',
)

if tuned_pipeline is not None:
    st.subheader("Hyperparameter tuning (Random Forest)")
    c1, c2 = st.columns(2)
    c1.metric("Best CV accuracy", f"{tuned_score:.3f}")
    c2.json(tuned_params)

best_pipeline = None
best_label = None
if stack_pipe is not None:
    best_pipeline, best_label = stack_pipe, "Stacking Classifier"
elif tuned_pipeline is not None:
    best_pipeline, best_label = tuned_pipeline, "Tuned Random Forest"
elif trained_pipes:
    top_name = comparison_df.iloc[0]["Model"]
    best_pipeline, best_label = trained_pipes[top_name], top_name

if stack_pipe is not None:
    st.subheader("Stacking classifier")
    report = classification_report(y_test, stack_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, stack_pred, ax=ax)
        ax.set_title("Stacking Classifier - Confusion Matrix")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_estimator(stack_pipe, X_test, y_test, ax=ax2)
        ax2.set_title("Stacking Classifier - ROC Curve")
        st.pyplot(fig2)

if run_shap and tuned_pipeline is not None:
    st.subheader("SHAP feature importance")
    with st.spinner("Computing SHAP values..."):
        rf_model = tuned_pipeline.named_steps["model"]
        preprocessor = tuned_pipeline.named_steps["prep"]
        
        # Transform structural tabular array objects cleanly
        X_test_transformed = preprocessor.transform(X_test)
        
        # Resolves layout names natively following transform pipelines
        feature_names = preprocessor.get_feature_names_out()
        
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(np.array(shap_values).shape) == 3:
            shap_values = shap_values[:, :, 1]
            
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
        st.pyplot(fig3)

# Download the best pipeline
if best_pipeline is not None:
    buffer = io.BytesIO()
    joblib.dump(best_pipeline, buffer)
    st.download_button(
        f"Download best model ({best_label}) as .pkl",
        data=buffer.getvalue(),
        file_name="best_model.pkl",
        mime="application/octet-stream",
    )

st.divider()
st.subheader("Predict a new customer")

if best_pipeline is None:
    st.info("Train at least one model above to enable predictions.")
else:
    st.caption(f"Using **{best_label}** for predictions.")
    with st.form("predict_form"):
        cols = st.columns(3)
        inputs = {}
        for i, col_name in enumerate(raw_feature_cols):
            target_col = cols[i % 3]
            if col_name in raw_numeric_cols:
                series = df[col_name]
                inputs[col_name] = target_col.number_input(
                    col_name,
                    value=float(series.median()),
                    step=1.0,
                )
            else:
                options = sorted(df[col_name].dropna().unique().tolist())
                inputs[col_name] = target_col.selectbox(col_name, options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        customer_df = pd.DataFrame([inputs])
        customer_df = add_engineered_features(customer_df)
        prediction = best_pipeline.predict(customer_df)[0]
        probability = (
            best_pipeline.predict_proba(customer_df)[0][1]
            if hasattr(best_pipeline, "predict_proba")
            else None
        )

        if prediction == 1:
            st.success("Prediction: Subscribed")
        else:
            st.warning("Prediction: Not subscribed")
        if probability is not None:
            st.metric("Subscription probability", f"{probability:.1%}")
