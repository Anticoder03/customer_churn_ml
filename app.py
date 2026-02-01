import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🌲",
    layout="centered"
)

st.title("📉 Customer Churn Prediction")
st.write("Random Forest model to predict customer churn")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# -----------------------------
# Data Cleaning
# -----------------------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# -----------------------------
# Define Columns
# -----------------------------
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numerical_cols = ['SeniorCitizen',  'tenure', 'MonthlyCharges', 'TotalCharges']

# -----------------------------
# Encode Categorical Columns (STORE ENCODERS)
# -----------------------------
label_encoders = {}

for col in categorical_cols + ['Churn']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Split Data
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# -----------------------------
# Sidebar Input (FIXED)
# -----------------------------
st.sidebar.header("🔢 Enter Customer Details")

def user_input():
    input_data = {}

    st.sidebar.subheader("Categorical Features")
    for col in categorical_cols:
        options = label_encoders[col].classes_
        selected = st.sidebar.selectbox(col, options)
        input_data[col] = label_encoders[col].transform([selected])[0]

    st.sidebar.subheader("Numerical Features")

    # SeniorCitizen as dropdown (0/1)
    input_data['SeniorCitizen'] = st.sidebar.selectbox(
        "SeniorCitizen",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    # Remaining numeric features as sliders
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        input_data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    # 🔥 CRITICAL FIX: enforce same column order as training
    return pd.DataFrame([input_data])[X.columns]

    input_data = {}

    st.sidebar.subheader("Categorical Features")
    for col in categorical_cols:
        options = label_encoders[col].classes_
        selected = st.sidebar.selectbox(col, options)
        input_data[col] = label_encoders[col].transform([selected])[0]

    st.sidebar.subheader("Numerical Features")
    for col in numerical_cols:
        input_data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    return pd.DataFrame([input_data])

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN ({probability:.2%})")
    else:
        st.success(f"✅ Customer is NOT likely to churn ({probability:.2%})")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df.head(10))
