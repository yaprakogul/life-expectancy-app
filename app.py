# ---------------------------------------------------------
# Streamlit app: train & compare models for Life Expectancy
# Requires:
#   - data/Life Expectancy Data.csv
#   - src/ as a package (with __init__.py)
#   - src/model.py (build_model_pipeline)
#   - src/features.py (FeatureMaker used inside model pipeline)
# Run:
#   streamlit run app.py
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.model import build_model_pipeline

# ---------------- Page & constants ----------------
st.set_page_config(page_title="Life Expectancy – Modeling", layout="wide")
st.title("Life Expectancy – Modeling Demo")

TARGET = "Life expectancy "  # note the trailing space in the column name!

# ---------------- Data loading ----------------
@st.cache_data
def load_data(path: str = "data/Life Expectancy Data.csv") -> pd.DataFrame:
    """Load raw CSV."""
    return pd.read_csv(path)

df = load_data()
st.subheader("Raw data sample")
st.dataframe(df.head(), use_container_width=True)

# Basic cleaning for target
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found. Check the CSV headers.")
    st.stop()

# Drop rows where target is missing (cannot train on NaN y)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# ---------------- Sidebar: model selection & params ----------------
st.sidebar.header("Model settings")

model_name = st.sidebar.selectbox(
    "Choose model",
    ["Linear", "Ridge", "Lasso", "RandomForest"],
    index=0
)

params = {}
if model_name in ["Ridge", "Lasso"]:
    # Regularization strength
    params["alpha"] = st.sidebar.slider("alpha", 0.0001, 5.0, 1.0, step=0.1)
elif model_name == "RandomForest":
    params["n_estimators"]    = st.sidebar.slider("n_estimators", 50, 400, 150, step=25)
    params["max_depth"]       = st.sidebar.slider("max_depth", 2, 30, 10, step=1)
    params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 10, 2, step=1)

# ---------------- Train/test split ----------------
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------------- Build & fit pipeline (features + preprocessing + model) ----------------
# Note: We skip caching here; training is fast on this dataset.
model = build_model_pipeline(model_name, **params)
model.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2   = r2_score(y_test, y_pred)

st.subheader("Model Performance (Test)")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("RMSE (years)", f"{rmse:.2f}")
with c2:
    st.metric("MSE (years²)", f"{mse:.2f}")
with c3:
    st.metric("R²", f"{r2:.2f}")

# ---------------- Actual vs Predicted ----------------
st.subheader("Actual vs Predicted Life Expectancy")
results = pd.DataFrame({
    "Actual":    y_test.reset_index(drop=True),
    "Predicted": y_pred
})
st.line_chart(results, use_container_width=True)

# ---------------- Quick exploration (optional) ----------------
st.sidebar.header("Explore numeric feature")
num_cols = list(X.select_dtypes(include="number").columns)
if num_cols:
    sel = st.sidebar.selectbox("Numeric feature", num_cols)
    st.write(f"Distribution of **{sel}** (value_counts)")
    st.bar_chart(X[sel].value_counts())
else:
    st.info("No numeric features found in X.")
