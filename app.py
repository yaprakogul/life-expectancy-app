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
import os, joblib

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

use_saved = st.sidebar.checkbox("Use saved model if available", value=True)
model_path = "models/best_pipeline.pkl"


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

# ---------------- Build & fit OR load ----------------
if use_saved and os.path.exists(model_path):
    # Load saved pipeline (features + preprocessor + regressor)
    model = joblib.load(model_path)
else:
    # Train from scratch with the selected model & params
    from src.model import build_model_pipeline
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
    st.metric("R²", f"{r2:.4f}")

# ---------------- Actual vs Predicted ----------------
st.subheader("Actual vs Predicted Life Expectancy")
results = pd.DataFrame({
    "Actual":    y_test.reset_index(drop=True),
    "Predicted": y_pred
})
st.line_chart(results, use_container_width=True)

# ---------------- Diagnostics (Residuals & Importances) ----------------
st.subheader("Diagnostics")

# Residuals = Actual - Predicted
residuals = y_test.reset_index(drop=True) - y_pred

# 1) Residuals histogram
st.markdown("**Residuals histogram**")
hist, bin_edges = np.histogram(residuals, bins=30)
hist_df = pd.DataFrame({
    "bin_left": bin_edges[:-1],
    "bin_right": bin_edges[1:],
    "count": hist,
})
hist_df["mid"] = (hist_df["bin_left"] + hist_df["bin_right"]) / 2
st.bar_chart(hist_df.set_index("mid")["count"])

# 2) Residuals vs Predicted
st.markdown("**Residuals vs Predicted**")
res_df = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})
st.scatter_chart(res_df, x="Predicted", y="Residual")

# 3) Feature importances
reg = model.named_steps.get("regressor", None)
if reg is not None and hasattr(reg, "feature_importances_"):
    try:
        pre = model.named_steps["preprocessing"]
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(reg.feature_importances_))]

    imp_df = (pd.DataFrame({"feature": feature_names,
                            "importance": reg.feature_importances_})
              .sort_values("importance", ascending=False)
              .head(20)
              .reset_index(drop=True))
    st.markdown("**Top feature importances**")
    st.dataframe(imp_df, use_container_width=True)

    # ---------------- Predict on New CSV ----------------
st.subheader("Predict on New Data (Upload CSV)")

uploaded = st.file_uploader(
    "Upload a CSV with the same feature columns used for training (no target).",
    type="csv"
)

if uploaded is not None:
    new_df = pd.read_csv(uploaded)

    # If user accidentally included target, drop it
    if TARGET in new_df.columns:
        new_df = new_df.drop(columns=[TARGET])

    # Required columns check (model expects training-time feature columns)
    required_cols = list(X.columns)   # X = df.drop(TARGET, axis=1)
    missing = [c for c in required_cols if c not in new_df.columns]
    extra   = [c for c in new_df.columns if c not in required_cols]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        # Reorder to match training column order; ignore extras
        new_df = new_df[required_cols]

        # Predict
        preds = model.predict(new_df)

        # Show & allow download
        out = new_df.copy()
        out["prediction"] = preds
        st.write("### Predictions (first 50 rows)")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            label="Download predictions as CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

    if extra:
        st.info(f"Extra columns ignored: {extra}")

# ---------------- Quick exploration (optional) ----------------
st.sidebar.header("Explore numeric feature")
num_cols = list(X.select_dtypes(include="number").columns)
if num_cols:
    sel = st.sidebar.selectbox("Numeric feature", num_cols)
    st.write(f"Distribution of **{sel}** (value_counts)")
    st.bar_chart(X[sel].value_counts())
else:
    st.info("No numeric features found in X.")
