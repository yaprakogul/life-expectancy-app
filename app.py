import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocess import load_data, build_preprocessing_pipeline, prepare_train_test

st.title("Life Expectancy Prediction App")

df = load_data("data/Life Expectancy Data.csv")
st.write("### Raw data sample", df.head())

pipeline, num_cols, cat_cols = build_preprocessing_pipeline(df)

X_train, X_test, y_train, y_test = prepare_train_test(df, pipeline)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

st.write("### Model Performance on Test Set")
st.metric(label="Mean Squared Error", value=f"{mse:.2f}")
st.metric(label="R² Score",           value=f"{r2:.2f}")

results = pd.DataFrame({
    "Actual":  y_test.reset_index(drop=True),
    "Predicted": y_pred
})
st.write("### Actual vs Predicted Life Expectancy")
st.line_chart(results)

st.sidebar.header("Options")
feature = st.sidebar.selectbox("Select numeric feature to view", num_cols)
st.write(f"#### Distribution of {feature}")
st.bar_chart(df[feature].value_counts())

