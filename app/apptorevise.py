import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Title ----------------
st.title("Life Expectancy Interactive Analysis")

# ---------------- Load & Cache Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Life Expectancy Data.csv")

df = load_data()

# ---------------- User Inputs ----------------
# Country and Year selection
country = st.selectbox("Select Country", df["Country"].sort_values().unique())
year = st.slider(
    "Select Year",
    int(df["Year"].min()),
    int(df["Year"].max()),
    int(df["Year"].median())
)

# Feature selection
numeric_columns = (
    df.select_dtypes(include="number")
      .columns
      .drop(["Year", "Life expectancy "])
      .tolist()
)
feature = st.selectbox("Select Feature", numeric_columns)

st.markdown(f"**Selected:** Country = {country}  |  Year = {year}  |  Feature = {feature}")

# ---------------- Filtered Subsets ----------------
sub_country = df[df["Country"] == country]
sub_year    = sub_country[sub_country["Year"] == year]

# ---------------- 1) Trend + Highlight ----------------
st.subheader(f"Life Expectancy Trend for {country} (highlighting {year})")

# Base chart with axes
chart_base = alt.Chart(sub_country).encode(
    x=alt.X("Year:O", title="Year"),
    y=alt.Y("Life expectancy :Q", title="Life Expectancy")
)

# Line for overall trend
line = chart_base.mark_line(color="steelblue", size=2)

# Highlight selected year with a red dot
highlight = chart_base.transform_filter(
    alt.datum.Year == year
).mark_circle(color="red", size=100)

# Combine and render
trend_chart = (line + highlight).properties(width=700, height=300).interactive()
st.altair_chart(trend_chart, use_container_width=True)

# ---------------- 2) Scatter Plot (Feature vs. Life Expectancy) ----------------
st.subheader(f"{feature} vs. Life Expectancy (Scatter Plot)")
scatter = (
    alt.Chart(sub_country)
       .mark_circle(size=60)
       .encode(
           x=alt.X(f"{feature}:Q", title=feature),
           y=alt.Y("Life expectancy :Q", title="Life Expectancy"),
           tooltip=[feature, "Year", "Life expectancy "]
       )
       .interactive()
       .properties(width=700, height=400)
)
st.altair_chart(scatter, use_container_width=True)


# ---------------- 4) Filtered Table View ----------------
st.subheader(f"Filter Records by {feature} Range")
min_val, max_val = st.slider(
    f"{feature} range",
    float(df[feature].min()),
    float(df[feature].max()),
    (float(df[feature].min()), float(df[feature].max()))
)
filtered = df[(df[feature] >= min_val) & (df[feature] <= max_val)]
st.write(f"Number of records with {feature} between **{min_val:.2f} and {max_val:.2f}**: {len(filtered)}")
st.dataframe(filtered[["Country", "Year", feature, "Life expectancy "]].reset_index(drop=True))

# ---------------- 5) Year-specific Metric ----------------
st.subheader(f"Life Expectancy in {year} for {country}")
if not sub_year.empty:
    val = sub_year["Life expectancy "].iloc[0]
    st.metric(label=f"{year}", value=f"{val:.2f} years")
else:
    st.warning(f"No data found for {country} in {year}.")
