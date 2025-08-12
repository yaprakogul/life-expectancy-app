This repository contains a Streamlit app and EDA notebooks for exploring global life expectancy data.
## Exploratory Data Analysis :
- Loaded dataset: 2938 rows × 22 cols  
- Missing values found mostly in `GDP`, `BMI`, `Schooling` columns  
- Life expectancy histogram: roughly normal distribution (mean ~70)  
- Missing-value heatmap plotted  



## Data Cleaning & Preprocessing 
- **Missing value imputation**  
   - Numeric columns: filled with **median** using `SimpleImputer(strategy="median")`.  
   - Categorical columns: filled with **most frequent** using `SimpleImputer(strategy="most_frequent")`.  
- **One-Hot Encoding** of all categorical features (`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`), creating columns like `Country_Turkey`, `Status_Developing`, etc.  
- **Standard Scaling** of all numeric features (`StandardScaler`) to zero mean and unit variance.  
- **Train/Test Split** using `train_test_split(test_size=0.2, random_state=42)`.  
- **Pipeline** construction combining imputation, encoding, and scaling into a single `Pipeline` + `ColumnTransformer` for reproducible preprocessing.




## Feature Engineering & Modeling
### Additions
- **`src/features.py`**
  - `FeatureMaker` introduces general-purpose features:
    - `GDP_Income_interaction = GDP * Income composition of resources`
    - `Mortality_per_School = Adult Mortality / (Schooling + 1e-3)`
    - `BMI_sq = BMI^2`
    - `log_GDP = log1p(max(GDP, 0))`
- **`src/model.py`**
  - Single end-to-end pipeline: **`FeatureMaker → Preprocessor → Regressor`**
  - **Preprocessor**:
    - `SimpleImputer` (numeric = `median`, categorical = `most_frequent`)
    - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
    - `StandardScaler` for numeric columns
- **`app.py`**
  - Sidebar model selector: `Linear`, `Ridge`, `Lasso`, `RandomForest`
  - Basic hyperparameters in the UI (e.g., `alpha`, `n_estimators`, `max_depth`)
  - Displays **RMSE, MSE, R²** and an **Actual vs Predicted** line chart

---

### Evaluation Protocol
- **Target column:** `"Life expectancy "`  
  > Note: there is a trailing space in the column name.
- **Cleaning:** remove rows with missing target values (`df.dropna(subset=[TARGET])`)
- **Split:** `train_test_split(test_size=0.2, random_state=42)`
- **Metrics:**
  - **RMSE** (primary): average error in years
  - **MSE:** squared error (years²)
  - **R²:** 0 = as good as predicting the mean, 1 = perfect, `< 0` = worse than mean
- **Baseline (reference):** a naïve predictor that always outputs the training mean

---

### How to Run
```bash
streamlit run app.py


