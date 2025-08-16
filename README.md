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


## Model Training & Comparison
-This section focuses on model selection and evaluation using various regression algorithms. All models are built on top of a unified pipeline that includes feature engineering and preprocessing steps.
### Models Used
- `LinearRegression`  
- `Ridge`  
- `Lasso`  
- `RandomForestRegressor`  
Each model was trained and tested on the same dataset using consistent preprocessing and feature transformation logic.

### Metrics
-Evaluation was conducted using the following metrics:

- **RMSE (Root Mean Squared Error):** Indicates the average error in prediction (in years).
- **MSE (Mean Squared Error):** Penalizes larger errors more strongly than RMSE.
- **R² Score:** Reflects how much variance in the target variable is explained by the model.

All models were also compared against a **baseline predictor** that always outputs the mean of the training target values.

### Visual Feedback
A side-by-side comparison chart of model performances was included to help identify the most effective regressor. Additionally, an **Actual vs Predicted** line chart is displayed after each run to visualize accuracy and highlight under- or overfitting.


### Finalization – Deployment and Testing
## CSV Prediction Integration
-File Upload Component added to the Streamlit app for uploading .csv test files.
-Incoming test data must match the structure of the training features — the target column should be excluded.
-Predictions are made using the fully-trained pipeline and displayed in a preview table (first 50 rows).
-Users can download the predictions as a CSV using the "Download predictions" button.

##Custom Test CSVs
Example CSVs were constructed and tested with realistic data, such as:
Turkey (2025): Moderate alcohol use, no HIV/AIDS presence.
Format was aligned with the training features and passed through the same preprocessing pipeline.


##Feature Fixes
-Verified inclusion of all original columns such as percentage expenditure, which was initially missing from test cases.
-Ensured compatibility between training and test pipelines using column consistency checks.

##UI Enhancements
Clean sidebar layout for model selection and hyperparameter tuning.
Improved display for metrics and plots, including:

Residuals vs Predicted

Actual vs Predicted

Predictions displayed in real-time after upload.
