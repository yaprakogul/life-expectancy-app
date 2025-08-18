# Life Expectancy Predictor

This repository contains a Streamlit app and EDA notebooks for exploring and predicting global life expectancy data.

---

## 🌎 Exploratory Data Analysis (EDA)

* **Dataset Loading:** The dataset was loaded, consisting of 2938 rows and 22 columns.
* **Missing Values:** Missing values were found predominantly in the `GDP`, `BMI`, and `Schooling` columns.
* **Life Expectancy Distribution:** A histogram of life expectancy showed a roughly normal distribution with a mean around 70 years.
* **Missing Value Heatmap:** A heatmap was plotted to visualize the distribution of missing values.

---

## 🧹 Data Cleaning & Preprocessing

* **Missing Value Imputation:**
    * Numeric columns were filled with the **median** using `SimpleImputer(strategy="median")`.
    * Categorical columns were filled with the **most frequent** value using `SimpleImputer(strategy="most_frequent")`.
* **One-Hot Encoding:** All categorical features were one-hot encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, creating new columns like `Country_Turkey`, `Status_Developing`, etc.
* **Standard Scaling:** All numeric features were scaled to a zero mean and unit variance using `StandardScaler`.
* **Train/Test Split:** The dataset was split into training and testing sets using `train_test_split(test_size=0.2, random_state=42)`.
* **Pipeline Construction:** A `Pipeline` and `ColumnTransformer` were built to combine the imputation, encoding, and scaling steps, ensuring reproducible preprocessing.

---

## 🧠 Feature Engineering & Modeling

### ✨ Additions

* **`src/features.py`:**
    * A `FeatureMaker` class was introduced to create new, general-purpose features, including:
        * `GDP_Income_interaction = GDP * Income composition of resources`
        * `Mortality_per_School = Adult Mortality / (Schooling + 1e-3)`
        * `BMI_sq = BMI^2`
        * `log_GDP = log1p(max(GDP, 0))`
* **`src/model.py`:**
    * A single end-to-end pipeline was created: `FeatureMaker` → `Preprocessor` → `Regressor`.
    * The **Preprocessor** includes `SimpleImputer`, `OneHotEncoder`, and `StandardScaler` for comprehensive data transformation.
* **`app.py`:**
    * A sidebar model selector for `Linear`, `Ridge`, `Lasso`, and `RandomForest` was implemented.
    * Basic hyperparameters (e.g., `alpha`, `n_estimators`, `max_depth`) can be tuned via the UI.
    * The app displays **RMSE, MSE, R²** scores and an **Actual vs Predicted** line chart.

### 🧪 Evaluation Protocol

* **Target Column:** The target variable is `"Life expectancy "` (note the trailing space in the column name).
* **Cleaning:** Rows with missing target values were removed using `df.dropna(subset=[TARGET])`.
* **Split:** The data was split using `train_test_split(test_size=0.2, random_state=42)`.
* **Metrics:**
    * **RMSE** (primary): The average error in years.
    * **MSE:** Squared error (years²).
    * **R²:** A score from 0 to 1, where 1 is a perfect fit.
* **Baseline:** A naive predictor that always outputs the training mean was used as a reference point.

---

## 📊 Model Training & Comparison

This section focuses on model selection and evaluation using various regression algorithms, all built on a unified pipeline.

### 🔧 Models Used

* `LinearRegression`
* `Ridge`
* `Lasso`
* `RandomForestRegressor`

Each model was trained and tested on the same dataset with consistent preprocessing and feature transformation logic.

### 📏 Metrics

Evaluation was conducted using the following metrics:

* **RMSE (Root Mean Squared Error):** Indicates the average prediction error in years.
* **MSE (Mean Squared Error):** Penalizes larger errors more strongly.
* **R² Score:** Reflects how much variance in the target variable is explained by the model.

All models were compared against a **baseline predictor** that always outputs the mean of the training target values.

### 📈 Visual Feedback

A side-by-side comparison chart of model performances was included. Additionally, an **Actual vs Predicted** line chart is displayed after each run to visualize accuracy and highlight under- or overfitting.

---

## 🚀 Finalization & Deployment

### 📥 CSV Prediction Integration

* A file upload component was added to the Streamlit app for `.csv` test files.
* Incoming test data must match the structure of the training features (excluding the target column).
* Predictions are made using the fully-trained pipeline and displayed in a preview table.
* Users can download the predictions as a CSV.

### 🧾 Custom Test CSVs

Example CSVs were constructed and tested with realistic data, such as **Turkey (2025)**. The format was aligned with the training features and passed through the same preprocessing pipeline.

### 🔧 Feature Fixes

* Verified the inclusion of all original columns, such as `percentage expenditure`, which was initially missing from test cases.
* Ensured compatibility between training and test pipelines through column consistency checks.

### 🎨 UI Enhancements

* A clean sidebar layout was created for model selection and hyperparameter tuning.
* The display for metrics and plots was improved, including:
    * Residuals vs Predicted plot
    * Actual vs Predicted plot
* Predictions are displayed in real-time after a file is uploaded.