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
