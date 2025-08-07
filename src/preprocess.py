# src/preprocess.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data(path="data/Life Expectancy Data.csv"):
    """Load raw CSV into a DataFrame."""
    return pd.read_csv(path)

def build_preprocessing_pipeline(df, target_col="Life expectancy "):
    """
    Build a sklearn Pipeline that:
     - imputes missing values
     - encodes categoricals
     - scales numericals
    Returns the pipeline and feature lists.
    """
    # 1) Identify columns
    all_num = df.select_dtypes(include="number").columns
    num_cols = all_num.drop(target_col)
    cat_cols = df.select_dtypes(include="object").columns

    # 2) Numeric sub-pipeline: impute → scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 3) Categorical sub-pipeline: impute → one-hot
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 4) Combine
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # 5) Full pipeline (without model)
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor)
    ])

    return full_pipeline, num_cols, cat_cols

def prepare_train_test(df, pipeline, target_col="Life expectancy ", test_size=0.2, random_state=42):
    """
    Split df into X_train, X_test, y_train, y_test,
    then fit_transform the train X and transform the test X.
    Drops rows where target is NaN.
    """
    # 0) Drop rows where target is missing
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

    # 1) Separate features & target
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3) Apply pipeline
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared  = pipeline.transform(X_test)

    return X_train_prepared, X_test_prepared, y_train, y_test


if __name__ == "__main__":
    # 1) Load
    df = load_data()

    # 2) Build pipeline
    pipeline, num_cols, cat_cols = build_preprocessing_pipeline(df)

    # 3) Prepare train/test
    X_train_prep, X_test_prep, y_train, y_test = prepare_train_test(df, pipeline)

    print("Train shape:", X_train_prep.shape)
    print("Test  shape:", X_test_prep.shape)
