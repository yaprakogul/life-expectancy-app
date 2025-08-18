import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from .features import FeatureMaker

def build_preprocessor():
    """
    Preprocessor that dynamically selects numeric/object columns.
    Works with new engineered features too.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, selector(dtype_include="number")),
        ("cat", categorical_pipeline, selector(dtype_include=object))
    ])
    return preprocessor

def get_regressor(name: str, **params):
    name = name.lower()
    if name == "linear":
        return LinearRegression(**params)
    if name == "ridge":
        return Ridge(**params)
    if name == "lasso":
        return Lasso(**params)
    if name == "randomforest":
        return RandomForestRegressor(random_state=42, **params)
    raise ValueError(f"Unknown model: {name}")

def build_model_pipeline(model_name: str, **model_params):
    """
    Full pipeline: FeatureMaker -> Preprocessor -> Regressor
    """
    pipe = Pipeline([
        ("features", FeatureMaker()),
        ("preprocessing", build_preprocessor()),
        ("regressor", get_regressor(model_name, **model_params)),
    ])
    return pipe
