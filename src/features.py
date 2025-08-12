from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureMaker(BaseEstimator, TransformerMixin):
    """
    Add simple, domain-agnostic features.
    Safe operations (avoid divide-by-zero and missing columns).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No learning needed here
        return self

    def transform(self, X):
        X = X.copy()

        # 1) Interaction: GDP * Income composition of resources
        if {"GDP", "Income composition of resources"}.issubset(X.columns):
            X["GDP_Income_interaction"] = (
                X["GDP"] * X["Income composition of resources"]
            )

        # 2) Ratio: Adult Mortality / (Schooling + eps)
        if {"Adult Mortality", "Schooling"}.issubset(X.columns):
            X["Mortality_per_School"] = (
                X["Adult Mortality"] / (X["Schooling"].astype(float) + 1e-3)
            )

        # 3) Nonlinearity: BMI squared
        if "BMI" in X.columns:
            X["BMI_sq"] = X["BMI"] ** 2

        # 4) Optional log transform (safe): log(GDP + 1)
        if "GDP" in X.columns:
            X["log_GDP"] = np.log1p(np.clip(X["GDP"], a_min=0, a_max=None))

        return X
