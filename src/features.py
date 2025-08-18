import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureMaker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Drop columns
        df.drop(columns=[
            c for c in ["Income composition of resources", "thinness 5-9 years"]
            if c in df.columns
        ], inplace=True)

        # 1. Mortality per schooling
        if "Adult Mortality" in df.columns and "Schooling" in df.columns:
            df["Mortality_per_School"] = df["Adult Mortality"] / (df["Schooling"] + 1e-3)

        # 2. log GDP
        if "GDP" in df.columns:
            df["log_GDP"] = np.log(df["GDP"] + 1e-3)

        # 3. Schooling * Total expenditure
        if "Schooling" in df.columns and "Total expenditure" in df.columns:
            df["school_health_index"] = df["Schooling"] * df["Total expenditure"]

        # 4. Vaccine access index
        vaccine_cols = ["Hepatitis B", "Polio", "Diphtheria"]
        if all(col in df.columns for col in vaccine_cols):
            df["vaccine_index"] = df[vaccine_cols].mean(axis=1)

        return df
