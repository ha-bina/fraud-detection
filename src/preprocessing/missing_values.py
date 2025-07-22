# src/preprocessing/missing_values.py

import pandas as pd

def handle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop or impute missing values.
    Currently drops rows with any missing values.
    """
    print("Missing values before handling:\n", df.isnull().sum())
    df = df.dropna()  # You could use fillna() or SimpleImputer if imputation is preferred
    print("Missing values after handling:\n", df.isnull().sum())
    return df
