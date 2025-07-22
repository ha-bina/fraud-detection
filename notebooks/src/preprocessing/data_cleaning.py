# src/preprocessing/data_cleaning.py

import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing duplicates and correcting data types.
    """
    print(f"Shape before cleaning: {df.shape}")
    df = df.drop_duplicates()
    df = df.convert_dtypes()  # Optional: converts to optimal types
    print(f"Shape after cleaning: {df.shape}")
    return df
