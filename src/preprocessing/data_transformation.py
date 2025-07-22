# src/preprocessing/data_transformation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def transform(df: pd.DataFrame):
    """
    Transforms data:
    - Encode categorical
    - Scale numerical
    - Handle class imbalance using SMOTE
    - Split into training/test sets
    """
    X = df.drop(columns=['is_fraud', 'signup_time', 'purchase_time', 'ip_address', 'ip_int'])
    y = df['is_fraud']

    # Identify feature types
    categorical = ['source', 'browser', 'device', 'country']
    numerical = X.select_dtypes(include=np.number).columns.tolist()
    if 'user_id' in numerical:
        numerical.remove('user_id')  # Remove ID if not a useful feature

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    # Split before resampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Fit-transform on training data only
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    # Apply SMOTE to training set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pre, y_train)

    print("Training set class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    return X_train_resampled, y_train_resampled, X_test_pre, y_test
