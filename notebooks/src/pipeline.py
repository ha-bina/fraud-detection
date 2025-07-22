# pipeline.py

import pandas as pd

# Import preprocessing modules
from src.preprocessing import (
    missing_values,
    data_cleaning,
    feature_engineering,
    data_transformation
)

def main():
    # Step 1: Load raw data
    fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    
    # Step 2: Handle missing values
    fraud_df = missing_values.handle(fraud_df)
    
    # Step 3: Clean the data
    fraud_df = data_cleaning.clean(fraud_df)
    
    # Step 4: Feature engineering (IP merge, time-based features, etc.)
    fraud_df = feature_engineering.add_features(fraud_df, ip_df)
    
    # Step 5: Transform data (scaling, encoding, class balancing, split)
    X_train, y_train, X_test, y_test = data_transformation.transform(fraud_df)
    
    # Optional: Save intermediate outputs
    pd.DataFrame(X_train).to_csv('data/processed/X_train_resampled.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)

    print("\n✅ Task 1 completed successfully.")

if __name__ == '__main__':
    main()
