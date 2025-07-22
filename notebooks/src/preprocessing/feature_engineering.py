# src/preprocessing/feature_engineering.py

import pandas as pd
from src.utils.ip_converter import ip_to_int

def add_features(df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering:
    - Convert IPs
    - Merge with geolocation
    - Add time-based features
    - Compute velocity and frequency
    """
    # Convert IP to int
    df['ip_int'] = df['ip_address'].apply(ip_to_int)
    ip_df['lower_bound_int'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
    ip_df['upper_bound_int'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)

    # Merge based on IP range
    def get_country(ip):
        match = ip_df[(ip_df['lower_bound_int'] <= ip) & (ip_df['upper_bound_int'] >= ip)]
        return match['country'].values[0] if not match.empty else 'Unknown'

    df['country'] = df['ip_int'].apply(get_country)

    # Time features
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600

    # Frequency & velocity features
    tx_count = df.groupby('user_id').size().rename('user_tx_count')
    df = df.merge(tx_count, on='user_id')
    df['velocity'] = df['user_tx_count'] / (df['time_since_signup'] + 1e-6)

    return df
