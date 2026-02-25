
import pandas as pd
import numpy as np

def create_features(df, lags, rolling_windows, date_features=True):
    """
    Creates lag and rolling features.
    """
    df_feat = df.copy()
    
    # Lags
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat['LOG_SUNSPOTS'].shift(lag)
        
    # Rolling stats
    for window in rolling_windows:
        df_feat[f'roll_mean_{window}'] = df_feat['LOG_SUNSPOTS'].rolling(window).mean()
        df_feat[f'roll_std_{window}'] = df_feat['LOG_SUNSPOTS'].rolling(window).std()
        
    # Date features
    if date_features:
        df_feat['dayofyear'] = df_feat.index.dayofyear
        df_feat['year'] = df_feat.index.year
        df_feat['sin_dayofyear'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365)
        df_feat['cos_dayofyear'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365)
        
    return df_feat

def prepare_target(df, shift=-5):
    """
    Creates target variable (shifted future value).
    """
    df['target'] = df['LOG_SUNSPOTS'].shift(shift)
    df = df.dropna().copy()
    return df

def build_residual_lags(residuals, max_lag=5):
    df = pd.DataFrame({'resid': residuals})
    for lag in range(1, max_lag + 1):
        df[f'resid_lag_{lag}'] = df['resid'].shift(lag)
    return df.dropna()
