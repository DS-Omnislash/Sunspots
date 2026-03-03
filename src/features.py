
import pandas as pd
import numpy as np

_SOLAR_CYCLE_REF = pd.Timestamp('2008-12-01')  # known solar minimum
_SOLAR_CYCLE_DAYS = 4015                         # ~11-year cycle

def create_features(df, lags, rolling_windows, date_features=True):
    """
    Creates lag and rolling features.
    """
    df_feat = df.copy()

    # Lags
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat['LOG_SUNSPOTS'].shift(lag)

    # Momentum: short-term minus long-term lag
    if 1 in lags and 30 in lags:
        df_feat['momentum_1_30'] = df_feat['lag_1'] - df_feat['lag_30']

    # Rolling stats
    for window in rolling_windows:
        df_feat[f'roll_mean_{window}'] = df_feat['LOG_SUNSPOTS'].rolling(window).mean()
        df_feat[f'roll_std_{window}'] = df_feat['LOG_SUNSPOTS'].rolling(window).std()
        df_feat[f'roll_min_{window}'] = df_feat['LOG_SUNSPOTS'].rolling(window).min()
        df_feat[f'roll_max_{window}'] = df_feat['LOG_SUNSPOTS'].rolling(window).max()

    # Date features
    if date_features:
        df_feat['dayofyear'] = df_feat.index.dayofyear
        df_feat['year'] = df_feat.index.year
        df_feat['sin_dayofyear'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 365)
        df_feat['cos_dayofyear'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 365)

        # Solar cycle phase (11-year cycle anchored to 2008-12 minimum)
        days_from_ref = (df_feat.index - _SOLAR_CYCLE_REF).days
        df_feat['sin_solar_cycle'] = np.sin(2 * np.pi * days_from_ref / _SOLAR_CYCLE_DAYS)
        df_feat['cos_solar_cycle'] = np.cos(2 * np.pi * days_from_ref / _SOLAR_CYCLE_DAYS)

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
