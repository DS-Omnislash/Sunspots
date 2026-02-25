
import pytest
import pandas as pd
import numpy as np
from src.features import create_features
from src.model import HybridEVTModel

def test_feature_creation():
    dates = pd.date_range(start='2020-01-01', periods=100)
    df = pd.DataFrame({'SUNSPOTS': np.abs(np.random.randn(100) * 100)}, index=dates)
    df['LOG_SUNSPOTS'] = np.log1p(df['SUNSPOTS'])
    
    df_feat = create_features(df, lags=[1, 2], rolling_windows=[3])
    assert 'lag_1' in df_feat.columns
    assert 'roll_mean_3' in df_feat.columns
    assert 'sin_dayofyear' in df_feat.columns

def test_hybrid_model_fit():
    # Create dummy data
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(np.random.rand(50)) # Log scale target
    X_val = pd.DataFrame(np.random.rand(10, 5), columns=[f'feat_{i}' for i in range(5)])
    
    model = HybridEVTModel(max_lag_resid=2)
    pred = model.fit_predict_val(X, y, X_val)
    
    assert len(pred) == 10
    assert not np.isnan(pred).any()
