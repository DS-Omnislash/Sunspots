import numpy as np
import pandas as pd
import pytest

from src.model import evt_tail_correction, HybridEVTModel


# ── evt_tail_correction ───────────────────────────────────────────────────────

def test_evt_correction_preserves_length():
    forecast = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    actual = np.array([12.0, 18.0, 35.0, 38.0, 55.0])
    corrected = evt_tail_correction(forecast, actual)
    assert len(corrected) == len(forecast)


def test_evt_correction_empty_actual_returns_forecast():
    forecast = np.array([10.0, 20.0])
    corrected = evt_tail_correction(forecast, np.array([]))
    np.testing.assert_array_equal(corrected, forecast)


def test_evt_correction_returns_floats():
    forecast = np.array([5.0, 15.0, 25.0])
    actual = np.array([6.0, 14.0, 30.0])
    corrected = evt_tail_correction(forecast, actual)
    assert corrected.dtype in (np.float32, np.float64)


# ── HybridEVTModel ────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    n_train, n_val, n_feat = 200, 10, 5
    X_train = pd.DataFrame(rng.standard_normal((n_train, n_feat)),
                            columns=[f'f{i}' for i in range(n_feat)])
    y_train = pd.Series(rng.standard_normal(n_train))
    X_val = pd.DataFrame(rng.standard_normal((n_val, n_feat)),
                          columns=[f'f{i}' for i in range(n_feat)])
    return X_train, y_train, X_val


def test_hybrid_model_output_length(tiny_dataset):
    X_train, y_train, X_val = tiny_dataset
    model = HybridEVTModel(
        ridge_alpha=1.0,
        resid_model_type='lgb',
        resid_params={'n_estimators': 10, 'verbosity': -1, 'n_jobs': 1},
        max_lag_resid=5,
    )
    preds = model.fit_predict_val(X_train, y_train, X_val)
    assert len(preds) == len(X_val)


def test_hybrid_model_no_nans(tiny_dataset):
    X_train, y_train, X_val = tiny_dataset
    model = HybridEVTModel(
        ridge_alpha=1.0,
        resid_model_type='lgb',
        resid_params={'n_estimators': 10, 'verbosity': -1, 'n_jobs': 1},
        max_lag_resid=5,
    )
    preds = model.fit_predict_val(X_train, y_train, X_val)
    assert not np.any(np.isnan(preds))


def test_hybrid_model_with_xgb(tiny_dataset):
    X_train, y_train, X_val = tiny_dataset
    model = HybridEVTModel(
        ridge_alpha=1.0,
        resid_model_type='xgb',
        resid_params={'n_estimators': 10, 'verbosity': 0, 'n_jobs': 1},
        max_lag_resid=5,
    )
    preds = model.fit_predict_val(X_train, y_train, X_val)
    assert len(preds) == len(X_val)
