import numpy as np
import pandas as pd
import pytest

from src.features import create_features, prepare_target, build_residual_lags

LAGS = [1, 2, 3, 5, 30]
WINDOWS = [7, 30]


@pytest.fixture
def sample_df():
    dates = pd.date_range('2000-01-01', periods=500, freq='D')
    sunspots = np.random.default_rng(42).integers(0, 200, size=500).astype(float)
    return pd.DataFrame(
        {'SUNSPOTS': sunspots, 'LOG_SUNSPOTS': np.log1p(sunspots)},
        index=dates,
    )


# ── create_features ───────────────────────────────────────────────────────────

def test_lag_columns_created(sample_df):
    df_feat = create_features(sample_df, LAGS, WINDOWS)
    for lag in LAGS:
        assert f'lag_{lag}' in df_feat.columns


def test_rolling_columns_created(sample_df):
    df_feat = create_features(sample_df, LAGS, WINDOWS)
    for w in WINDOWS:
        for stat in ('mean', 'std', 'min', 'max'):
            assert f'roll_{stat}_{w}' in df_feat.columns


def test_momentum_created_when_lags_present(sample_df):
    df_feat = create_features(sample_df, [1, 30], [7])
    assert 'momentum_1_30' in df_feat.columns


def test_momentum_absent_when_lag_missing(sample_df):
    df_feat = create_features(sample_df, [1, 2], [7])
    assert 'momentum_1_30' not in df_feat.columns


def test_date_features_created(sample_df):
    df_feat = create_features(sample_df, [1], [7], date_features=True)
    for col in ('sin_dayofyear', 'cos_dayofyear', 'sin_solar_cycle', 'cos_solar_cycle'):
        assert col in df_feat.columns


def test_date_features_skipped(sample_df):
    df_feat = create_features(sample_df, [1], [7], date_features=False)
    assert 'sin_dayofyear' not in df_feat.columns


def test_create_features_preserves_row_count(sample_df):
    df_feat = create_features(sample_df, LAGS, WINDOWS)
    assert len(df_feat) == len(sample_df)


def test_solar_cycle_features_bounded(sample_df):
    df_feat = create_features(sample_df, [1], [7])
    assert df_feat['sin_solar_cycle'].between(-1, 1).all()
    assert df_feat['cos_solar_cycle'].between(-1, 1).all()


# ── prepare_target ────────────────────────────────────────────────────────────

def test_prepare_target_adds_column(sample_df):
    df_feat = create_features(sample_df, [1], [7])
    df_out = prepare_target(df_feat, shift=-5)
    assert 'target' in df_out.columns


def test_prepare_target_no_nans(sample_df):
    df_feat = create_features(sample_df, [1], [7])
    df_out = prepare_target(df_feat, shift=-5)
    assert df_out['target'].isna().sum() == 0


def test_prepare_target_shortens_df(sample_df):
    df_feat = create_features(sample_df, [1], [7])
    df_out = prepare_target(df_feat, shift=-5)
    assert len(df_out) < len(df_feat)


# ── build_residual_lags ───────────────────────────────────────────────────────

def test_residual_lag_columns(sample_df):
    residuals = np.random.randn(100)
    df = build_residual_lags(residuals, max_lag=5)
    assert 'resid' in df.columns
    for lag in range(1, 6):
        assert f'resid_lag_{lag}' in df.columns


def test_residual_lags_no_nans():
    residuals = np.random.randn(100)
    df = build_residual_lags(residuals, max_lag=5)
    assert df.isna().sum().sum() == 0
