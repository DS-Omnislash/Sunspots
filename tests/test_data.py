import numpy as np
import pandas as pd
import pytest

from src.data import load_data


@pytest.fixture
def local_csv(tmp_path):
    """Write a minimal pre-processed CSV and return its path."""
    dates = pd.date_range('2000-01-01', periods=10, freq='D')
    df = pd.DataFrame(
        {
            'SUNSPOTS': np.arange(10, dtype=float),
            'LOG_SUNSPOTS': np.log1p(np.arange(10, dtype=float)),
        },
        index=dates,
    )
    df.index.name = 'Date'
    path = tmp_path / 'sunspots.csv'
    df.to_csv(path)
    return str(path)


def test_load_from_local_cache(local_csv):
    df = load_data('http://fake-url', save_path=local_csv)
    assert 'SUNSPOTS' in df.columns
    assert 'LOG_SUNSPOTS' in df.columns
    assert len(df) == 10


def test_index_is_datetime(local_csv):
    df = load_data('http://fake-url', save_path=local_csv)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_sunspots_non_negative(tmp_path):
    """Even if the cached file has negatives, load_data clips them."""
    dates = pd.date_range('2000-01-01', periods=5, freq='D')
    df = pd.DataFrame(
        {
            'SUNSPOTS': [-5.0, 0.0, 10.0, 20.0, 30.0],
            'LOG_SUNSPOTS': np.log1p([0.0, 0.0, 10.0, 20.0, 30.0]),
        },
        index=dates,
    )
    df.index.name = 'Date'
    path = tmp_path / 'sunspots_neg.csv'
    df.to_csv(path)

    result = load_data('http://fake-url', save_path=str(path))
    assert (result['SUNSPOTS'] >= 0).all()
