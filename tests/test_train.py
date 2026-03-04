import numpy as np
import pandas as pd
import pytest

from src.train import expanding_walk_forward_splits


@pytest.fixture
def xy():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((200, 3)), columns=['a', 'b', 'c'])
    y = pd.Series(rng.standard_normal(200))
    return X, y


def test_correct_number_of_splits(xy):
    X, y = xy
    splits = list(expanding_walk_forward_splits(X, y, initial_train_size=100, val_size=10, step_size=10))
    # (200 - 100) / 10 = 10 splits
    assert len(splits) == 10


def test_val_size_constant(xy):
    X, y = xy
    for _, _, Xval, yval in expanding_walk_forward_splits(X, y, initial_train_size=100, val_size=10, step_size=10):
        assert len(Xval) == 10
        assert len(yval) == 10


def test_train_window_grows(xy):
    X, y = xy
    sizes = [
        len(Xtr)
        for Xtr, _, _, _ in expanding_walk_forward_splits(X, y, initial_train_size=100, val_size=10, step_size=10)
    ]
    assert sizes == sorted(sizes)


def test_no_train_val_overlap(xy):
    X, y = xy
    for Xtr, _, Xval, _ in expanding_walk_forward_splits(X, y, initial_train_size=100, val_size=10, step_size=10):
        assert set(Xtr.index).isdisjoint(set(Xval.index))


def test_zero_splits_when_data_too_short():
    X = pd.DataFrame(np.ones((50, 2)))
    y = pd.Series(np.ones(50))
    splits = list(expanding_walk_forward_splits(X, y, initial_train_size=100, val_size=10, step_size=10))
    assert splits == []
