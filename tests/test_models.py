from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tfmbench.config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def tiny_frames():
    rng = np.random.default_rng(42)
    rows = 240
    target = np.tile([0, 1], rows // 2)
    frame = pd.DataFrame(index=np.arange(rows))
    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            frame[feature] = pd.Categorical(np.where(rng.random(rows) > 0.5, "a", "b"))
        else:
            frame[feature] = rng.normal(size=rows) + target * 0.15
    return frame.iloc[:160], target[:160], frame.iloc[160:200], target[160:200], frame.iloc[200:], target[200:]


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_lightgbm_default_fit_predict_interface():
    from tfmbench.models.lgbm import fit_default, predict_probability

    X_train, y_train, X_validation, y_validation, X_test, _ = tiny_frames()
    fit = fit_default(X_train, y_train, X_validation, y_validation)
    probability = predict_probability(fit, X_test)
    assert probability.shape == (len(X_test),)
    assert np.isfinite(probability).all()
    assert ((0 <= probability) & (probability <= 1)).all()


def test_realmlp_prepares_nullable_numbers_and_preserves_categories(monkeypatch):
    from sklearn.utils.validation import check_array

    from tfmbench.models.realmlp import fit_model, predict_probability

    X_train = pd.DataFrame({
        "term": pd.array([36, 60, None, 60], dtype="Float32"),
        "grade": pd.Categorical(["A", "B", "A", None]),
        "emp_length": pd.array([None] * 4, dtype="Float32"),
    })
    holdout = pd.DataFrame({
        "term": pd.array([None, 36], dtype="Float32"),
        "grade": pd.Categorical(["A", None], categories=["A", "B"]),
        "emp_length": pd.array([None, 4], dtype="Float32"),
    })

    def check_holdout(frame):
        check_array(frame, dtype=None, ensure_all_finite="allow-nan")
        assert frame.columns.equals(holdout.columns)
        assert frame["term"].tolist() == [60, 36]
        assert frame["emp_length"].tolist() == [0, 4]
        pd.testing.assert_series_equal(frame["grade"], holdout["grade"])

    class FakeRealMLP:
        classes_ = [0, 1]

        def __init__(self, **kwargs):
            pass

        def fit(self, train, y, validation, y_validation, *, cat_col_names):
            check_array(train, dtype=None, ensure_all_finite="allow-nan")
            assert "grade" in cat_col_names
            assert train["term"].dtype == np.dtype("float32")
            assert train["term"].tolist() == [36, 60, 60, 60]
            check_holdout(validation)

        def predict_proba(self, frame):
            check_holdout(frame)
            return np.tile([0.75, 0.25], (len(frame), 1))

    monkeypatch.setitem(sys.modules, "pytabkit", SimpleNamespace(RealMLP_TD_Classifier=FakeRealMLP))
    fit = fit_model(X_train, [0, 1, 0, 1], holdout, [0, 1])
    np.testing.assert_array_equal(predict_probability(fit, holdout), [0.25, 0.25])
    assert pd.isna(X_train.loc[2, "term"])


@pytest.mark.skipif(importlib.util.find_spec("pytabkit") is None, reason="pytabkit not installed")
def test_realmlp_fit_predict_interface_when_cuda_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tfmbench.models.realmlp import fit_model, predict_probability

    X_train, y_train, X_validation, y_validation, X_test, _ = tiny_frames()
    fit = fit_model(X_train, y_train, X_validation, y_validation)
    probability = predict_probability(fit, X_test)
    assert probability.shape == (len(X_test),)
    assert np.isfinite(probability).all()
