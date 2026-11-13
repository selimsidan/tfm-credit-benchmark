from __future__ import annotations

import importlib.util

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
