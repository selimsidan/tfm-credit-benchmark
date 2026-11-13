"""RealMLP neural baseline wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ..config import CATEGORICAL_COLUMNS, SEED


@dataclass
class RealMLPFit:
    """Fitted RealMLP model and timing details."""

    model: Any
    fit_seconds: float
    details: dict[str, Any]


def fit_model(X_train, y_train, X_validation, y_validation) -> RealMLPFit:
    """Fit tuned-default RealMLP with the signed validation set."""
    try:
        from pytabkit import RealMLP_TD_Classifier
    except ImportError:
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

    model = RealMLP_TD_Classifier(
        device="cuda",
        random_state=SEED,
        n_cv=1,
        n_refit=0,
        val_metric_name="cross_entropy",
        use_ls=False,
        verbosity=2,
    )
    started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        X_validation,
        y_validation,
        cat_col_names=list(CATEGORICAL_COLUMNS),
    )
    seconds = time.perf_counter() - started
    return RealMLPFit(
        model=model,
        fit_seconds=seconds,
        details={
            "model": "RealMLP_TD_Classifier",
            "device": "cuda",
            "random_state": SEED,
            "n_cv": 1,
            "n_refit": 0,
            "val_metric_name": "cross_entropy",
            "use_ls": False,
            "validation_rows": int(len(y_validation)),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
        },
    )


def predict_probability(fit: RealMLPFit, X_test) -> np.ndarray:
    """Predict positive-class probabilities."""
    probabilities = np.asarray(fit.model.predict_proba(X_test), dtype=float)
    classes = list(fit.model.classes_)
    return probabilities[:, classes.index(1)]
