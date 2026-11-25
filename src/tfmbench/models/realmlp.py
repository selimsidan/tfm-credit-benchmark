"""RealMLP neural baseline wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import CATEGORICAL_COLUMNS, SEED


@dataclass
class RealMLPFit:
    """Fitted RealMLP model and timing details."""

    model: Any
    fit_seconds: float
    details: dict[str, Any]
    numeric_fill_values: pd.Series


def _prepare_input(frame: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    # Nullable numeric dtypes can trigger sklearn to cast the entire frame,
    # including string categories. RealMLP also rejects missing continuous values.
    numeric = frame[fill_values.index].astype(np.float32).fillna(fill_values)
    return pd.concat([frame.drop(columns=fill_values.index), numeric], axis=1)[frame.columns]


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
    numeric_columns = X_train.columns.difference(CATEGORICAL_COLUMNS, sort=False)
    fill_values = X_train[numeric_columns].astype(np.float32).median().fillna(0.0)
    model.fit(
        _prepare_input(X_train, fill_values),
        y_train,
        _prepare_input(X_validation, fill_values),
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
            "numeric_missing_values": "training median, or zero for all-missing columns",
            "numeric_fill_values": fill_values.to_dict(),
        },
        numeric_fill_values=fill_values,
    )


def predict_probability(fit: RealMLPFit, X_test) -> np.ndarray:
    """Predict positive-class probabilities."""
    probabilities = np.asarray(
        fit.model.predict_proba(_prepare_input(X_test, fit.numeric_fill_values)), dtype=float
    )
    classes = list(fit.model.classes_)
    return probabilities[:, classes.index(1)]
