"""Default and Optuna-tuned LightGBM wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    CATEGORICAL_COLUMNS,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    MAX_OPTUNA_TRIALS,
    SEED,
)


@dataclass
class LightGBMFit:
    """One fitted booster and its reproducibility details."""

    model: Any
    fit_seconds: float
    tuning_seconds: float
    details: dict[str, Any]
    trials: pd.DataFrame | None = None


def _datasets(X_train, y_train, X_validation, y_validation):
    import lightgbm as lgb

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=list(CATEGORICAL_COLUMNS),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_validation,
        label=y_validation,
        reference=train_set,
        categorical_feature=list(CATEGORICAL_COLUMNS),
        free_raw_data=False,
    )
    return train_set, valid_set


def _base_params() -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": -1,
    }


def fit_default(X_train, y_train, X_validation, y_validation) -> LightGBMFit:
    """Fit the default reference with validation early stopping."""
    import lightgbm as lgb

    train_set, valid_set = _datasets(X_train, y_train, X_validation, y_validation)
    started = time.perf_counter()
    model = lgb.train(
        _base_params(),
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    seconds = time.perf_counter() - started
    return LightGBMFit(
        model=model,
        fit_seconds=seconds,
        tuning_seconds=0.0,
        details={
            "kind": "default",
            "best_iteration": int(model.best_iteration),
            "validation_log_loss": float(model.best_score["validation"]["binary_logloss"]),
            "parameters": _base_params(),
        },
    )

def fit_tuned(
    X_train,
    y_train,
    X_validation,
    y_validation,
    *,
    study_path: Path,
    timeout_minutes: float,
    max_trials: int = MAX_OPTUNA_TRIALS,
) -> LightGBMFit:
    """Resume TPE tuning, then refit the selected parameters once."""
    import lightgbm as lgb
    import optuna

    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_path.resolve()}"
    study = optuna.create_study(
        study_name=f"lgbm_n{len(y_train)}",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial) -> float:
        params = {
            **_base_params(),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 1023, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [255, 511]),
        }
        train_set, valid_set = _datasets(X_train, y_train, X_validation, y_validation)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[valid_set],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return float(booster.best_score["validation"]["binary_logloss"])

    remaining = max(0, max_trials - len(study.trials))
    tuning_started = time.perf_counter()
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=float(timeout_minutes) * 60,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    tuning_seconds = time.perf_counter() - tuning_started
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("LightGBM tuning completed without a successful trial")

    best_params = {**_base_params(), **study.best_params}
    if "bagging_fraction" in best_params:
        best_params["bagging_freq"] = 1
    train_set, valid_set = _datasets(X_train, y_train, X_validation, y_validation)
    fit_started = time.perf_counter()
    model = lgb.train(
        best_params,
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    fit_seconds = time.perf_counter() - fit_started
    return LightGBMFit(
        model=model,
        fit_seconds=fit_seconds,
        tuning_seconds=tuning_seconds,
        details={
            "kind": "optuna_tuned",
            "study_name": study.study_name,
            "study_path": str(study_path),
            "completed_trials": sum(
                trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
            ),
            "best_trial_number": int(study.best_trial.number),
            "best_parameters": study.best_params,
            "best_iteration": int(model.best_iteration),
            "validation_log_loss": float(model.best_score["validation"]["binary_logloss"]),
        },
        trials=study.trials_dataframe(),
    )


def predict_probability(fit: LightGBMFit, X_test) -> np.ndarray:
    """Predict positive-class probabilities from the selected iteration."""
    return np.asarray(
        fit.model.predict(X_test, num_iteration=fit.model.best_iteration), dtype=float
    )
