"""TabPFN-3.5 full-context fit and memory-safe query inference."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
import time
from typing import Any

import numpy as np

from ..config import INITIAL_PREDICTION_CHUNK_ROWS, MIN_PREDICTION_CHUNK_ROWS, SEED


@dataclass
class TabPFNFit:
    """Fitted TabPFN model and immutable context details."""

    model: Any
    fit_seconds: float
    train_rows: int
    details: dict[str, Any]


def is_oom_error(exc: BaseException) -> bool:
    """Recognize CUDA out-of-memory exceptions without hiding other errors."""
    text = f"{type(exc).__name__} {exc}".lower()
    return "outofmemory" in text or "out of memory" in text or "cuda oom" in text


def predict_with_chunk_retries(
    model: Any,
    X,
    *,
    initial_chunk_rows: int = INITIAL_PREDICTION_CHUNK_ROWS,
    minimum_chunk_rows: int = MIN_PREDICTION_CHUNK_ROWS,
) -> tuple[np.ndarray, int, int]:
    """Retry only query chunks after OOM and never change fitted context."""
    chunk_rows = int(initial_chunk_rows)
    retries = 0
    attempted: list[int] = []
    while chunk_rows >= minimum_chunk_rows:
        attempted.append(chunk_rows)
        parts: list[np.ndarray] = []
        try:
            for start in range(0, len(X), chunk_rows):
                part = np.asarray(model.predict_proba(X.iloc[start : start + chunk_rows]))
                parts.append(part)
                if start == 0 or (start // chunk_rows) % 50 == 0:
                    print(
                        f"predicted {min(start + chunk_rows, len(X)):,}/{len(X):,} "
                        f"rows with chunk={chunk_rows}"
                    )
            probabilities = np.concatenate(parts, axis=0)
            classes = list(model.classes_)
            return probabilities[:, classes.index(1)], chunk_rows, retries
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            retries += 1
            print(
                f"OOM with query chunk={chunk_rows}; retrying without changing "
                f"the fitted training context"
            )
            parts.clear()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
            chunk_rows //= 2
    raise RuntimeError(
        f"TabPFN-3.5 could not score after query chunk retries {attempted}. "
        "The training context was not subsampled. Use a higher-VRAM GPU."
    )


def fit_model(
    X_train, y_train, *, fit_mode: str = "fit_with_cache"
) -> TabPFNFit:
    """Fit TabPFN-3.5 on every supplied training row."""
    allowed_modes = {"low_memory", "fit_preprocessors", "fit_with_cache"}
    if fit_mode not in allowed_modes:
        raise ValueError(f"fit_mode must be one of {sorted(allowed_modes)}")
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(
            "Missing TABPFN_TOKEN. Accept the TabPFN-3.5 model license and add "
            "TABPFN_TOKEN as a Colab secret with notebook access."
        )
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        n_estimators=8,
        device="cuda",
        fit_mode=fit_mode,
        memory_saving_mode=True,
        keep_cache_on_device=False,
        random_state=SEED,
        show_progress_bar=True,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    inference_config = model.get_inference_config()
    paths = getattr(model, "model_path", None)
    if isinstance(paths, (str, os.PathLike)):
        model_paths = [str(paths)]
    elif paths is None:
        model_paths = []
    else:
        model_paths = [str(path) for path in paths]
    return TabPFNFit(
        model=model,
        fit_seconds=fit_seconds,
        train_rows=len(y_train),
        details={
            "library": "tabpfn",
            "package_version": importlib.metadata.version("tabpfn"),
            "model_version": "v3.5",
            "model_paths": model_paths,
            "resolved_inference_config": inference_config,
            "n_estimators": 8,
            "fit_mode": fit_mode,
            "memory_saving_mode": True,
            "keep_cache_on_device": False,
            "random_state": SEED,
            "training_context_rows": int(len(y_train)),
            "row_subsampling": None,
        },
    )
