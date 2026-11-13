"""Binary metrics, paired uncertainty, and durable run artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ._io import atomic_csv, atomic_json, atomic_parquet, sha256_file
from .config import METHODS


@dataclass(frozen=True)
class RunArtifactSet:
    """Paths written for one completed method and training size."""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    """Calculate the benchmark's four primary binary metrics."""
    y = np.asarray(y_true, dtype=np.uint8)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape:
        raise ValueError("y_true and probability must be aligned one-dimensional arrays")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between zero and one")
    if np.unique(y).tolist() != [0, 1]:
        raise ValueError("Metrics require both binary target classes")
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "test_rows": int(len(y)),
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    metric: str,
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Estimate a paired percentile interval for A minus B."""
    y = np.asarray(y_true, dtype=np.uint8)
    a = np.asarray(probability_a, dtype=float)
    b = np.asarray(probability_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("Paired bootstrap inputs must have identical lengths")
    functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": lambda truth, score: float(roc_auc_score(truth, score)),
        "log_loss": lambda truth, score: float(log_loss(truth, score, labels=[0, 1])),
    }
    if metric not in functions:
        raise ValueError("metric must be roc_auc or log_loss")
    measure = functions[metric]
    observed = measure(y, a) - measure(y, b)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    attempts = 0
    max_attempts = n_resamples * 2
    while len(differences) < n_resamples and attempts < max_attempts:
        indices = rng.integers(0, len(y), len(y))
        attempts += 1
        if np.unique(y[indices]).size < 2:
            continue
        differences.append(measure(y[indices], a[indices]) - measure(y[indices], b[indices]))
    if len(differences) != n_resamples:
        raise RuntimeError("Could not draw enough valid paired bootstrap samples")
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "metric": metric,
        "difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": n_resamples,
        "seed": seed,
    }


def run_output_dir(experiment_root: Path, method: str, train_size: int) -> Path:
    """Return the signed artifact directory for a method and size."""
    if method not in METHODS:
        raise ValueError(f"Unknown artifact method: {method}")
    return experiment_root / method / f"n{int(train_size)}"


def is_complete_run(output_dir: Path) -> bool:
    """Return true only for a valid atomic completion marker."""
    path = output_dir / "metrics.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("complete") is True)


def key_library_versions() -> dict[str, str | None]:
    """Collect installed benchmark library versions without importing them."""
    names = [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "pytabkit",
        "tabpfn",
        "torch",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_safe(value: Any) -> Any:
    """Convert nested library objects into stable JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    return str(value)


def _result_card(metrics: dict[str, Any], resources: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {metrics['method']} at {int(metrics['train_size']):,} training rows",
            "",
            f"- ROC-AUC: {metrics['roc_auc']:.6f}",
            f"- PR-AUC: {metrics['pr_auc']:.6f}",
            f"- Log loss: {metrics['log_loss']:.6f}",
            f"- Brier score: {metrics['brier_score']:.6f}",
            f"- Fit seconds: {float(resources.get('fit_seconds', 0.0)):.3f}",
            f"- Test prediction seconds: {float(resources.get('test_predict_seconds', 0.0)):.3f}",
            f"- Test rows per second: {float(resources.get('test_rows_per_second', 0.0)):.3f}",
            f"- Peak host RAM GiB: {float(resources.get('peak_host_ram_bytes', 0)) / 1024**3:.3f}",
            f"- Peak GPU memory GiB: {float(resources.get('peak_gpu_memory_bytes', 0)) / 1024**3:.3f}",
            "",
            f"Manifest hash: `{metrics['manifest_hash']}`",
            "",
        ]
    )


def _save_diagnostic_plots(
    output_dir: Path, y_true: np.ndarray, probability: np.ndarray, method: str
) -> list[str]:
    """Save ROC, precision-recall, and reliability plots for one run."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    observed, predicted = calibration_curve(
        y_true, probability, n_bins=15, strategy="quantile"
    )
    specifications = [
        ("roc_curve.png", fpr, tpr, "False-positive rate", "True-positive rate", True),
        ("precision_recall_curve.png", recall, precision, "Recall", "Precision", False),
        ("calibration.png", predicted, observed, "Mean predicted probability", "Observed frequency", True),
    ]
    names: list[str] = []
    for filename, x, y, xlabel, ylabel, diagonal in specifications:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(x, y, linewidth=2, label=method)
        if diagonal:
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{method}: {filename[:-4].replace('_', ' ')}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
        names.append(filename)
    return names


def save_run_artifacts(
    output_dir: Path,
    *,
    method: str,
    train_size: int,
    row_ids: np.ndarray,
    y_true: np.ndarray,
    probability: np.ndarray,
    resources: dict[str, Any],
    run_details: dict[str, Any],
    split_manifest: dict[str, Any],
    repo_commit: str | None = None,
    extra_frames: dict[str, pd.DataFrame] | None = None,
) -> RunArtifactSet:
    """Write one restart-safe run package with metrics last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Completed run already exists: {output_dir}")
    row_ids = np.asarray(row_ids, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.uint8)
    probability = np.asarray(probability, dtype=float)
    if not (len(row_ids) == len(y_true) == len(probability)):
        raise ValueError("Prediction artifact arrays must have identical lengths")
    predictions = pd.DataFrame(
        {"source_row_id": row_ids, "y_true": y_true, "probability": probability}
    )
    predictions_path = output_dir / "test_predictions.parquet"
    atomic_parquet(predictions_path, predictions)
    metrics = {
        "method": method,
        "train_size": int(train_size),
        **binary_metrics(y_true, probability),
        "manifest_hash": split_manifest["manifest_hash"],
        "test_row_ids_sha256": split_manifest["splits"]["test"]["row_ids_sha256"],
        "complete": True,
    }
    atomic_csv(output_dir / "metrics.csv", pd.DataFrame([metrics]))
    resources = _json_safe(resources)
    run_details = _json_safe(run_details)
    atomic_json(output_dir / "resources.json", resources)
    atomic_csv(output_dir / "resources.csv", pd.DataFrame([resources]))
    atomic_json(output_dir / "run_details.json", run_details)
    for filename, frame in (extra_frames or {}).items():
        atomic_csv(output_dir / filename, frame)
    (output_dir / "result_card.md").write_text(_result_card(metrics, resources))
    figure_names = _save_diagnostic_plots(output_dir, y_true, probability, method)
    artifact_names = [
        "test_predictions.parquet",
        "metrics.csv",
        "resources.json",
        "resources.csv",
        "run_details.json",
        "result_card.md",
        *figure_names,
        *(extra_frames or {}).keys(),
    ]
    checksums = {name: sha256_file(output_dir / name) for name in artifact_names}
    atomic_json(output_dir / "artifact_checksums.json", checksums)
    run_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "train_size": int(train_size),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "test_row_ids_sha256": split_manifest["splits"]["test"]["row_ids_sha256"],
        "repo_commit": repo_commit,
        "library_versions": key_library_versions(),
        "artifact_checksums_file": "artifact_checksums.json",
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    atomic_json(metrics_path, metrics)
    return RunArtifactSet(output_dir, metrics_path, predictions_path)


def sibling_metrics_table(experiment_root: Path) -> pd.DataFrame:
    """Load all completed compatible metric rows under an experiment."""
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for path in sorted((experiment_root / method).glob("n*/metrics.json")):
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if payload.get("complete") is True:
                rows.append(payload)
    columns = [
        "method",
        "train_size",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "test_rows",
        "manifest_hash",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["train_size", "roc_auc"], ascending=[True, False])
