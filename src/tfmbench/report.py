"""Cross-model performance review, figures, and portable review bundle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from ._io import atomic_csv, atomic_json, sha256_array, sha256_file
from .config import METHODS, PILOT_SIZES, BenchmarkConfig
from .evaluate import paired_bootstrap_difference


@dataclass(frozen=True)
class PerformanceReview:
    """Review status and paths written by the comparison step."""

    status: str
    review_dir: Path
    comparison_path: Path
    inventory: pd.DataFrame


def _expected_runs() -> list[tuple[str, int]]:
    sizes = [*PILOT_SIZES, 700_000]
    return [
        *((method, size) for method in ("lgbm_default", "lgbm_tuned", "tabpfn35") for size in sizes),
        ("realmlp", 700_000),
    ]


def _inventory_run(
    experiment_root: Path, method: str, size: int, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    run_dir = experiment_root / method / f"n{size}"
    row = {
        "method": method,
        "train_size": size,
        "relative_path": str(run_dir.relative_to(experiment_root)),
        "status": "missing",
        "reason": "run folder does not exist",
    }
    if not run_dir.exists():
        return row, None, None
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        row.update(status="partial", reason="metrics.json completion marker is missing")
        return row, None, None
    try:
        metrics = json.loads(metrics_path.read_text())
        resources = json.loads((run_dir / "resources.json").read_text())
        predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    except Exception as exc:
        row.update(status="failed", reason=f"could not load artifacts: {type(exc).__name__}: {exc}")
        return row, None, None
    if metrics.get("manifest_hash") != manifest["manifest_hash"]:
        row.update(status="incompatible", reason="split manifest hash differs")
        return row, metrics, resources
    expected_test_hash = manifest["splits"]["test"]["row_ids_sha256"]
    actual_test_hash = sha256_array(predictions["source_row_id"].to_numpy(dtype=np.int64))
    if actual_test_hash != expected_test_hash:
        row.update(status="incompatible", reason="test row identifier hash differs")
        return row, metrics, resources
    if len(predictions) != 100_000 or predictions["y_true"].nunique() != 2:
        row.update(status="incompatible", reason="test predictions are not the signed 100,000 binary rows")
        return row, metrics, resources
    checksums_path = run_dir / "artifact_checksums.json"
    if checksums_path.is_file():
        checksums = json.loads(checksums_path.read_text())
        for filename, expected in checksums.items():
            path = run_dir / filename
            if not path.is_file() or sha256_file(path) != expected:
                row.update(status="failed", reason=f"artifact checksum mismatch: {filename}")
                return row, metrics, resources
    row.update(status="complete", reason="all signed artifacts verified")
    metrics["relative_path"] = row["relative_path"]
    return row, metrics, resources


def _prediction_frame(experiment_root: Path, method: str, size: int) -> pd.DataFrame:
    return pd.read_parquet(
        experiment_root / method / f"n{size}" / "test_predictions.parquet"
    ).sort_values("source_row_id").reset_index(drop=True)


def _plot_700k_curves(experiment_root: Path, figure_dir: Path, available: set[str]) -> None:
    import matplotlib.pyplot as plt

    colors = {"lgbm_tuned": "#2b6cb0", "realmlp": "#38a169", "tabpfn35": "#c53030"}
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
    prevalence = None
    for method in ("lgbm_tuned", "realmlp", "tabpfn35"):
        if method not in available:
            continue
        frame = _prediction_frame(experiment_root, method, 700_000)
        y = frame["y_true"].to_numpy()
        probability = frame["probability"].to_numpy()
        prevalence = float(y.mean())
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        observed, predicted = calibration_curve(y, probability, n_bins=15, strategy="quantile")
        ax_roc.plot(fpr, tpr, label=method, color=colors[method])
        ax_pr.plot(recall, precision, label=method, color=colors[method])
        ax_cal.plot(predicted, observed, marker="o", label=method, color=colors[method])
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    if prevalence is not None:
        ax_pr.axhline(prevalence, color="black", linestyle="--", linewidth=1)
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves at 700k")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves at 700k")
    ax_cal.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Reliability at 700k")
    for figure, axis, name in (
        (fig_roc, ax_roc, "roc_curves_700k.png"),
        (fig_pr, ax_pr, "precision_recall_curves_700k.png"),
        (fig_cal, ax_cal, "calibration_700k.png"),
    ):
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / name, dpi=180)
        plt.close(figure)


def _plot_learning_and_time(
    metrics: pd.DataFrame,
    resources: pd.DataFrame,
    review_dir: Path,
    experiment_root: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = review_dir / "figures"
    for metric in ("roc_auc", "log_loss"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in ("lgbm_tuned", "tabpfn35"):
            rows = metrics[metrics["method"] == method].sort_values("train_size")
            if not rows.empty:
                ax.plot(rows["train_size"], rows[metric], marker="o", label=method)
        ax.set(xlabel="Training rows", ylabel=metric.replace("_", " ").title(), title=f"Learning curve: {metric.replace('_', ' ')}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"learning_curve_{metric}.png", dpi=180)
        plt.close(fig)
    pilot_path = experiment_root / "pilot" / "pilot_report.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.is_file() else {}
    for resource_name, title in (
        ("fit_seconds", "Fit time"),
        ("test_predict_seconds", "Test prediction time"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in ("lgbm_tuned", "tabpfn35"):
            rows = resources[resources["method"] == method].sort_values("train_size")
            if not rows.empty:
                ax.plot(rows["train_size"], rows[resource_name], marker="o", label=method)
        projection = pilot.get("projections")
        if projection:
            key = "fit_power_law" if resource_name == "fit_seconds" else "predict_power_law"
            model = projection[key]
            x = np.linspace(50_000, 700_000, 100)
            y = model["a"] * x ** model["b"]
            ax.plot(x, y, linestyle="--", color="gray", label="TabPFN pilot projection")
        ax.set(xlabel="Training rows", ylabel="Seconds", title=title)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{resource_name}_scaling.png", dpi=180)
        plt.close(fig)


def _comparison_markdown(
    status: str,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    inventory: pd.DataFrame,
) -> str:
    lines = [
        "# Credit benchmark performance review",
        "",
        f"Review status: **{status}**",
        "",
        "## Completed runs",
        "",
    ]
    if metrics.empty:
        lines.append("No compatible completed runs were found.")
    else:
        lines.append(metrics.to_markdown(index=False))
    lines.extend(["", "## Paired 700k comparisons", ""])
    lines.append(bootstrap.to_markdown(index=False) if not bootstrap.empty else "The required 700k pairs are not complete yet.")
    lines.extend(["", "## Run inventory", "", inventory.to_markdown(index=False), "", "## Figures", ""])
    lines.extend(
        [
            "- `figures/roc_curves_700k.png`",
            "- `figures/precision_recall_curves_700k.png`",
            "- `figures/calibration_700k.png`",
            "- `figures/learning_curve_roc_auc.png`",
            "- `figures/learning_curve_log_loss.png`",
            "- `figures/fit_seconds_scaling.png`",
            "- `figures/test_predict_seconds_scaling.png`",
            "",
            "ROC-AUC and PR-AUC are higher-is-better. Log loss and Brier score are lower-is-better. Resource use is reported separately and is not folded into a composite score.",
            "",
        ]
    )
    return "\n".join(lines)


def build_performance_review(
    output_root: str | Path,
    experiment_name: str = "lending_club_900k",
    *,
    bootstrap_resamples: int = 2_000,
) -> PerformanceReview:
    """Verify completed runs and build the Drive performance-review bundle."""
    config = BenchmarkConfig(output_root=output_root, experiment_name=experiment_name)
    config.validate()
    manifest = json.loads((config.shared_dir / "split_manifest.json").read_text())
    review_dir = config.experiment_root / "performance_review"
    figure_dir = review_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for method, size in _expected_runs():
        inventory, metrics, resources = _inventory_run(
            config.experiment_root, method, size, manifest
        )
        inventory_rows.append(inventory)
        if inventory["status"] == "complete" and metrics is not None and resources is not None:
            metric_rows.append(metrics)
            resource_rows.append({"method": method, "train_size": size, **resources})
    inventory_frame = pd.DataFrame(inventory_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    resources_frame = pd.DataFrame(resource_rows)
    core = {("lgbm_tuned", 700_000), ("realmlp", 700_000), ("tabpfn35", 700_000)}
    complete = {
        (row.method, int(row.train_size))
        for row in inventory_frame.itertuples()
        if row.status == "complete"
    }
    incompatible = inventory_frame[inventory_frame["status"].isin(["incompatible", "failed"])]
    expected = set(_expected_runs())
    status = "FAIL" if not incompatible.empty else "PASS" if expected <= complete else "WARN"
    atomic_csv(review_dir / "run_inventory.csv", inventory_frame)
    atomic_csv(review_dir / "all_metrics.csv", metrics_frame)
    atomic_csv(review_dir / "resource_summary.csv", resources_frame)
    if not metrics_frame.empty:
        wide = metrics_frame[metrics_frame["train_size"] == 700_000].pivot(
            index="method", columns="train_size", values=["roc_auc", "pr_auc", "log_loss", "brier_score"]
        )
        wide.columns = [f"{metric}_{size}" for metric, size in wide.columns]
        atomic_csv(review_dir / "metrics_700k_wide.csv", wide.reset_index())
        best_rows = []
        full = metrics_frame[metrics_frame["train_size"] == 700_000]
        for metric, ascending in (("roc_auc", False), ("pr_auc", False), ("log_loss", True), ("brier_score", True)):
            if not full.empty:
                best = full.sort_values(metric, ascending=ascending).iloc[0]
                best_rows.append({"metric": metric, "method": best["method"], "value": best[metric], "direction": "lower" if ascending else "higher"})
        atomic_csv(review_dir / "best_by_metric.csv", pd.DataFrame(best_rows))
    bootstrap_rows: list[dict[str, Any]] = []
    if core <= complete:
        tab = _prediction_frame(config.experiment_root, "tabpfn35", 700_000)
        for baseline in ("lgbm_tuned", "realmlp"):
            other = _prediction_frame(config.experiment_root, baseline, 700_000)
            if not tab[["source_row_id", "y_true"]].equals(other[["source_row_id", "y_true"]]):
                raise ValueError(f"Paired comparison rows do not align for {baseline}")
            for metric in ("roc_auc", "log_loss"):
                row = paired_bootstrap_difference(
                    tab["y_true"].to_numpy(),
                    tab["probability"].to_numpy(),
                    other["probability"].to_numpy(),
                    metric,
                    n_resamples=bootstrap_resamples,
                    seed=0,
                )
                bootstrap_rows.append({"comparison": f"tabpfn35_minus_{baseline}", **row})
        _plot_700k_curves(config.experiment_root, figure_dir, {method for method, _ in core})
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    atomic_csv(review_dir / "paired_bootstrap.csv", bootstrap_frame)
    if not metrics_frame.empty and not resources_frame.empty:
        _plot_learning_and_time(metrics_frame, resources_frame, review_dir, config.experiment_root)
    review_status = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest_hash": manifest["manifest_hash"],
        "completed_runs": len(complete),
        "expected_runs": len(_expected_runs()),
        "core_700k_complete": core <= complete,
        "issues": incompatible.to_dict(orient="records"),
    }
    atomic_json(review_dir / "review_status.json", review_status)
    comparison = {
        **review_status,
        "metrics_file": "all_metrics.csv",
        "resources_file": "resource_summary.csv",
        "bootstrap_file": "paired_bootstrap.csv",
        "figures_dir": "figures",
    }
    atomic_json(review_dir / "comparison.json", comparison)
    comparison_path = review_dir / "comparison.md"
    comparison_path.write_text(
        _comparison_markdown(status, metrics_frame, bootstrap_frame, inventory_frame)
    )
    bundle_path = review_dir / "review_bundle.zip"
    with zipfile.ZipFile(bundle_path.with_suffix(".zip.tmp"), "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in review_dir.rglob("*"):
            if path.is_file() and path.name not in {"review_bundle.zip", "review_bundle.zip.tmp"}:
                bundle.write(path, path.relative_to(review_dir))
    bundle_path.with_suffix(".zip.tmp").replace(bundle_path)
    print(f"Performance review status: {status}")
    print(f"Review folder: {review_dir}")
    return PerformanceReview(status, review_dir, comparison_path, inventory_frame)
