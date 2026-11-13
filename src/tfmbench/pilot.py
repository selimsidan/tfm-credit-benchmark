"""Pilot orchestration, scaling projections, and full-run safety gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import atomic_csv, atomic_json
from .config import (
    COMPUTE_UNITS_PER_HOUR,
    MAX_OPTUNA_TRIALS,
    MAX_SESSION_HOURS,
    MEMORY_GATE_FRACTION,
    PILOT_SIZES,
    PILOT_TUNING_MINUTES,
    BenchmarkConfig,
)
from .resources import hardware_info
from .run import run_method


def fit_power_law(sizes: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Fit t = a times n to the power b in log space."""
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(sizes) < 2 or (sizes <= 0).any() or (values <= 0).any():
        raise ValueError("Power-law fitting requires at least two positive observations")
    exponent, log_scale = np.polyfit(np.log(sizes), np.log(values), 1)
    return {"a": float(np.exp(log_scale)), "b": float(exponent)}


def predict_power_law(model: dict[str, float], size: int) -> float:
    """Evaluate a fitted power law."""
    return float(model["a"] * float(size) ** model["b"])


def fit_linear(sizes: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Fit value = intercept plus slope times rows."""
    slope, intercept = np.polyfit(np.asarray(sizes, float), np.asarray(values, float), 1)
    return {"intercept": float(intercept), "slope": float(slope)}


def predict_linear(model: dict[str, float], size: int) -> float:
    """Evaluate a fitted linear model and clamp physical resources at zero."""
    return max(0.0, float(model["intercept"] + model["slope"] * float(size)))


def evaluate_gate_checks(
    runs: list[dict[str, Any]],
    projections: dict[str, Any] | None,
    hardware: dict[str, Any],
    *,
    max_session_hours: float = MAX_SESSION_HOURS,
) -> list[dict[str, Any]]:
    """Evaluate every automatic pilot gate with an explicit reason."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, reason: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "reason": reason})

    expected = {(method, size) for method in ("lgbm_tuned", "tabpfn35") for size in PILOT_SIZES}
    finished = {(row["method"], int(row["train_size"])) for row in runs if row.get("status") == "complete"}
    add("all_runs_finished", expected <= finished, f"finished {len(expected & finished)}/{len(expected)} required runs")
    finite = all(
        math.isfinite(float(row.get(metric, math.nan)))
        for row in runs
        if row.get("status") == "complete" and row.get("method") in {"lgbm_tuned", "tabpfn35"}
        for metric in ("roc_auc", "pr_auc", "log_loss", "brier_score")
    ) and expected <= finished
    add("finite_metrics", finite, "all required primary metrics are finite" if finite else "a required metric is missing or non-finite")
    auc_rows = [row for row in runs if row.get("method") in {"lgbm_tuned", "tabpfn35"}]
    auc_ok = len(auc_rows) == 6 and all(0.60 <= float(row.get("roc_auc", math.nan)) <= 0.85 for row in auc_rows)
    add("auc_sanity_range", auc_ok, "every required ROC-AUC is within [0.60, 0.85]" if auc_ok else "a required ROC-AUC is missing or outside [0.60, 0.85]")
    tab_rows = sorted(
        (row for row in runs if row.get("method") == "tabpfn35" and row.get("status") == "complete"),
        key=lambda row: row["train_size"],
    )
    monotonic = len(tab_rows) == 3 and all(
        float(later["roc_auc"]) >= float(earlier["roc_auc"]) - 0.01
        for earlier, later in zip(tab_rows, tab_rows[1:])
    )
    add("tabpfn_auc_stability", monotonic, "TabPFN ROC-AUC never drops by more than 0.01" if monotonic else "TabPFN ROC-AUC drops by more than 0.01 or is incomplete")
    no_exhaustion = len(tab_rows) == 3 and all(row.get("oom_exhausted") is not True for row in tab_rows)
    add("oom_retry_not_exhausted", no_exhaustion, "all TabPFN query inference completed" if no_exhaustion else "a TabPFN query retry was exhausted or is incomplete")
    if projections is None:
        add("projected_gpu_memory", False, "projection unavailable")
        add("projected_host_memory", False, "projection unavailable")
        add("projected_session_time", False, "projection unavailable")
        return checks
    gpu_limit = float(hardware.get("gpu_vram_bytes", 0)) * MEMORY_GATE_FRACTION
    host_limit = float(hardware.get("system_ram_bytes", 0)) * MEMORY_GATE_FRACTION
    projected_gpu = float(projections["peak_gpu_memory_bytes_700k"])
    projected_host = float(projections["peak_host_ram_bytes_700k"])
    projected_hours = float(projections["tabpfn_total_hours_700k"])
    add("projected_gpu_memory", projected_gpu < gpu_limit, f"projected {projected_gpu / 1e9:.2f} GB versus 90 percent limit {gpu_limit / 1e9:.2f} GB")
    add("projected_host_memory", projected_host < host_limit, f"projected {projected_host / 1024**3:.2f} GiB versus 90 percent limit {host_limit / 1024**3:.2f} GiB")
    add("projected_session_time", projected_hours < max_session_hours, f"projected {projected_hours:.2f} hours versus limit {max_session_hours:.2f} hours")
    return checks


def _load_run(experiment_root: Path, method: str, size: int) -> dict[str, Any]:
    output_dir = experiment_root / method / f"n{size}"
    metrics = json.loads((output_dir / "metrics.json").read_text())
    resources = json.loads((output_dir / "resources.json").read_text())
    return {
        "status": "complete",
        **metrics,
        **{
            "fit_seconds": resources["fit_seconds"],
            "test_predict_seconds": resources["test_predict_seconds"],
            "peak_host_ram_bytes": resources["peak_host_ram_bytes"],
            "peak_gpu_memory_bytes": resources["peak_gpu_memory_bytes"],
            "oom_retries": resources.get("oom_retries", 0),
            "oom_exhausted": False,
        },
    }


def _project(runs: list[dict[str, Any]], compute_units_per_hour: float) -> dict[str, Any]:
    tab = sorted(
        (row for row in runs if row.get("method") == "tabpfn35" and row.get("status") == "complete"),
        key=lambda row: row["train_size"],
    )
    if len(tab) != 3:
        raise ValueError("Three complete TabPFN pilot runs are required for projection")
    sizes = np.asarray([row["train_size"] for row in tab], dtype=float)
    fit_model = fit_power_law(sizes, np.asarray([row["fit_seconds"] for row in tab]))
    predict_model = fit_power_law(sizes, np.asarray([row["test_predict_seconds"] for row in tab]))
    gpu_model = fit_linear(sizes, np.asarray([row["peak_gpu_memory_bytes"] for row in tab]))
    host_model = fit_linear(sizes, np.asarray([row["peak_host_ram_bytes"] for row in tab]))
    fit_700 = predict_power_law(fit_model, 700_000)
    predict_700 = predict_power_law(predict_model, 700_000)
    hours = (fit_700 + predict_700) / 3600
    return {
        "fit_power_law": fit_model,
        "predict_power_law": predict_model,
        "gpu_memory_linear": gpu_model,
        "host_memory_linear": host_model,
        "fit_seconds_700k": fit_700,
        "test_predict_seconds_700k": predict_700,
        "peak_gpu_memory_bytes_700k": predict_linear(gpu_model, 700_000),
        "peak_host_ram_bytes_700k": predict_linear(host_model, 700_000),
        "tabpfn_total_hours_700k": hours,
        "compute_units_per_hour": float(compute_units_per_hour),
        "projected_compute_units_700k": hours * float(compute_units_per_hour),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Pilot report",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        f"Manifest hash: `{report['manifest_hash']}`",
        "",
        "## Runs",
        "",
        "| Method | Rows | ROC-AUC | Log loss | Fit seconds | Predict seconds | Peak GPU GiB | Peak host GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row.get('method')} | {int(row.get('train_size', 0)):,} | "
            f"{float(row.get('roc_auc', math.nan)):.6f} | {float(row.get('log_loss', math.nan)):.6f} | "
            f"{float(row.get('fit_seconds', math.nan)):.1f} | {float(row.get('test_predict_seconds', math.nan)):.1f} | "
            f"{float(row.get('peak_gpu_memory_bytes', 0)) / 1024**3:.2f} | "
            f"{float(row.get('peak_host_ram_bytes', 0)) / 1024**3:.2f} |"
        )
    lines.extend(["", "## Projections", ""])
    if report["projections"]:
        projection = report["projections"]
        lines.extend(
            [
                f"- Total TabPFN time at 700k: {projection['tabpfn_total_hours_700k']:.2f} hours",
                f"- Compute units at 700k: {projection['projected_compute_units_700k']:.2f}",
                f"- Peak GPU memory at 700k: {projection['peak_gpu_memory_bytes_700k'] / 1e9:.2f} GB",
                f"- Peak host RAM at 700k: {projection['peak_host_ram_bytes_700k'] / 1024**3:.2f} GiB",
            ]
        )
    else:
        lines.append("Projection unavailable because the required pilot runs did not finish.")
    lines.extend(["", "## Gate checks", ""])
    for check in report["checks"]:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'}: {check['name']}: {check['reason']}")
    return "\n".join(lines) + "\n"


def run_pilot(
    output_root: str | Path,
    experiment_name: str = "lending_club_900k",
    *,
    force_rerun: bool = False,
    tuning_minutes: float = PILOT_TUNING_MINUTES,
    max_trials: int = MAX_OPTUNA_TRIALS,
    compute_units_per_hour: float = COMPUTE_UNITS_PER_HOUR,
    max_session_hours: float = MAX_SESSION_HOURS,
) -> dict[str, Any]:
    """Run the staged pilot and write an auditable PASS or FAIL report."""
    config = BenchmarkConfig(output_root=output_root, experiment_name=experiment_name)
    config.validate()
    manifest = json.loads((config.shared_dir / "split_manifest.json").read_text())
    pilot_dir = config.experiment_root / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    error: str | None = None
    for size in PILOT_SIZES:
        print(f"Starting pilot size {size:,}")
        try:
            run_method(
                "lgbm",
                size,
                output_root,
                experiment_name,
                force_rerun=force_rerun,
                tuning_minutes=tuning_minutes,
                max_trials=max_trials,
            )
            runs.append(_load_run(config.experiment_root, "lgbm_tuned", size))
            run_method(
                "tabpfn35",
                size,
                output_root,
                experiment_name,
                force_rerun=force_rerun,
            )
            runs.append(_load_run(config.experiment_root, "tabpfn35", size))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            runs.append({"status": "failed", "train_size": size, "error": error})
            print(f"Pilot stopped at {size:,}: {error}")
            break
    try:
        projections = _project(runs, compute_units_per_hour)
    except (ValueError, FloatingPointError):
        projections = None
    hardware = hardware_info()
    checks = evaluate_gate_checks(
        runs, projections, hardware, max_session_hours=max_session_hours
    )
    verdict = "PASS" if checks and all(check["passed"] for check in checks) else "FAIL"
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_hash": manifest["manifest_hash"],
        "verdict": verdict,
        "runs": runs,
        "projections": projections,
        "checks": checks,
        "hardware": hardware,
        "error": error,
    }
    atomic_json(pilot_dir / "pilot_report.json", report)
    (pilot_dir / "pilot_report.md").write_text(_markdown_report(report))
    if projections:
        atomic_csv(pilot_dir / "projections.csv", pd.DataFrame([projections]))
    print(f"Pilot verdict: {verdict}")
    if projections:
        print(
            f"Projected 700k TabPFN: {projections['tabpfn_total_hours_700k']:.2f} hours, "
            f"{projections['projected_compute_units_700k']:.2f} compute units"
        )
    return report
