"""Notebook-facing model orchestration and command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import pandas as pd

from .config import (
    FULL_TUNING_MINUTES,
    MAX_OPTUNA_TRIALS,
    PILOT_TUNING_MINUTES,
    BenchmarkConfig,
)
from .data import PreparedSplits, load_prepared_splits
from .evaluate import (
    is_complete_run,
    run_output_dir,
    save_run_artifacts,
    sibling_metrics_table,
)
from .resources import ResourceMonitor, hardware_info, require_cuda, require_tabpfn_full_hardware
from .split import verify_manifest


def _repo_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare_output(
    output_dir: Path, force_rerun: bool, expected_manifest_hash: str
) -> bool:
    if is_complete_run(output_dir) and not force_rerun:
        completed = json.loads((output_dir / "metrics.json").read_text())
        if completed.get("manifest_hash") != expected_manifest_hash:
            raise RuntimeError(
                f"Completed run at {output_dir} belongs to a different split manifest. "
                "Set FORCE_RERUN=True to replace this exact run folder."
            )
        print(f"Skipping completed run: {output_dir}")
        return False
    if output_dir.exists():
        if not force_rerun:
            raise RuntimeError(
                f"Partial run artifacts exist at {output_dir}. Inspect them, then set "
                "FORCE_RERUN=True to replace this exact run folder."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return True


def _resource_row(
    *,
    fit_seconds: float,
    test_predict_seconds: float,
    test_rows: int,
    monitor: ResourceMonitor,
    tuning_seconds: float = 0.0,
    oom_retries: int = 0,
) -> dict[str, Any]:
    return {
        "fit_seconds": float(fit_seconds),
        "tuning_seconds": float(tuning_seconds),
        "test_predict_seconds": float(test_predict_seconds),
        "test_rows_per_second": float(test_rows / test_predict_seconds),
        "peak_host_ram_bytes": int(monitor.peak_host_ram_bytes),
        "peak_gpu_memory_bytes": int(monitor.peak_gpu_memory_bytes),
        "oom_retries": int(oom_retries),
        **hardware_info(),
    }


def _save_lgbm_run(
    data: PreparedSplits,
    config: BenchmarkConfig,
    *,
    kind: str,
    train_size: int,
    force_rerun: bool,
    tuning_minutes: float,
    max_trials: int,
) -> Path:
    from .models import lgbm

    method = f"lgbm_{kind}"
    output_dir = run_output_dir(config.experiment_root, method, train_size)
    if not _prepare_output(output_dir, force_rerun, data.manifest["manifest_hash"]):
        return output_dir
    with ResourceMonitor() as monitor:
        if kind == "default":
            fit = lgbm.fit_default(
                data.X_train, data.y_train, data.X_validation, data.y_validation
            )
        elif kind == "tuned":
            fit = lgbm.fit_tuned(
                data.X_train,
                data.y_train,
                data.X_validation,
                data.y_validation,
                study_path=config.experiment_root / "studies" / f"lgbm_n{train_size}.sqlite3",
                timeout_minutes=tuning_minutes,
                max_trials=max_trials,
            )
        else:
            raise ValueError(f"Unknown LightGBM kind: {kind}")
        prediction_started = time.perf_counter()
        probability = lgbm.predict_probability(fit, data.X_test)
        prediction_seconds = time.perf_counter() - prediction_started
    resources = _resource_row(
        fit_seconds=fit.fit_seconds,
        tuning_seconds=fit.tuning_seconds,
        test_predict_seconds=prediction_seconds,
        test_rows=len(data.y_test),
        monitor=monitor,
    )
    extra = {"tuning_trials.csv": fit.trials} if fit.trials is not None else None
    fit.model.save_model(str(output_dir / "model.txt"))
    save_run_artifacts(
        output_dir,
        method=method,
        train_size=train_size,
        row_ids=data.test_row_ids,
        y_true=data.y_test,
        probability=probability,
        resources=resources,
        run_details=fit.details,
        split_manifest=data.manifest,
        repo_commit=_repo_commit(),
        extra_frames=extra,
    )
    return output_dir


def _save_realmlp_run(
    data: PreparedSplits,
    config: BenchmarkConfig,
    *,
    train_size: int,
    force_rerun: bool,
) -> Path:
    from .models import realmlp

    output_dir = run_output_dir(config.experiment_root, "realmlp", train_size)
    if not _prepare_output(output_dir, force_rerun, data.manifest["manifest_hash"]):
        return output_dir
    require_cuda("RealMLP")
    with ResourceMonitor() as monitor:
        fit = realmlp.fit_model(
            data.X_train, data.y_train, data.X_validation, data.y_validation
        )
        prediction_started = time.perf_counter()
        probability = realmlp.predict_probability(fit, data.X_test)
        prediction_seconds = time.perf_counter() - prediction_started
    resources = _resource_row(
        fit_seconds=fit.fit_seconds,
        test_predict_seconds=prediction_seconds,
        test_rows=len(data.y_test),
        monitor=monitor,
    )
    save_run_artifacts(
        output_dir,
        method="realmlp",
        train_size=train_size,
        row_ids=data.test_row_ids,
        y_true=data.y_test,
        probability=probability,
        resources=resources,
        run_details=fit.details,
        split_manifest=data.manifest,
        repo_commit=_repo_commit(),
    )
    return output_dir


def _verify_pilot_gate(config: BenchmarkConfig, ignore: bool) -> dict[str, Any] | None:
    report_path = config.experiment_root / "pilot" / "pilot_report.json"
    if ignore:
        print("WARNING: IGNORE_PILOT_GATE=True; the 700,000-row safety gate is overridden")
        return json.loads(report_path.read_text()) if report_path.is_file() else None
    if not report_path.is_file():
        raise RuntimeError(
            "Missing pilot_report.json. Run 01_pilot.ipynb and obtain a PASS before "
            "starting the 700,000-row TabPFN run."
        )
    report = json.loads(report_path.read_text())
    if report.get("verdict") != "PASS":
        raise RuntimeError(
            f"Pilot verdict is {report.get('verdict')!r}, not PASS. Review pilot_report.md."
        )
    if report.get("manifest_hash") != json.loads(
        (config.shared_dir / "split_manifest.json").read_text()
    ).get("manifest_hash"):
        raise RuntimeError("Pilot report manifest hash does not match the current shared split")
    return report


def _save_tabpfn_run(
    data: PreparedSplits,
    config: BenchmarkConfig,
    *,
    train_size: int,
    force_rerun: bool,
    ignore_pilot_gate: bool,
    predict_validation: bool,
    fit_mode: str = "fit_with_cache",
) -> Path:
    from .models import tabpfn35

    output_dir = run_output_dir(config.experiment_root, "tabpfn35", train_size)
    if not _prepare_output(output_dir, force_rerun, data.manifest["manifest_hash"]):
        return output_dir
    require_cuda("TabPFN-3.5")
    pilot_report = None
    if train_size == 700_000:
        pilot_report = _verify_pilot_gate(config, ignore_pilot_gate)
        require_tabpfn_full_hardware()
        if pilot_report:
            print("Pilot projection:", pilot_report.get("projections", {}))
    with ResourceMonitor() as monitor:
        fit = tabpfn35.fit_model(data.X_train, data.y_train, fit_mode=fit_mode)
        validation_details: dict[str, Any] = {"validation_predictions_enabled": False}
        if predict_validation:
            _, validation_chunk, validation_retries = tabpfn35.predict_with_chunk_retries(
                fit.model, data.X_validation
            )
            validation_details = {
                "validation_predictions_enabled": True,
                "validation_chunk_rows": validation_chunk,
                "validation_oom_retries": validation_retries,
            }
        prediction_started = time.perf_counter()
        probability, chunk_rows, retries = tabpfn35.predict_with_chunk_retries(
            fit.model, data.X_test
        )
        prediction_seconds = time.perf_counter() - prediction_started
    resources = _resource_row(
        fit_seconds=fit.fit_seconds,
        test_predict_seconds=prediction_seconds,
        test_rows=len(data.y_test),
        monitor=monitor,
        oom_retries=retries,
    )
    details = {
        **fit.details,
        **validation_details,
        "test_chunk_rows": chunk_rows,
        "test_oom_retries": retries,
    }
    save_run_artifacts(
        output_dir,
        method="tabpfn35",
        train_size=train_size,
        row_ids=data.test_row_ids,
        y_true=data.y_test,
        probability=probability,
        resources=resources,
        run_details=details,
        split_manifest=data.manifest,
        repo_commit=_repo_commit(),
    )
    return output_dir


def run_method(
    method: str,
    train_size: int,
    output_root: str | Path,
    experiment_name: str = "lending_club_900k",
    *,
    force_rerun: bool = False,
    ignore_pilot_gate: bool = False,
    tuning_minutes: float | None = None,
    max_trials: int = MAX_OPTUNA_TRIALS,
    predict_validation: bool = False,
) -> pd.DataFrame:
    """Run one benchmark method and return the current sibling table."""
    config = BenchmarkConfig(output_root=output_root, experiment_name=experiment_name)
    config.validate()
    if method == "realmlp" and train_size != 700_000:
        raise ValueError("RealMLP runs only at 700,000 training rows")
    target_methods = {
        "lgbm": ("lgbm_default", "lgbm_tuned"),
        "lgbm_default": ("lgbm_default",),
        "lgbm_tuned": ("lgbm_tuned",),
        "realmlp": ("realmlp",),
        "tabpfn35": ("tabpfn35",),
    }.get(method)
    if target_methods is None:
        raise ValueError("method must be lgbm, lgbm_default, lgbm_tuned, realmlp, or tabpfn35")
    if not force_rerun and all(
        is_complete_run(run_output_dir(config.experiment_root, name, train_size))
        for name in target_methods
    ):
        manifest = verify_manifest(config.shared_dir, config)
        for name in target_methods:
            completed_path = run_output_dir(config.experiment_root, name, train_size) / "metrics.json"
            completed = json.loads(completed_path.read_text())
            if completed.get("manifest_hash") != manifest["manifest_hash"]:
                raise RuntimeError(
                    f"Completed run {name} at n{train_size} belongs to a different split. "
                    "Set FORCE_RERUN=True to replace it."
                )
        print("All requested run artifacts are already complete")
        table = sibling_metrics_table(config.experiment_root)
        print(table.to_string(index=False))
        return table
    if method == "tabpfn35" and train_size == 700_000:
        _verify_pilot_gate(config, ignore_pilot_gate)
        require_tabpfn_full_hardware()
    data = load_prepared_splits(output_root, experiment_name, train_size=train_size)
    budget = tuning_minutes
    if budget is None:
        budget = FULL_TUNING_MINUTES if train_size == 700_000 else PILOT_TUNING_MINUTES
    if method in {"lgbm", "lgbm_default"}:
        _save_lgbm_run(
            data,
            config,
            kind="default",
            train_size=train_size,
            force_rerun=force_rerun,
            tuning_minutes=budget,
            max_trials=max_trials,
        )
    if method in {"lgbm", "lgbm_tuned"}:
        _save_lgbm_run(
            data,
            config,
            kind="tuned",
            train_size=train_size,
            force_rerun=force_rerun,
            tuning_minutes=budget,
            max_trials=max_trials,
        )
    elif method == "realmlp":
        _save_realmlp_run(data, config, train_size=train_size, force_rerun=force_rerun)
    elif method == "tabpfn35":
        _save_tabpfn_run(
            data,
            config,
            train_size=train_size,
            force_rerun=force_rerun,
            ignore_pilot_gate=ignore_pilot_gate,
            predict_validation=predict_validation,
        )
    elif method not in {"lgbm", "lgbm_default", "lgbm_tuned"}:
        raise ValueError("method must be lgbm, lgbm_default, lgbm_tuned, realmlp, or tabpfn35")
    table = sibling_metrics_table(config.experiment_root)
    print(table.to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--train-size", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-name", default="lending_club_900k")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--ignore-pilot-gate", action="store_true")
    parser.add_argument("--tuning-minutes", type=float)
    parser.add_argument("--max-trials", type=int, default=MAX_OPTUNA_TRIALS)
    args = parser.parse_args()
    run_method(
        args.method,
        args.train_size,
        args.output_root,
        args.experiment_name,
        force_rerun=args.force_rerun,
        ignore_pilot_gate=args.ignore_pilot_gate,
        tuning_minutes=args.tuning_minutes,
        max_trials=args.max_trials,
    )


if __name__ == "__main__":
    main()
