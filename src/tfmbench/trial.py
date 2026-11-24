"""Signed 75k trial derived from the immutable full benchmark cohort."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ._io import (
    atomic_csv,
    atomic_json,
    atomic_npz,
    canonical_hash,
    sha256_array,
    sha256_file,
)
from .config import (
    FEATURE_COLUMNS,
    ROW_ID_COLUMN,
    SEED,
    TARGET_COLUMN,
    TRIAL_EXPERIMENT_NAME,
    TRIAL_TEST_ROWS,
    TRIAL_TRAIN_ROWS,
    TRIAL_VALIDATION_ROWS,
    BenchmarkConfig,
)
from .data import PreparedSplits
from .evaluate import sibling_metrics_table
from .features import apply_category_schema, fit_category_schema, leakage_auc_table
from .split import load_split_arrays, verify_manifest


@dataclass(frozen=True)
class TrialConfig:
    """Paths and fixed row counts for the derived trial."""

    output_root: Path | str
    experiment_name: str = TRIAL_EXPERIMENT_NAME
    source_experiment_name: str = "lending_club_900k"
    seed: int = SEED
    train_rows: int = TRIAL_TRAIN_ROWS
    validation_rows: int = TRIAL_VALIDATION_ROWS
    test_rows: int = TRIAL_TEST_ROWS

    @property
    def experiment_root(self) -> Path:
        return Path(self.output_root).expanduser().resolve() / self.experiment_name

    @property
    def shared_dir(self) -> Path:
        return self.experiment_root / "shared"

    @property
    def source_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(self.output_root, self.source_experiment_name)

    def validate(self) -> None:
        if self.seed != SEED:
            raise ValueError("The 75k trial seed must remain 42")
        expected = (TRIAL_TRAIN_ROWS, TRIAL_VALIDATION_ROWS, TRIAL_TEST_ROWS)
        actual = (self.train_rows, self.validation_rows, self.test_rows)
        if actual != expected:
            raise ValueError(
                "The trial split must be exactly 75,000/10,000/20,000 rows"
            )
        if self.experiment_name == self.source_experiment_name:
            raise ValueError("Trial and source experiment names must differ")


@dataclass(frozen=True)
class TrialPreparationResult:
    """Artifacts created or verified for the trial."""

    experiment_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    leakage_report: pd.DataFrame
    reused: bool


def _rows_by_ids(frame: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    indexed = frame.set_index(ROW_ID_COLUMN, drop=False)
    try:
        return indexed.loc[row_ids].reset_index(drop=True)
    except KeyError as exc:
        raise ValueError("Trial row identifiers are missing from the source cohort") from exc


def _stratified_subset(
    row_ids: np.ndarray, target_by_id: pd.Series, size: int, seed: int
) -> np.ndarray:
    if size > len(row_ids):
        raise ValueError(f"Cannot select {size:,} rows from a {len(row_ids):,}-row split")
    selected, _ = train_test_split(
        np.asarray(row_ids, dtype=np.int64),
        train_size=size,
        random_state=seed,
        shuffle=True,
        stratify=target_by_id.loc[row_ids].to_numpy(dtype=np.uint8),
    )
    return np.asarray(selected, dtype=np.int64)


def _split_details(ids: np.ndarray, target_by_id: pd.Series) -> dict[str, Any]:
    counts = target_by_id.loc[ids].value_counts().sort_index()
    return {
        "rows": int(len(ids)),
        "class_counts": {str(int(key)): int(value) for key, value in counts.items()},
        "positive_rate": float(target_by_id.loc[ids].mean()),
        "row_ids_sha256": sha256_array(ids),
    }


def _load_source_contract(config: TrialConfig) -> tuple[dict[str, Any], Any]:
    source = config.source_config
    source.validate()
    manifest = verify_manifest(source.shared_dir, source)
    arrays = load_split_arrays(
        source.shared_dir / manifest["artifacts"]["split_indices_file"]
    )
    return manifest, arrays


def _load_source(config: TrialConfig) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    manifest, arrays = _load_source_contract(config)
    source = config.source_config
    cohort = pd.read_parquet(source.shared_dir / manifest["cohort"]["parquet_file"])
    return manifest, cohort, arrays


def prepare_trial(
    output_root: str | Path,
    source_experiment_name: str = "lending_club_900k",
    trial_experiment_name: str = TRIAL_EXPERIMENT_NAME,
    *,
    force_rebuild: bool = False,
) -> TrialPreparationResult:
    """Create or verify the deterministic 75k/10k/20k child experiment."""
    config = TrialConfig(output_root, trial_experiment_name, source_experiment_name)
    config.validate()
    config.shared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.shared_dir / "trial_manifest.json"
    leakage_path = config.shared_dir / "leakage_report.csv"
    if manifest_path.is_file() and not force_rebuild:
        manifest = verify_trial_manifest(config)
        leakage = pd.read_csv(leakage_path)
        print(f"Verified trial manifest: {manifest['manifest_hash']}")
        return TrialPreparationResult(
            config.experiment_root, manifest_path, manifest, leakage, True
        )

    source_manifest, cohort, source_arrays = _load_source(config)
    target_by_id = cohort.set_index(ROW_ID_COLUMN)[TARGET_COLUMN]
    train_ids = _stratified_subset(
        source_arrays.train, target_by_id, config.train_rows, config.seed
    )
    validation_ids = _stratified_subset(
        source_arrays.validation, target_by_id, config.validation_rows, config.seed
    )
    test_ids = _stratified_subset(
        source_arrays.test, target_by_id, config.test_rows, config.seed
    )
    indices_path = config.shared_dir / "trial_indices.npz"
    atomic_npz(
        indices_path,
        train=train_ids,
        validation=validation_ids,
        test=test_ids,
    )

    training = _rows_by_ids(cohort, train_ids)
    schema = fit_category_schema(training)
    schema_path = config.shared_dir / "feature_schema.json"
    atomic_json(schema_path, schema)
    leakage = leakage_auc_table(apply_category_schema(training, schema))
    atomic_csv(leakage_path, leakage)
    failed = leakage[leakage["status"] == "FAIL"]
    if not failed.empty:
        details = ", ".join(
            f"{row.feature}={row.auc:.4f}" for row in failed.itertuples()
        )
        raise ValueError(f"Trial leakage check failed: {details}")

    split_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "profile": "lending_club_75k_trial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "experiment_name": source_experiment_name,
            "manifest_hash": source_manifest["manifest_hash"],
            "dataset_revision": source_manifest["dataset"]["revision"],
            "cohort_content_hash": source_manifest["cohort"]["content_hash"],
        },
        "split_contract": {
            "seed": config.seed,
            "train_rows": config.train_rows,
            "validation_rows": config.validation_rows,
            "test_rows": config.test_rows,
            "sampling": "independent stratified subsets of the signed source partitions",
        },
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "count": len(FEATURE_COLUMNS),
            "schema_file": schema_path.name,
            "schema_sha256": sha256_file(schema_path),
        },
        "artifacts": {
            "indices_file": indices_path.name,
            "indices_sha256": sha256_file(indices_path),
            "leakage_report_file": leakage_path.name,
            "leakage_report_sha256": sha256_file(leakage_path),
        },
        "splits": {
            name: _split_details(ids, target_by_id) for name, ids in split_ids.items()
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    atomic_json(manifest_path, payload)
    manifest = verify_trial_manifest(config)
    print(f"Created trial manifest: {manifest['manifest_hash']}")
    return TrialPreparationResult(
        config.experiment_root, manifest_path, manifest, leakage, False
    )


def verify_trial_manifest(config: TrialConfig) -> dict[str, Any]:
    """Verify the child manifest, parent identity, and all signed row arrays."""
    config.validate()
    path = config.shared_dir / "trial_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing trial manifest: {path}. Run 06_prepare_75k_trial.ipynb first."
        )
    manifest = json.loads(path.read_text())
    stored_hash = manifest.get("manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if stored_hash != canonical_hash(unsigned):
        raise ValueError("Trial manifest hash mismatch")
    contract = manifest.get("split_contract", {})
    expected_contract = {
        "seed": config.seed,
        "train_rows": config.train_rows,
        "validation_rows": config.validation_rows,
        "test_rows": config.test_rows,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Trial split contract mismatch for {key}")
    if manifest.get("features", {}).get("columns") != list(FEATURE_COLUMNS):
        raise ValueError("Trial feature list mismatch")

    source_manifest, source_arrays = _load_source_contract(config)
    source = manifest.get("source", {})
    if source.get("experiment_name") != config.source_experiment_name:
        raise ValueError("Trial source experiment name mismatch")
    if source.get("manifest_hash") != source_manifest["manifest_hash"]:
        raise ValueError("Trial source manifest hash mismatch")

    artifact_checks = {
        manifest["artifacts"]["indices_file"]: manifest["artifacts"]["indices_sha256"],
        manifest["artifacts"]["leakage_report_file"]: manifest["artifacts"]["leakage_report_sha256"],
        manifest["features"]["schema_file"]: manifest["features"]["schema_sha256"],
    }
    for filename, expected_hash in artifact_checks.items():
        artifact = config.shared_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Trial artifact checksum mismatch: {filename}")

    with np.load(config.shared_dir / manifest["artifacts"]["indices_file"], allow_pickle=False) as bundle:
        arrays = {
            name: np.asarray(bundle[name], dtype=np.int64)
            for name in ("train", "validation", "test")
        }
    for name, values in arrays.items():
        details = manifest["splits"][name]
        if len(values) != details["rows"] or sha256_array(values) != details["row_ids_sha256"]:
            raise ValueError(f"Trial split array mismatch: {name}")
    if set(arrays["train"]) & set(arrays["validation"]):
        raise ValueError("Trial train and validation rows overlap")
    if set(arrays["train"]) & set(arrays["test"]):
        raise ValueError("Trial train and test rows overlap")
    if set(arrays["validation"]) & set(arrays["test"]):
        raise ValueError("Trial validation and test rows overlap")
    source_membership = {
        "train": set(source_arrays.train),
        "validation": set(source_arrays.validation),
        "test": set(source_arrays.test),
    }
    for name, values in arrays.items():
        if not set(values) <= source_membership[name]:
            raise ValueError(f"Trial {name} rows are outside the signed source partition")
    return manifest


def load_trial_splits(
    output_root: str | Path,
    source_experiment_name: str = "lending_club_900k",
    trial_experiment_name: str = TRIAL_EXPERIMENT_NAME,
) -> PreparedSplits:
    """Load model-ready trial frames from the source cohort and child manifest."""
    config = TrialConfig(output_root, trial_experiment_name, source_experiment_name)
    manifest = verify_trial_manifest(config)
    source_manifest, cohort, _ = _load_source(config)
    if source_manifest["manifest_hash"] != manifest["source"]["manifest_hash"]:
        raise ValueError("Source manifest changed while loading the trial")
    with np.load(config.shared_dir / manifest["artifacts"]["indices_file"], allow_pickle=False) as bundle:
        ids = {
            name: np.asarray(bundle[name], dtype=np.int64)
            for name in ("train", "validation", "test")
        }
    schema = json.loads(
        (config.shared_dir / manifest["features"]["schema_file"]).read_text()
    )

    def load(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows = apply_category_schema(_rows_by_ids(cohort, ids[name]), schema)
        return (
            rows.loc[:, FEATURE_COLUMNS].copy(),
            rows[TARGET_COLUMN].to_numpy(dtype=np.uint8),
            rows[ROW_ID_COLUMN].to_numpy(dtype=np.int64),
        )

    X_train, y_train, train_ids = load("train")
    X_validation, y_validation, validation_ids = load("validation")
    X_test, y_test, test_ids = load("test")
    return PreparedSplits(
        X_train,
        y_train,
        train_ids,
        X_validation,
        y_validation,
        validation_ids,
        X_test,
        y_test,
        test_ids,
        manifest,
    )


def run_trial_method(
    method: str,
    output_root: str | Path,
    source_experiment_name: str = "lending_club_900k",
    trial_experiment_name: str = TRIAL_EXPERIMENT_NAME,
    *,
    force_rerun: bool = False,
    tuning_minutes: float = 15,
    max_trials: int = 30,
    tabpfn_fit_mode: str = "fit_with_cache",
    predict_validation: bool = False,
) -> pd.DataFrame:
    """Run one or both model variants against the signed 75k trial."""
    from .run import _save_lgbm_run, _save_realmlp_run, _save_tabpfn_run

    config = TrialConfig(output_root, trial_experiment_name, source_experiment_name)
    config.validate()
    data = load_trial_splits(
        output_root, source_experiment_name, trial_experiment_name
    )
    if method == "lgbm":
        _save_lgbm_run(
            data,
            config,
            kind="default",
            train_size=config.train_rows,
            force_rerun=force_rerun,
            tuning_minutes=tuning_minutes,
            max_trials=max_trials,
        )
        _save_lgbm_run(
            data,
            config,
            kind="tuned",
            train_size=config.train_rows,
            force_rerun=force_rerun,
            tuning_minutes=tuning_minutes,
            max_trials=max_trials,
        )
    elif method == "realmlp":
        _save_realmlp_run(
            data, config, train_size=config.train_rows, force_rerun=force_rerun
        )
    elif method == "tabpfn35":
        _save_tabpfn_run(
            data,
            config,
            train_size=config.train_rows,
            force_rerun=force_rerun,
            ignore_pilot_gate=False,
            predict_validation=predict_validation,
            fit_mode=tabpfn_fit_mode,
        )
    else:
        raise ValueError("Trial method must be lgbm, realmlp, or tabpfn35")
    table = sibling_metrics_table(config.experiment_root)
    print(table.to_string(index=False))
    return table


def build_trial_performance_review(
    output_root: str | Path,
    source_experiment_name: str = "lending_club_900k",
    trial_experiment_name: str = TRIAL_EXPERIMENT_NAME,
    *,
    bootstrap_resamples: int = 2_000,
):
    """Verify and compare all four saved 75k trial results."""
    from .report import build_fixed_size_performance_review

    config = TrialConfig(output_root, trial_experiment_name, source_experiment_name)
    manifest = verify_trial_manifest(config)
    return build_fixed_size_performance_review(
        config.experiment_root,
        manifest,
        train_size=config.train_rows,
        bootstrap_resamples=bootstrap_resamples,
    )
