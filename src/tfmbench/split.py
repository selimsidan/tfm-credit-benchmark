"""Deterministic split creation, persistence, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ._io import atomic_json, atomic_npz, canonical_hash, sha256_array, sha256_file
from .config import BenchmarkConfig, FEATURE_COLUMNS, ROW_ID_COLUMN, TARGET_COLUMN


@dataclass(frozen=True)
class SplitArrays:
    """Stable source-row identifiers for every signed split."""

    cohort: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    pilot_50000: np.ndarray
    pilot_100000: np.ndarray
    pilot_200000: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(value, dtype=np.int64) for name, value in self.__dict__.items()}


def _stratified_sample(
    row_ids: np.ndarray, target: np.ndarray, size: int, seed: int
) -> np.ndarray:
    selected, _ = train_test_split(
        row_ids,
        train_size=size,
        random_state=seed,
        shuffle=True,
        stratify=target,
    )
    return np.asarray(selected, dtype=np.int64)


def create_split_arrays(
    row_ids: np.ndarray,
    target: np.ndarray,
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    pilot_sizes: tuple[int, int, int],
    seed: int,
) -> SplitArrays:
    """Create exact disjoint splits and nested stratified pilot subsets."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    target = np.asarray(target, dtype=np.uint8)
    if len(row_ids) != train_rows + validation_rows + test_rows:
        raise ValueError("Cohort size does not equal requested split sizes")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Cohort row identifiers must be unique")
    train_ids, remainder_ids, train_y, remainder_y = train_test_split(
        row_ids,
        target,
        train_size=train_rows,
        random_state=seed,
        shuffle=True,
        stratify=target,
    )
    validation_ids, test_ids = train_test_split(
        remainder_ids,
        train_size=validation_rows,
        test_size=test_rows,
        random_state=seed,
        shuffle=True,
        stratify=remainder_y,
    )
    lookup = pd.Series(target, index=row_ids)
    size_50, size_100, size_200 = pilot_sizes
    pilot_200 = _stratified_sample(
        train_ids, lookup.loc[train_ids].to_numpy(), size_200, seed
    )
    pilot_100 = _stratified_sample(
        pilot_200, lookup.loc[pilot_200].to_numpy(), size_100, seed
    )
    pilot_50 = _stratified_sample(
        pilot_100, lookup.loc[pilot_100].to_numpy(), size_50, seed
    )
    return SplitArrays(
        cohort=row_ids,
        train=np.asarray(train_ids, dtype=np.int64),
        validation=np.asarray(validation_ids, dtype=np.int64),
        test=np.asarray(test_ids, dtype=np.int64),
        pilot_50000=pilot_50,
        pilot_100000=pilot_100,
        pilot_200000=pilot_200,
    )


def _class_counts(ids: np.ndarray, target_by_id: pd.Series) -> dict[str, int]:
    counts = target_by_id.loc[ids].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def build_manifest(
    *,
    config: BenchmarkConfig,
    dataset_revision: str,
    dataset_sha256: str,
    cohort_path: Path,
    cohort_content_hash: str,
    cohort: pd.DataFrame,
    arrays: SplitArrays,
    indices_path: Path,
    feature_schema_path: Path,
    leakage_report_path: Path,
) -> dict[str, Any]:
    """Build the complete manifest and its canonical contract hash."""
    target_by_id = cohort.set_index(ROW_ID_COLUMN)[TARGET_COLUMN]
    array_payload = arrays.as_dict()
    split_details = {
        name: {
            "rows": int(len(values)),
            "class_counts": _class_counts(values, target_by_id),
            "row_ids_sha256": sha256_array(values),
        }
        for name, values in array_payload.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repo_id": "codesignal/lending-club-loan-accepted",
            "filename": "accepted_2007_to_2018Q4.csv",
            "revision": dataset_revision,
            "file_sha256": dataset_sha256,
        },
        "cohort": {
            "rows": int(len(cohort)),
            "class_counts": _class_counts(arrays.cohort, target_by_id),
            "row_id_kind": "zero-based physical CSV data-row ordinal",
            "content_hash": cohort_content_hash,
            "parquet_file": cohort_path.name,
            "parquet_sha256": sha256_file(cohort_path),
        },
        "split_contract": {
            "seed": config.seed,
            "train_rows": config.train_rows,
            "validation_rows": config.validation_rows,
            "test_rows": config.test_rows,
            "pilot_sizes": list(config.pilot_sizes),
            "sampling": "stratified random cohort and nested stratified pilot subsets",
        },
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "count": len(FEATURE_COLUMNS),
            "schema_file": feature_schema_path.name,
            "schema_sha256": sha256_file(feature_schema_path),
        },
        "artifacts": {
            "split_indices_file": indices_path.name,
            "split_indices_sha256": sha256_file(indices_path),
            "leakage_report_file": leakage_report_path.name,
            "leakage_report_sha256": sha256_file(leakage_report_path),
        },
        "splits": split_details,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def save_split_arrays(path: Path, arrays: SplitArrays) -> None:
    """Persist all split arrays atomically."""
    atomic_npz(path, **arrays.as_dict())


def load_split_arrays(path: Path) -> SplitArrays:
    """Load signed split arrays."""
    with np.load(path, allow_pickle=False) as bundle:
        values = {name: np.asarray(bundle[name], dtype=np.int64) for name in SplitArrays.__annotations__}
    return SplitArrays(**values)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the signed split manifest atomically."""
    atomic_json(path, manifest)


def verify_manifest(shared_dir: Path, config: BenchmarkConfig) -> dict[str, Any]:
    """Verify the complete persisted data and split contract."""
    path = shared_dir / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    manifest = json.loads(path.read_text())
    stored_hash = manifest.pop("manifest_hash", None)
    computed_hash = canonical_hash(manifest)
    manifest["manifest_hash"] = stored_hash
    if stored_hash != computed_hash:
        raise ValueError("Split manifest hash mismatch")
    expected_contract = {
        "seed": config.seed,
        "train_rows": config.train_rows,
        "validation_rows": config.validation_rows,
        "test_rows": config.test_rows,
        "pilot_sizes": list(config.pilot_sizes),
    }
    for key, expected in expected_contract.items():
        if manifest["split_contract"].get(key) != expected:
            raise ValueError(f"Split manifest contract mismatch for {key}")
    if manifest["features"]["columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Split manifest feature list mismatch")
    checks = {
        manifest["cohort"]["parquet_file"]: manifest["cohort"]["parquet_sha256"],
        manifest["features"]["schema_file"]: manifest["features"]["schema_sha256"],
        manifest["artifacts"]["split_indices_file"]: manifest["artifacts"]["split_indices_sha256"],
        manifest["artifacts"]["leakage_report_file"]: manifest["artifacts"]["leakage_report_sha256"],
    }
    for filename, expected_hash in checks.items():
        artifact = shared_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Shared artifact checksum mismatch: {filename}")
    arrays = load_split_arrays(shared_dir / manifest["artifacts"]["split_indices_file"])
    for name, values in arrays.as_dict().items():
        details = manifest["splits"][name]
        if len(values) != details["rows"] or sha256_array(values) != details["row_ids_sha256"]:
            raise ValueError(f"Split array mismatch: {name}")
    if set(arrays.train) & set(arrays.validation) or set(arrays.train) & set(arrays.test):
        raise ValueError("Training rows overlap validation or test rows")
    if set(arrays.validation) & set(arrays.test):
        raise ValueError("Validation rows overlap test rows")
    if not set(arrays.pilot_50000) <= set(arrays.pilot_100000) <= set(arrays.pilot_200000) <= set(arrays.train):
        raise ValueError("Pilot subsets are not nested inside training")
    return manifest
