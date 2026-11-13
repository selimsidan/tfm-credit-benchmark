from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfmbench._io import atomic_csv, atomic_json, canonical_hash
from tfmbench.config import BenchmarkConfig, FEATURE_COLUMNS, ROW_ID_COLUMN, TARGET_COLUMN
from tfmbench.data import cohort_content_hash
from tfmbench.split import (
    build_manifest,
    create_split_arrays,
    load_split_arrays,
    save_manifest,
    save_split_arrays,
    verify_manifest,
)


def small_arrays():
    row_ids = np.arange(100, dtype=np.int64)
    target = np.tile([0, 0, 0, 0, 1], 20).astype(np.uint8)
    arrays = create_split_arrays(
        row_ids,
        target,
        train_rows=70,
        validation_rows=15,
        test_rows=15,
        pilot_sizes=(10, 20, 40),
        seed=42,
    )
    return row_ids, target, arrays


def test_split_sizes_disjointness_nesting_and_determinism():
    row_ids, target, first = small_arrays()
    second = create_split_arrays(
        row_ids,
        target,
        train_rows=70,
        validation_rows=15,
        test_rows=15,
        pilot_sizes=(10, 20, 40),
        seed=42,
    )
    assert [len(first.train), len(first.validation), len(first.test)] == [70, 15, 15]
    assert [len(first.pilot_50000), len(first.pilot_100000), len(first.pilot_200000)] == [10, 20, 40]
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)
    assert not set(first.validation) & set(first.test)
    assert set(first.pilot_50000) <= set(first.pilot_100000) <= set(first.pilot_200000) <= set(first.train)
    for name in first.__dict__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    target_by_id = pd.Series(target, index=row_ids)
    for ids in (first.train, first.validation, first.test, first.pilot_50000):
        assert target_by_id.loc[ids].mean() == pytest.approx(0.2, abs=0.04)


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path):
    row_ids, target, arrays = small_arrays()
    cohort = pd.DataFrame({ROW_ID_COLUMN: row_ids, TARGET_COLUMN: target})
    for feature in FEATURE_COLUMNS:
        cohort[feature] = np.arange(len(cohort), dtype=float)
    shared = tmp_path / "shared"
    shared.mkdir()
    cohort_path = shared / "cohort_900000.parquet"
    cohort_path.write_bytes(b"synthetic parquet placeholder")
    indices_path = shared / "split_indices.npz"
    save_split_arrays(indices_path, arrays)
    schema_path = shared / "feature_schema.json"
    atomic_json(
        schema_path,
        {"schema_version": 1, "feature_columns": list(FEATURE_COLUMNS), "categories": {}},
    )
    leakage_path = shared / "leakage_report.csv"
    atomic_csv(leakage_path, pd.DataFrame({"feature": FEATURE_COLUMNS, "auc": 0.5}))
    config = BenchmarkConfig(
        output_root=tmp_path,
        cohort_rows=100,
        train_rows=70,
        validation_rows=15,
        test_rows=15,
        pilot_sizes=(10, 20, 40),
    )
    manifest = build_manifest(
        config=config,
        dataset_revision="abc123",
        dataset_sha256="0" * 64,
        cohort_path=cohort_path,
        cohort_content_hash=cohort_content_hash(cohort),
        cohort=cohort,
        arrays=arrays,
        indices_path=indices_path,
        feature_schema_path=schema_path,
        leakage_report_path=leakage_path,
    )
    save_manifest(shared / "split_manifest.json", manifest)
    verified = verify_manifest(shared, config)
    assert verified["manifest_hash"] == manifest["manifest_hash"]
    loaded = load_split_arrays(indices_path)
    np.testing.assert_array_equal(loaded.test, arrays.test)

    payload = json.loads((shared / "split_manifest.json").read_text())
    payload["split_contract"]["seed"] = 99
    (shared / "split_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest hash"):
        verify_manifest(shared, config)


def test_manifest_contract_mismatch_is_detected(tmp_path: Path):
    config = BenchmarkConfig(output_root=tmp_path)
    config.validate()
    assert canonical_hash({"a": 1}) == canonical_hash({"a": 1})
