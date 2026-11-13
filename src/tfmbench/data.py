"""Dataset download, cleaning, cohort caching, and preparation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ._io import atomic_csv, atomic_json, atomic_parquet, canonical_hash, sha256_file
from .config import (
    CHARGED_OFF_STATUSES,
    COMPLETED_STATUSES,
    DATASET_FILENAME,
    DATASET_REPO_ID,
    FEATURE_COLUMNS,
    ROW_ID_COLUMN,
    SOURCE_COLUMNS,
    SOURCE_DTYPES,
    TARGET_COLUMN,
    BenchmarkConfig,
)
from .features import (
    apply_category_schema,
    fit_category_schema,
    leakage_auc_table,
    prepare_feature_frame,
)
from .split import (
    SplitArrays,
    build_manifest,
    create_split_arrays,
    load_split_arrays,
    save_manifest,
    save_split_arrays,
    verify_manifest,
)


@dataclass(frozen=True)
class PreparationResult:
    """Paths and summaries produced by data preparation."""

    experiment_root: Path
    cohort_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    leakage_report: pd.DataFrame
    reused: bool


@dataclass(frozen=True)
class PreparedSplits:
    """Shared model-ready train, validation, and test frames."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    train_row_ids: np.ndarray
    X_validation: pd.DataFrame
    y_validation: np.ndarray
    validation_row_ids: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    test_row_ids: np.ndarray
    manifest: dict[str, Any]


def resolve_dataset_revision(token: str | None = None) -> str:
    """Resolve the current immutable dataset commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(DATASET_REPO_ID)
    if not info.sha:
        raise RuntimeError("Hugging Face did not return a dataset revision SHA")
    return str(info.sha)


def download_dataset_csv(revision: str, token: str | None = None) -> Path:
    """Download the exact public dataset revision through the Hub cache."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=DATASET_FILENAME,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return Path(path)


def clean_lending_club_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep completed loans and map charged-off outcomes to one."""
    if "loan_status" not in frame:
        raise ValueError("Input data is missing loan_status")
    cleaned = frame[frame["loan_status"].notna()].copy()
    cleaned["loan_status"] = cleaned["loan_status"].astype("string").str.strip()
    cleaned = cleaned[cleaned["loan_status"].isin(COMPLETED_STATUSES)].copy()
    cleaned[TARGET_COLUMN] = cleaned["loan_status"].isin(CHARGED_OFF_STATUSES).astype(np.uint8)
    return cleaned


def read_eligible_rows(csv_path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Read only whitelisted source columns and return eligible prepared rows."""
    parts: list[pd.DataFrame] = []
    source_rows = 0
    eligible_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=list(SOURCE_COLUMNS),
            dtype=SOURCE_DTYPES,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        source_rows += len(chunk)
        chunk[ROW_ID_COLUMN] = chunk.index.to_numpy(dtype=np.int64)
        eligible = clean_lending_club_frame(chunk)
        if not eligible.empty:
            prepared = prepare_feature_frame(eligible)
            parts.append(prepared)
            eligible_rows += len(prepared)
        print(
            f"read chunk={chunk_number} source_rows={source_rows:,} "
            f"eligible_rows={eligible_rows:,}"
        )
    if not parts:
        raise ValueError("No completed loans were found in the dataset")
    result = pd.concat(parts, ignore_index=True)
    counts = result[TARGET_COLUMN].value_counts().sort_index().to_dict()
    print(f"Eligible completed loans: {len(result):,}")
    print(f"Eligible class counts: {counts}")
    print(f"Eligible positive rate: {result[TARGET_COLUMN].mean():.6f}")
    return result


def sample_cohort(eligible: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Draw the exact fixed stratified cohort."""
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible):,} eligible rows are available; need {size:,}")
    selected, _ = train_test_split(
        np.arange(len(eligible)),
        train_size=size,
        random_state=seed,
        shuffle=True,
        stratify=eligible[TARGET_COLUMN].to_numpy(),
    )
    cohort = eligible.iloc[selected].reset_index(drop=True)
    if cohort[ROW_ID_COLUMN].nunique() != size:
        raise AssertionError("Sampled cohort row identifiers are not unique")
    print(f"Cohort rows: {len(cohort):,}")
    print(f"Cohort class counts: {cohort[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    print(f"Cohort positive rate: {cohort[TARGET_COLUMN].mean():.6f}")
    return cohort


def cohort_content_hash(frame: pd.DataFrame) -> str:
    """Hash cohort values, column order, and dtypes independently of parquet."""
    digest = hashlib.sha256()
    digest.update("|".join(frame.columns).encode())
    digest.update("|".join(str(dtype) for dtype in frame.dtypes).encode())
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _rows_by_ids(frame: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    indexed = frame.set_index(ROW_ID_COLUMN, drop=False)
    try:
        return indexed.loc[row_ids].reset_index(drop=True)
    except KeyError as exc:
        raise ValueError("Saved split contains row identifiers missing from the cohort") from exc


def _build_cohort(
    config: BenchmarkConfig,
    revision: str,
    token: str | None,
    expected_dataset_sha256: str | None = None,
) -> tuple[pd.DataFrame, str]:
    csv_path = download_dataset_csv(revision, token)
    digest = sha256_file(csv_path)
    if expected_dataset_sha256 is not None and digest != expected_dataset_sha256:
        raise ValueError(
            "Downloaded dataset checksum does not match the checksum stored in the split manifest"
        )
    eligible = read_eligible_rows(csv_path, config.csv_chunksize)
    cohort = sample_cohort(eligible, config.cohort_rows, config.seed)
    return cohort, digest


def _repair_cohort_cache(
    config: BenchmarkConfig,
    manifest: dict[str, Any],
    token: str | None,
) -> dict[str, Any]:
    """Rebuild a missing or corrupt cohort at the manifest's pinned revision."""
    cohort_path = config.shared_dir / manifest["cohort"]["parquet_file"]
    cohort, _ = _build_cohort(
        config,
        manifest["dataset"]["revision"],
        token,
        manifest["dataset"]["file_sha256"],
    )
    content_hash = cohort_content_hash(cohort)
    if content_hash != manifest["cohort"]["content_hash"]:
        raise ValueError("Rebuilt cohort content does not match the signed manifest")
    atomic_parquet(cohort_path, cohort)
    manifest["cohort"]["parquet_sha256"] = sha256_file(cohort_path)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = canonical_hash(unsigned)
    save_manifest(config.shared_dir / "split_manifest.json", manifest)
    return manifest


def prepare_data(
    output_root: str | os.PathLike[str],
    experiment_name: str = "lending_club_900k",
    *,
    revision: str | None = None,
    force_rebuild_split: bool = False,
    hf_token: str | None = None,
    csv_chunksize: int = 100_000,
) -> PreparationResult:
    """Build or verify the immutable cohort, split, schema, and leakage report."""
    config = BenchmarkConfig(
        output_root=Path(output_root),
        experiment_name=experiment_name,
        dataset_revision=revision,
        csv_chunksize=csv_chunksize,
    )
    config.validate()
    shared_dir = config.shared_dir
    shared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shared_dir / "split_manifest.json"
    cohort_path = shared_dir / "cohort_900000.parquet"
    leakage_path = shared_dir / "leakage_report.csv"

    if manifest_path.is_file() and not force_rebuild_split:
        raw_manifest = json.loads(manifest_path.read_text())
        if revision is not None and revision != raw_manifest["dataset"]["revision"]:
            raise ValueError(
                "Configured dataset revision differs from the saved split manifest. "
                "Set FORCE_REBUILD_SPLIT=True to create a new contract."
            )
        expected_cache_sha = raw_manifest["cohort"]["parquet_sha256"]
        if not cohort_path.is_file() or sha256_file(cohort_path) != expected_cache_sha:
            print("Cohort cache is missing or corrupt; rebuilding the pinned cohort")
            _repair_cohort_cache(config, raw_manifest, hf_token)
        manifest = verify_manifest(shared_dir, config)
        leakage = pd.read_csv(leakage_path)
        print(f"Verified split manifest: {manifest['manifest_hash']}")
        return PreparationResult(
            config.experiment_root, cohort_path, manifest_path, manifest, leakage, True
        )

    resolved_revision = revision or resolve_dataset_revision(hf_token)
    print(f"Dataset revision: {resolved_revision}")
    cohort, dataset_sha256 = _build_cohort(config, resolved_revision, hf_token)
    atomic_parquet(cohort_path, cohort)
    content_hash = cohort_content_hash(cohort)

    arrays = create_split_arrays(
        cohort[ROW_ID_COLUMN].to_numpy(),
        cohort[TARGET_COLUMN].to_numpy(),
        train_rows=config.train_rows,
        validation_rows=config.validation_rows,
        test_rows=config.test_rows,
        pilot_sizes=config.pilot_sizes,
        seed=config.seed,
    )
    indices_path = shared_dir / "split_indices.npz"
    save_split_arrays(indices_path, arrays)

    training = _rows_by_ids(cohort, arrays.train)
    schema = fit_category_schema(training)
    schema_path = shared_dir / "feature_schema.json"
    atomic_json(schema_path, schema)
    training = apply_category_schema(training, schema)
    leakage = leakage_auc_table(training)
    atomic_csv(leakage_path, leakage)
    print(leakage.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    failed = leakage[leakage["status"] == "FAIL"]
    if not failed.empty:
        details = ", ".join(f"{row.feature}={row.auc:.4f}" for row in failed.itertuples())
        raise ValueError(f"Leakage check failed: {details}")

    manifest = build_manifest(
        config=config,
        dataset_revision=resolved_revision,
        dataset_sha256=dataset_sha256,
        cohort_path=cohort_path,
        cohort_content_hash=content_hash,
        cohort=cohort,
        arrays=arrays,
        indices_path=indices_path,
        feature_schema_path=schema_path,
        leakage_report_path=leakage_path,
    )
    save_manifest(manifest_path, manifest)
    verify_manifest(shared_dir, config)
    print(f"Created split manifest: {manifest['manifest_hash']}")
    return PreparationResult(
        config.experiment_root, cohort_path, manifest_path, manifest, leakage, False
    )


def load_prepared_splits(
    output_root: str | os.PathLike[str],
    experiment_name: str = "lending_club_900k",
    *,
    train_size: int = 700_000,
) -> PreparedSplits:
    """Load one signed training size with common validation and test rows."""
    config = BenchmarkConfig(output_root=Path(output_root), experiment_name=experiment_name)
    config.validate()
    manifest = verify_manifest(config.shared_dir, config)
    cohort = pd.read_parquet(config.shared_dir / manifest["cohort"]["parquet_file"])
    arrays = load_split_arrays(config.shared_dir / manifest["artifacts"]["split_indices_file"])
    training_ids = {
        50_000: arrays.pilot_50000,
        100_000: arrays.pilot_100000,
        200_000: arrays.pilot_200000,
        700_000: arrays.train,
    }.get(train_size)
    if training_ids is None:
        raise ValueError("train_size must be one of 50,000, 100,000, 200,000, or 700,000")
    schema = json.loads((config.shared_dir / manifest["features"]["schema_file"]).read_text())

    def load(ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows = apply_category_schema(_rows_by_ids(cohort, ids), schema)
        return (
            rows.loc[:, FEATURE_COLUMNS].copy(),
            rows[TARGET_COLUMN].to_numpy(dtype=np.uint8),
            rows[ROW_ID_COLUMN].to_numpy(dtype=np.int64),
        )

    X_train, y_train, train_ids = load(training_ids)
    X_validation, y_validation, validation_ids = load(arrays.validation)
    X_test, y_test, test_ids = load(arrays.test)
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
