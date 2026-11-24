from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfmbench.config import TRIAL_TEST_ROWS, TRIAL_TRAIN_ROWS, TRIAL_VALIDATION_ROWS
from tfmbench.models.tabpfn35 import fit_model
from tfmbench.report import build_fixed_size_performance_review
from tfmbench.trial import TrialConfig, _stratified_subset


def test_trial_config_is_separate_and_fixed(tmp_path: Path):
    config = TrialConfig(tmp_path)
    config.validate()
    assert config.experiment_root == tmp_path / "lending_club_75k_trial"
    assert config.source_config.experiment_root == tmp_path / "lending_club_900k"
    assert (config.train_rows, config.validation_rows, config.test_rows) == (
        TRIAL_TRAIN_ROWS,
        TRIAL_VALIDATION_ROWS,
        TRIAL_TEST_ROWS,
    )
    with pytest.raises(ValueError, match="must differ"):
        TrialConfig(tmp_path, "same", "same").validate()


def test_trial_subset_is_stratified_and_deterministic():
    row_ids = np.arange(1_000, dtype=np.int64)
    target = pd.Series(np.tile([0, 0, 0, 0, 1], 200), index=row_ids)
    first = _stratified_subset(row_ids, target, 750, 42)
    second = _stratified_subset(row_ids, target, 750, 42)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 750
    assert target.loc[first].mean() == pytest.approx(0.2, abs=0.01)


def test_tabpfn_fit_mode_is_validated_before_loading_weights():
    with pytest.raises(ValueError, match="fit_mode"):
        fit_model(np.zeros((2, 1)), np.array([0, 1]), fit_mode="not-a-mode")


def test_empty_fixed_size_review_is_restart_safe(tmp_path: Path):
    manifest = {
        "manifest_hash": "trial-hash",
        "splits": {"test": {"rows": 20_000, "row_ids_sha256": "test-hash"}},
    }
    review = build_fixed_size_performance_review(
        tmp_path, manifest, train_size=75_000, bootstrap_resamples=10
    )
    assert review.status == "WARN"
    assert len(review.inventory) == 4
    assert list(pd.read_csv(review.review_dir / "all_metrics.csv").columns)[:2] == [
        "method",
        "train_size",
    ]
    assert (review.review_dir / "review_bundle.zip").is_file()
