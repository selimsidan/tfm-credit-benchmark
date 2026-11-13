from __future__ import annotations

import numpy as np
import pytest

import tfmbench.evaluate as evaluate
from tfmbench.evaluate import binary_metrics, paired_bootstrap_difference


def test_binary_metrics_on_known_predictions():
    y = np.array([0, 0, 1, 1], dtype=np.uint8)
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(y, probability)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["test_rows"] == 4
    assert 0 < metrics["log_loss"] < 0.3
    assert 0 < metrics["brier_score"] < 0.1


def test_binary_metrics_reject_invalid_probability():
    with pytest.raises(ValueError, match="between zero and one"):
        binary_metrics(np.array([0, 1]), np.array([0.2, 1.2]))


def test_paired_bootstrap_identical_models_has_zero_interval():
    y = np.tile([0, 1], 50)
    probability = np.where(y == 1, 0.8, 0.2)
    for metric in ("roc_auc", "log_loss"):
        result = paired_bootstrap_difference(
            y, probability, probability, metric, n_resamples=100, seed=0
        )
        assert result["difference"] == 0.0
        assert result["ci95_low"] == 0.0
        assert result["ci95_high"] == 0.0


def test_run_artifacts_write_metrics_completion_marker_last(tmp_path, monkeypatch):
    calls = []
    real_atomic_json = evaluate.atomic_json

    def fake_parquet(path, frame):
        path.write_bytes(b"parquet placeholder")

    def tracked_json(path, payload):
        calls.append(path.name)
        real_atomic_json(path, payload)

    monkeypatch.setattr(evaluate, "atomic_parquet", fake_parquet)
    monkeypatch.setattr(evaluate, "atomic_json", tracked_json)
    monkeypatch.setattr(evaluate, "_save_diagnostic_plots", lambda *args: [])
    manifest = {
        "manifest_hash": "manifest",
        "splits": {"test": {"row_ids_sha256": "test-hash"}},
    }
    evaluate.save_run_artifacts(
        tmp_path,
        method="lgbm_default",
        train_size=50_000,
        row_ids=np.arange(4),
        y_true=np.array([0, 0, 1, 1]),
        probability=np.array([0.1, 0.2, 0.8, 0.9]),
        resources={"fit_seconds": 1.0, "test_predict_seconds": 0.5},
        run_details={"test": True},
        split_manifest=manifest,
    )
    assert calls[-1] == "metrics.json"
    assert evaluate.is_complete_run(tmp_path)
