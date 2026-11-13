from __future__ import annotations

import numpy as np
import pytest

from tfmbench.config import PILOT_SIZES, BenchmarkConfig
from tfmbench.pilot import (
    evaluate_gate_checks,
    fit_linear,
    fit_power_law,
    predict_linear,
    predict_power_law,
)
from tfmbench.run import _verify_pilot_gate


def test_projection_functions_recover_known_relationships():
    sizes = np.array([50_000, 100_000, 200_000], dtype=float)
    power_values = 3e-8 * sizes**2
    power = fit_power_law(sizes, power_values)
    assert power["b"] == pytest.approx(2.0)
    assert predict_power_law(power, 700_000) == pytest.approx(3e-8 * 700_000**2)
    linear_values = 1000 + 4 * sizes
    linear = fit_linear(sizes, linear_values)
    assert linear["slope"] == pytest.approx(4.0)
    assert predict_linear(linear, 700_000) == pytest.approx(2_801_000)


def passing_runs():
    rows = []
    for method in ("lgbm_tuned", "tabpfn35"):
        for index, size in enumerate(PILOT_SIZES):
            rows.append(
                {
                    "status": "complete",
                    "method": method,
                    "train_size": size,
                    "roc_auc": 0.70 + index * 0.01,
                    "pr_auc": 0.4,
                    "log_loss": 0.5,
                    "brier_score": 0.16,
                    "oom_exhausted": False,
                }
            )
    return rows


def passing_projection():
    return {
        "peak_gpu_memory_bytes_700k": 20e9,
        "peak_host_ram_bytes_700k": 20 * 1024**3,
        "tabpfn_total_hours_700k": 10.0,
    }


def test_all_gate_checks_pass_with_safe_inputs():
    hardware = {"gpu_vram_bytes": 40e9, "system_ram_bytes": 60 * 1024**3}
    checks = evaluate_gate_checks(passing_runs(), passing_projection(), hardware)
    assert checks
    assert all(check["passed"] for check in checks)


@pytest.mark.parametrize(
    "mutation,failed_name",
    [
        (lambda runs, projection: runs.pop(), "all_runs_finished"),
        (lambda runs, projection: runs[0].update(roc_auc=0.90), "auc_sanity_range"),
        (lambda runs, projection: projection.update(tabpfn_total_hours_700k=25), "projected_session_time"),
        (lambda runs, projection: projection.update(peak_gpu_memory_bytes_700k=39e9), "projected_gpu_memory"),
    ],
)
def test_gate_failures_are_named(mutation, failed_name):
    runs = passing_runs()
    projection = passing_projection()
    mutation(runs, projection)
    hardware = {"gpu_vram_bytes": 40e9, "system_ram_bytes": 60 * 1024**3}
    checks = evaluate_gate_checks(runs, projection, hardware)
    assert not next(check for check in checks if check["name"] == failed_name)["passed"]


def test_full_tabpfn_gate_refuses_missing_pilot_report(tmp_path):
    config = BenchmarkConfig(output_root=tmp_path)
    with pytest.raises(RuntimeError, match="Missing pilot_report.json"):
        _verify_pilot_gate(config, ignore=False)
