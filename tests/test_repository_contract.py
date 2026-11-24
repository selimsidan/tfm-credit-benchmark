from __future__ import annotations

import json
from pathlib import Path

from tfmbench.config import (
    COHORT_ROWS,
    MAX_ALLOWED_ROWS,
    PILOT_SIZES,
    TEST_ROWS,
    TRAIN_ROWS,
    TRIAL_TEST_ROWS,
    TRIAL_TRAIN_ROWS,
    TRIAL_VALIDATION_ROWS,
    VALIDATION_ROWS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_row_contract():
    assert COHORT_ROWS == 900_000
    assert COHORT_ROWS <= MAX_ALLOWED_ROWS == 1_000_000
    assert TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS == COHORT_ROWS
    assert PILOT_SIZES == (50_000, 100_000, 200_000)
    assert (TRIAL_TRAIN_ROWS, TRIAL_VALIDATION_ROWS, TRIAL_TEST_ROWS) == (
        75_000,
        10_000,
        20_000,
    )


def test_repository_contains_no_em_dash():
    suffixes = {".py", ".toml", ".txt", ".md", ".ipynb", ".json", ".yaml", ".yml"}
    offenders = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes and ".git" not in path.parts:
            if "\N{EM DASH}" in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_prepare_notebook_is_valid_and_thin():
    path = ROOT / "notebooks" / "00_prepare_data.ipynb"
    if not path.exists():
        return
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "EDIT THIS CELL ONLY" in source
    assert "https://github.com/selimsidan/tfm-credit-benchmark.git" in source
    assert "prepare_data" in source
    assert "FORCE_REBUILD_SPLIT" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            compile("".join(cell["source"]), "00_prepare_data.ipynb", "exec")


def test_all_notebooks_are_valid_thin_colab_workflows():
    full = {
        "00_prepare_data.ipynb",
        "01_pilot.ipynb",
        "02_lightgbm_700k.ipynb",
        "03_realmlp_700k.ipynb",
        "04_tabpfn35_700k.ipynb",
        "05_compare.ipynb",
    }
    trial = {
        "06_prepare_75k_trial.ipynb",
        "07_lightgbm_75k.ipynb",
        "08_realmlp_75k.ipynb",
        "09_tabpfn35_75k.ipynb",
        "10_compare_75k.ipynb",
    }
    notebook_dir = ROOT / "notebooks"
    assert {path.name for path in notebook_dir.glob("*.ipynb")} == full | trial
    for path in notebook_dir.glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "EDIT THIS CELL ONLY" in source
        assert "https://github.com/selimsidan/tfm-credit-benchmark.git" in source
        assert "DRIVE_OUTPUT_DIR='/content/drive/MyDrive/tfm_credit_benchmark'" in source
        assert "RUN_TESTS=True" in source
        assert "src_path=str(repo/'src')" in source
        assert "tfmbench package:" in source
        if path.name in full:
            assert "EXPERIMENT_NAME='lending_club_900k'" in source
            assert "FORCE_REBUILD_SPLIT=False" in source
            assert "FORCE_RERUN=False" in source
        else:
            assert "SOURCE_EXPERIMENT_NAME='lending_club_900k'" in source
            assert "TRIAL_EXPERIMENT_NAME='lending_club_75k_trial'" in source
        assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
    for name in (
        "01_pilot.ipynb",
        "03_realmlp_700k.ipynb",
        "04_tabpfn35_700k.ipynb",
        "08_realmlp_75k.ipynb",
        "09_tabpfn35_75k.ipynb",
    ):
        notebook = json.loads((notebook_dir / name).read_text())
        assert notebook["metadata"]["accelerator"] == "GPU"
    pilot_source = (notebook_dir / "01_pilot.ipynb").read_text()
    tabpfn_source = (notebook_dir / "04_tabpfn35_700k.ipynb").read_text()
    trial_tabpfn_source = (notebook_dir / "09_tabpfn35_75k.ipynb").read_text()
    assert "TABPFN_TOKEN" in pilot_source
    assert "TABPFN_TOKEN" in tabpfn_source
    assert "TABPFN_TOKEN" in trial_tabpfn_source
    assert "TABPFN_FIT_MODE='auto'" in trial_tabpfn_source
    assert "IGNORE_PILOT_GATE=False" in tabpfn_source


def test_tabpfn_wrapper_locks_full_context_settings():
    source = (ROOT / "src" / "tfmbench" / "models" / "tabpfn35.py").read_text()
    assert "ModelVersion.V3_5" in source
    assert "n_estimators=8" in source
    assert 'device="cuda"' in source
    assert 'fit_mode: str = "fit_with_cache"' in source
    assert "fit_mode=fit_mode" in source
    assert "memory_saving_mode=True" in source
    assert "keep_cache_on_device=False" in source
    assert '"row_subsampling": None' in source
    assert "SUBSAMPLE_SAMPLES" not in source
