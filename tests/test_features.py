from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfmbench.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LEAKAGE_AND_ID_EXCLUSIONS,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
)
from tfmbench.features import (
    apply_category_schema,
    enforce_leakage_check,
    fit_category_schema,
    leakage_auc_table,
)


def feature_frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    target = np.tile([0, 1], rows // 2)
    values: dict[str, object] = {
        ROW_ID_COLUMN: np.arange(rows),
        TARGET_COLUMN: target,
    }
    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            values[feature] = np.where(rng.random(rows) > 0.5, "left", "right")
        else:
            values[feature] = rng.normal(size=rows)
    return pd.DataFrame(values)


def test_feature_whitelist_does_not_contain_exact_exclusions():
    exact = {name for name in LEAKAGE_AND_ID_EXCLUSIONS if "*" not in name}
    assert len(FEATURE_COLUMNS) == 26
    assert not set(FEATURE_COLUMNS) & exact


def test_category_schema_is_fixed_from_training():
    training = feature_frame()
    schema = fit_category_schema(training)
    validation = training.iloc[:4].copy()
    validation.loc[validation.index[0], "grade"] = "UNSEEN"
    converted = apply_category_schema(validation, schema)
    assert str(converted["grade"].dtype) == "category"
    assert pd.isna(converted.loc[validation.index[0], "grade"])


def test_leakage_table_format_and_planted_feature():
    frame = feature_frame()
    frame["loan_amnt"] = frame[TARGET_COLUMN]
    table = leakage_auc_table(frame)
    assert list(table.columns) == [
        "feature",
        "kind",
        "auc",
        "missing_count",
        "unique_count",
        "status",
    ]
    planted = table.set_index("feature").loc["loan_amnt"]
    assert planted["auc"] == 1.0
    assert planted["status"] == "FAIL"
    with pytest.raises(ValueError, match="loan_amnt"):
        enforce_leakage_check(frame)


def test_inverse_leakage_is_also_detected():
    frame = feature_frame()
    frame["loan_amnt"] = 1 - frame[TARGET_COLUMN]
    assert leakage_auc_table(frame).set_index("feature").loc["loan_amnt", "auc"] == 1.0
