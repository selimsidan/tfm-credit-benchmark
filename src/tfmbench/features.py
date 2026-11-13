"""Deterministic shared feature preparation and leakage checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LEAKAGE_AUC_THRESHOLD,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_month_year(values: pd.Series, *, earliest: bool = False) -> tuple[pd.Series, pd.Series]:
    """Parse Lending Club month-year strings into numeric year and month."""
    text = values.astype("string").str.strip().str.lower()
    parts = text.str.extract(r"^(?P<month>[a-z]{3})-(?P<year>\d{2}|\d{4})$")
    month = parts["month"].map(MONTHS).astype("Float64")
    raw_year = pd.to_numeric(parts["year"], errors="coerce").astype("Float64")
    is_short = parts["year"].str.len().eq(2)
    year = raw_year.where(~is_short, raw_year + 2000)
    if not earliest:
        return year, month
    return year, month


def _credit_history_months(issue: pd.Series, earliest: pd.Series) -> pd.Series:
    """Compute complete months from earliest credit line to loan issue."""
    issue_year, issue_month = _parse_month_year(issue)
    early_year, early_month = _parse_month_year(earliest, earliest=True)
    short_early = earliest.astype("string").str.extract(r"-(\d{2})$")[0].notna()
    needs_previous_century = short_early & early_year.gt(issue_year)
    early_year = early_year.where(~needs_previous_century, early_year - 100)
    months = (issue_year - early_year) * 12 + (issue_month - early_month)
    invalid = months.notna() & months.lt(0)
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} rows where earliest_cr_line is after issue_d"
        )
    return months.astype("Float32")


def parse_percentage(values: pd.Series) -> pd.Series:
    """Parse percentage strings to numeric percentage points."""
    text = values.astype("string").str.strip().str.removesuffix("%")
    return pd.to_numeric(text, errors="coerce").astype("Float32")


def parse_term(values: pd.Series) -> pd.Series:
    """Parse loan term strings to months."""
    extracted = values.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Float32")


def parse_emp_length(values: pd.Series) -> pd.Series:
    """Map employment length strings to ordinal years."""
    text = values.astype("string").str.strip().str.lower()
    result = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    result = result.mask(text.str.startswith("<"), 0)
    result = result.mask(text.isin(["n/a", "na", "none", ""]), np.nan)
    return result.astype("Float32")


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply shared deterministic transforms and return signed columns."""
    required = {ROW_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS[:-1], "issue_d", "earliest_cr_line"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required preparation columns: {missing}")
    result = frame[[ROW_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS[:-1]]].copy()
    result["int_rate"] = parse_percentage(frame["int_rate"])
    result["revol_util"] = parse_percentage(frame["revol_util"])
    result["term"] = parse_term(frame["term"])
    result["emp_length"] = parse_emp_length(frame["emp_length"])
    result["credit_history_months"] = _credit_history_months(
        frame["issue_d"], frame["earliest_cr_line"]
    )
    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("string").str.strip()
    return result[[ROW_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]]


def fit_category_schema(training: pd.DataFrame) -> dict[str, Any]:
    """Fit category levels from the complete training split only."""
    categories = {
        column: sorted(training[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }
    return {
        "schema_version": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "categories": categories,
    }


def apply_category_schema(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Apply saved training categories without learning from other splits."""
    if schema.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Feature schema does not match the configured feature list")
    result = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        categories = schema["categories"][column]
        values = result[column].where(result[column].isin(categories))
        result[column] = pd.Categorical(values, categories=categories)
    return result


def _numeric_auc(values: pd.Series, target: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0 or numeric.nunique(dropna=True) < 2:
        return 0.5
    filled = numeric.fillna(float(numeric.median()))
    auc = float(roc_auc_score(target, filled))
    return max(auc, 1.0 - auc)


def _categorical_auc(values: pd.Series, target: pd.Series) -> float:
    encoded = values.astype("string").fillna("__MISSING__")
    rates = pd.DataFrame({"value": encoded, "target": target}).groupby(
        "value", observed=True
    )["target"].mean()
    scores = encoded.map(rates).astype(float)
    if scores.nunique() < 2:
        return 0.5
    auc = float(roc_auc_score(target, scores))
    return max(auc, 1.0 - auc)


def leakage_auc_table(training: pd.DataFrame) -> pd.DataFrame:
    """Calculate a direction-normalized univariate ROC-AUC per feature."""
    target = training[TARGET_COLUMN].astype(np.uint8)
    if target.nunique() != 2:
        raise ValueError("Leakage check requires both target classes")
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        values = training[feature]
        kind = "categorical" if feature in CATEGORICAL_COLUMNS else "numeric"
        auc = _categorical_auc(values, target) if kind == "categorical" else _numeric_auc(values, target)
        rows.append(
            {
                "feature": feature,
                "kind": kind,
                "auc": auc,
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
                "status": "FAIL" if auc > LEAKAGE_AUC_THRESHOLD else "PASS",
            }
        )
    return pd.DataFrame(rows).sort_values(["auc", "feature"], ascending=[False, True]).reset_index(drop=True)


def enforce_leakage_check(training: pd.DataFrame) -> pd.DataFrame:
    """Return the full table or raise if a feature exceeds the threshold."""
    table = leakage_auc_table(training)
    failed = table[table["status"] == "FAIL"]
    if not failed.empty:
        details = ", ".join(f"{row.feature}={row.auc:.4f}" for row in failed.itertuples())
        raise ValueError(
            f"Univariate leakage check exceeded AUC {LEAKAGE_AUC_THRESHOLD:.2f}: {details}"
        )
    return table
