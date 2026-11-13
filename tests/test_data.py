from __future__ import annotations

import numpy as np
import pandas as pd

from tfmbench.config import FEATURE_COLUMNS, ROW_ID_COLUMN, TARGET_COLUMN
from tfmbench.data import clean_lending_club_frame, sample_cohort
from tfmbench.features import prepare_feature_frame


def synthetic_source() -> pd.DataFrame:
    rows = 6
    frame = pd.DataFrame(
        {
            "loan_amnt": [1000.0] * rows,
            "term": [" 36 months", "60 months", "36 months", "36 months", "60 months", "36 months"],
            "int_rate": ["10.5%", "20%", "9%", "12%", "11%", "8%"],
            "installment": [32.0] * rows,
            "grade": ["A", "D", "B", "C", "B", "A"],
            "sub_grade": ["A1", "D2", "B3", "C1", "B2", "A2"],
            "emp_length": ["< 1 year", "10+ years", "2 years", None, "1 year", "n/a"],
            "home_ownership": ["RENT"] * rows,
            "annual_inc": [50_000.0] * rows,
            "verification_status": ["Verified"] * rows,
            "purpose": ["credit_card"] * rows,
            "addr_state": ["CA"] * rows,
            "dti": [12.0] * rows,
            "delinq_2yrs": [0.0] * rows,
            "fico_range_low": [700.0] * rows,
            "fico_range_high": [704.0] * rows,
            "inq_last_6mths": [1.0] * rows,
            "open_acc": [8.0] * rows,
            "pub_rec": [0.0] * rows,
            "revol_bal": [5000.0] * rows,
            "revol_util": ["40%", "50%", None, "30%", "20%", "10%"],
            "total_acc": [15.0] * rows,
            "mort_acc": [1.0] * rows,
            "pub_rec_bankruptcies": [0.0] * rows,
            "application_type": ["Individual"] * rows,
            "issue_d": ["Jan-2018"] * rows,
            "earliest_cr_line": ["Dec-68", "Jan-2008", "Jan-18", None, "Jun-1999", "Jan-2010"],
            "loan_status": [
                "Fully Paid",
                "Charged Off",
                "Current",
                None,
                "Does not meet the credit policy. Status:Charged Off",
                "Does not meet the credit policy. Status:Fully Paid",
            ],
        }
    )
    frame[ROW_ID_COLUMN] = np.arange(rows, dtype=np.int64)
    return frame


def test_cleaning_target_and_footer_row():
    cleaned = clean_lending_club_frame(synthetic_source())
    assert cleaned[ROW_ID_COLUMN].tolist() == [0, 1, 4, 5]
    assert cleaned[TARGET_COLUMN].tolist() == [0, 1, 1, 0]


def test_feature_parsing_and_credit_history_century():
    prepared = prepare_feature_frame(clean_lending_club_frame(synthetic_source()))
    assert list(prepared.columns) == [ROW_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    assert prepared["term"].tolist() == [36.0, 60.0, 60.0, 36.0]
    assert prepared["int_rate"].tolist() == [10.5, 20.0, 11.0, 8.0]
    assert prepared["emp_length"].iloc[0] == 0
    assert prepared["emp_length"].iloc[1] == 10
    assert pd.isna(prepared["emp_length"].iloc[3])
    assert prepared["credit_history_months"].iloc[0] == 589


def test_stratified_cohort_is_exact_and_deterministic():
    frame = pd.DataFrame(
        {ROW_ID_COLUMN: np.arange(200), TARGET_COLUMN: np.repeat([0, 1], [160, 40])}
    )
    first = sample_cohort(frame, 100, 42)
    second = sample_cohort(frame, 100, 42)
    assert first[ROW_ID_COLUMN].tolist() == second[ROW_ID_COLUMN].tolist()
    assert first[TARGET_COLUMN].value_counts().to_dict() == {0: 80, 1: 20}
