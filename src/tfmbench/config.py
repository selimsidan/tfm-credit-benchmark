"""Immutable benchmark contract and user-editable defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 42
MAX_ALLOWED_ROWS = 1_000_000
COHORT_ROWS = 900_000
TRAIN_ROWS = 700_000
VALIDATION_ROWS = 100_000
TEST_ROWS = 100_000
PILOT_SIZES = (50_000, 100_000, 200_000)

# The exploratory profile is derived from the immutable 900k cohort. It is a
# separate signed experiment, so none of these values alter the full benchmark.
TRIAL_TRAIN_ROWS = 75_000
TRIAL_VALIDATION_ROWS = 10_000
TRIAL_TEST_ROWS = 20_000
TRIAL_EXPERIMENT_NAME = "lending_club_75k_trial"

DATASET_REPO_ID = "codesignal/lending-club-loan-accepted"
DATASET_FILENAME = "accepted_2007_to_2018Q4.csv"
TARGET_COLUMN = "target"
ROW_ID_COLUMN = "source_row_id"

COMPLETED_STATUSES = (
    "Fully Paid",
    "Charged Off",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
)
CHARGED_OFF_STATUSES = (
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
)

RAW_FEATURE_COLUMNS = (
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "application_type",
)
FEATURE_COLUMNS = (*RAW_FEATURE_COLUMNS, "credit_history_months")
CATEGORICAL_COLUMNS = (
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
)
DATE_COLUMNS = ("issue_d", "earliest_cr_line")
SOURCE_COLUMNS = (*RAW_FEATURE_COLUMNS, "loan_status", *DATE_COLUMNS)

# Documented exclusions include exact names and prefix patterns ending in *.
LEAKAGE_AND_ID_EXCLUSIONS = (
    "total_pymnt*",
    "total_rec_*",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_*",
    "last_credit_pull_d",
    "last_fico_*",
    "out_prncp*",
    "next_pymnt_d",
    "debt_settlement_flag",
    "settlement_*",
    "hardship_*",
    "pymnt_plan",
    "funded_amnt_inv",
    "policy_code",
    "url",
    "desc",
    "title",
    "emp_title",
    "id",
    "member_id",
    "zip_code",
)

STRING_SOURCE_COLUMNS = {
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "revol_util",
    "application_type",
    "loan_status",
    "issue_d",
    "earliest_cr_line",
}
SOURCE_DTYPES = {
    name: ("string" if name in STRING_SOURCE_COLUMNS else "float64")
    for name in SOURCE_COLUMNS
}

COMPUTE_UNITS_PER_HOUR = 5.4
MAX_SESSION_HOURS = 20.0
LEAKAGE_AUC_THRESHOLD = 0.80
MEMORY_GATE_FRACTION = 0.90
PILOT_TUNING_MINUTES = 15
FULL_TUNING_MINUTES = 45
MAX_OPTUNA_TRIALS = 60
EARLY_STOPPING_ROUNDS = 200
MAX_BOOST_ROUNDS = 5_000
INITIAL_PREDICTION_CHUNK_ROWS = 512
MIN_PREDICTION_CHUNK_ROWS = 128
MIN_TABPFN_VRAM_BYTES = 40_000_000_000
MIN_HIGH_RAM_BYTES = 40 * 1024**3
SUPPORTED_TABPFN_GPU_NAMES = ("A100", "H100", "H200", "B100", "B200", "GB200")
METHODS = ("lgbm_default", "lgbm_tuned", "realmlp", "tabpfn35")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Paths and fixed values used by data preparation."""

    output_root: Path | str
    experiment_name: str = "lending_club_900k"
    dataset_revision: str | None = None
    seed: int = SEED
    cohort_rows: int = COHORT_ROWS
    train_rows: int = TRAIN_ROWS
    validation_rows: int = VALIDATION_ROWS
    test_rows: int = TEST_ROWS
    pilot_sizes: tuple[int, ...] = PILOT_SIZES
    csv_chunksize: int = 100_000

    @property
    def experiment_root(self) -> Path:
        return Path(self.output_root).expanduser().resolve() / self.experiment_name

    @property
    def shared_dir(self) -> Path:
        return self.experiment_root / "shared"

    def validate(self) -> None:
        """Reject any configuration that violates the signed row contract."""
        total = self.train_rows + self.validation_rows + self.test_rows
        if self.cohort_rows != COHORT_ROWS or total != self.cohort_rows:
            raise ValueError(
                "The benchmark cohort must be exactly 900,000 rows with the "
                "700,000/100,000/100,000 split"
            )
        if self.cohort_rows > MAX_ALLOWED_ROWS:
            raise ValueError("The benchmark must never use more than 1,000,000 rows")
        if tuple(self.pilot_sizes) != PILOT_SIZES:
            raise ValueError("Pilot sizes must be exactly 50,000, 100,000, and 200,000")
        if max(self.pilot_sizes) > self.train_rows:
            raise ValueError("Pilot subsets must fit inside the training split")
        if self.seed != SEED:
            raise ValueError("The shared split seed must remain 42")


def validate_feature_contract() -> None:
    """Validate whitelist uniqueness and documented exclusions."""
    if len(FEATURE_COLUMNS) != 26 or len(set(FEATURE_COLUMNS)) != 26:
        raise AssertionError("The benchmark must contain 26 unique features")
    exact_exclusions = {name for name in LEAKAGE_AND_ID_EXCLUSIONS if "*" not in name}
    overlap = sorted(set(FEATURE_COLUMNS) & exact_exclusions)
    if overlap:
        raise AssertionError(f"Whitelisted features overlap exclusions: {overlap}")


validate_feature_contract()
