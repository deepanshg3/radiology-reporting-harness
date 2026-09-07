"""Safely load and validate the inference dataset.

IMPORTANT: ``test.csv`` is the ONLY inference source. ``train.csv`` contains a
ground-truth ``report`` column and must never be used as input to the runtime
pipeline (no training loop, no evaluation loop over train, no embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import SAMPLE_SUBMISSION_CSV, TEST_CSV

CASE_ID_COL = "case_id"
EXPECTED_TEST_ROW_COUNT = 132  # verified against the real test.csv and sample_submission.csv

REQUIRED_COLUMNS = frozenset(
    {
        CASE_ID_COL,
        "modality",
        "body_part",
        "study_description",
        "patient_age_band",
        "patient_sex",
        "template_content",
        "dictation",
    }
)

# Columns that must be non-empty for every case (an empty template or dictation
# is itself an edge case the pipeline must handle, so emptiness here is only
# flagged when the whole value is blank).
NON_NULL_COLUMNS = frozenset(
    {CASE_ID_COL, "modality", "body_part", "template_content", "dictation"}
)

SUBMISSION_COLUMNS = frozenset({CASE_ID_COL, "report"})


class DataValidationError(ValueError):
    """Raised when the inference dataset violates the expected contract."""


@dataclass(frozen=True)
class TestCase:
    """A single inference row with clean, string-typed fields."""

    case_id: str
    modality: str
    body_part: str
    study_description: str
    patient_age_band: str
    patient_sex: str
    template_content: str
    dictation: str


@dataclass(frozen=True)
class TestDataset:
    """Ordered collection of inference rows."""

    cases: tuple[TestCase, ...]

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as raw strings, preserving exact cell contents.

    ``dtype=str`` keeps numeric-looking and whitespace content verbatim;
    ``keep_default_na=False`` turns empty cells into empty strings instead of
    NaN, so we can distinguish "empty" from "missing" explicitly.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_dataframe(df: pd.DataFrame, expected_rows: int = EXPECTED_TEST_ROW_COUNT) -> None:
    """Validate a test dataframe against the dataset contract.

    Raises DataValidationError (never silently fixes anything).
    """
    errors: list[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        errors.append("Dataset is empty.")

    if CASE_ID_COL in df.columns:
        dupes = int(df[CASE_ID_COL].duplicated().sum())
        if dupes:
            errors.append(f"{dupes} duplicate case_id value(s).")
        blanks = int((df[CASE_ID_COL].astype(str).str.strip() == "").sum())
        if blanks:
            errors.append(f"{blanks} blank case_id value(s).")

    if len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(df)}.")

    for col in sorted(NON_NULL_COLUMNS & set(df.columns)):
        blanks = int((df[col].astype(str).str.strip() == "").sum())
        if blanks:
            errors.append(f"Column '{col}' has {blanks} blank value(s).")

    if errors:
        raise DataValidationError("test.csv validation failed:\n" + "\n".join(errors))


def _to_case(record: dict) -> TestCase:
    return TestCase(
        case_id=str(record[CASE_ID_COL]).strip(),
        modality=str(record["modality"]).strip(),
        body_part=str(record["body_part"]).strip(),
        study_description=str(record["study_description"]).strip(),
        patient_age_band=str(record["patient_age_band"]).strip(),
        patient_sex=str(record["patient_sex"]).strip(),
        template_content=str(record["template_content"]),
        dictation=str(record["dictation"]),
    )


def load_test_cases(path: Path | str = TEST_CSV) -> TestDataset:
    """Load, validate, and return the inference cases in original row order."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Test CSV not found: {path}")

    df = _read_csv(path)
    validate_dataframe(df)

    return TestDataset(cases=tuple(_to_case(record) for record in df.to_dict("records")))


def load_sample_submission(path: Path | str = SAMPLE_SUBMISSION_CSV) -> pd.DataFrame:
    """Load sample_submission.csv as the structural contract for output."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Sample submission not found: {path}")
    df = _read_csv(path)
    missing = SUBMISSION_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"sample_submission.csv missing required columns: {sorted(missing)}"
        )
    return df


def verify_structural_contract(test_cases: TestDataset, submission: pd.DataFrame) -> None:
    """Ensure the submission skeleton covers exactly the test case_ids.

    Errors are raised rather than auto-corrected so a bad run can never produce
    a silently corrupted submission.
    """
    errors: list[str] = []
    test_ids = set(test_cases.case_ids)
    sub_ids = set(submission[CASE_ID_COL].astype(str).str.strip())

    if len(submission) != len(test_cases):
        errors.append(
            f"Submission has {len(submission)} rows; expected {len(test_cases)}."
        )
    if len(sub_ids) != len(submission[CASE_ID_COL]):
        errors.append("Submission contains duplicate case_id values.")
    if fatal := sorted(test_ids - sub_ids):
        errors.append(f"Missing from submission: {len(fatal)} case_id(s), e.g. {fatal[:3]}.")
    if extra := sorted(sub_ids - test_ids):
        errors.append(f"Unexpected in submission: {len(extra)} case_id(s), e.g. {extra[:3]}.")
    if errors:
        raise DataValidationError("Submission contract violated:\n" + "\n".join(errors))