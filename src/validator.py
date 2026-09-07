"""Final-submission validation against the Kaggle contract.

``sample_submission.csv`` defines the structural contract: a file with exactly
``case_id,report`` and one row per test case. This module refuses to emit a
submission until the checkpoint state satisfies that contract, so a partially
completed run can never be confused with a finished one.

Report *text* is additionally screened on finalization: any ``[`` or ``]``
still present (unresolved placeholders like ``[generic]`` or bracketed template
prose) rejects the submission.

Completion is tracked separately (see ``outputs/test_predictions.complete``);
this module is the enforcement point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import file_utils
from src.checkpoint_manager import Status
from src.data_loader import TestCase
from src.errors import SubmissionError
from src.template_editor import GENERIC_PLACEHOLDER, LATERALITY_PLACEHOLDERS, SECTION_HEADER_RE

FINDINGS_HEADER = "FINDINGS:"
IMPRESSION_HEADER = "IMPRESSION:"

KNOWN_PLACEHOLDERS = LATERALITY_PLACEHOLDERS + (GENERIC_PLACEHOLDER,)

# Values (number [+/-] unit) that must survive from the dictation into the
# report; a missing one means the editor/model likely dropped a measurement.
_MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mL|ml|mm|cm|cc|mmHg|mm\s*hg|kPa|degrees?|deg)"
)

_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def find_bracketed_spans(report: str) -> list[str]:
    """Return bracket-delimited spans present in the *final report text*.

    Any ``[``/``]`` character in generated report text is treated as leftover
    template placeholder or bracketed template prose — including safe-looking
    bracketed phrases such as ``[ and demonstrate normal post-contrast
    enhancement]`` that must never reach the submitted report. A lone or
    unbalanced bracket also counts. This inspects the emitted report, not the
    model's JSON payload.
    """
    text = report or ""
    if "[" not in text and "]" not in text:
        return []
    spans = _BRACKET_RE.findall(text)
    if not spans:
        spans = ["["] if "[" in text else ["]"]
    return spans


def collect_success_rows(
    test_cases: list[TestCase], state: dict
) -> list[dict]:
    """SUCCESS rows for every test case, in exact test.csv order."""
    rows: list[dict] = []
    for case in test_cases:
        record = state.get(case.case_id)
        if record is not None and record.status == Status.SUCCESS:
            rows.append({"case_id": case.case_id, "report": record.report})
    return rows


def find_submission_errors(test_cases: list[TestCase], state: dict) -> list[str]:
    """Return a list of contract violations (empty list means the run is final).

    Inspects each SUCCESS report's *text* (not the model JSON) and rejects any
    report that still contains bracketed placeholder/template text.
    """
    errors: list[str] = []
    test_ids = {case.case_id for case in test_cases}

    for case in test_cases:
        record = state.get(case.case_id)
        if record is None:
            errors.append(f"- missing: {case.case_id} (never attempted)")
        elif record.status != Status.SUCCESS:
            errors.append(f"- not succeeded: {case.case_id} (status={record.status})")
        elif not (record.report or "").strip():
            errors.append(f"- empty report: {case.case_id}")
        else:
            bracketed = find_bracketed_spans(record.report)
            if bracketed:
                errors.append(
                    f"- bracketed placeholder/template text in report: "
                    f"{case.case_id} ({', '.join(repr(s) for s in bracketed)})"
                )

    if not test_cases:
        errors.append("- no test cases provided")

    unexpected = [
        case_id
        for case_id, record in state.items()
        if record.status == Status.SUCCESS and case_id not in test_ids
    ]
    if unexpected:
        errors.append(
            f"- unexpected case_id(s) in state (not in test.csv): {unexpected[:5]}"
        )

    # Defensive duplicate check (state is a dict, but never trust by assumption).
    success_ids = [case_id for case_id, rec in state.items() if rec.status == Status.SUCCESS]
    if len(success_ids) != len(set(success_ids)):
        errors.append("- duplicate success case_id(s) found in state")

    max_errors = 10
    limited = errors[:max_errors]
    if len(errors) > max_errors:
        limited.append(f"- ... and {len(errors) - max_errors} more issue(s)")
    return limited


def build_submission(
    test_cases: list[TestCase], state: dict, path: Path | str
) -> pd.DataFrame:
    """Write ``submission.csv`` only if the full contract holds.

    Raises SubmissionError listing every violation otherwise.
    """
    errors = find_submission_errors(test_cases, state)
    if errors:
        raise SubmissionError(
            "Final submission rejected (run not complete/valid):\n"
            + "\n".join(errors)
        )

    df = pd.DataFrame(collect_success_rows(test_cases, state), columns=["case_id", "report"])
    file_utils.atomic_write_csv(df, path)
    return df


@dataclass(frozen=True)
class ReportValidation:
    """Per-report quality assessment computed right after edits are applied.

    - ``errors`` make the report unusable and force a retry (e.g. the structure
      is broken, an expected header is missing).
    - ``warnings`` are recorded in the checkpoint but never block the case.
    """

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _split_blocks(report: str) -> tuple[str | None, str | None]:
    """Return ``(findings_body, impression_body)`` using *whole-line* headers.

    Matching is per-line and exact so that labels like ``OTHER FINDINGS:``
    (which contain the substring 'FINDINGS:') can never be mistaken for the
    FINDINGS block header. ``None`` means the structure is broken.
    """
    lines = report.split("\n")
    if not lines or lines[0] != FINDINGS_HEADER:
        return None, None
    if IMPRESSION_HEADER not in lines:
        return None, None
    impression_index = lines.index(IMPRESSION_HEADER)
    if impression_index == 0:
        return None, None
    findings = "\n".join(lines[1:impression_index])
    impression = "\n".join(lines[impression_index + 1:])
    return findings, impression


def validate_report(report: str, case: TestCase) -> ReportValidation:
    """Lightweight structural + quality check on a single generated report."""
    errors: list[str] = []
    warnings: list[str] = []
    text = report or ""

    if not text.strip():
        errors.append("report is empty")
        return ReportValidation(warnings=warnings, errors=errors)

    bracketed = find_bracketed_spans(text)
    if bracketed:
        errors.append(
            "report contains bracketed placeholder/template text: "
            + ", ".join(repr(span) for span in bracketed)
        )

    findings, impression = _split_blocks(text)
    if findings is None:
        errors.append(
            f"report must begin with a {FINDINGS_HEADER!r} header line and "
            f"contain one {IMPRESSION_HEADER!r} header line"
        )
    else:
        if not findings.strip("\n"):
            errors.append("report has an empty FINDINGS block")
        if not impression.strip():
            errors.append("report has an empty IMPRESSION")

    # ---- warnings: quality issues that do not block the case -----------------
    leftovers = [
        token for token in KNOWN_PLACEHOLDERS if token in text
    ]
    if leftovers:
        warnings.append(
            f"unresolved placeholder(s) remain in final report: {leftovers}"
        )

    if findings is not None:
        labels = [
            SECTION_HEADER_RE.match(line).group(1)
            for line in findings.splitlines()
            if SECTION_HEADER_RE.match(line)
        ]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            warnings.append(f"duplicate section label(s) in final report: {duplicates}")

    missing_values = []
    for value in _MEASUREMENT_RE.findall(case.dictation or ""):
        needle = value.strip().lower()
        if needle and needle not in text.lower():
            missing_values.append(value.strip())
    if missing_values:
        warnings.append(
            f"measured value(s) stated in dictation absent from report: {missing_values}"
        )

    return ReportValidation(warnings=warnings, errors=errors)