"""Sequential inference pipeline with incremental persistence.

Behavior
--------
- Processes one case at a time (no concurrency). After *every* case the result
  is appended to the checkpoint (fsynced) and the working CSVs are rebuilt, so
  the process can be killed at any point and resumed later.
- ``SUCCESS`` cases are skipped on resume; ``FAILED`` cases are retried;
  never-attempted cases are processed. A case is never re-asked after a
  SUCCESS record exists.
- Inter-case and retry-backoff sleeps come from settings; setting
  ``REQUEST_DELAY_SECONDS=0`` disables all sleeps (used in local tests).
- Guarded by ``run.py``: a run that ignores an existing checkpoint is refused
  unless ``--resume`` or ``--fresh`` is passed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import file_utils
from src.checkpoint_manager import CaseRecord, CheckpointManager, Status
from src.config import Settings
from src.data_loader import TestCase
from src.errors import ErrorCategory, InvalidModelOutput, classify_error, is_retryable
from src.gemini_client import ReportGenerator
from src.template_editor import apply_edits
from src.validator import build_submission, validate_report

LOGGER = logging.getLogger("radiology_harness")


def _edit_summary(template_content: str, report: str) -> dict:
    """Deterministic per-case edit bookkeeping for experiment logging.

    Returns the edited section count, whether IMPRESSION changed, and a
    character-level edit ratio. Pure Python, safe to call on any report.
    """
    try:
        from src.analyze_training_edits import _char_edit_ratio
        from src.template_editor import parse_template

        template = parse_template(template_content)
        report_parsed = parse_template(report)
        edited = [
            s.label
            for s in template.sections
            if (s.label not in {r.label: r for r in report_parsed.sections})
            or (
                s.label in {r.label for r in report_parsed.sections}
                and s.body.strip() != next(
                    (r.body for r in report_parsed.sections if r.label == s.label), ""
                ).strip()
            )
        ]
        impression_changed = (
            template.after_impression.strip()
            != report_parsed.after_impression.strip()
        )
        return {
            "edited_sections": edited,
            "edited_section_count": len(edited),
            "impression_changed": impression_changed,
            "char_edit_ratio": round(_char_edit_ratio(template_content, report), 4),
        }
    except Exception:  # noqa: BLE001 — logging must never crash the pipeline
        return {}


@dataclass(frozen=True)
class Plan:
    """What this run will (and will not) touch."""

    to_process: list[TestCase]
    skipped_success: list[TestCase]

    @property
    def summary(self) -> str:
        return (
            f"plan: {len(self.to_process)} to process, "
            f"{len(self.skipped_success)} already-successful (skipped), "
            f"{len(self.to_process) + len(self.skipped_success)} planned"
        )


@dataclass
class RunSummary:
    plan: Plan
    succeeded: int = 0
    failed: int = 0
    call_count: int = 0
    started_at: str = ""
    finished_at: str = ""

    def __str__(self) -> str:
        return (
            f"succeeded={self.succeeded} failed={self.failed} "
            f"calls={self.call_count} ({self.plan.summary})"
        )


class Pipeline:
    """Orchestrates a single incremental inference run over test.csv."""

    def __init__(
        self,
        *,
        settings: Settings,
        test_cases: list[TestCase],
        checkpoint_manager: CheckpointManager,
        generator: ReportGenerator,
        predictions_path: Path,
        failures_path: Path,
        complete_marker_path: Path,
        details_log_path: Path | None = None,
        logger=None,
    ):
        self.settings = settings
        self.test_cases = test_cases
        self.checkpoint_manager = checkpoint_manager
        self.generator = generator
        self.predictions_path = Path(predictions_path)
        self.failures_path = Path(failures_path)
        self.complete_marker_path = Path(complete_marker_path)
        self.details_log_path = Path(details_log_path) if details_log_path else None
        self.logger = logger or LOGGER
        # REQUEST_DELAY_SECONDS == 0 disables all sleeps (test mode).
        self.delay_enabled = settings.request_delay_seconds > 0
        self._index_by_id = {
            case.case_id: idx for idx, case in enumerate(test_cases)
        }

    # ------------------------------------------------------------------ plan
    def build_plan(self, *, resume: bool, limit: int | None = None) -> Plan:
        to_process: list[TestCase] = []
        skipped: list[TestCase] = []
        for case in self.test_cases:
            record = self.checkpoint_manager.get(case.case_id)
            if resume and record is not None and record.is_success:
                skipped.append(case)
            else:
                to_process.append(case)
        if limit is not None:
            to_process = to_process[: max(0, int(limit))]
        return Plan(to_process=to_process, skipped_success=skipped)

    # ------------------------------------------------------------------- run
    def run(self, plan: Plan) -> RunSummary:
        summary = RunSummary(
            plan=plan,
            started_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        # Rebuild working CSVs from persisted state first: picks up any progress
        # from a previous run that was interrupted before its last sync.
        self.sync_outputs()

        total = len(plan.to_process)
        for index, case in enumerate(plan.to_process):
            record = self._process_case(case)
            summary.call_count += record.attempts
            if record.is_success:
                summary.succeeded += 1
            else:
                summary.failed += 1
            if total > 1:
                self.logger.info(
                    "[%s %3d/%3d] %s attempts=%d %.2fs",
                    record.status, index + 1, total, case.case_id,
                    record.attempts, record.latency_seconds,
                )
            if record.warnings:
                self.logger.warning(
                    "case %s warnings:\n%s", case.case_id, record.warnings
                )
            # Persist and refresh working outputs immediately per case.
            self.sync_outputs()
            if index < len(plan.to_process) - 1:
                self._sleep(self.settings.request_delay_seconds)

        summary.finished_at = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        return summary

    def _process_case(self, test_case: TestCase) -> CaseRecord:
        """One case, end-to-end: generate -> parse -> apply -> validate.

        The generator proposes structured :class:`~src.response_parser.Edits`;
        ``template_editor`` performs all text surgery; the report validator
        rejects structurally unusable reports (retried as INVALID_OUTPUT).
        """
        attempts = 0
        last_error: BaseException | None = None
        outcome: tuple[str, list[str]] | None = None
        started = time.monotonic()

        while attempts < self.settings.max_retries:
            attempts += 1
            try:
                edits = self.generator.generate(test_case, attempt=attempts)
                applied = apply_edits(test_case.template_content, edits, case=test_case)
                validation = validate_report(applied.report, test_case)
                if not validation.is_valid:
                    raise InvalidModelOutput(
                        "report validation failed: " + "; ".join(validation.errors)
                    )
                outcome = (applied.report, applied.warnings + validation.warnings)
                break
            except Exception as exc:  # noqa: BLE001 — classify and retry
                last_error = exc
                category = classify_error(exc)
                if attempts >= self.settings.max_retries:
                    break
                if not is_retryable(category):
                    self.logger.warning(
                        "case %s: non-retryable failure (%s)", test_case.case_id, category
                    )
                    break
                self._sleep(self.settings.retry_backoff_base_seconds * (2 ** (attempts - 1)))

        latency = time.monotonic() - started
        row_index = self._index_by_id[test_case.case_id]

        if outcome is not None:
            report, warnings = outcome
            record = self.checkpoint_manager.record_success(
                case_id=test_case.case_id,
                row_index=row_index,
                report=report,
                attempts=attempts,
                latency_seconds=latency,
                model=self.generator.model_name,
                warnings="\n".join(warnings),
            )
            self._log_details(test_case, record, report=report, warnings=warnings)
            return record

        category = classify_error(last_error) if last_error else ErrorCategory.GENERIC
        message = (
            f"{type(last_error).__name__}: {last_error}" if last_error
            else "all attempts exhausted without an exception"
        )
        record = self.checkpoint_manager.record_failure(
            case_id=test_case.case_id,
            row_index=row_index,
            attempts=attempts,
            latency_seconds=latency,
            error_category=category,
            error_message=str(message)[:2000],
            model=self.generator.model_name,
        )
        self._log_details(test_case, record, report="", errors=message)
        return record

    def _log_details(self, test_case: TestCase, record: CaseRecord, *, report: str, warnings=None, errors: str = "") -> None:
        """Append one line to the experiment details log, if configured."""
        if self.details_log_path is None:
            return
        entry = {
            "case_id": test_case.case_id,
            "status": record.status,
            "attempts": record.attempts,
            "latency_seconds": record.latency_seconds,
            "model": record.model,
            "warnings": warnings or [],
            "errors": errors,
            "report": report,
        }
        entry.update(_edit_summary(test_case.template_content, report or ""))
        self.details_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.details_log_path.open("a", encoding="utf-8").write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )

    # ------------------------------------------------------- output refresh
    def sync_outputs(self) -> None:
        """Rebuild predictions CSV and failures CSV from checkpoint state.

        Both are atomic replacements and (re)generated from state, so there
        are never duplicate rows even across many runs.
        """
        state = self.checkpoint_manager.state
        predictions = []
        for case in self.test_cases:
            rec = state.get(case.case_id)
            if rec is not None and rec.is_success:
                predictions.append({"case_id": rec.case_id, "report": rec.report})
        file_utils.atomic_write_csv(
            pd.DataFrame(predictions, columns=["case_id", "report"]),
            self.predictions_path,
        )

        failures = []
        for case in self.test_cases:
            rec = state.get(case.case_id)
            if rec is not None and rec.is_failed:
                failures.append(
                    {
                        "case_id": rec.case_id,
                        "row_index": rec.row_index,
                        "timestamp": rec.timestamp,
                        "error_category": rec.error_category or "",
                        "error_message": rec.error_message or "",
                        "attempts": rec.attempts,
                    }
                )
        file_utils.atomic_write_csv(
            pd.DataFrame(
                failures,
                columns=["case_id", "row_index", "timestamp",
                         "error_category", "error_message", "attempts"],
            ),
            self.failures_path,
        )

    # ------------------------------------------------------------- finalize
    def finalize_submission(self, submission_path: Path) -> pd.DataFrame:
        """Validate the full contract and write the final submission.

        Only called when every test case has succeeded; refuses otherwise.
        Also writes the completion marker that separates a finished run from a
        partially written working CSV.
        """
        df = build_submission(
            self.test_cases, self.checkpoint_manager.state, submission_path
        )
        marker = (
            "status: COMPLETE\n"
            f"cases: {len(df)}\n"
            f"timestamp: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}\n"
        )
        file_utils.atomic_write_text(marker, self.complete_marker_path)
        return df

    @property
    def remaining_cases(self) -> list[TestCase]:
        """Test cases not yet in a SUCCESS state (failed or never attempted)."""
        return [
            case for case in self.test_cases
            if not self.checkpoint_manager.is_success(case.case_id)
        ]

    def _sleep(self, seconds: float) -> None:
        if self.delay_enabled and seconds and seconds > 0:
            time.sleep(seconds)