"""Durable, append-only checkpointing of per-case outcomes.

Design
------
- Source of truth is a JSONL file (``checkpoints/checkpoints.jsonl``).
- Every attempted case is appended as exactly one JSON record, fsynced
  immediately. There is never a full-file rewrite, so losing the process loses
  at most the case that was mid-write.
- The in-memory view is a mapping ``case_id -> latest record``. Because the
  log is append-only, a ``FAILED`` record is simply superseded by a later
  ``SUCCESS`` record for the same case; failures.csv views are therefore free
  of duplicates and dropped once a case succeeds.
- Readers tolerate a trailing partial line from an interrupted write.

Status model
------------
- SUCCESS  — case produced a report; skipped on resume, included in output.
- FAILED   — exhausted retries / permanent error; eligible for retry on resume.
- (absence of any record) — never attempted.
There is no explicit PENDING state; "not attempted" is represented by the
absence of a checkpoint record, which keeps resume logic simple and honest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import file_utils

UTC = timezone.utc


class Status:
    """Per-case outcome states stored in the checkpoint."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

    VALID = frozenset({SUCCESS, FAILED})


@dataclass(frozen=True)
class CaseRecord:
    """One checkpoint record for one test case (latest record wins)."""

    case_id: str
    row_index: int
    status: str
    report: str
    attempts: int
    latency_seconds: float
    error_category: str | None = None
    error_message: str | None = None
    model: str = ""
    timestamp: str = ""
    warnings: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "CaseRecord":
        """Build a record, defaulting optional fields for forward tolerance."""
        return cls(
            case_id=str(data.get("case_id", "")).strip(),
            row_index=int(data.get("row_index", -1)),
            status=str(data.get("status", "")),
            report=str(data.get("report", "")),
            attempts=int(data.get("attempts", 1)),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            error_category=data.get("error_category"),
            error_message=data.get("error_message"),
            model=str(data.get("model", "")),
            timestamp=str(data.get("timestamp", "")),
            warnings=str(data.get("warnings", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_success(self) -> bool:
        return self.status == Status.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.status == Status.FAILED


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class CheckpointManager:
    """Loads on start-up, appends per-case records, serves latest-per-case state."""

    def __init__(self, path: Path | str, logger=None):
        self.path = Path(path)
        self._logger = logger or _NullLogger()
        self._records: list[CaseRecord] = []
        self._state: dict[str, CaseRecord] = {}
        self.corrupt_line_count = 0
        self._load()

    # ------------------------------------------------------------------ state
    def _load(self) -> None:
        raw_records, corrupt_lines = file_utils.read_jsonl_tolerant(self.path)
        self.corrupt_line_count = len(corrupt_lines)
        if corrupt_lines:
            self._logger.warning(
                "checkpoint %s: %d unreadable line(s) skipped: %s",
                self.path, len(corrupt_lines), corrupt_lines[:20],
            )
        for raw in raw_records:
            try:
                record = CaseRecord.from_dict(raw)
            except (TypeError, ValueError) as exc:
                self._logger.warning("checkpoint line skipped (unable to parse): %s", exc)
                continue
            if not record.case_id or record.status not in Status.VALID:
                self._logger.warning(
                    "checkpoint line skipped (invalid case_id/status): %r", raw
                )
                continue
            self._records.append(record)
            # latest record for a case_id wins (append-only log semantics)
            self._state[record.case_id] = record

    @property
    def state(self) -> dict[str, CaseRecord]:
        """Latest record per case_id (copy is safest for callers)."""
        return dict(self._state)

    @property
    def records(self) -> list[CaseRecord]:
        return list(self._records)

    def get(self, case_id: str) -> CaseRecord | None:
        return self._state.get(case_id)

    def is_success(self, case_id: str) -> bool:
        record = self._state.get(case_id)
        return record is not None and record.is_success

    def is_failed(self, case_id: str) -> bool:
        record = self._state.get(case_id)
        return record is not None and record.is_failed

    @property
    def attempted_count(self) -> int:
        return len(self._state)

    @property
    def success_count(self) -> int:
        return sum(1 for rec in self._state.values() if rec.is_success)

    @property
    def failure_count(self) -> int:
        return sum(1 for rec in self._state.values() if rec.is_failed)

    # --------------------------------------------------------------- writers
    def _append(self, record: CaseRecord) -> CaseRecord:
        file_utils.append_jsonl(record.to_dict(), self.path)
        self._records.append(record)
        self._state[record.case_id] = record
        return record

    def record_success(
        self,
        *,
        case_id: str,
        row_index: int,
        report: str,
        attempts: int,
        latency_seconds: float,
        model: str,
        warnings: str = "",
    ) -> CaseRecord:
        return self._append(
            CaseRecord(
                case_id=case_id,
                row_index=row_index,
                status=Status.SUCCESS,
                report=report,
                attempts=attempts,
                latency_seconds=round(float(latency_seconds), 3),
                model=model,
                timestamp=_utc_now(),
                warnings=warnings,
            )
        )

    def record_failure(
        self,
        *,
        case_id: str,
        row_index: int,
        attempts: int,
        latency_seconds: float,
        error_category: str,
        error_message: str,
        model: str,
    ) -> CaseRecord:
        return self._append(
            CaseRecord(
                case_id=case_id,
                row_index=row_index,
                status=Status.FAILED,
                report="",
                attempts=attempts,
                latency_seconds=round(float(latency_seconds), 3),
                error_category=error_category,
                error_message=error_message,
                model=model,
                timestamp=_utc_now(),
            )
        )


class _NullLogger:
    def warning(self, *args, **kwargs):  # noqa: D401
        pass