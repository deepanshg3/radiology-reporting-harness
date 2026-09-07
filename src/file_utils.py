"""Low-level, crash-safe file primitives.

Two guarantees that matter for this project:

- JSONL checkpoints are append-only. A record is written, flushed, and
  fsynced as one unit, so an interrupted write leaves at worst a trailing
  partial line that readers must tolerate.
- CSV/text replacements go through a temp file + ``os.replace`` so a reader
  never observes a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def append_jsonl(record: dict, path: Path | str) -> None:
    """Append one JSON record to *path*, fsynced for durability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_tolerant(path: Path | str) -> tuple[list[dict], list[int]]:
    """Read JSONL records, returning ``(records, corrupt_line_numbers)``.

    An empty file or a final partial line (crash during append) is tolerated:
    the partial line is reported as corrupt and skipped. Any corrupt *middle*
    line is also skipped and reported, so a corrupted checkpoint never silently
    poisons resume logic.
    """
    path = Path(path)
    if not path.is_file():
        return [], []

    records: list[dict] = []
    corrupt: list[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                corrupt.append(line_no)
    return records, corrupt


def atomic_write_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Atomically replace *path* with the serialized dataframe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(text: str, path: Path | str) -> None:
    """Atomically replace *path* with a block of text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise