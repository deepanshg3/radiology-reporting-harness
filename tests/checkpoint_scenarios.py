#!/usr/bin/env python3
"""Local, network-free end-to-end scenarios for checkpoint / resume behavior.

Scenarios
---------
A: `--limit 2 --mock`            -> 2 SUCCESS cases, persisted immediately.
B: `--limit 3 --mock --mock-fail`-> 2 SUCCESS + 1 FAILED (retries exhausted),
                                    failure recorded in failures.csv.
C: resume after a partial+failure -> SUCCESS cases skipped (proven via a call
   log), FAILED case retried and promoted to SUCCESS, never-attempted cases
   processed, no duplicate rows, idempotent third run.
D: submission export is rejected while incomplete and accepted once complete.

Every scenario uses a throwaway 132-row test.csv and temp artifact dirs; the
real data/*.csv files are never touched and no network requests occur.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
RUN = ROOT / "run.py"

PREDICTIONS_COLUMNS = ["case_id", "report"]
FAILURES_COLUMNS = [
    "case_id", "row_index", "timestamp",
    "error_category", "error_message", "attempts",
]

_chks = {}
_checks_run = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _checks_run
    _checks_run += 1
    _chks[name] = bool(condition)
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))


def make_fake_test_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "modality", "body_part", "study_description",
        "patient_age_band", "patient_sex", "template_content", "dictation",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i in range(1, 133):
            case_id = f"case_{i:04d}"
            writer.writerow(
                [
                    case_id, "XRAY", "Knee", "XR TEST", "40-44", "female",
                    f"FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal study for {case_id}.",
                    "normal",
                ]
            )


def run_cli(test_csv: Path, ckdir: Path, outdir: Path, *cli_args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        REQUEST_DELAY_SECONDS="0",
        RETRY_BACKOFF_BASE_SECONDS="0",
        MAX_RETRIES="3",
    )
    cmd = [
        str(VENV_PY), str(RUN),
        "--mode", "test",
        "--test-csv", str(test_csv),
        "--checkpoints-dir", str(ckdir),
        "--outputs-dir", str(outdir),
        *cli_args,
    ]
    return subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary(proc: subprocess.CompletedProcess) -> str:
    return f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def scenario_a(fake_csv: Path) -> None:
    print("\n=== Scenario A: 2 successful cases persisted ===")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ck, out, log = base / "ck", base / "out", base / "calls.jsonl"
        proc = run_cli(fake_csv, ck, out, "--limit", "2", "--mock", "--mock-log", str(log))
        check("A: exit 0", proc.returncode == 0, summary(proc))

        state = read_jsonl(ck / "checkpoints.jsonl")
        check("A: exactly 2 checkpoint records", len(state) == 2, repr(state))
        check(
            "A: both SUCCESS",
            [r["status"] for r in state] == ["SUCCESS", "SUCCESS"],
            repr(state),
        )
        check(
            "A: case_ids match test.csv order",
            [r["case_id"] for r in state] == ["case_0001", "case_0002"],
        )
        rows = read_csv_rows(out / "test_predictions.csv")
        check("A: predictions has 2 rows", len(rows) == 2)
        check(
            "A: predictions columns exact",
            list(rows[0].keys()) == PREDICTIONS_COLUMNS,
            repr(list(rows[0].keys())) if rows else "no rows",
        )
        check("A: non-empty reports", all(r["report"].strip() for r in rows))
        failures = read_csv_rows(out / "failures.csv")
        check("A: failures.csv is header-only", len(failures) == 0)
        check("A: no submission / marker yet",
              not (out / "submission.csv").exists()
              and not (out / "test_predictions.complete").exists())
        calls = read_jsonl(log)
        check("A: exactly 2 mock calls (case_0001, case_0002)",
              [c["case_id"] for c in calls] == ["case_0001", "case_0002"])


def scenario_b(fake_csv: Path) -> None:
    print("\n=== Scenario B: one FAILED case recorded ===")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ck, out, log = base / "ck", base / "out", base / "calls.jsonl"
        proc = run_cli(fake_csv, ck, out, "--limit", "3", "--mock",
                       "--mock-fail-case", "case_0003", "--mock-log", str(log))
        check("B: exit 0", proc.returncode == 0, summary(proc))

        state = read_jsonl(ck / "checkpoints.jsonl")
        check("B: 3 checkpoint records", len(state) == 3)
        check(
            "B: statuses SUCCESS,SUCCESS,FAILED",
            [(r["status"], r["case_id"]) for r in state]
            == [("SUCCESS", "case_0001"), ("SUCCESS", "case_0002"), ("FAILED", "case_0003")],
            repr(state),
        )
        failed = state[-1]
        check("B: failed record has attempts=MAX_RETRIES(3)", failed["attempts"] == 3)
        check("B: failed record error_category=MOCK", failed["error_category"] == "MOCK")
        check("B: failed record has error_message", bool(failed["error_message"]))
        check("B: failed record has timestamp", bool(failed["timestamp"]))

        failures = read_csv_rows(out / "failures.csv")
        check("B: failures.csv lists exactly case_0003",
              [(r["case_id"], r["attempts"], r["error_category"]) for r in failures]
              == [("case_0003", "3", "MOCK")], repr(failures))
        check("B: failures.csv columns exact", list(failures[0].keys()) == FAILURES_COLUMNS)
        rows = read_csv_rows(out / "test_predictions.csv")
        check("B: predictions contains only the 2 successes",
              [r["case_id"] for r in rows] == ["case_0001", "case_0002"])
        calls = read_jsonl(log)
        check(
            "B: attempts recorded (1+1+3 calls)",
            [c["case_id"] for c in calls] == [
                "case_0001", "case_0002",
                "case_0003", "case_0003", "case_0003",
            ],
            repr(calls),
        )


def scenario_c(fake_csv: Path) -> None:
    print("\n=== Scenario C: resume skips successes, retries the failed case ===")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ck, out, log = base / "ck", base / "out", base / "calls.jsonl"

        # Run 1: first 3 cases, case_0003 fails (3 attempts).
        p1 = run_cli(fake_csv, ck, out, "--limit", "3", "--mock",
                     "--mock-fail-case", "case_0003", "--mock-log", str(log))
        check("C1: run 1 exit 0", p1.returncode == 0, summary(p1))

        # Run 2: resume, process up to 2 more -> retry case_0003 + case_0004.
        p2 = run_cli(fake_csv, ck, out, "--limit", "2", "--resume", "--mock",
                     "--mock-log", str(log))
        check("C2: resume exit 0", p2.returncode == 0, summary(p2))

        state = read_jsonl(ck / "checkpoints.jsonl")
        check("C2: 5 total records on disk (3 + 2, case_0003 twice)",
              len(state) == 5, repr(state))

        # Latest-record-wins view:
        latest = {}
        for rec in state:
            latest[rec["case_id"]] = rec
        check("C2: case_0003 promoted to SUCCESS", latest["case_0003"]["status"] == "SUCCESS", repr(latest["case_0003"]))
        check("C2: case_0004 processed", latest.get("case_0004", {}).get("status") == "SUCCESS")

        # Call log proves skipped cases are never re-queried.
        calls = read_jsonl(log)
        count = {}
        for c in calls:
            count.setdefault(c["case_id"], 0)
            count[c["case_id"]] += 1
        check("C2: case_0001 called exactly once", count.get("case_0001") == 1, repr(count))
        check("C2: case_0002 called exactly once", count.get("case_0002") == 1, repr(count))
        check("C2: case_0003 3 (fail) + 1 (retry)", count.get("case_0003") == 4, repr(count))
        check("C2: case_0004 called once", count.get("case_0004") == 1, repr(count))

        rows = read_csv_rows(out / "test_predictions.csv")
        check("C2: predictions lists 4 unique successes",
              [r["case_id"] for r in rows] == ["case_0001", "case_0002", "case_0003", "case_0004"])
        check("C2: no duplicate case_id in predictions",
              len({r["case_id"] for r in rows}) == len(rows))
        failures = read_csv_rows(out / "failures.csv")
        check("C2: failures.csv is clean (case_0003 succeeded)",
              len(failures) == 0, repr(failures))
        check("C2: no submission yet (run is partial)", not (out / "submission.csv").exists())

        # Run 3: resume again, limit 1 -> next case (case_0005). Idempotent skip.
        p3 = run_cli(fake_csv, ck, out, "--limit", "1", "--resume", "--mock",
                     "--mock-log", str(log))
        check("C3: resume (limit 1) exit 0", p3.returncode == 0, summary(p3))
        calls3 = read_jsonl(log)
        count3 = {}
        for c in calls3:
            count3.setdefault(c["case_id"], 0)
            count3[c["case_id"]] += 1
        check("C3: case_0001..0004 call counts unchanged",
              count3.get("case_0001") == 1 and count3.get("case_0002") == 1
              and count3.get("case_0003") == 4 and count3.get("case_0004") == 1,
              repr(count3))
        check("C3: case_0005 now processed once", count3.get("case_0005") == 1, repr(count3))


def scenario_d(fake_csv: Path) -> None:
    print("\n=== Scenario D: submission export contract enforcement ===")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ck, out = base / "ck", base / "out"

        # Partial state: only 2 successes.
        p1 = run_cli(fake_csv, ck, out, "--limit", "2", "--mock")
        check("D1: partial run exit 0", p1.returncode == 0, summary(p1))

        p2 = run_cli(fake_csv, ck, out, "--export-submission", "--resume")
        check("D2: export rejected while incomplete (rc!=0)", p2.returncode != 0)
        check("D2: rejection names missing cases", "case_0003" in p2.stderr or "case_0003" in p2.stdout)
        check("D2: no submission.csv written", not (out / "submission.csv").exists())

        # Full mock run (no limit) completes all 132.
        p3 = run_cli(fake_csv, ck, out, "--resume", "--mock")
        check("D3: full resume run exit 0", p3.returncode == 0, summary(p3))
        sub = read_csv_rows(out / "submission.csv")
        check("D4: submission.csv has 132 rows", len(sub) == 132, f"{len(sub)} rows")
        check("D4: exactly case_id,report", list(sub[0].keys()) == ["case_id", "report"])
        check("D4: all test case_ids present exactly once",
              [r["case_id"] for r in sub]
              == [f"case_{i:04d}" for i in range(1, 133)])
        check("D4: all reports non-empty", all(r["report"].strip() for r in sub))
        check("D4: completion marker written",
              (out / "test_predictions.complete").exists())


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fake_csv = Path(tmp) / "test.csv"
        make_fake_test_csv(fake_csv)
        scenario_a(fake_csv)
        scenario_b(fake_csv)
        scenario_c(fake_csv)
        scenario_d(fake_csv)

    print(f"\n{_checks_run} checks run.")
    failed = [name for name, ok in _chks.items() if not ok]
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print("ALL SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())