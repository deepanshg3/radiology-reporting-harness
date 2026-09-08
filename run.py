#!/usr/bin/env python3
"""CLI entry point for the Radiology Reporting Harness.

Local (no-API) usage runs against the deterministic mock:

    python run.py --mode test --mock --limit 2
    python run.py --mode test --mock --limit 2 --resume
    python run.py --mode test --mock                       # full mock run

Real (Gemini) usage reads GEMINI_API_KEY / GEMINI_MODEL from .env:

    python run.py --mode test --limit 5                    # 5 real cases
    python run.py --mode test --resume                     # continue a run
    python run.py --mode test --export-submission          # validate + export
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src import config
from src.checkpoint_manager import CheckpointManager
from src.data_loader import load_test_cases
from src.errors import SubmissionError
from src.gemini_client import create_report_generator
from src.pipeline import Pipeline
from src.validator import build_submission, find_submission_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Incremental, resumable report generation for data/test.csv.",
    )
    parser.add_argument(
        "--mode",
        choices=["test"],
        default="test",
        help="Only 'test' (production inference) is supported.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=config.TEST_CSV,
        help="Path to the inference CSV (default: data/test.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N cases this run (after skipping successes).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from disk: skip SUCCESS, retry FAILED, process the rest.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint progress and reprocess everything.",
    )
    parser.add_argument(
        "--export-submission",
        action="store_true",
        help="Validate checkpoint state and write submission.csv (no processing).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic local mock generator (no network).",
    )
    parser.add_argument(
        "--mock-fail-case",
        action="append",
        default=None,
        metavar="CASE_ID",
        help="Mock: fail this case_id (repeatable). Used for retry tests.",
    )
    parser.add_argument(
        "--mock-log",
        type=Path,
        default=None,
        help="Mock: append one JSON line per generator call to this file.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=config.CHECKPOINTS_DIR,
        help="Directory for the append-only checkpoint (default: checkpoints/).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=config.OUTPUTS_DIR,
        help="Directory for predictions/failures/submission (default: outputs/).",
    )
    parser.add_argument(
        "--details-log",
        type=Path,
        default=None,
        help="Optional JSONL path to record per-case experiment details (report, "
             "edited-section count, impression change, latency). Never writes to "
             "submission.csv.",
    )
    return parser


def _wipe_run_artifacts(paths: list[Path]) -> list[Path]:
    """Delete checkpoint/output artifacts for a ``--fresh`` run."""
    removed: list[Path] = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                removed.append(path)
        except OSError as exc:
            logging.getLogger("radiology_harness").warning(
                "--fresh: could not remove %s: %s", path, exc
            )
    return removed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("radiology_harness")

    settings = config.load_settings()
    dataset = load_test_cases(args.test_csv)
    test_cases = list(dataset.cases)
    logger.info("loaded %d test cases from %s", len(test_cases), args.test_csv)

    checkpoint_path = args.checkpoints_dir / config.CHECKPOINT_FILE
    predictions_path = args.outputs_dir / "test_predictions.csv"
    failures_path = args.outputs_dir / "failures.csv"
    submission_path = args.outputs_dir / "submission.csv"
    complete_marker_path = args.outputs_dir / "test_predictions.complete"

    if args.fresh:
        for removed in _wipe_run_artifacts(
            [checkpoint_path, predictions_path, failures_path,
             submission_path, complete_marker_path]
        ):
            logger.info("--fresh: removed %s", removed)

    manager = CheckpointManager(checkpoint_path, logger=logger)
    logger.info(
        "checkpoint: %d attempted (%d SUCCESS, %d FAILED) @ %s",
        manager.attempted_count,
        manager.success_count,
        manager.failure_count,
        checkpoint_path,
    )

    # --- standalone export: validate current state, no processing -----------
    if args.export_submission:
        errors = find_submission_errors(test_cases, manager.state)
        if errors:
            logger.error("submission export refused:\n%s", "\n".join(errors))
            return 1
        df = build_submission(test_cases, manager.state, submission_path)
        logger.info("submission written: %s (%d rows)", submission_path, len(df))
        return 0

    # --- guard against accidental full re-processing -------------------------
    if manager.attempted_count and not (args.resume or args.fresh):
        build_parser().error(
            f"checkpoint already holds {manager.attempted_count} case record(s). "
            "Pass --resume to continue incrementally, or --fresh to start over."
        )

    if not args.mock:
        try:
            config.require_gemini_api_key(settings)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1

    if args.mock:
        generator = create_report_generator(
            mock=True,
            fail_case_ids=set(args.mock_fail_case or []),
            log_path=args.mock_log,
        )
    else:
        generator = create_report_generator(settings=settings, mock=False)

    pipeline = Pipeline(
        settings=settings,
        test_cases=test_cases,
        checkpoint_manager=manager,
        generator=generator,
        predictions_path=predictions_path,
        failures_path=failures_path,
        complete_marker_path=complete_marker_path,
        details_log_path=args.details_log,
        logger=logger,
    )

    plan = pipeline.build_plan(resume=args.resume, limit=args.limit)
    logger.info("%s", plan.summary)

    try:
        summary = pipeline.run(plan)
    except KeyboardInterrupt:
        logger.warning("interrupted by user; progress is on disk — resume with --resume")
        return 130

    logger.info("run complete: %s", summary)
    for case in plan.skipped_success:
        logger.debug("skipped (already SUCCESS): %s", case.case_id)

    remaining = pipeline.remaining_cases
    if remaining:
        logger.warning(
            "INCOMPLETE: %d case(s) still pending/failed (e.g. %s). "
            "Next: python run.py --mode test --resume",
            len(remaining),
            [c.case_id for c in remaining[:3]],
        )
        for recorder in manager.state.values():
            if recorder.is_failed:
                logger.warning(
                    "FAILED %s attempts=%d category=%s %s",
                    recorder.case_id, recorder.attempts,
                    recorder.error_category, recorder.error_message,
                )
        return 0

    try:
        pipeline.finalize_submission(submission_path)
    except SubmissionError as exc:
        logger.error("final submission rejected:\n%s", exc)
        return 1
    logger.info("ALL CASES SUCCESSFUL — submission written: %s", submission_path)
    logger.info("completion marker: %s", complete_marker_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())