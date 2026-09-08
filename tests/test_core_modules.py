#!/usr/bin/env python3
"""Network-free unit + integration checks for the Step-3 core modules.

Covers the 11 required test points:
  1. Sectioned-template parsing (labels + bodies, incl. grouped headers).
  2. Prose-template parsing and whole-block ``FINDINGS`` editing.
  3. ``apply_edits`` surgical replacement; untouched text stays byte-identical.
  4. Invalid edit rejection: unknown labels, empty FINDINGS, empty IMPRESSION.
  5. IMPRESSION replacement vs ``None`` (keep template impression).
  6. Placeholder resolution: [generic], [left/right], [_laterality_],
     [left/right/bilateral]; unresolvable laterality -> warning, not an error.
  7. ``response_parser`` strict validation of the model payload.
  8. ``prompt_builder`` structure: turns, derived labels, few-shot capping.
  9. ``validate_report`` errors vs warnings (placeholders, measurements);
     bracketed placeholder/template text is a hard rejection.
 10. Error classification: APIError codes -> AUTH/CONFIG/RATE_LIMIT/TRANSIENT;
     INVALID_OUTPUT retryable; AUTH/CONFIG non-retryable.
 11. End-to-end mock run: all SUCCESS, warnings persisted, resume skips via a
     call log, ``--fresh`` wipes state.
 12. Finalization rejects any report containing ``[``/``]`` (text inspection).
 13. Apply-time bracket stripping: final reports never contain ``[``/``]``.

No API keys, no network, no real data/*.csv files are touched.
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
sys.path.insert(0, str(ROOT))

from src.checkpoint_manager import CaseRecord, Status
from src.data_loader import TestCase
from src.errors import (
    ErrorCategory,
    InvalidModelOutput,
    MockFailure,
    SubmissionError,
    TemplateError,
    classify_error,
    is_retryable,
)
from src.gemini_client import _mock_edits
from src.prompt_builder import build_prompt, load_few_shot_examples, load_system_prompt
from src.response_parser import Edits, parse_edits, parse_model_text, strip_fences
from src.template_editor import apply_edits, parse_template, strip_bracketed_text
from src.validator import (
    build_submission,
    find_bracketed_spans,
    find_submission_errors,
    validate_report,
)

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
RUN = ROOT / "run.py"

_chks: dict[str, bool] = {}
_checks_run = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _checks_run
    _checks_run += 1
    _chks[name] = bool(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
SECTIONED = (
    "FINDINGS:\n"
    "BONES: Normal alignment. No fracture.\n"
    "JOINT SPACE: Preserved.\n"
    "SOFT TISSUES: Unremarkable.\n"
    "OTHER FINDINGS:\n"
    "Mild joint effusion.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute osseous abnormality of the knee."
)

GROUPED = (
    "FINDINGS:\n"
    "Menisci: \n"
    "Medial meniscus: Intact.\n"
    "Lateral meniscus: Intact.\n"
    "\n"
    "IMPRESSION:\n"
    "Intact menisci."
)

PROSE = (
    "FINDINGS:\n"
    "The spinal alignment is maintained.\n"
    "No fracture is identified.\n"
    "\n"
    "IMPRESSION:\n"
    "Unremarkable study."
)

GENERIC_TEMPLATE = (
    "FINDINGS:\n"
    "VERTEBRAE: Normal.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute abnormality evident in the [generic] spine."
)

LATERALITY_TEMPLATE = (
    "FINDINGS:\n"
    "BONES: Normal.\n"
    "\n"
    "IMPRESSION:\n"
    "Normal [left/right] ankle."
)

LATERALITY_UNDERSCORE_TEMPLATE = (
    "FINDINGS:\n"
    "DEEP VEINS: Patent.\n"
    "\n"
    "IMPRESSION:\n"
    "No deep venous thrombosis in the [_laterality_] lower extremity."
)

BILATERAL_TEMPLATE = (
    "FINDINGS:\n"
    "BONES: Normal.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute derangement of the [left/right/bilateral] knee."
)


def make_case(**kw: object) -> TestCase:
    defaults: dict[str, object] = dict(
        case_id="case_0001",
        modality="XRAY",
        body_part="Knee",
        study_description="XR KNEE",
        patient_age_band="40-44",
        patient_sex="female",
        template_content=SECTIONED,
        dictation="normal",
    )
    defaults.update(kw)
    return TestCase(**defaults)  # type: ignore[arg-type]


def _expect_invalid(name: str, fn) -> None:
    try:
        fn()
        check(name, False, "expected InvalidModelOutput, none raised")
    except InvalidModelOutput:
        check(name, True)


# --------------------------------------------------------------------------- #
# 1 + 2. Template parsing (sectioned + grouped + prose)
# --------------------------------------------------------------------------- #
def test_parse_sectioned() -> None:
    parsed = parse_template(SECTIONED)
    check("parse: sectioned template detected",
          parsed.is_sectioned and parsed.section_labels == [
              "BONES", "JOINT SPACE", "SOFT TISSUES", "OTHER FINDINGS"])
    bodies = {s.label: s.body for s in parsed.sections}
    check("parse: inline-body section", bodies["BONES"] == "Normal alignment. No fracture.")
    check("parse: multiline-body section", bodies["OTHER FINDINGS"] == "Mild joint effusion.")
    check("parse: impression captured", parsed.after_impression.startswith("\nNo acute osseous"))


def test_parse_grouped() -> None:
    parsed = parse_template(GROUPED)
    check("parse: grouped header expands",
          parsed.section_labels == ["Menisci", "Medial meniscus", "Lateral meniscus"])
    by_label = {s.label: s.body for s in parsed.sections}
    check("parse: group header has empty body", by_label["Menisci"] == "")
    check("parse: child sections bodies",
          by_label["Medial meniscus"] == "Intact." and by_label["Lateral meniscus"] == "Intact.")


def test_parse_prose() -> None:
    parsed = parse_template(PROSE)
    check("parse: prose detected (no sections)",
          not parsed.is_sectioned and parsed.section_labels == [])


def test_parse_malformed() -> None:
    try:
        parse_template("Just some text")
        check("parse: missing FINDINGS raises", False)
    except TemplateError:
        check("parse: missing FINDINGS raises", True)
    try:
        parse_template("FINDINGS:\nA: b\n\nIMPRESSION:\nX.\nIMPRESSION:\nY.")
        check("parse: duplicate IMPRESSION raises", False)
    except TemplateError:
        check("parse: duplicate IMPRESSION raises", True)


# --------------------------------------------------------------------------- #
# 3 + 4 + 5. Applying edits
# --------------------------------------------------------------------------- #
def test_apply_single_section() -> None:
    res = apply_edits(
        SECTIONED,
        parse_edits({"findings_edits": {"BONES": "No fracture. Joint alignment maintained."}}),
        case=make_case(),
    )
    expected = (
        "FINDINGS:\n"
        "BONES: No fracture. Joint alignment maintained.\n"
        "JOINT SPACE: Preserved.\n"
        "SOFT TISSUES: Unremarkable.\n"
        "OTHER FINDINGS:\n"
        "Mild joint effusion.\n"
        "\n"
        "IMPRESSION:\n"
        "No acute osseous abnormality of the knee."
    )
    check("apply: single section replaced, rest verbatim", res.report == expected, repr(res.report))


def test_apply_impression() -> None:
    res = apply_edits(
        SECTIONED,
        parse_edits({"findings_edits": {}, "impression": "Normal study."}),
        case=make_case(),
    )
    check("apply: impression replaced", res.report.endswith("IMPRESSION:\nNormal study."))
    kept = apply_edits(
        SECTIONED,
        parse_edits({"findings_edits": {}, "impression": None}),
        case=make_case(),
    )
    check("apply: impression=None keeps template byte-identical",
          kept.report == SECTIONED, repr(kept.report))


def test_apply_rejections() -> None:
    case = make_case()
    _expect_invalid("apply: unknown label raises", lambda: apply_edits(
        SECTIONED, parse_edits({"findings_edits": {"BONE": "x"}}), case=case))
    _expect_invalid("apply: prose rejects section labels", lambda: apply_edits(
        PROSE, parse_edits({"findings_edits": {"BONES": "x"}}), case=case))
    _expect_invalid("apply: empty whole-block FINDINGS raises", lambda: apply_edits(
        PROSE, parse_edits({"findings_edits": {"FINDINGS": ""}}), case=case))
    _expect_invalid("apply: empty IMPRESSION raises", lambda: apply_edits(
        SECTIONED, parse_edits({"findings_edits": {}, "impression": "   "}), case=case))


def test_apply_prose_whole_block() -> None:
    res = apply_edits(
        PROSE,
        parse_edits({
            "findings_edits": {"FINDINGS": "No fracture. Alignment maintained."},
            "impression": "Unremarkable.",
        }),
        case=make_case(body_part="Cervical spine"),
    )
    check("apply: prose whole-FINDINGS replaced",
          res.report == "FINDINGS:\nNo fracture. Alignment maintained.\n\nIMPRESSION:\nUnremarkable.",
          repr(res.report))


# --------------------------------------------------------------------------- #
# 6. Placeholder resolution
# --------------------------------------------------------------------------- #
def test_placeholder_generic() -> None:
    res = apply_edits(
        GENERIC_TEMPLATE,
        parse_edits({"findings_edits": {}}),
        case=make_case(body_part="Lumbar spine", dictation="normal"),
    )
    check("placeholder: [generic] -> lumbar", "the lumbar spine." in res.report, repr(res.report))
    check("placeholder: [generic] no warning", res.warnings == [], repr(res.warnings))


def test_placeholder_laterality() -> None:
    res = apply_edits(
        LATERALITY_TEMPLATE,
        parse_edits({"findings_edits": {}}),
        case=make_case(body_part="Ankle", dictation="MRI of the left ankle was performed"),
    )
    check("placeholder: [left/right] -> left", res.report.endswith("IMPRESSION:\nNormal left ankle."),
          repr(res.report))

    res2 = apply_edits(
        LATERALITY_UNDERSCORE_TEMPLATE,
        parse_edits({"findings_edits": {}}),
        case=make_case(dictation="The right common femoral vein is patent"),
    )
    check("placeholder: [_laterality_] -> right", "in the right lower extremity" in res2.report,
          repr(res2.report))

    res3 = apply_edits(
        BILATERAL_TEMPLATE,
        parse_edits({"findings_edits": {}}),
        case=make_case(dictation="Bilateral knee pain"),
    )
    check("placeholder: [left/right/bilateral] -> bilateral", "of the bilateral knee" in res3.report,
          repr(res3.report))


def test_placeholder_unresolvable() -> None:
    res = apply_edits(
        LATERALITY_TEMPLATE,
        parse_edits({"findings_edits": {}}),
        case=make_case(body_part="Ankle", dictation="No acute abnormality"),
    )
    check("placeholder: unresolvable -> warning emitted", any(
        "left unresolved" in w for w in res.warnings), repr(res.warnings))
    check("placeholder: unresolvable -> bracket-stripped deterministically",
          "[" not in res.report and "]" not in res.report,
          repr(res.report))


def test_apply_strips_brackets() -> None:
    bracketed = (
        "FINDINGS:\n"
        "PROSTATE: The prostate gland is normal in size"
        "[ and demonstrate normal post-contrast enhancement].\n"
        "BLADDER: Unremarkable.\n"
        "\n"
        "IMPRESSION:\nUnremarkable MRI."
    )
    res = apply_edits(
        bracketed,
        parse_edits({"findings_edits": {
            "PROSTATE": "The prostate gland is normal in size"
                        "[ and demonstrate normal post-contrast enhancement]."}}),
        case=make_case(dictation="Unremarkable"),
    )
    check("strip: sectioned bracketed template prose removed",
          "[" not in res.report and "]" not in res.report,
          repr(res.report))
    check("strip: bracketed clause text preserved",
          "size and demonstrate normal post-contrast enhancement." in res.report,
          repr(res.report))

    prose_res = apply_edits(
        PROSE,
        parse_edits({"findings_edits": {
            "FINDINGS": "Normal alignment[ and normal width] of the cervical spine."}}),
        case=make_case(body_part="Cervical spine"),
    )
    check("strip: prose whole-block edit bracket-free",
          "[" not in prose_res.report and "]" not in prose_res.report
          and "of the cervical spine." in prose_res.report,
          repr(prose_res.report))

    impression_res = apply_edits(
        SECTIONED,
        parse_edits({"findings_edits": {}, "impression": "Fine. [no comment]"}),
        case=make_case(),
    )
    check("strip: bracketed text in IMPRESSION removed",
          "no comment" in impression_res.report
          and "[" not in impression_res.report and "]" not in impression_res.report,
          repr(impression_res.report))

    check("strip: strip_bracketed_text unit", strip_bracketed_text("a[b]c[d e]f") == "abcd ef")
    check("strip: lone bracket removed", strip_bracketed_text("Normal] alignment.") == "Normal alignment.")
    check("strip: clean text unchanged", strip_bracketed_text("No brackets here") == "No brackets here")


# --------------------------------------------------------------------------- #
# 7. response_parser
# --------------------------------------------------------------------------- #
def test_response_parser_ok() -> None:
    edits = parse_edits({"findings_edits": {"BONES": "  Normal.\n"}, "impression": "  Fine.  "})
    check("parser: Edits fields normalized",
          isinstance(edits, Edits) and edits.findings_edits["BONES"] == "Normal."
          and edits.impression == "Fine.")
    none_imp = parse_edits({"findings_edits": {}, "impression": None})
    check("parser: impression null ok", none_imp.impression is None)
    empty = parse_edits({"findings_edits": {}})
    check("parser: findings_edits empty dict ok", empty.findings_edits == {})


def test_response_parser_invalid() -> None:
    _expect_invalid("parser: non-dict payload", lambda: parse_edits(["x"]))
    _expect_invalid("parser: unknown key", lambda: parse_edits(
        {"findings_edits": {}, "case_id": "123"}))
    _expect_invalid("parser: missing findings_edits", lambda: parse_edits({"impression": "x"}))
    _expect_invalid("parser: non-dict findings_edits", lambda: parse_edits(
        {"findings_edits": ["BONES"]}))
    _expect_invalid("parser: non-string value", lambda: parse_edits(
        {"findings_edits": {"BONES": 5}}))
    _expect_invalid("parser: blank key", lambda: parse_edits(
        {"findings_edits": {"  ": "x"}}))
    _expect_invalid("parser: non-string impression", lambda: parse_edits(
        {"findings_edits": {}, "impression": 7}))


def test_response_parser_model_text() -> None:
    fenced = "```json\n{\"findings_edits\": {\"BONES\": \"ok\"}}\n```"
    edits = parse_model_text(fenced)
    check("parser: fenced model text accepted", edits.findings_edits["BONES"] == "ok")
    check("parser: strip_fences handles plain text", strip_fences("no fence") == "no fence")
    _expect_invalid("parser: non-JSON text", lambda: parse_model_text("I am not JSON"))
    _expect_invalid("parser: empty text", lambda: parse_model_text(""))


# --------------------------------------------------------------------------- #
# 8. prompt_builder
# --------------------------------------------------------------------------- #
def test_prompt_builder() -> None:
    system = load_system_prompt()
    check("prompt: system prompt loads", "findings_edits" in system and "impression" in system)

    prompt = build_prompt(make_case())
    last_role, last_text = prompt.turns[-1]
    check("prompt: final turn is user", last_role == "user")
    check("prompt: labels derived from template", "BONES" in last_text and "OTHER FINDINGS" in last_text)
    check("prompt: template + dictation embedded",
          "No acute osseous abnormality" in last_text and "normal" in last_text)
    check("prompt: system instruction attached", prompt.system_instruction == system)

    prompt_prose = build_prompt(make_case(template_content=PROSE))
    check("prompt: prose case lists FINDINGS", "- FINDINGS" in prompt_prose.turns[-1][1])

    examples = load_few_shot_examples()
    check("prompt: few-shot file loads exactly 4 curated examples",
          len(examples) == 4, repr(len(examples)))
    prompt_fs = build_prompt(make_case(), few_shot=examples, max_examples=5)
    roles = [r for r, _ in prompt_fs.turns]
    check("prompt: few-shot expands to user/model turns",
          len(prompt_fs.turns) == 2 * len(examples) + 1
          and roles[:2] == ["user", "model"] and roles[-1] == "user",
          repr(roles))
    check("prompt: each few-shot example has valid structure",
          all(set(e) >= {"template", "dictation", "output"} for e in examples))

    many = [dict(examples[0]) | {"name": f"e{i}"} for i in range(6)]
    capped = build_prompt(make_case(), few_shot=many, max_examples=3)
    check("prompt: few-shot capped at max_examples",
          len(capped.turns) == 2 * 3 + 1, repr(len(capped.turns)))


# --------------------------------------------------------------------------- #
# 9. validate_report
# --------------------------------------------------------------------------- #
def test_validate_report() -> None:
    case = make_case()
    ok = validate_report(
        "FINDINGS:\nBONES: Normal.\nOTHER FINDINGS:\nNone.\n\nIMPRESSION:\nFine.", case)
    check("validator: 'OTHER FINDINGS' does not break header checks",
          ok.is_valid and ok.errors == [], repr(ok.errors))

    missing = validate_report("FINDINGS:\nBONES: Normal.\n\nIMPRESSION:", case)
    check("validator: empty IMPRESSION -> error", not missing.is_valid)
    missing2 = validate_report("Some report", case)
    check("validator: no FINDINGS header -> error", not missing2.is_valid)

    placeholder_case = make_case(dictation="No acute abnormality.")
    warned = validate_report(
        "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal [left/right] ankle.", placeholder_case)
    check("validator: leftover placeholder -> warning", any(
        "placeholder" in w for w in warned.warnings), repr(warned.warnings))
    check("validator: leftover placeholder -> ERROR (rejection)", not warned.is_valid,
          repr(warned.errors))

    generic = validate_report(
        "FINDINGS:\nNo acute abnormality evident in the [generic] spine.\n\nIMPRESSION:\nNormal.",
        make_case())
    check("validator: [generic] leftover -> rejected", not generic.is_valid and any(
        "bracket" in e.lower() for e in generic.errors), repr(generic.errors))

    bracketed_prose = validate_report(
        "FINDINGS:\nPROSTATE: The prostate gland is normal in size"
        "[ and demonstrate normal post-contrast enhancement].\n\nIMPRESSION:\nUnremarkable.",
        make_case())
    check("validator: bracketed template prose -> rejected", not bracketed_prose.is_valid
          and "[ and demonstrate normal post-contrast enhancement]" in bracketed_prose.errors[0],
          repr(bracketed_prose.errors))

    stray = validate_report(
        "FINDINGS:\nNormal] alignment maintained.\n\nIMPRESSION:\nFine.",
        make_case())
    check("validator: lone ']' -> rejected", not stray.is_valid, repr(stray.errors))
    check("validator: find_bracketed_spans finds lone bracket",
          find_bracketed_spans("Normal] alignment.") == ["]"],
          repr(find_bracketed_spans("Normal] alignment.")))
    check("validator: clean report -> no brackets",
          find_bracketed_spans("FINDINGS:\nBONES: Normal.\n") == [])

    measure = validate_report(
        "FINDINGS:\nNormal.\n\nIMPRESSION:\nFine.",
        make_case(dictation="A 121.3 mL collection is noted"),
    )
    check("validator: dropped measurement -> warning", any(
        "121.3 mL" in w for w in measure.warnings), repr(measure.warnings))


def test_finalization_rejects_brackets() -> None:
    """Finalization inspects final report *text* and rejects any brackets."""

    def rec(case_id: str, report: str) -> CaseRecord:
        return CaseRecord(
            case_id=case_id, row_index=0, status=Status.SUCCESS,
            report=report, attempts=1, latency_seconds=0.1, model="mock",
        )

    out_dir = Path(tempfile.mkdtemp(prefix="radio-finalize-"))
    case = make_case()
    clean = {case.case_id: rec(case.case_id, case.template_content)}
    check("finalize: clean report passes", find_submission_errors([case], clean) == [],
          repr(find_submission_errors([case], clean)))

    df = build_submission([case], clean, out_dir / "submission.csv")
    check("finalize: clean report written", len(df) == 1 and df.iloc[0]["case_id"] == case.case_id)

    bad_state = {case.case_id: rec(
        case.case_id,
        "FINDINGS:\nNo acute abnormality evident in the [generic] spine.\n\nIMPRESSION:\nNormal.",
    )}
    ferr = find_submission_errors([case], bad_state)
    check("finalize: [generic] report rejected", any(
        "bracket" in e.lower() and case.case_id in e for e in ferr), repr(ferr))

    try:
        build_submission([case], bad_state, out_dir / "submission2.csv")
        check("finalize: build_submission raises SubmissionError", False)
    except SubmissionError:
        check("finalize: build_submission raises SubmissionError", True)

    prose_state = {case.case_id: rec(
        case.case_id,
        "FINDINGS:\nPROSTATE: Normal in size[ and demonstrate normal post-contrast enhancement]."
        "\n\nIMPRESSION:\nFine.",
    )}
    perr = find_submission_errors([case], prose_state)
    check("finalize: bracketed template prose rejected", bool(perr),
          repr(perr))

    stray_state = {case.case_id: rec(
        case.case_id, "FINDINGS:\nNormal] alignment.\n\nIMPRESSION:\nFine.")}
    check("finalize: lone ']' rejected", bool(find_submission_errors([case], stray_state)))


# --------------------------------------------------------------------------- #
# 10. error classification
# --------------------------------------------------------------------------- #
class _FakeApiError(Exception):
    code: int | None

    def __init__(self, code: int | None):
        self.code = code
        super().__init__(code)


_FakeApiError.__name__ = "APIError"


def test_error_classification() -> None:
    check("classify: 401 -> AUTH", classify_error(_FakeApiError(401)) == ErrorCategory.AUTH)
    check("classify: 403 -> AUTH", classify_error(_FakeApiError(403)) == ErrorCategory.AUTH)
    check("classify: 400 -> CONFIG", classify_error(_FakeApiError(400)) == ErrorCategory.CONFIG)
    check("classify: 404 -> CONFIG", classify_error(_FakeApiError(404)) == ErrorCategory.CONFIG)
    check("classify: 429 -> RATE_LIMIT", classify_error(_FakeApiError(429)) == ErrorCategory.RATE_LIMIT)
    check("classify: 502 -> TRANSIENT", classify_error(_FakeApiError(502)) == ErrorCategory.TRANSIENT)
    check("classify: invalid model output", classify_error(InvalidModelOutput("x"))
          == ErrorCategory.INVALID_OUTPUT)
    check("classify: mock failure", classify_error(MockFailure("x")) == ErrorCategory.MOCK)
    check("retryable: INVALID_OUTPUT yes",
          is_retryable(ErrorCategory.INVALID_OUTPUT))
    check("retryable: RATE_LIMIT yes", is_retryable(ErrorCategory.RATE_LIMIT))
    check("retryable: AUTH no", not is_retryable(ErrorCategory.AUTH))
    check("retryable: CONFIG no", not is_retryable(ErrorCategory.CONFIG))


# --------------------------------------------------------------------------- #
# 11. End-to-end mock run (subprocess, temp dirs)
# --------------------------------------------------------------------------- #
def _write_fake_test_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "modality", "body_part", "study_description",
              "patient_age_band", "patient_sex", "template_content", "dictation"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i in range(1, 133):
            case_id = f"case_{i:04d}"
            template = (
                "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal study for " + case_id + "."
            )
            dictation = "normal" if i != 2 else "normal "*40 + "121.3 mL volume."
            writer.writerow([case_id, "XRAY", "Knee", "XR TEST", "40-44",
                             "female", template, dictation])


def _run_cli(test_csv: Path, ckdir: Path, outdir: Path, *args: str):
    env = dict(os.environ)
    env.update(REQUEST_DELAY_SECONDS="0", RETRY_BACKOFF_BASE_SECONDS="0", MAX_RETRIES="1")
    cmd = [str(VENV_PY), str(RUN), "--mode", "test", "--test-csv", str(test_csv),
           "--checkpoints-dir", str(ckdir), "--outputs-dir", str(outdir), *args]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def test_e2e_mock_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        csv_path = base / "test.csv"
        _write_fake_test_csv(csv_path)
        ck, out = base / "ck", base / "out"
        log = base / "calls.jsonl"

        p1 = _run_cli(csv_path, ck, out, "--limit", "3", "--mock", "--mock-log", str(log))
        check("e2e: mock run 1 exit 0", p1.returncode == 0, f"rc={p1.returncode}\n{p1.stdout}\n{p1.stderr}")
        records = _read_jsonl(ck / "checkpoints.jsonl")
        check("e2e: 3 SUCCESS records", len(records) == 3
              and all(r["status"] == "SUCCESS" for r in records), repr(records))
        check("e2e: warnings persisted on success", bool(records[1].get("warnings")),
              repr(records[1].get("warnings")))
        check("e2e: model= mock recorded", records[0].get("model") == "mock")

        p2 = _run_cli(csv_path, ck, out, "--resume", "--mock", "--limit", "1",
                      "--mock-log", str(log))
        calls_after = _read_jsonl(log)
        check("e2e: resume processes 1 new case (4 total calls, first 3 never re-queried)",
              len(calls_after) == 4 and calls_after[3]["case_id"] == "case_0004",
              repr(len(calls_after)))

        p3 = _run_cli(csv_path, ck, out, "--fresh", "--limit", "3", "--mock")
        fresh_records = _read_jsonl(ck / "checkpoints.jsonl")
        check("e2e: --fresh wipes prior state (3 records, not 6)",
              len(fresh_records) == 3, repr(len(fresh_records)))


# --------------------------------------------------------------------------- #
def main() -> int:
    test_parse_sectioned()
    test_parse_grouped()
    test_parse_prose()
    test_parse_malformed()
    test_apply_single_section()
    test_apply_impression()
    test_apply_rejections()
    test_apply_prose_whole_block()
    test_placeholder_generic()
    test_placeholder_laterality()
    test_placeholder_unresolvable()
    test_apply_strips_brackets()
    test_response_parser_ok()
    test_response_parser_invalid()
    test_response_parser_model_text()
    test_prompt_builder()
    test_validate_report()
    test_finalization_rejects_brackets()
    test_error_classification()
    test_e2e_mock_run()

    print(f"\n{_checks_run} checks run.")
    failed = [name for name, ok in _chks.items() if not ok]
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print("ALL CORE MODULE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())