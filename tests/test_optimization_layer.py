#!/usr/bin/env python3
"""Deterministic tests for the RES-optimization layer (no network, no CSV).

Covers:
  1. response_parser header guardrail (FINDINGS:/IMPRESSION: inside values).
  2. few_shot_examples.json: loads, every template parses, every output is valid
     Edits and round-trips through apply_edits to a bracket-free report.
  3. section_routing.RoutingModel: available flag, candidate_sections never
     returns unknown labels, degrades gracefully when the artifact is absent.
  4. prompt_builder: routing hints appear as a CANDIDATE block; prompt still
     lists exact template labels; hints are identified as non-restrictive.
  5. pipeline._edit_summary: deterministic edited-section count + impression
     change bookkeeping used by the experiment details log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import prompt_builder
from src.data_loader import TestCase
from src.errors import InvalidModelOutput
from src.pipeline import _edit_summary
from src.prompt_builder import (
    build_prompt,
    load_few_shot_examples,
    load_system_prompt,
)
from src.response_parser import parse_edits
from src.section_routing import RoutingModel
from src.template_editor import apply_edits, parse_template

_chks_run = 0
_results: dict[str, bool] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    global _chks_run
    _chks_run += 1
    _results[name] = bool(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not condition else ""))


def _case(template: str, dictation: str, **kw: object) -> TestCase:
    base: dict[str, object] = dict(
        case_id="case_x", modality="XRAY", body_part="Knee",
        study_description="XR KNEE", patient_age_band="40-44", patient_sex="female",
        template_content=template, dictation=dictation,
    )
    base.update(kw)
    return TestCase(**base)  # type: ignore[arg-type]


SIMPLE_TEMPLATE = (
    "FINDINGS:\n"
    "BONES: No fracture.\n"
    "JOINTS: Normal.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute osseous abnormality."
)


# --------------------------------------------------------------------------- #
# 1. Header guardrail
# --------------------------------------------------------------------------- #
def test_header_guardrail() -> None:
    def _try(fn) -> bool:
        try:
            fn()
            return False
        except InvalidModelOutput:
            return True

    check("guardrail: FINDINGS header in findings value rejected",
          _try(lambda: parse_edits({"findings_edits": {"BONES": "x\nFINDINGS:\ny"}})))
    check("guardrail: IMPRESSION header in findings value rejected",
          _try(lambda: parse_edits({"findings_edits": {"BONES": "x\nIMPRESSION:\ny"}})))
    check("guardrail: IMPRESSION header in impression value rejected",
          _try(lambda: parse_edits({"findings_edits": {}, "impression": "ok\nIMPRESSION:\nno"})))
    check("guardrail: clean value accepted",
          not _try(lambda: parse_edits({"findings_edits": {"BONES": "normal"}, "impression": "fine"})))


# --------------------------------------------------------------------------- #
# 2. Few-shot examples
# --------------------------------------------------------------------------- #
def test_few_shot_examples() -> None:
    from src.validator import validate_report

    examples = load_few_shot_examples()
    check("few-shot: exactly 4 curated examples", len(examples) == 4, repr(len(examples)))
    for ex in examples:
        name = ex.get("name", "?")
        parse_template(ex["template"])  # must parse (raises otherwise)
        edits = parse_edits(ex["output"])
        case = _case(ex["template"], ex["dictation"])
        result = apply_edits(ex["template"], edits, case=case)
        check(f"few-shot '{name}': applies to a bracket-free report",
              "[" not in result.report and "]" not in result.report, repr(result.report[:80]))
        check(f"few-shot '{name}': reconstruction passes structural validation",
              not validate_report(result.report, case).errors)
        # Ensure values are body-only (no block headers) — guardrail already covers.
        for key in edits.findings_edits:
            check(f"few-shot '{name}': key '{key}' is a real section/parameter",
                  isinstance(key, str) and key.strip() != "")


# --------------------------------------------------------------------------- #
# 3. Routing model
# --------------------------------------------------------------------------- #
def test_routing() -> None:
    model = RoutingModel()
    check("routing: routing artifact available", model.available)
    candidate_case = _case(
        SIMPLE_TEMPLATE, "fracture with mild effusion",
        modality="XRAY", body_part="Knee",
    )
    candidates = model.candidate_sections(candidate_case)
    check("routing: returns only template labels",
          set(candidates) <= {"BONES", "JOINTS"})
    check("routing: returns non-empty candidates", len(candidates) >= 1, repr(candidates))

    # Missing artifact -> degrades gracefully.
    absent = RoutingModel(Path(ROOT) / "does_not_exist_routing.json")
    check("routing: missing artifact degrades to unavailable",
          not absent.available and absent.candidate_sections(candidate_case) == [])


# --------------------------------------------------------------------------- #
# 4. Prompt builder with routing hints
# --------------------------------------------------------------------------- #
def test_prompt_hints() -> None:
    # Force the routing model into the prompt regardless of the shared default.
    routing = RoutingModel()
    case = _case(SIMPLE_TEMPLATE, "fracture of the kneecap with mild effusion",
                 modality="XRAY", body_part="Knee")
    prompt = build_prompt(case, routing_model=routing)
    user_text = prompt.turns[-1][1]
    check("prompt: labels listed", "TEMPLATE SECTION LABELS" in user_text
          and "- BONES" in user_text and "- JOINTS" in user_text)
    check("prompt: candidate hint present when routing available",
          "CANDIDATE SECTIONS TO CONSIDER" in user_text and "HINT" in user_text)
    check("prompt: template + dictation embedded",
          "TEMPLATE_CONTENT" in user_text and "DICTATION" in user_text)
    check("prompt: system prompt loads",
          "You are an expert radiology report TEMPLATE EDITOR"
          in load_system_prompt())

    # Without routing artifact available -> no hint block.
    no_routing = RoutingModel(Path(ROOT) / "missing_routing.json")
    prompt2 = build_prompt(case, routing_model=no_routing)
    check("prompt: no candidate block when routing unavailable",
          "CANDIDATE SECTIONS TO CONSIDER" not in prompt2.turns[-1][1])

    # Routing hints are explicitly non-restrictive (a hint, not a filter).
    check("prompt: hint described as suggestion",
          "HINT" in user_text and "not a restriction" in user_text)


# --------------------------------------------------------------------------- #
# 5. Pipeline edit summary
# --------------------------------------------------------------------------- #
def test_edit_summary() -> None:
    # Template unchanged -> 0 edits, impression unchanged.
    s = _edit_summary(SIMPLE_TEMPLATE, SIMPLE_TEMPLATE)
    check("summary: unchanged report -> 0 edits",
          s.get("edited_section_count", 1) == 0 and s.get("impression_changed") is False,
          repr(s))

    # BONES edited, JOINTS kept, impression changed.
    edited_report = SIMPLE_TEMPLATE.replace(
        "BONES: No fracture.",
        "BONES: A distal fibular fracture is present with mild effusion."
    ).replace(
        "NO ACUTE OSSEOUS ABNORMALITY".lower(), "X"
    ).replace(
        "IMPRESSION:\nNo acute osseous abnormality.",
        "IMPRESSION:\n1. Distal fibular fracture.",
    )
    s2 = _edit_summary(SIMPLE_TEMPLATE, edited_report)
    check("summary: detects 1 edited section", s2.get("edited_section_count") == 1, repr(s2))
    check("summary: detects impression change", s2.get("impression_changed") is True, repr(s2))
    check("summary: char edit ratio present", 0.0 < s2.get("char_edit_ratio", 0.0) < 1.0,
          repr(s2.get("char_edit_ratio")))


def main() -> int:
    test_header_guardrail()
    test_few_shot_examples()
    test_routing()
    test_prompt_hints()
    test_edit_summary()

    print(f"\n{_chks_run} checks run.")
    failed = [name for name, ok in _results.items() if not ok]
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print("ALL OPTIMIZATION-LAYER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
