#!/usr/bin/env python3
"""Offline, network-free tests for dynamic training-example retrieval.

Covers the 12 required verification points:

  1. train.csv loads exactly 636 rows.
  2. Retrieval is deterministic (same instance and across a fresh index).
  3. Every test case returns <= 3 examples.
  4. Examples actually come from train.csv.
  5. Retrieved examples carry real template/dictation/report data.
  6. The same test case retrieves the same examples repeatedly.
  7. Same modality/body_part cases are preferred when strong matches exist.
  8. Highly similar templates outrank unrelated templates.
  9. Retrieved examples are correctly inserted into the Gemini prompt.
 10. The old static examples are NOT simultaneously added.
 11. Existing prompt/response parsing behavior (static few_shot path) still works.
 12. Derived structured edits reconstruct the human reference report.

No API keys, no network calls. The real data/train.csv is read read-only.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.analyze_training_edits import normalize
from src.data_loader import TestCase, load_test_cases
from src.example_retriever import (
    DEFAULT_MAX_EXAMPLES,
    ExampleRetriever,
    RetrievalWeights,
    derive_structured_output,
    template_similarity_score,
)
from src.prompt_builder import (
    build_prompt,
    load_few_shot_examples,
    retrieval_prompt_builder,
)
from src.response_parser import parse_edits
from src.template_editor import apply_edits

_chks_run = 0
_results: dict[str, bool] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    global _chks_run
    _chks_run += 1
    _results[name] = bool(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not condition else ""))


TRAIN_CSV = ROOT / "data" / "train.csv"
TEST_CSV = ROOT / "data" / "test.csv"


def _case(
    *, case_id: str = "case_q", modality: str = "XRAY", body_part: str = "Knee",
    template: str, dictation: str,
) -> TestCase:
    return TestCase(
        case_id=case_id, modality=modality, body_part=body_part,
        study_description="", patient_age_band="", patient_sex="",
        template_content=template, dictation=dictation,
    )


KNEE_TEMPLATE = (
    "FINDINGS:\n"
    "BONES: No fracture or focal osseous lesion.\n"
    "JOINTS: No dislocation. The joint spaces are normal.\n"
    "SOFT TISSUES: The soft tissues are unremarkable.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute osseous abnormality."
)

KNEE_SIMILAR_TEMPLATE = (
    "FINDINGS:\n"
    "BONES: No fracture or focal osseous lesion.\n"
    "JOINTS: The joint spaces are normal.\n"
    "SOFT TISSUES: The soft tissues are unremarkable.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute osseous abnormality."
)

KNEE_SIMILAR_REPORT = (
    "FINDINGS:\n"
    "BONES: No fracture or focal osseous lesion.\n"
    "JOINTS: There is mild joint space narrowing. No dislocation.\n"
    "SOFT TISSUES: The soft tissues are unremarkable.\n"
    "\n"
    "IMPRESSION:\n"
    "1. Mild joint space narrowing.\n2. No acute osseous abnormality."
)

CHEST_TEMPLATE = (
    "FINDINGS:\n"
    "LUNGS: Clear.\n"
    "HEART: Normal size.\n"
    "DIAPHRAGM: Unremarkable.\n"
    "PLEURAL SPACES: No effusion.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute cardiopulmonary abnormality."
)

CHEST_REPORT_EDITED = (
    "FINDINGS:\n"
    "LUNGS: Mild left basilar opacity.\n"
    "HEART: Normal size.\n"
    "DIAPHRAGM: Unremarkable.\n"
    "PLEURAL SPACES: No effusion.\n"
    "\n"
    "IMPRESSION:\n"
    "1. Mild left basilar opacity.\n2. No acute cardiopulmonary abnormality."
)


def _write_train_csv(path: Path, rows: list[list]) -> None:
    header = ["case_id", "modality", "body_part", "study_description",
              "patient_age_band", "patient_sex", "template_content",
              "dictation", "report"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# 12. derive_structured_output round-trips the human reference
# --------------------------------------------------------------------------- #
def test_derive_output_round_trip() -> None:
    outputs = derive_structured_output(KNEE_SIMILAR_TEMPLATE, KNEE_SIMILAR_REPORT)
    check("derive: JOINTS body is the verbatim reference body",
          outputs["findings_edits"]["JOINTS"] == "There is mild joint space narrowing. No dislocation.",
          repr(outputs["findings_edits"]))
    check("derive: impression derived from reference",
          outputs["impression"] == "1. Mild joint space narrowing.\n2. No acute osseous abnormality.",
          repr(outputs["impression"]))
    check("derive: BONES unchanged -> no edit",
          "BONES" not in outputs["findings_edits"], repr(outputs["findings_edits"]))

    edits = parse_edits(outputs)
    case = _case(template=KNEE_SIMILAR_TEMPLATE, dictation="mild joint space narrowing, no fracture")
    reconstructed = apply_edits(KNEE_SIMILAR_TEMPLATE, edits, case=case).report
    check("derive: re-applying edits reconstructs the reference (normalized)",
          normalize(reconstructed) == normalize(KNEE_SIMILAR_REPORT),
          f"len rec={len(reconstructed)} ref={len(KNEE_SIMILAR_REPORT)}")

    no_edit = derive_structured_output(KNEE_TEMPLATE, KNEE_TEMPLATE)
    check("derive: unchanged report -> empty edits + null impression",
          no_edit == {"findings_edits": {}, "impression": None}, repr(no_edit))


# --------------------------------------------------------------------------- #
# 1. train.csv loads exactly 636 rows
# --------------------------------------------------------------------------- #
def test_train_row_count() -> None:
    retriever = ExampleRetriever(TRAIN_CSV)
    check("retrieval: train.csv indexes exactly 636 rows",
          retriever.row_count == 636, repr(retriever.row_count))
    check("retrieval: no skipped rows", retriever.skipped_rows == 0,
          repr(retriever.skipped_rows))
    check("retrieval: weights match documented defaults",
          RetrievalWeights().components == {
              "template_similarity": 0.55,
              "modality_bodypart": 0.12,
              "dictation_similarity": 0.28,
              "edited_section_relevance": 0.05,
          })


# --------------------------------------------------------------------------- #
# 2 + 6. Determinism and repeatability
# --------------------------------------------------------------------------- #
def test_determinism() -> None:
    retriever = ExampleRetriever(TRAIN_CSV)
    cases = load_test_cases(TEST_CSV)
    probe = cases.cases[7]

    first = retriever.retrieve(probe, k=3)
    second = retriever.retrieve(probe, k=3)
    check("retrieval: repeated call returns identical ranked examples",
          [(r.example.case_id, r.score) for r in first]
          == [(r.example.case_id, r.score) for r in second],
          repr([r.example.case_id for r in first]))

    fresh = ExampleRetriever(TRAIN_CSV)
    third = fresh.retrieve(probe, k=3)
    check("retrieval: a fresh index ranks identically (reproducible)",
          [(r.example.case_id, r.score) for r in first]
          == [(r.example.case_id, r.score) for r in third],
          repr([(r.example.case_id, r.score) for r in first])
          + " vs " + repr([(r.example.case_id, r.score) for r in third]))


# --------------------------------------------------------------------------- #
# 3 + 4 + 5 + 7. Full-coverage sanity over all 132 test cases
# --------------------------------------------------------------------------- #
def test_full_coverage() -> None:
    retriever = ExampleRetriever(TRAIN_CSV)
    cases = load_test_cases(TEST_CSV)
    train_ids = retriever.train_case_ids()

    train_df = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    by_id = {row["case_id"]: row for row in train_df.to_dict("records")}

    results = retriever.retrieve_all(cases, k=3)

    check("retrieval: all 132 test cases were retrieved", len(results) == 132,
          repr(len(results)))
    check("retrieval: every case returns <= 3 examples",
          all(len(items) <= 3 for items in results.values()),
          repr(sorted({len(items) for items in results.values()})))

    all_from_train = True
    real_data = True
    for case in cases.cases:
        for result in results[case.case_id]:
            eid = result.example.case_id
            if eid not in train_ids:
                all_from_train = False
            row = by_id.get(eid)
            if row is None:
                real_data = False
                continue
            if (row["template_content"] != result.example.template_content
                    or row["dictation"] != result.example.dictation
                    or row["report"] != result.example.report
                    or row["modality"] != result.example.modality
                    or row["body_part"] != result.example.body_part):
                real_data = False
    check("retrieval: every example case_id exists in train.csv", all_from_train)
    check("retrieval: examples carry the real training template/dictation/report",
          real_data)

    # 7. Priority: same template structure first; when none exists, modality/
    #    body part is the fallback signal (the RES-driving hierarchy).
    template_precedence = 0
    template_supported = 0
    mb_fallback = 0
    mb_required = 0
    for case in cases.cases:
        fam = {
            row.case_id for row in retriever.rows
            if normalize(row.template_content) == normalize(case.template_content)
        }
        exact_mb = {
            row.case_id for row in retriever.rows
            if row.modality == case.modality and row.body_part == case.body_part
        }
        got = [r.example.case_id for r in results[case.case_id]]
        if fam:
            template_supported += 1
            template_precedence += int(bool(fam & set(got)))
        elif exact_mb:
            mb_required += 1
            mb_fallback += int(bool(exact_mb & set(got)))
    check(
        "retrieval: exact-template example in top-3 whenever template support exists",
        template_supported > 0 and template_precedence == template_supported,
        f"template precedence {template_precedence}/{template_supported}",
    )
    check(
        "retrieval: exact modality/body_part is the fallback when no template support",
        mb_required > 0 and mb_fallback == mb_required,
        f"mb fallback {mb_fallback}/{mb_required}",
    )
    check("retrieval: template-supported and mb-required exhaustive",
          template_supported + mb_required == sum(
              1 for case in cases.cases
              if any(
                  normalize(row.template_content) == normalize(case.template_content)
                  or (row.modality == case.modality and row.body_part == case.body_part)
                  for row in retriever.rows
              )
          ))
    # Also: dictation similarity should actually discriminate WITHIN the
    # exact-template tier (the fixed dictation index, not the old constant).
    avg_strongest_ds = []
    for case in cases.cases:
        fam = {
            row.case_id for row in retriever.rows
            if normalize(row.template_content) == normalize(case.template_content)
        }
        if not fam:
            continue
        same = [r for r in results[case.case_id] if r.example.case_id in fam]
        if same:
            avg_strongest_ds.append(max(r.signals["dictation_similarity"] for r in same))
    check("retrieval: dictation similarity discriminates within the exact-template tier",
          avg_strongest_ds and
          any(ds != avg_strongest_ds[0] for ds in avg_strongest_ds[1:]),
          f"n={len(avg_strongest_ds)} distinct={len(set(round(d,4) for d in avg_strongest_ds))}")


# --------------------------------------------------------------------------- #
# 8. Similar templates outrank unrelated templates
# --------------------------------------------------------------------------- #
def test_template_similarity_ranks_higher() -> None:
    query = _case(
        template=KNEE_TEMPLATE,
        dictation="no fracture; mild joint space narrowing",
    )

    with tempfile.TemporaryDirectory() as tmp:
        train_csv = Path(tmp) / "train.csv"
        _write_train_csv(
            train_csv,
            [
                # Similar-template row: same XRAY/Knee, same sections/order.
                ["t_1", "XRAY", "Knee", "", "", "", KNEE_SIMILAR_TEMPLATE,
                 "no fracture; mild joint space narrowing", KNEE_SIMILAR_REPORT],
                # Unrelated-template row: same XRAY/Knee, chest sections.
                ["t_2", "XRAY", "Knee", "", "", "", CHEST_TEMPLATE,
                 "no fracture; mild joint space narrowing", CHEST_REPORT_EDITED],
            ],
        )
        retriever = ExampleRetriever(train_csv)
        results = retriever.retrieve(query, k=2)

        check("retrieval: 2 candidates considered", len(results) == 2,
              repr([r.example.case_id for r in results]))
        check("retrieval: identical-template row ranked above unrelated",
              results[0].example.case_id == "t_1",
              repr([r.example.case_id for r in results]))
        check("retrieval: template_similarity signal much higher for t_1",
              results[0].signals["template_similarity"]
              > results[1].signals["template_similarity"],
              repr([r.signals["template_similarity"] for r in results]))
        check("retrieval: t_2 still ranked (backfill, not a failure)",
              results[1].example.case_id == "t_2")


# --------------------------------------------------------------------------- #
# 9 + 10. Prompt injection: dynamic yes, static no
# --------------------------------------------------------------------------- #
def test_prompt_injection() -> None:
    retriever = ExampleRetriever(TRAIN_CSV)
    cases = load_test_cases(TEST_CSV)
    case = cases.cases[0]

    retrieved = retriever.retrieve(case, k=DEFAULT_MAX_EXAMPLES)
    prompt = build_prompt(case, example_retriever=retriever)

    expected_length = 2 * len(retrieved) + 1
    check("prompt: dynamic few-shot expands one user/model pair per example",
          len(prompt.turns) == expected_length and len(retrieved) == 3,
          f"turns={len(prompt.turns)} examples={len(retrieved)}")

    roles = [role for role, _ in prompt.turns]
    check("prompt: roles alternate user/model then the real case",
          roles == ["user", "model"] * len(retrieved) + ["user"], repr(roles))

    for index, result in enumerate(retrieved):
        user_text, model_text = prompt.turns[2 * index][1], prompt.turns[2 * index + 1][1]
        check(f"prompt: example {index + 1} labelled REAL TRAINING EXAMPLE",
              user_text.startswith(f"REAL TRAINING EXAMPLE {index + 1}\n\n"),
              repr(user_text[:60]))
        check(f"prompt: example {index + 1} shows the real derived edits",
              model_text == json.dumps(
                  result.example.derived_output, ensure_ascii=False, indent=2
              ),
              repr(model_text[:80]))
        check(f"prompt: example {index + 1} embeds real template + dictation",
              result.example.template_content in user_text
              and result.example.dictation in user_text)

    last_user = prompt.turns[-1][1]
    check("prompt: final turn carries the current test case",
          case.template_content in last_user and case.dictation in last_user)

    # 10. Static examples are NOT combined with the dynamic ones.
    static = load_few_shot_examples()
    retrieved_templates = {r.example.template_content for r in retrieved}
    embedded_templates = {
        prompt.turns[2 * i][1].split("TEMPLATE_CONTENT:\n", 1)[1]
        .split("\n\nDICTATION:", 1)[0]
        for i in range(len(retrieved))
    }
    check("prompt: exactly the retrieved templates are embedded (no static extras)",
          embedded_templates == retrieved_templates,
          f"{len(embedded_templates)} embedded vs {len(retrieved)} retrieved")
    check("prompt: static file still holds 4 entries but none were force-added",
          len(static) == 4 and len(retrieved) == 3,
          f"static={len(static)} dynamic={len(retrieved)}")


# --------------------------------------------------------------------------- #
# 11. Static few_shot path (existing prompt behavior) still works
# --------------------------------------------------------------------------- #
def test_static_prompt_path() -> None:
    static = load_few_shot_examples()
    probe = _case(template=KNEE_TEMPLATE, dictation="normal")
    prompt = build_prompt(probe, few_shot=static, max_examples=5)
    check("prompt: static few-shot expands to 2N+1 turns",
          len(prompt.turns) == 2 * len(static) + 1, repr(len(prompt.turns)))
    check("prompt: cap honoured for static examples",
          len(build_prompt(probe, few_shot=static, max_examples=2).turns) == 5)
    check("prompt: static examples carry no REAL TRAINING EXAMPLE header",
          all("REAL TRAINING EXAMPLE" not in prompt.turns[2 * i][1]
              for i in range(len(static))))

    # retrieval_prompt_builder falls back to static when the index is absent.
    from src import prompt_builder as pb
    original_default = pb.default_retriever
    pb.default_retriever = lambda: None  # simulate "train.csv unavailable"
    try:
        fallback = retrieval_prompt_builder(_case(
            template=KNEE_TEMPLATE, dictation="normal"
        ))
        check("prompt: retrieval_prompt_builder falls back to static when index is absent",
              len(fallback.turns) == 2 * 3 + 1,  # static capped at max_examples=3
              repr(len(fallback.turns)))
        check("prompt: static fallback carries no REAL TRAINING EXAMPLE header",
              all("REAL TRAINING EXAMPLE" not in fallback.turns[2 * i][1]
                  for i in range(len(static))))
    finally:
        pb.default_retriever = original_default

    check("prompt: retrieval_prompt_builder is stable on real data",
          len(retrieval_prompt_builder(_case(template=KNEE_TEMPLATE,
                                             dictation="normal")).turns) == 7)


# --------------------------------------------------------------------------- #
# Self-exclusion guard + missing-file behaviour
# --------------------------------------------------------------------------- #
def test_safety_guards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train_csv = Path(tmp) / "train.csv"
        _write_train_csv(
            train_csv,
            [
                ["case_self", "XRAY", "Knee", "", "", "", KNEE_TEMPLATE,
                 "normal", KNEE_TEMPLATE],
                ["t_1", "XRAY", "Knee", "", "", "", KNEE_TEMPLATE,
                 "no fracture; mild joint space narrowing", KNEE_SIMILAR_REPORT],
            ],
        )
        retriever = ExampleRetriever(train_csv)
        self_query = _case(case_id="case_self", template=KNEE_TEMPLATE,
                           dictation="no fracture; mild joint space narrowing")
        results = retriever.retrieve(self_query, k=3)
        check("retrieval: a test case cannot retrieve itself from train.csv",
              all(r.example.case_id != "case_self" for r in results),
              repr([r.example.case_id for r in results]))
        check("retrieval: backfill still returns the other candidate",
              len(results) == 1 and results[0].example.case_id == "t_1",
              repr([r.example.case_id for r in results]))

    try:
        ExampleRetriever(Path(tmp) / "nope" / "train.csv")
        check("retrieval: missing train.csv raises", False)
    except FileNotFoundError:
        check("retrieval: missing train.csv raises", True)


def main() -> int:
    test_derive_output_round_trip()
    test_train_row_count()
    test_determinism()
    test_full_coverage()
    test_template_similarity_ranks_higher()
    test_prompt_injection()
    test_static_prompt_path()
    test_safety_guards()

    print(f"\n{_chks_run} checks run.")
    failed = [name for name, ok in _results.items() if not ok]
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print("ALL RETRIEVAL-TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())