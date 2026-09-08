#!/usr/bin/env python3
"""Deterministic data-fidelity tests for prompts/few_shot_examples.json.

Verifies (offline only, no API / no CSV writes):
  1. The file contains exactly 4 examples.
  2. Every example's template + dictation appear verbatim in data/train.csv,
     and the output impression (when non-null) comes from that row's report.
  3. Re-applying the example's output to its template through template_editor
     reproduces the training reference report exactly under the project's
     whitespace normalization (normalize()).
  4. Unchanged template sections are NOT emitted in findings_edits, and every
     emitted value is the complete reference body (no fabricated wording).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.analyze_training_edits import normalize
from src.data_loader import TestCase
from src.prompt_builder import load_few_shot_examples
from src.response_parser import parse_edits
from src.template_editor import apply_edits, parse_template

_chks_run = 0
_results: dict[str, bool] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    global _chks_run
    _chks_run += 1
    _results[name] = bool(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not condition else ""))


def _make_case(row: pd.Series) -> TestCase:
    return TestCase(
        case_id=row["case_id"],
        modality=row["modality"],
        body_part=row["body_part"],
        study_description=row["study_description"],
        patient_age_band=row["patient_age_band"],
        patient_sex=row["patient_sex"],
        template_content=row["template_content"],
        dictation=row["dictation"],
    )


def main() -> int:
    examples = load_few_shot_examples()
    check("file has exactly 4 examples", len(examples) == 4, repr(len(examples)))

    train = pd.read_csv(ROOT / "data" / "train.csv", dtype=str, keep_default_na=False)

    for ex in examples:
        name = ex["name"]
        template = ex["template"]
        dictation = ex["dictation"]
        output = ex["output"]

        # --- (2) template + dictation exist verbatim in train.csv ------------
        row = train[(train["template_content"] == template)
                    & (train["dictation"] == dictation)]
        check(f"'{name}': exact template+dictation row in train.csv",
              len(row) == 1, f"matched {len(row)} rows")
        if len(row) != 1:
            continue
        reference = row.iloc[0]["report"]

        # --- (3) reconstruction == reference under project normalization -----
        case = _make_case(row.iloc[0])
        edits = parse_edits(output)
        reconstructed = apply_edits(template, edits, case=case).report
        check(f"'{name}': reconstructs reference (normalized)",
              normalize(reconstructed) == normalize(reference),
              f"len rec={len(reconstructed)} ref={len(reference)}")

        # --- (4) untouched sections omitted; values are complete ref bodies ---
        template_parsed = parse_template(template)
        reference_parsed = parse_template(reference)
        ref_bodies = {s.label: s.body for s in reference_parsed.sections}
        unchanged = [
            s.label for s in template_parsed.sections
            if s.label in ref_bodies and normalize(s.body) == normalize(ref_bodies[s.label])
        ]
        check(f"'{name}': no edits for unchanged sections",
              not set(output["findings_edits"]) & set(unchanged),
              repr(set(output["findings_edits"]) & set(unchanged)))
        check(f"'{name}': every edit is a complete verbatim reference body",
              all(output["findings_edits"][k] == ref_bodies.get(k) for k in output["findings_edits"]),
              repr([k for k in output["findings_edits"] if output["findings_edits"][k] != ref_bodies.get(k)]))

        # --- impression fidelity: non-null must equal the reference impression
        if output["impression"] is not None:
            ref_imp = reference_parsed.after_impression.strip("\n")
            check(f"'{name}': impression matches reference verbatim",
                  output["impression"] == ref_imp,
                  repr(output["impression"])[:120] + " vs " + repr(ref_imp)[:120])

    print(f"\n{_chks_run} checks run.")
    failed = [n for n, ok in _results.items() if not ok]
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print("ALL FEW-SHOT DATA-FIDELITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())