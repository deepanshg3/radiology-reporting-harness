#!/usr/bin/env python3
"""Deterministic tests for src/analyze_training_edits.py.

These tests exercise the pure text-processing helpers and the per-row analysis
without any CSV, network, or LLM. They verify:

  1. normalize() collapses whitespace / case.
  2. _char_edit_ratio: identical -> 0.0, disjoint -> 1.0, edit -> in between.
  3. _diff_additions_deletions counts added/deleted characters correctly.
  4. _substring_presence respects word-for-word matching (whitespace tolerant).
  5. _classify_sections: edited vs unchanged vs added, including renamed/moved
     labels detected by verbatim-body fallback.
  6. analyze_row: whole FINDINGS/IMPRESSION flags, section bookkeeping, char
     edit ratio, placeholder resolution.
  7. aggregate(): sanity arithmetic over a small batch (counts, medians,
     percentage, top-labels, zero-addition preservation).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyze_training_edits import (
    _char_edit_ratio,
    _classify_sections,
    _diff_additions_deletions,
    _substring_presence,
    aggregate,
    analyze_row,
    normalize,
)
from src.template_editor import parse_template

_chks_run = 0
_results: dict[str, bool] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    global _chks_run
    _chks_run += 1
    _results[name] = bool(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not condition else ""))


# --------------------------------------------------------------------------- #
# 1. normalize
# --------------------------------------------------------------------------- #
def test_normalize() -> None:
    check("normalize: lowercases + collapses whitespace",
          normalize("  BONES:  No\n fracture.  ") == "bones: no fracture.")
    check("normalize: empty -> empty", normalize("   ") == "")
    check("normalize: preserves punctuation/words",
          normalize("Mild JOINT-space narrowing") == "mild joint-space narrowing")


# --------------------------------------------------------------------------- #
# 2. _char_edit_ratio
# --------------------------------------------------------------------------- #
def test_char_edit_ratio() -> None:
    check("edit ratio: identical text -> 0.0",
          abs(_char_edit_ratio("FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nFine.",
                               "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nFine.")) < 1e-9)
    check("edit ratio: whitespace-only diff -> 0.0",
          abs(_char_edit_ratio("a b c", "a   b\nc")) < 1e-9)
    check("edit ratio: disjoint texts -> 1.0",
          abs(_char_edit_ratio("aaaa", "bbbb") - 1.0) < 1e-9)
    check("edit ratio: partial edit between extremes",
          0.0 < _char_edit_ratio("the quick brown fox", "the quick red fox") < 1.0)


# --------------------------------------------------------------------------- #
# 3. _diff_additions_deletions
# --------------------------------------------------------------------------- #
def test_diff_additions_deletions() -> None:
    added, deleted = _diff_additions_deletions("abc", "abcdef")
    check("diff: insertion adds, deletes nothing", added == 3 and deleted == 0,
          f"added={added} deleted={deleted}")
    added, deleted = _diff_additions_deletions("abcdef", "abc")
    check("diff: deletion deletes, adds nothing", added == 0 and deleted == 3,
          f"added={added} deleted={deleted}")
    added, deleted = _diff_additions_deletions("abc", "xyz")
    check("diff: full replace counts both sides", added == 3 and deleted == 3,
          f"added={added} deleted={deleted}")
    added, deleted = _diff_additions_deletions("same", "same")
    check("diff: identical -> zero", added == 0 and deleted == 0,
          f"added={added} deleted={deleted}")


# --------------------------------------------------------------------------- #
# 4. _substring_presence
# --------------------------------------------------------------------------- #
def test_substring_presence() -> None:
    check("substring: verbatim present (whitespace tolerant)", _substring_presence(
        "No\n fracture. ", "There is no fracture."))
    check("substring: not present when wording differs", not _substring_presence(
        "No fracture.", "There was a fracture."))
    check("substring: empty needle -> always present", _substring_presence("", "anything"))
    check("substring: case insensitive", _substring_presence("No Fracture", "no fracture."))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
TEMPLATE_A = (
    "FINDINGS:\n"
    "BONES: No fracture.\n"
    "JOINTS: Normal joint spaces.\n"
    "SOFT TISSUES: Unremarkable.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute osseous abnormality."
)

# Report: BONES edited, JOINTS & SOFT TISSUES unchanged, IMPRESSION edited.
REPORT_A = (
    "FINDINGS:\n"
    "BONES: Mild degenerative changes are present. No fracture.\n"
    "JOINTS: Normal joint spaces.\n"
    "SOFT TISSUES: Unremarkable.\n"
    "\n"
    "IMPRESSION:\n"
    "1. Mild degenerative changes.\n2. No acute osseous abnormality."
)

# Report where BONES was RENAMED to "OSSEOUS" but body verbatim.
REPORT_RENAMED = (
    "FINDINGS:\n"
    "OSSEOUS: No fracture.\n"
    "JOINTS: Normal joint spaces.\n"
    "SOFT TISSUES: Unremarkable.\n"
    "\n"
    "IMPRESSION:\n"
    "No acute osseous abnormality."
)

# Report identical to template except whitespace.
REPORT_WS_ONLY = (
    "FINDINGS:\nBONES: No fracture.\nJOINTS: Normal joint spaces.\n"
    "SOFT TISSUES: Unremarkable.\n\nIMPRESSION:\nNo acute osseous abnormality."
)

TEMPLATE_PLACEHOLDER = (
    "FINDINGS:\n"
    "BONES: Normal.\n"
    "\n"
    "IMPRESSION:\n"
    "Normal [left/right] ankle."
)
REPORT_PLACEHOLDER_RESOLVED = (
    "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal left ankle."
)


# --------------------------------------------------------------------------- #
# 5. _classify_sections
# --------------------------------------------------------------------------- #
def test_classify_sections() -> None:
    t = parse_template(TEMPLATE_A)

    edited, unchanged, added = _classify_sections(t, parse_template(REPORT_A))
    check("classify: edited = BONES only", edited == ["BONES"], repr(edited))
    check("classify: unchanged = JOINTS, SOFT TISSUES",
          sorted(unchanged) == ["JOINTS", "SOFT TISSUES"], repr(unchanged))
    check("classify: no added sections on REPORT_A", added == [], repr(added))

    edited, unchanged, added = _classify_sections(t, parse_template(REPORT_RENAMED))
    check("classify: renamed label with verbatim body -> unchanged",
          edited == [] and "BONES" in unchanged, f"edited={edited} unchanged={unchanged}")
    check("classify: renamed label counted as added section",
          "OSSEOUS" in added, repr(added))

    edited, unchanged, _ = _classify_sections(t, parse_template(REPORT_WS_ONLY))
    check("classify: whitespace-only report -> all unchanged",
          edited == [] and len(unchanged) == 3, f"edited={edited} unchanged={unchanged}")


# --------------------------------------------------------------------------- #
# 6. analyze_row
# --------------------------------------------------------------------------- #
def _row(template: str, report: str, **kw: object) -> dict:
    base: dict[str, object] = {
        "case_id": "case_x", "modality": "XRAY", "body_part": "Knee",
        "template_content": template, "report": report,
    }
    base.update(kw)
    return base


def test_analyze_row() -> None:
    a = analyze_row(_row(TEMPLATE_A, REPORT_A))
    check("analyze: findings changed", a.findings_changed)
    check("analyze: impression changed", a.impression_changed)
    check("analyze: edited sections = BONES", a.edited_sections == ["BONES"], repr(a.edited_sections))
    check("analyze: unchanged sections = 2", len(a.unchanged_sections) == 2, repr(a.unchanged_sections))
    check("analyze: template section count = 3", a.template_section_count == 3)
    check("analyze: char edit ratio in (0,1)", 0.0 < a.char_edit_ratio < 1.0, repr(a.char_edit_ratio))
    check("analyze: has additions", a.added_chars > 0, repr(a.added_chars))
    check("analyze: not preserved exactly", not a.preserved_exactly)
    check("analyze: no placeholders in this template", a.template_placeholders == [])

    ws = analyze_row(_row(TEMPLATE_A, REPORT_WS_ONLY))
    check("analyze: whitespace-only report flagged as NO findings / impression change",
          not ws.findings_changed and not ws.impression_changed)
    check("analyze: whitespace-only report has zero section edits",
          ws.edited_sections == [], repr(ws.edited_sections))
    check("analyze: whitespace-only report preserved exactly",
          ws.preserved_exactly, f"added={ws.added_chars} deleted={ws.deleted_chars}")
    check("analyze: whitespace-only char ratio == 0",
          abs(ws.char_edit_ratio) < 1e-9, repr(ws.char_edit_ratio))

    # Report where the JOINTS body is shortened -> deletions, no additions there.
    report_short = REPORT_A.replace("JOINTS: Normal joint spaces.",
                                    "JOINTS: Narrow joint space.")
    short = analyze_row(_row(TEMPLATE_A, report_short))
    check("analyze: shortening produces deletions", short.deleted_chars > 0,
          repr(short.deleted_chars))

    ph = analyze_row(_row(TEMPLATE_PLACEHOLDER, REPORT_PLACEHOLDER_RESOLVED))
    check("analyze: placeholder detected in template",
          ph.template_placeholders == ["[left/right]"], repr(ph.template_placeholders))
    check("analyze: placeholder resolved in report (none unresolved)",
          ph.unresolved_placeholders == [] and ph.resolved_placeholder_count == 1,
          f"{ph.unresolved_placeholders} / {ph.resolved_placeholder_count}")


# --------------------------------------------------------------------------- #
# 7. aggregate
# --------------------------------------------------------------------------- #
def test_aggregate() -> None:
    # 2 reports: A (edited) and a whitespace-only (not edited).
    rows = [
        analyze_row(_row(TEMPLATE_A, REPORT_A, case_id="c1")),
        analyze_row(_row(TEMPLATE_A, REPORT_WS_ONLY, case_id="c2")),
    ]
    stats = aggregate(rows)
    check("aggregate: total cases", stats["total_training_cases"] == 2)
    check("aggregate: findings edited count = 1", stats["cases_with_findings_edits"] == 1)
    check("aggregate: impression edited count = 1", stats["cases_with_impression_edits"] == 1)
    check("aggregate: median edited sections = 0.5", stats["median_edited_sections"] == 0.5,
          repr(stats["median_edited_sections"]))
    check("aggregate: top modified label BONES once",
          list(stats["top_20_modified_section_labels"][0]) == ["BONES", 1],
          repr(stats["top_20_modified_section_labels"]))
    check("aggregate: only_impression_changed 0", stats["only_impression_changed"] == 0)
    check("aggregate: both_changed 1", stats["both_findings_and_impression_changed"] == 1)
    check("aggregate: percent mostly unchanged = 100.0 (both rows keep >= half sections)",
          stats["percentage_reports_mostly_unchanged"] == 100.0,
          repr(stats["percentage_reports_mostly_unchanged"]))
    # avg edited = (1 + 0)/2 = 0.5
    check("aggregate: average edited sections = 0.5",
          abs(stats["average_edited_sections"] - 0.5) < 1e-9,
          repr(stats["average_edited_sections"]))


def main() -> int:
    test_normalize()
    test_char_edit_ratio()
    test_diff_additions_deletions()
    test_substring_presence()
    test_classify_sections()
    test_analyze_row()
    test_aggregate()

    print(f"\n{_chks_run} checks run.")
    failed = [name for name, ok in _results.items() if not ok]
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        return 1
    print("ALL TRAINING-EDIT ANALYSIS TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
