"""Offline, deterministic analysis of how human reference reports edit their templates.

Reads ``data/train.csv`` (636 cases) and, for every row, compares the
``template_content`` against the ground-truth ``report`` with *pure Python text
processing* — no LLM, no Gemini, no network.

It reuses the existing template parser (:func:`src.template_editor.parse_template`)
for BOTH the template and the report. Because every train report starts with
``FINDINGS:`` and contains exactly one ``IMPRESSION:`` (verified empirically),
the same parser yields a structural view (labelled sections + IMPRESSION text)
for each column, so section-level edits can be measured without inventing a
second parser.

Nothing here writes to train.csv / test.csv / submission.csv and the production
inference pipeline is never touched. Outputs are written to
``outputs/training_edit_analysis.txt`` and ``outputs/training_edit_analysis.json``.

Run::

    python -m src.analyze_training_edits            # default: train.csv
    python src/analyze_training_edits.py --csv path --out-dir path
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Allow direct execution (`python src/analyze_training_edits.py`) by putting the
# project root on sys.path, mirroring how tests bootstrap the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import OUTPUTS_DIR, TRAIN_CSV
from src.errors import TemplateError
from src.template_editor import (
    FINDINGS_HEADER,
    IMPRESSION_HEADER,
    LATERALITY_PLACEHOLDERS,
    GENERIC_PLACEHOLDER,
    parse_template,
)

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace so comparison ignores whitespace.

    This keeps the comparison deterministic and tolerant of the small
    whitespace/formatting differences humans introduce while still treating any
    *word* change as a real edit.
    """
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _char_edit_ratio(a: str, b: str) -> float:
    """Character-level edit ratio in [0,1] between two normalized texts.

    0.0 = identical (ignoring whitespace), 1.0 = nothing in common. Computed
    from :func:`difflib.SequenceMatcher` matching blocks, purely deterministic.
    """
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 0.0
    sm = difflib.SequenceMatcher(None, na, nb, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    total = len(na) + len(nb)
    return 1.0 - (2.0 * matched / total) if total else 0.0


def _diff_additions_deletions(a: str, b: str) -> tuple[int, int]:
    """Return (added_chars, deleted_chars) between normalized A and B.

    Uses difflib opcodes; ``added`` counts characters present in B but not A,
    ``deleted`` counts characters present in A but not B.
    """
    na, nb = normalize(a), normalize(b)
    added = deleted = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, na, nb, autojunk=False
    ).get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "replace":
            added += j2 - j1
            deleted += i2 - i1
    return added, deleted


def _substring_presence(needle: str, haystack: str) -> bool:
    """True when a normalized needle appears word-for-word within a normalized haystack."""
    nn, nh = normalize(needle), normalize(haystack)
    if not nn:
        return True
    return nn in nh


# --------------------------------------------------------------------------- #
# Per-row dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RowAnalysis:
    """All deterministic measurements computed for one training row."""

    case_id: str
    modality: str
    body_part: str

    # Whole-report
    findings_changed: bool = False
    impression_changed: bool = False
    char_edit_ratio: float = 0.0
    added_chars: int = 0
    deleted_chars: int = 0

    # Section bookkeeping
    edited_sections: list[str] = field(default_factory=list)
    unchanged_sections: list[str] = field(default_factory=list)
    section_edits_by_modality: list[str] = field(default_factory=list)
    sectioned: bool = True
    template_section_count: int = 0
    preserved_exactly: bool = False

    # Placeholders
    template_placeholders: list[str] = field(default_factory=list)
    unresolved_placeholders: list[str] = field(default_factory=list)
    resolved_placeholder_count: int = 0


def _classify_sections(template, report) -> tuple[list[str], list[str], list[str]]:
    """Classify each template section as edited / unchanged / added.

    Returns (edited_labels, unchanged_labels, added_labels).

    Rules (deterministic):
      * A template section is **unchanged** when its exact body (whitespace
        normalized) appears verbatim somewhere in the report's FINDINGS region —
        even if the human moved/renamed the label or reordered sections.
      * Otherwise it is **edited** (its wording changed or it was removed).
      * A section label that appears in the report but not in the template is an
        **added** section.
    """
    template_by_label = {s.label: s.body for s in template.sections}
    report_region = _findings_region_text(report)

    report_by_label = {s.label: s.body for s in report.sections}

    edited: list[str] = []
    unchanged: list[str] = []
    for label, body in template_by_label.items():
        if label in report_by_label:
            # Same label retained: compare bodies directly (whitespace-tolerant).
            # This treats any wording change — including expansion that still
            # contains the original sentence — as an edit.
            is_unchanged = normalize(body) == normalize(report_by_label[label])
        else:
            # Label renamed/moved (or the section merged into prose): fall back
            # to verbatim body presence. A body preserved word-for-word was
            # relocated, not rewritten, so it counts as unchanged.
            is_unchanged = _substring_presence(body, report_region)
        (unchanged if is_unchanged else edited).append(label)

    added = [s.label for s in report.sections if s.label not in template_by_label]
    return edited, unchanged, added


def _findings_region_text(parsed) -> str:
    """Reconstruct the raw FINDINGS region text from a parsed template."""
    return "\n".join(parsed.region_lines)


def _analyze_placeholders(template_content: str, report_content: str) -> tuple[list[str], list[str], int]:
    """Inspect placeholder usage: which placeholders the template had and how the
    report handled them (still present => unresolved; otherwise resolved).
    """
    template_placeholders: list[str] = []
    for token in LATERALITY_PLACEHOLDERS + (GENERIC_PLACEHOLDER,):
        if token in template_content:
            template_placeholders.append(token)

    unresolved = [t for t in template_placeholders if t in report_content]
    resolved_count = len(template_placeholders) - len(unresolved)
    return template_placeholders, unresolved, resolved_count


def analyze_row(row: dict) -> RowAnalysis:
    """Compute deterministic edit statistics for one training row."""
    template_content = row["template_content"]
    report_content = row["report"]
    case_id = row["case_id"]

    try:
        template = parse_template(template_content)
        report = parse_template(report_content)
    except TemplateError:
        # Should not happen on real data (all 636 parse), but stay defensive.
        return RowAnalysis(case_id=case_id, modality=str(row.get("modality", "")),
                           body_part=str(row.get("body_part", "")))

    analysis = RowAnalysis(
        case_id=case_id,
        modality=str(row.get("modality", "")),
        body_part=str(row.get("body_part", "")),
        sectioned=template.is_sectioned,
        template_section_count=len(template.sections),
    )

    # --- Whole-report FINDINGS / IMPRESSION changes -------------------------
    template_region = _findings_region_text(template)
    report_region = _findings_region_text(report)
    analysis.findings_changed = normalize(template_region) != normalize(report_region)

    template_imp = template.after_impression
    report_imp = report.after_impression
    analysis.impression_changed = normalize(template_imp) != normalize(report_imp)

    # --- Character-level edit ratio + additions/deletions -------------------
    analysis.char_edit_ratio = _char_edit_ratio(template_content, report_content)
    analysis.added_chars, analysis.deleted_chars = _diff_additions_deletions(
        template_content, report_content
    )

    # --- Section-level edits ------------------------------------------------
    edited, unchanged, added = _classify_sections(template, report)
    analysis.edited_sections = edited
    analysis.unchanged_sections = unchanged
    analysis.section_edits_by_modality = ["edited" for _ in edited] + [
        "added" for _ in added
    ]

    # A report "preserved template text exactly" when nothing was added/deleted
    # at the character level (whitespace-tolerant) and no section was edited.
    analysis.preserved_exactly = (
        analysis.added_chars == 0
        and analysis.deleted_chars == 0
        and not analysis.edited_sections
    )

    # --- Placeholder patterns ----------------------------------------------
    (analysis.template_placeholders,
     analysis.unresolved_placeholders,
     analysis.resolved_placeholder_count) = _analyze_placeholders(
        template_content, report_content
    )

    return analysis


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _median(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def aggregate(rows: list[RowAnalysis]) -> dict:
    """Turn per-row analyses into human/machine readable aggregate statistics."""
    total = len(rows)

    findings_edited = sum(1 for r in rows if r.findings_changed)
    findings_unchanged = total - findings_edited
    impression_edited = sum(1 for r in rows if r.impression_changed)
    impression_unchanged = total - impression_edited

    edit_counts = [len(r.edited_sections) for r in rows]
    edit_counts_dist = Counter(edit_counts)

    # Section label modification frequency (how often each template section is edited)
    section_edit_freq = Counter()
    for r in rows:
        for label in r.edited_sections:
            section_edit_freq[label] += 1

    # Localized vs spread: single edited section vs many
    localized = sum(1 for r in rows if len(r.edited_sections) == 1)
    spread = sum(1 for r in rows if len(r.edited_sections) >= 2)
    no_section_edits = sum(1 for r in rows if len(r.edited_sections) == 0)

    # Theme combinations over FINDINGS/IMPRESSION
    only_impression = sum(
        1 for r in rows if r.impression_changed and not r.findings_changed
    )
    only_findings = sum(
        1 for r in rows if r.findings_changed and not r.impression_changed
    )
    both_changed = sum(1 for r in rows if r.findings_changed and r.impression_changed)

    # Percentage of reports where most of the template remained unchanged.
    # "Most" => at least half of the template sections are unchanged AND the
    # char edit ratio is <= 0.5 (>= half the characters preserved).
    mostly_unchanged = sum(
        1
        for r in rows
        if r.template_section_count
        and len(r.unchanged_sections) >= (r.template_section_count / 2)
    )
    low_edit_ratio = sum(1 for r in rows if r.char_edit_ratio <= 0.5)

    char_ratios = [r.char_edit_ratio for r in rows]
    added_chars = [r.added_chars for r in rows]
    deleted_chars = [r.deleted_chars for r in rows]

    # Placeholder stats
    rows_with_placeholder = sum(1 for r in rows if r.template_placeholders)
    assert len(rows) == total
    placeholder_types = Counter()
    for r in rows:
        for p in dict.fromkeys(r.template_placeholders):
            placeholder_types[p] += 1
    unresolved_cases = sum(
        1 for r in rows if r.unresolved_placeholders
    )

    # Body-part / modality breakdown of edit behavior
    by_bodypart_findings_edit = defaultdict(lambda: [0, 0])
    for r in rows:
        k = f"{r.modality} / {r.body_part}"
        by_bodypart_findings_edit[k][int(r.findings_changed)] += 1

    return {
        "total_training_cases": total,
        "cases_with_findings_edits": findings_edited,
        "cases_with_no_findings_edits": findings_unchanged,
        "cases_with_impression_edits": impression_edited,
        "cases_with_no_impression_edits": impression_unchanged,
        "average_edited_sections": round(statistics.fmean(edit_counts), 3),
        "median_edited_sections": _median(edit_counts),
        "edited_section_counts_distribution": {
            str(k): v for k, v in sorted(edit_counts_dist.items())
        },
        "cases_with_zero_section_edits": no_section_edits,
        "cases_with_one_edited_section_localized": localized,
        "cases_with_multi_section_edits": spread,
        "only_impression_changed": only_impression,
        "only_findings_changed": only_findings,
        "both_findings_and_impression_changed": both_changed,
        "top_20_modified_section_labels": section_edit_freq.most_common(20),
        "percentage_reports_mostly_unchanged": round(
            (mostly_unchanged / total) * 100, 2
        ),
        "percentage_reports_low_char_edit_ratio_le_0_5": round(
            (low_edit_ratio / total) * 100, 2
        ),
        "average_char_edit_ratio": round(statistics.fmean(char_ratios), 4),
        "median_char_edit_ratio": _median(char_ratios),
        "average_added_chars": round(statistics.fmean(added_chars), 2),
        "average_deleted_chars": round(statistics.fmean(deleted_chars), 2),
        "placeholder": {
            "cases_with_placeholder_in_template": rows_with_placeholder,
            "placeholder_type_counts": dict(placeholder_types),
            "cases_with_unresolved_placeholder_in_report": unresolved_cases,
        },
        "by_modality_bodypart": {
            k: {"findings_edited": v[1], "total": sum(v)}
            for k, v in sorted(by_bodypart_findings_edit.items())
        },
    }


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #


def _write_text_report(stats: dict, path: Path) -> None:
    lines: list[str] = []
    add = lines.append
    add("=" * 74)
    add("TRAINING EDIT ANALYSIS — template vs. ground-truth report (train.csv)")
    add("Deterministic text-only analysis. No LLM involved.")
    add("=" * 74)
    add("")
    add(f"Total training cases:                      {stats['total_training_cases']}")
    add("")
    add("--- FINDINGS / IMPRESSION ---")
    add(f"  Cases with FINDINGS edits:               {stats['cases_with_findings_edits']}")
    add(f"  Cases with NO FINDINGS edits:            {stats['cases_with_no_findings_edits']}")
    add(f"  Cases with IMPRESSION edits:             {stats['cases_with_impression_edits']}")
    add(f"  Cases with NO IMPRESSION edits:          {stats['cases_with_no_impression_edits']}")
    add("")
    add("--- Section-edit magnitude ---")
    add(f"  Average edited sections per case:        {stats['average_edited_sections']}")
    add(f"  Median edited sections per case:         {stats['median_edited_sections']}")
    add("  Distribution of edited-section counts:")
    for k, v in stats["edited_section_counts_distribution"].items():
        add(f"      {k:>3} section(s): {v}")
    add(f"  Cases with zero section edits:           {stats['cases_with_zero_section_edits']}")
    add(f"  Cases with exactly 1 edited section:     {stats['cases_with_one_edited_section_localized']}")
    add(f"  Cases with >= 2 edited sections:         {stats['cases_with_multi_section_edits']}")
    add("")
    add("--- FINDINGS vs IMPRESSION theme ---")
    add(f"  Only IMPRESSION changed:                 {stats['only_impression_changed']}")
    add(f"  Only FINDINGS changed:                   {stats['only_findings_changed']}")
    add(f"  Both FINDINGS and IMPRESSION changed:    {stats['both_findings_and_impression_changed']}")
    add("")
    add("--- Template preservation ---")
    add(f"  % reports where >= half of sections unchanged: "
        f"{stats['percentage_reports_mostly_unchanged']}%")
    add(f"  % reports with char-edit-ratio <= 0.5:    "
        f"{stats['percentage_reports_low_char_edit_ratio_le_0_5']}%")
    add(f"  Average char edit ratio:                 {stats['average_char_edit_ratio']}")
    add(f"  Median char edit ratio:                  {stats['median_char_edit_ratio']}")
    add(f"  Average chars added (per case):          {stats['average_added_chars']}")
    add(f"  Average chars deleted (per case):        {stats['average_deleted_chars']}")
    add("")
    add("--- Top 20 most frequently modified section labels ---")
    add("  (label : times edited across all 636 cases)")
    for label, count in stats["top_20_modified_section_labels"]:
        add(f"      {label:<38} {count}")
    add("")
    add("--- Placeholder resolution ---")
    ph = stats["placeholder"]
    add(f"  Cases with placeholder in template:        {ph['cases_with_placeholder_in_template']}")
    for token, count in ph["placeholder_type_counts"].items():
        add(f"      {token:<28} in {count} templates")
    add(f"  Cases with placeholder unresolved in report: {ph['cases_with_unresolved_placeholder_in_report']}")
    add("")
    add("--- Edit propensity by modality / body part (FINDINGS edited) ---")
    for key, val in stats["by_modality_bodypart"].items():
        add(f"      {key:<28} {val['findings_edited']:>3}/{val['total']} edited")
    add("")
    add("=" * 74)
    add("Report written from train.csv only. Nothing modified in the pipeline.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_train(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"case_id", "template_content", "report"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {sorted(missing)}")
    return list(df.to_dict("records"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze how human reference reports edit their templates."
    )
    parser.add_argument("--csv", default=str(TRAIN_CSV), help="train.csv path")
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR), help="output directory")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    rows = _load_train(csv_path)
    analyses = [analyze_row(row) for row in rows]
    stats = aggregate(analyses)

    txt_path = out_dir / "training_edit_analysis.txt"
    json_path = out_dir / "training_edit_analysis.json"

    _write_text_report(stats, txt_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Analyzed {len(analyses)} training cases.")
    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")

    # Also print a concise summary to stdout.
    print("\n=== SUMMARY ===")
    print(f"FINDINGS edited: {stats['cases_with_findings_edits']}/{stats['total_training_cases']}")
    print(f"IMPRESSION edited: {stats['cases_with_impression_edits']}/{stats['total_training_cases']}")
    print(f"Avg edited sections: {stats['average_edited_sections']}  "
          f"(median {stats['median_edited_sections']})")
    print(f"Only IMPRESSION: {stats['only_impression_changed']} | "
          f"Only FINDINGS: {stats['only_findings_changed']} | "
          f"Both: {stats['both_findings_and_impression_changed']}")
    print("Top modified sections:")
    for label, count in stats["top_20_modified_section_labels"][:10]:
        print(f"   {label:<38} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
