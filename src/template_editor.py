"""Deterministic bridge between LLM-created edits and the real report text.

The LLM never writes the report; it only proposes *structured edits* (a map of
template section label -> replacement body, plus an optional full IMPRESSION).
Python owns all text surgery here so the round-trip is fully deterministic:

    parse_template(template)  ->  structural view (sections, boundaries)
    apply_edits(template, edits, case)  ->  final report + warnings

Design notes (verified against the real 132 templates)
------------------------------------------------------
- Every template starts with ``FINDINGS:`` and has exactly one ``IMPRESSION:``.
- 128/132 templates are *sectioned*: the FINDINGS block is a sequence of
  ``LABEL: body`` sections (labels matching ``^([A-Za-z][A-Za-z0-9 &/-()]+):``).
- 4/132 (cervical-spine X-ray) are *prose*: no section headers, so the whole
  FINDINGS block is edited as one unit under the reserved key ``FINDINGS``.
- Unedited regions are preserved byte-for-byte: edits only touch their own
  line slice, so untouched sections (and their blank-line separators) are
  carried over verbatim.
- Only the four known placeholders are special: ``[left/right]``,
  ``[left/right/bilateral]``, ``[_laterality_]`` (resolved from the dictation)
  and ``[generic]`` (resolved from the body part).
- As a deterministic safety net the final report is normalized so it never
  contains ``[`` or ``]``: anything that is still bracketed after placeholder
  resolution — including bracketed template prose such as
  ``[ and demonstrate normal post-contrast enhancement]`` or an unresolvable
  laterality placeholder — has its bracket characters removed at apply time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.data_loader import TestCase
from src.errors import InvalidModelOutput, TemplateError
from src.response_parser import Edits

FINDINGS_HEADER = "FINDINGS:"
IMPRESSION_HEADER = "IMPRESSION:"

# Matches a section header line ("LABEL: ..."); the label may itself contain
# spaces/slashes/parens/hyphens. Verified: on the real templates this pattern
# matches every header and zero body/prose lines.
SECTION_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 &/\-()]+):")

LATERALITY_PLACEHOLDERS = ("[left/right]", "[left/right/bilateral]", "[_laterality_]")
GENERIC_PLACEHOLDER = "[generic]"
_LATERALITY_WORD_RE = re.compile(r"\b(left|right|bilateral)\b", re.IGNORECASE)

# Treat the FINDINGS whole-block as the only valid section key on prose templates.
WHOLE_BLOCK_KEY = "FINDINGS"


@dataclass(frozen=True)
class TemplateSection:
    """One ``LABEL: body`` section inside a template's FINDINGS block.

    ``start_line``/``end_line`` are inclusive/exclusive indexes into the
    FINDINGS region of the template, used for surgical replacement that leaves
    every other line untouched.
    """

    label: str
    body: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ParsedTemplate:
    template_content: str
    region_lines: list[str]
    sections: tuple[TemplateSection, ...]
    after_impression: str

    @property
    def is_sectioned(self) -> bool:
        return len(self.sections) > 0

    @property
    def section_labels(self) -> list[str]:
        return [section.label for section in self.sections]


@dataclass(frozen=True)
class ApplyResult:
    report: str
    warnings: list[str]


def parse_template(template_content: str) -> ParsedTemplate:
    """Parse a template into its FINDINGS region, sections, and impression.

    Raises TemplateError when the template does not have the expected
    ``FINDINGS: ... IMPRESSION:`` structure.
    """
    text = template_content
    if not text.startswith(FINDINGS_HEADER):
        raise TemplateError(
            f"template does not start with {FINDINGS_HEADER!r}: {text[:80]!r}"
        )
    if text.count(IMPRESSION_HEADER) != 1:
        raise TemplateError(
            f"template must contain exactly one {IMPRESSION_HEADER!r} "
            f"(found {text.count(IMPRESSION_HEADER)})"
        )
    impression_index = text.index(IMPRESSION_HEADER)
    region = text[len(FINDINGS_HEADER):impression_index]
    region_lines = region.split("\n")
    after_impression = text[impression_index + len(IMPRESSION_HEADER):]

    sections = _extract_sections(region_lines)
    return ParsedTemplate(
        template_content=text,
        region_lines=region_lines,
        sections=sections,
        after_impression=after_impression,
    )


def _extract_sections(region_lines: list[str]) -> tuple[TemplateSection, ...]:
    """Find all header lines and their body line-slices in the FINDINGS region."""
    header_indexes: list[int] = []
    for index, line in enumerate(region_lines):
        if SECTION_HEADER_RE.match(line):
            header_indexes.append(index)

    sections: list[TemplateSection] = []
    for position, start in enumerate(header_indexes):
        next_header = header_indexes[position + 1] if position + 1 < len(header_indexes) else len(region_lines)
        # A section owns its header plus the non-header lines until the next
        # header; trailing blank lines are separators and stay untouched.
        end = next_header
        while end > start and region_lines[end - 1] == "":
            end -= 1
        header_line = region_lines[start]
        label = SECTION_HEADER_RE.match(header_line).group(1)
        # Body-on-the-same-line (e.g. "BONES: Normal.") belongs to the section.
        inline = header_line[len(label) + 1:].lstrip()
        following = region_lines[start + 1:end]
        body_items = [inline] + following if inline.strip() else following
        body = "\n".join(body_items).strip("\n")
        sections.append(
            TemplateSection(label=label, body=body, start_line=start, end_line=end)
        )
    return tuple(sections)


def _section_lines(label: str, body: str) -> list[str]:
    """Render the replacement lines for one section."""
    body = (body or "").strip("\n")
    if not body:
        return [f"{label}:"]
    if "\n" not in body:
        return [f"{label}: {body}"]
    return [f"{label}:"] + body.split("\n")


def strip_bracketed_text(text: str) -> str:
    """Deterministically remove *bracket characters* from generated report text.

    The bracketed clause is preserved (so ``signal intensity[ and demonstrate
    normal post-contrast enhancement].`` becomes ``signal intensity and
    demonstrate normal post-contrast enhancement.``); only ``[``/``]`` are
    removed, including lone/unbalanced brackets.
    """
    return text.replace("[", "").replace("]", "")


def apply_edits(template_content: str, edits: Edits, *, case: TestCase) -> ApplyResult:
    """Build the final report by applying *edits* to *template_content*.

    Returns the reconstructed report plus accumulated warnings (unresolvable
    placeholders). Raises InvalidModelOutput for unknown section labels or an
    empty whole-block edit, which the pipeline treats as a retryable failure.

    The final report is normalized so that no ``[``/``]`` characters survive:
    known placeholders are resolved first, then any remaining bracketed text
    (unresolvable placeholders, bracketed template prose) has its brackets
    deterministically stripped.
    """
    parsed = parse_template(template_content)
    warnings: list[str] = []

    region_lines = list(parsed.region_lines)

    if parsed.is_sectioned:
        allowed = {section.label for section in parsed.sections}
        _reject_unknown_labels(edits, allowed, parsed)
        # Bottom-up so line indexes stay valid while we splice.
        for section in sorted(parsed.sections, key=lambda s: s.start_line, reverse=True):
            if section.label not in edits.findings_edits:
                continue
            replacement = _section_lines(section.label, edits.findings_edits[section.label])
            region_lines[section.start_line:section.end_line] = replacement
    else:
        # Prose template: the whole FINDINGS block is the only editable unit.
        _reject_unknown_labels(edits, {WHOLE_BLOCK_KEY}, parsed)
        body = (edits.findings_edits.get(WHOLE_BLOCK_KEY) or "").strip("\n")
        if not body.strip():
            raise InvalidModelOutput("edits produced an empty FINDINGS block")
        # region is "\n<body>\n\n"; keep the leading "" fence line and every
        # trailing "" line of the separator, replace only the content in between.
        non_empty = [
            index for index, line in enumerate(parsed.region_lines) if line.strip()
        ]
        content_start = non_empty[0] if non_empty else 1
        content_end = non_empty[-1] + 1 if non_empty else len(parsed.region_lines)
        region_lines = (
            parsed.region_lines[:content_start]
            + body.split("\n")
            + parsed.region_lines[content_end:]
        )

    after_impression = parsed.after_impression
    if edits.impression is not None:
        new_impression = edits.impression.strip("\n")
        if not new_impression.strip():
            raise InvalidModelOutput("edits produced an empty IMPRESSION")
        after_impression = "\n" + new_impression

    report = (
        FINDINGS_HEADER
        + "\n".join(region_lines)
        + IMPRESSION_HEADER
        + after_impression
    )

    report, placeholder_warnings = _resolve_placeholders(
        report, case.body_part, case.dictation
    )
    warnings.extend(placeholder_warnings)
    report = strip_bracketed_text(report)
    return ApplyResult(report=report, warnings=warnings)


def _reject_unknown_labels(edits: Edits, allowed: set[str], parsed: ParsedTemplate) -> None:
    unknown = sorted(set(edits.findings_edits) - allowed)
    if unknown:
        kind = "prose template (only 'FINDINGS')" if not parsed.is_sectioned else "sectioned template"
        raise InvalidModelOutput(
            f"edit key(s) not in the {kind}: {unknown}; "
            f"available: {sorted(allowed)}"
        )


def infer_laterality(dictation: str) -> str | None:
    """Return 'left', 'right', or 'bilateral' when the dictation is decisive.

    Returns None when the laterality is absent or mutually ambiguous (equal
    counts of 'left' and 'right').
    """
    counts = {"left": 0, "right": 0, "bilateral": 0}
    for match in _LATERALITY_WORD_RE.finditer(dictation or ""):
        counts[match.group(1).lower()] += 1
    if counts["bilateral"]:
        return "bilateral"
    if counts["left"] != counts["right"]:
        return "left" if counts["left"] > counts["right"] else "right"
    return None


def _generic_term(body_part: str) -> str:
    """Map a body part to the small-cap anatomical term for ``[generic]``.

    body_part ``'Lumbar spine'`` -> ``'lumbar'`` so ``[generic] spine`` becomes
    ``lumbar spine``; ``'Thoracic spine'`` -> ``'thoracic'``.
    """
    term = (body_part or "").strip().lower()
    for suffix in (" spine", " joint", " bone", " region"):
        if term.endswith(suffix) and len(term) > len(suffix):
            return term[: -len(suffix)]
    return term


def _resolve_placeholders(
    report: str, body_part: str, dictation: str
) -> tuple[str, list[str]]:
    """Resolve known placeholders as a safety net behind the model's edits."""
    warnings: list[str] = []

    if any(token in report for token in LATERALITY_PLACEHOLDERS):
        laterality = infer_laterality(dictation)
        if laterality:
            for token in LATERALITY_PLACEHOLDERS:
                report = report.replace(token, laterality)
        else:
            remaining = [t for t in LATERALITY_PLACEHOLDERS if t in report]
            warnings.append(
                f"laterality placeholder(s) {remaining} left unresolved "
                "(dictation does not state left/right/bilateral)"
            )

    if GENERIC_PLACEHOLDER in report:
        term = _generic_term(body_part)
        if term:
            report = report.replace(GENERIC_PLACEHOLDER, term)
        else:
            warnings.append(
                f"{GENERIC_PLACEHOLDER} left unresolved (empty body_part)"
            )

    return report, warnings