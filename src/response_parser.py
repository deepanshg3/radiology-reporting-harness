"""Parse and validate the structured JSON the LLM is asked to produce.

Contract (also enforced by the ``response_json_schema`` sent to the model)::

    {
      "findings_edits": { "<exact template section label>": "<body text>", ... },
      "impression": "<full IMPRESSION text>"   // or null to keep the template's
    }

- ``findings_edits`` is required; keys must be the *exact* template section
  labels and values must be plain strings (the body-only text).
- ``impression``, when present, is the complete replacement IMPRESSION without
  the ``IMPRESSION:`` header; ``null`` means "keep the template impression".
- No other keys are allowed; the model must never emit ``case_id``, row ids,
  metadata, commentary, markdown, or code fences.

Any deviation raises ``InvalidModelOutput`` (a retryable pipeline error).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.errors import InvalidModelOutput

ALLOWED_KEYS = frozenset({"findings_edits", "impression"})

_FENCE_RE = re.compile(r"^\s*(```+|~~~+)[a-zA-Z]*\s*$")


@dataclass(frozen=True)
class Edits:
    """Validated, ready-to-apply edit instructions."""

    findings_edits: dict[str, str]
    impression: str | None = None


def strip_fences(text: str) -> str:
    """Remove markdown code fences a model may (incorrectly) wrap output in."""
    lines = text.splitlines()
    if not lines:
        return text
    if _FENCE_RE.match(lines[0]) and _FENCE_RE.match(lines[-1]):
        return "\n".join(lines[1:-1])
    return text


def parse_edits(payload: object) -> Edits:
    """Validate a decoded JSON payload and return the Edits it describes."""
    if not isinstance(payload, dict):
        raise InvalidModelOutput(
            f"expected a JSON object, got {type(payload).__name__}"
        )

    unknown = sorted(set(payload) - ALLOWED_KEYS)
    if unknown:
        raise InvalidModelOutput(
            f"unexpected top-level key(s): {unknown}; allowed: {sorted(ALLOWED_KEYS)}"
        )

    findings = payload.get("findings_edits")
    if not isinstance(findings, dict):
        raise InvalidModelOutput(
            f"'findings_edits' must be an object of label->body string, "
            f"got {type(findings).__name__ if findings is not None else 'null'}"
        )

    normalized: dict[str, str] = {}
    for label, value in findings.items():
        if not isinstance(label, str) or not label.strip():
            raise InvalidModelOutput("findings_edits contains a blank/empty key")
        if not isinstance(value, str):
            raise InvalidModelOutput(
                f"value for label {label!r} must be a string, "
                f"got {type(value).__name__}"
            )
        normalized[label.strip()] = value.strip()

    impression = payload.get("impression")
    if impression is not None and not isinstance(impression, str):
        raise InvalidModelOutput(
            f"'impression' must be a string or null, got {type(impression).__name__}"
        )

    return Edits(
        findings_edits=normalized,
        impression=impression.strip() if impression is not None else None,
    )


def parse_model_text(text: str) -> Edits:
    """Parse a raw model text response (used when ``.parsed`` is unavailable).

    Tolerates markdown code fences around the JSON, matching the defensive
    contract in :func:`strip_fences`.
    """
    cleaned = strip_fences(text or "").strip()
    if not cleaned:
        raise InvalidModelOutput("model returned an empty response")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidModelOutput(
            f"model output is not valid JSON: {exc}"
        ) from exc
    return parse_edits(payload)