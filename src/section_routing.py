"""Deterministic section-routing HINTS derived from training evidence.

At inference time this module turns a dictation + template into a small list of
*CANDIDATE* template sections whose edit is suggested by training data. It is a
pure hint layer:

  - It NEVER filters: Gemini may still edit any section the dictation requires.
  - It NEVER reads train.csv at inference (it loads a precomputed artifact from
    ``prompts/routing_model.json`` produced by ``src.routing_trainer``).
  - It NEVER memorizes a reference report or a case-specific answer.

Without a routing-model artifact the module degrades gracefully to an empty
candidate list, so the pipeline is unchanged for any run without routing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import PROMPTS_DIR
from src.data_loader import TestCase
from src.template_editor import parse_template

ROUTING_MODEL_PATH = PROMPTS_DIR / "routing_model.json"

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}\b")
_STOPWORDS = frozenset(
    {
        "the", "and", "are", "with", "was", "for", "from", "this", "there",
        "that", "have", "has", "not", "no", "any", "is", "of", "in", "on",
        "at", "to", "a", "an", "be", "or", "but", "right", "left", "mild",
        "moderate", "severe", "small", "large", "shows", "show", "noted",
        "note", "consistent", "seen", "findings", "within", "without", "due",
        "related", "also", "both", "report", "imaging", "study", "showing",
    }
)
# Keep candidate-section weighting modest so hints stay weak suggestions.
MAX_TERM_MATCHES = 4
MAX_CANDIDATES = 8


class RoutingModel:
    """Loaded routing artifact. Safe to construct even if no model exists."""

    def __init__(self, path: Path | str = ROUTING_MODEL_PATH):
        self.path = Path(path)
        self._data = self._load(self.path)
        self.by_mb: dict[str, dict[str, float]] = self._data.get(
            "by_modality_bodypart", {}
        )
        self.by_term: dict[str, dict[str, float]] = self._data.get("by_term", {})

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @property
    def available(self) -> bool:
        return bool(self.by_mb or self.by_term)

    def candidate_sections(self, test_case: TestCase) -> list[str]:
        """Return an ordered, de-duplicated list of candidate section labels.

        Candidates come from (1) the modality/body-part family and (2) dictation
        terms, intersected with the sections actually present in this template.
        """
        if not self.available:
            return []

        parsed = parse_template(test_case.template_content)
        available = set(parsed.section_labels)

        weighted: dict[str, float] = {}

        mb_key = f"{test_case.modality} / {test_case.body_part}"
        for label, weight in self.by_mb.get(mb_key, {}).items():
            if label in available:
                weighted[label] = max(weighted.get(label, 0.0), weight)

        # Term-based signal: gather from a bounded set of dictation terms.
        terms = _TOKEN_RE.findall((test_case.dictation or "").lower())
        covered = 0
        for term in terms:
            if term in _STOPWORDS or len(term) < 3:
                continue
            for label, weight in self.by_term.get(term, {}).items():
                if label in available:
                    weighted[label] = max(weighted.get(label, 0.0), weight)
            covered += 1
            if covered >= MAX_TERM_MATCHES:
                break

        ordered = sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))
        return [label for label, _weight in ordered][:MAX_CANDIDATES]
