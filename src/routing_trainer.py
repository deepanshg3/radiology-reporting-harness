"""Offline trainer for the section-routing model (train.csv ONLY, deterministic).

Train.csv is never read at inference time. This script consumes the 636
training rows and derives a *generalizing* routing model: a map from dictation
signals (modality/body_part and recurring lowercase terms) to the template
section labels that human reference reports most often edited given those
signals.

The output artifact (``routing_model.json``) is a plain JSON map of:

    {
      "by_modality_bodypart": { "XRAY / Knee": {"BONES": <weight>, ...}, ... },
      "by_term": {"fracture": {"BONES": <weight>, ...}, "effusion": {...}, ...}
    }

``weight`` is the empirical conditional probability that the section was edited
for rows matching the signal, gated by a minimum support (row count) to avoid
noisy one-off associations.

This is a HINT source, never an authoritative filter. The prompt presents these
as *candidate* sections; Gemini (and the deterministic editor/validator) remain
the final decision makers. Nothing here memorizes a reference report.

Run::

    python -m src.routing_trainer --train-csv data/train.csv \
          --out prompts/routing_model.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.config import PROMPTS_DIR, TRAIN_CSV
from src.analyze_training_edits import analyze_row

# Tokens shorter than this are noise ("a", "of", "the"...) and skipped when
# learning term->section associations.
MIN_TERM_LEN = 3
# A term must appear in at least this many training rows before its association
# is believed (guards against one-off spellings/reports).
MIN_TERM_SUPPORT = 6
# A (signal, section) association is kept only if this many rows showed it.
MIN_ASSOC_SUPPORT = 3
# Conditional probability above which a section is considered "suggested" by a
# term. Use a floor so moderately-common-but-not-dominant associations remain.
ASSOC_THRESHOLD = 0.25

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


def _termify(dictation: str) -> list[str]:
    tokens = _TOKEN_RE.findall((dictation or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= MIN_TERM_LEN]


def _section_weights(rows: list[dict]) -> dict[str, int]:
    """Count how often each section label was edited across the given rows."""
    counts: Counter[str] = Counter()
    for row in rows:
        for label in row["edited_sections"]:
            counts[label] += 1
    return dict(counts)


def _likelihood(weights: dict[str, int], total: int) -> dict[str, float]:
    """Convert raw counts into conditional probabilities p(edited | signal)."""
    return {label: round(count / total, 4) for label, count in weights.items()}


def _filter(signal_weights: dict[str, int], total: int) -> dict[str, float]:
    """Keep associations with enough support and a meaningful probability."""
    out: dict[str, float] = {}
    for label, count in signal_weights.items():
        if count >= MIN_ASSOC_SUPPORT and (count / total) >= ASSOC_THRESHOLD:
            out[label] = round(count / total, 4)
    return out


def build_routing_model(rows: list[dict]) -> dict:
    """Derive a routing model from per-row analyses.

    ``rows`` is a list of dictionaries with keys ``modality``, ``body_part``,
    ``dictation``, and ``edited_sections`` (the result of
    :func:`src.analyze_training_edits.analyze_row` plus the dictation text).
    """
    # modality/body_part -> (edited counts, total rows)
    mb_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    mb_total: Counter[tuple[str, str]] = Counter()
    # dictation term -> {label: count}
    term_counts: dict[str, Counter[str]] = defaultdict(Counter)
    term_total: Counter[str] = Counter()

    for row in rows:
        mb = (row["modality"], row["body_part"])
        mb_total[mb] += 1
        mb_counts[mb].update(row["edited_sections"])

        terms = _termify(row.get("dictation", ""))
        edited_here = set(row["edited_sections"])
        for term in terms:
            term_total[term] += 1
            for label in edited_here:
                term_counts[term][label] += 1

    by_mb = {
        f"{m} / {b}": _filter(dict(mb_counts[(m, b)]), mb_total[(m, b)])
        for (m, b) in mb_total
    }

    by_term = {}
    for term, total in term_total.items():
        if total < MIN_TERM_SUPPORT:
            continue
        weights = _filter(dict(term_counts[term]), total)
        if weights:
            # Keep the most predictive labels per term (cap size for the prompt).
            by_term[term] = dict(
                sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            )

    return {"by_modality_bodypart": by_mb, "by_term": by_term}


def _load_train(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"case_id", "modality", "body_part", "template_content", "dictation", "report"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {sorted(missing)}")
    analyses = [analyze_row(row) for row in df.to_dict("records")]
    out = []
    raw = df.to_dict("records")
    for analysis, rec in zip(analyses, raw):
        out.append(
            {
                "modality": analysis.modality,
                "body_part": analysis.body_part,
                "dictation": rec["dictation"],
                "edited_sections": analysis.edited_sections,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the routing model from train.csv.")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--out", default=str(PROMPTS_DIR / "routing_model.json"))
    args = parser.parse_args(argv)

    rows = _load_train(Path(args.train_csv))
    model = build_routing_model(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Trained routing model on {len(rows)} rows -> {out}")
    print(f"  by_modality_bodypart entries: {len(model['by_modality_bodypart'])}")
    print(f"  by_term entries: {len(model['by_term'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
