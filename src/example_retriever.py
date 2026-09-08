"""Offline, deterministic retrieval of the most relevant REAL training examples.

For every inference case this module selects up to ``k`` real rows from
``data/train.csv`` to inject into that case's Gemini prompt as few-shot
demonstrations. Retrieval is 100% offline and deterministic:

  - ``train.csv`` is read exactly once, at index build time. The Gemini client
    never reads it; ``prompt_builder`` only receives the selected examples.
  - No LLM / Gemini / API call is made anywhere in this module.
  - The index is built once and reused in memory: per-row normalized template
    text, section labels, a lightweight pure-Python TF-IDF space over the
    training dictations, and the structured edit output derived from each
    row's human reference report are all precomputed.

Scoring
-------
Ranking is lexicographic, in the data-driven priority order:

    same template structure  >  same modality/body part  >  similar dictated
    findings  >  similar edited sections

Implementation: each candidate gets four deterministic signals normalised to
[0, 1] (below). Candidates are first bucketed by their TEMPLATE tier
(:func:`template_tier`), then sorted within the tier by modality/body part,
then dictation similarity, then edited-section relevance, then train case_id
(final, deterministic tiebreak).

Of the 132 test cases, 117 share an EXACT (normalized) template with at least
one train row, so the template tier dominates the ranking: the primary job of a
retrieved example is to show how THAT template is minimally edited. Within a
same-template tier, dictation similarity breaks the tie so the example whose
findings resemble the current dictation (i.e. the most relevant minimal edit)
is preferred. ``RetrievalWeights`` still aggregates the signals into a single
*display* ``score`` for inspection, but the using the weighted mean for
ranking was shown to let dictation/edited-section compensation pull
near-template rows above exact-template rows, so the ranking is lexicographic.

  component                      +tier   what it measures
  ------------------------------ ------- -------------------------------------
  template_similarity            tier 0  label-set overlap, label ORDER, and
                                          vocabulary overlap of the normalized
                                          template text — the signal most tied
                                          to the leaderboard's template-edit
                                          fidelity metric; tiers are
                                          >=0.97 / >=0.90 / >=0.80 / else
  modality_bodypart              within  exact / partial modality + body_part
                                          agreement (secondary once the
                                          template tier matches; primary
                                          fallback for the 15 test templates
                                          with no exact train match)
  dictation_similarity           within  half TF-IDF cosine over DICTATION
                                          tokens (pure-Python space fitted on
                                          the training dictations) and half
                                          content-token Jaccard — the Jaccard
                                          term keeps discriminating WITHIN a
                                          same-template tier where the cosine
                                          saturates (verified: cosine stdev is
                                          ~0 inside every large template
                                          family)
  edited_section_relevance       within  overlap between the sections a
                                          training row's human report actually
                                          edited and the sections suggested for
                                          the test case by the existing
                                          section-routing model — a weak final
                                          tiebreak; a larger weight was shown
                                          to displace exact-template examples
                                          from the top 3

Retrieval safety
----------------
- A test case is ranked using ONLY its metadata, template, and dictation.
- A training row's ground-truth ``report`` is used only to (a) derive the
  example's structured edit output and (b) compute the edited-section signal.
- A test case can never retrieve itself: its ``case_id`` is excluded from the
  candidate pool (train/test ids are normally disjoint; this is a guard).
- At most ``k`` examples are returned. If fewer strong matches exist, the best
  remaining candidates backfill rather than failing.

Inspection CLI
--------------
    python -m src.example_retriever --test-csv data/test.csv --case-id <CASE_ID>
    python -m src.example_retriever --test-csv data/test.csv --inspect 5
    python -m src.example_retriever --test-csv data/test.csv --inspect-all
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.analyze_training_edits import (
    _findings_region_text,
    _substring_presence,
    analyze_row,
    normalize,
)
from src.config import TEST_CSV, TRAIN_CSV
from src.data_loader import TestCase, TestDataset, load_test_cases
from src.errors import TemplateError
from src.section_routing import RoutingModel
from src.template_editor import parse_template

# Reuse the project's existing deterministic token extraction + stopwords so
# retrieval and routing disagree as little as possible.
from src.section_routing import _STOPWORDS, _TOKEN_RE

DEFAULT_MAX_EXAMPLES = 3
_TRAIN_REQUIRED_COLUMNS = frozenset(
    {
        "case_id", "modality", "body_part", "template_content", "dictation", "report",
    }
)


# --------------------------------------------------------------------------- #
# Weights + ranking (changed in one place)
# --------------------------------------------------------------------------- #


# Template-structure tiers used by the lexicographic ranking. ``template
# similarity`` is the PRIMARY signal (117/132 test templates exist in train),
# so candidates are bucketed by it first; the weighted score is kept purely as
# a display/transparency number.
_TEMPLATE_TIER_EDGES = (0.97, 0.90, 0.80)


def template_tier(template_similarity: float) -> int:
    """0 = same/exact template family, 1 = near, 2 = moderate, 3 = unrelated."""
    for tier, edge in enumerate(_TEMPLATE_TIER_EDGES):
        if template_similarity >= edge:
            return tier
    return len(_TEMPLATE_TIER_EDGES)


@dataclass(frozen=True)
class RetrievalWeights:
    """Weights used for the *display* score (weighted mean of the four signals).

    The actual ranking is lexicographic (template tier -> modality/body part ->
    dictation similarity -> edited-section relevance -> case_id), matching the
    data-driven priority order:

        same template structure > same modality/body part > similar dictated
        findings > similar edited sections

    These weights only feed the ``score`` printed by the inspection CLI; they
    do not change the ranking order. They are exposed here (and asserted in
    tests) so the priority is reviewable in one place.
    """

    template_similarity: float = 0.55
    modality_bodypart: float = 0.12
    dictation_similarity: float = 0.28
    edited_section_relevance: float = 0.05

    @property
    def components(self) -> dict[str, float]:
        return {
            "template_similarity": self.template_similarity,
            "modality_bodypart": self.modality_bodypart,
            "dictation_similarity": self.dictation_similarity,
            "edited_section_relevance": self.edited_section_relevance,
        }

    @property
    def total(self) -> float:
        return sum(self.components.values())


# --------------------------------------------------------------------------- #
# Derived example output (from the human reference report, never invented)
# --------------------------------------------------------------------------- #


def derive_structured_output(template_content: str, report_content: str) -> dict:
    """Derive the structured edit payload that turns *template* into *report*.

    Only sections whose label survives in the reference WITH a changed body are
    emitted (verbatim reference body). A renamed/merged/removed section is not
    representable as a section edit and is skipped. The impression is emitted
    only when it actually changed. Everything comes from the real report — this
    is the "human reference / structured edits" shown to the model.
    """
    template = parse_template(template_content)
    report = parse_template(report_content)
    template_bodies = {s.label: s.body for s in template.sections}
    report_bodies = {s.label: s.body for s in report.sections}

    findings_edits: dict[str, str] = {}
    for label, body in template_bodies.items():
        if label in report_bodies:
            if normalize(report_bodies[label]) != normalize(body):
                findings_edits[label] = report_bodies[label]
        elif not _substring_presence(body, _findings_region_text(report)):
            pass  # removed entirely -> no label-based edit is representable

    impression = None
    if normalize(template.after_impression) != normalize(report.after_impression):
        stripped = report.after_impression.strip("\n")
        impression = stripped if stripped else None

    return {"findings_edits": findings_edits, "impression": impression}


# --------------------------------------------------------------------------- #
# Per-row index record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingExample:
    """One train.csv row plus its reusable precomputed features."""

    case_id: str
    modality: str
    body_part: str
    template_content: str
    dictation: str
    report: str
    section_labels: tuple[str, ...] = field(default_factory=tuple)
    edited_sections: tuple[str, ...] = field(default_factory=tuple)
    derived_output: dict = field(default_factory=dict)
    norm_tokens: frozenset = field(default_factory=frozenset)
    dictation_tokens: frozenset = field(default_factory=frozenset)
    is_sectioned: bool = True

    @classmethod
    def from_record(cls, record: dict) -> "TrainingExample | None":
        """Build a TrainingExample, or None when the row cannot be parsed."""
        try:
            parsed = parse_template(record["template_content"])
            parsed_report = parse_template(record["report"])
            analysis = analyze_row(record)
        except TemplateError:
            return None

        return cls(
            case_id=str(record["case_id"]).strip(),
            modality=str(record["modality"]).strip(),
            body_part=str(record["body_part"]).strip(),
            template_content=record["template_content"],
            dictation=record["dictation"],
            report=record["report"],
            section_labels=tuple(parsed.section_labels),
            edited_sections=tuple(analysis.edited_sections),
            derived_output=derive_structured_output(
                record["template_content"], record["report"]
            ),
            norm_tokens=_text_tokens(record["template_content"]),
            dictation_tokens=_text_tokens(record["dictation"]),
            is_sectioned=parsed.is_sectioned,
        )


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved example with its score and per-signal breakdown."""

    example: TrainingExample
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    def to_prompt_example(self) -> dict:
        """Shape the example for ``prompt_builder`` (template/dictation/output)."""
        return {
            "template": self.example.template_content,
            "dictation": self.example.dictation,
            "output": self.example.derived_output,
            "case_id": self.example.case_id,
        }


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def _bodypart_tokens(body_part: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", (body_part or "").lower()))


def _text_tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS
    )


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _label_order_ratio(labels_a: tuple[str, ...], labels_b: tuple[str, ...]) -> float:
    """How similarly the two label SEQUENCES are ordered (difflib ratio)."""
    return SequenceMatcher(None, list(labels_a), list(labels_b), autojunk=False).ratio()


# --------------------------------------------------------------------------- #
# Signal scorers (+ documentation of the exact deterministic rules)
# --------------------------------------------------------------------------- #


def modality_bodypart_score(q_mod: str, q_bp: str, r_mod: str, r_bp: str) -> float:
    """Compatibility of modality + body_part, exact to partial.

    Rules (deterministic):
      * exact modality AND exact body_part            -> 1.0
      * same modality, partial body_part token overlap -> 0.5 + 0.25 * overlap
      * same modality only                             -> 0.5
      * exact body_part only                           -> 0.6
      * different modality, partial body_part overlap  -> 0.25 * overlap
      * otherwise                                      -> 0.0
    """
    mod_ok = q_mod == r_mod
    bp_ok = (q_bp or "").lower() == (r_bp or "").lower()
    q_toks = _bodypart_tokens(q_bp)
    r_toks = _bodypart_tokens(r_bp)
    overlap = _jaccard(set(q_toks), set(r_toks))

    if mod_ok and bp_ok:
        return 1.0
    if mod_ok and overlap > 0:
        return 0.5 + 0.25 * overlap
    if mod_ok:
        return 0.5
    if bp_ok:
        return 0.6
    if overlap > 0:
        return 0.25 * overlap
    return 0.0


def template_similarity_score(
    q: "_QueryFeatures", e: TrainingExample
) -> float:
    """Template structure + normalized text similarity (0..1).

    * 0.40 label-set Jaccard (identical editable sections)
    * 0.25 label ORDER ratio (same sections, same order)
    * 0.35 token Jaccard over the normalized template text (shared wording)
    """
    jaccard = _jaccard(set(q.section_labels), set(e.section_labels))
    order = _label_order_ratio(q.section_labels, e.section_labels)
    text = _jaccard(set(q.norm_tokens), set(e.norm_tokens))
    return 0.40 * jaccard + 0.25 * order + 0.35 * text


def dictation_similarity_score(
    q: "_QueryFeatures",
    e: TrainingExample,
    tfidf: _TfidfIndex,
    vector: dict[str, float],
) -> float:
    """How much the test dictation resembles the training *dictation* (0..1).

    Half TF-IDF cosine (whole-corpus, term-weighted recall) and half
    content-token Jaccard. Samples with an identical template produce constant
    TF-IDF cosine (verified: stdev = 0.000 inside every large template family,
    because the corpus index used to be fitted on TEMPLATE text), so the
    Jaccard term is what lets us prefer, WITHIN a same-template family, the
    example whose dictated findings resemble the current case.
    """
    cosine = tfidf.cosine(q.dictation_tfidf, vector)
    jaccard = _jaccard(set(q.dictation_tokens), set(e.dictation_tokens))
    return 0.5 * cosine + 0.5 * jaccard


def edited_section_relevance_score(
    q: "_QueryFeatures", e: TrainingExample
) -> float:
    """How relevant the training row's EDITED sections are to the test case.

    Preference order: the routing model's candidate sections for the test case
    when the routing artifact is available; otherwise the test template's own
    section labels. Score is the fraction of those candidate sections the
    training row's human report actually edited.
    """
    edited = set(e.edited_sections)
    if not edited:
        return 0.0
    cands = q.candidate_sections
    if not cands:
        if not q.section_labels:
            return 0.0
        cands = q.section_labels
    return len(set(cands) & edited) / max(1, len(set(cands)))


# --------------------------------------------------------------------------- #
# Pure-Python TF-IDF (small, deterministic; no sklearn/scipy dependency)
# --------------------------------------------------------------------------- #


class _TfidfIndex:
    """Lazy, idempotent TF-IDF space over a tokenized document corpus."""

    def __init__(self, docs: list[frozenset]):
        self._docs = docs
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] | None = None

    def _fit(self) -> None:
        if self._vectors is not None:
            return
        n = max(1, len(self._docs))
        df: Counter[str] = Counter()
        for tokens in self._docs:
            df.update(tokens)
        # sklearn-style smoothed idf: log((1+n)/(1+df)) + 1
        self._idf = {
            term: math.log((1.0 + n) / (1.0 + count)) + 1.0
            for term, count in df.items()
        }
        vectors: list[dict[str, float]] = []
        for tokens in self._docs:
            counts = Counter(tokens)
            norm_sq = 0.0
            raw: dict[str, float] = {}
            for term, count in counts.items():
                weight = count * self._idf.get(term, 0.0)
                raw[term] = weight
                norm_sq += weight * weight
            inv_norm = 1.0 / math.sqrt(norm_sq) if norm_sq else 0.0
            vectors.append(
                {term: weight * inv_norm for term, weight in raw.items() if weight}
            )
        self._vectors = vectors

    @property
    def vectors(self) -> list[dict[str, float]]:
        self._fit()
        return self._vectors

    def query_vector(self, tokens: frozenset[str]) -> dict[str, float]:
        counts = Counter(tokens)
        norm_sq = 0.0
        raw: dict[str, float] = {}
        for term, count in counts.items():
            idf = self._idf.get(term)
            if not idf:
                continue
            weight = count * idf
            raw[term] = weight
            norm_sq += weight * weight
        inv_norm = 1.0 / math.sqrt(norm_sq) if norm_sq else 0.0
        return {term: weight * inv_norm for term, weight in raw.items()}

    @staticmethod
    def cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        small, big = (vec_a, vec_b) if len(vec_a) <= len(vec_b) else (vec_b, vec_a)
        if not small or not big:
            return 0.0
        return sum(weight * big.get(term, 0.0) for term, weight in small.items())


# --------------------------------------------------------------------------- #
# Query features (computed once per retrieval)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _QueryFeatures:
    case_id: str
    modality: str
    body_part: str
    section_labels: tuple[str, ...]
    norm_tokens: frozenset
    candidate_sections: tuple[str, ...]
    dictation_tokens: frozenset
    dictation_tfidf: dict[str, float]


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #


class ExampleRetriever:
    """In-memory, deterministic retrieval index over train.csv rows."""

    def __init__(
        self,
        train_csv_path: Path | str = TRAIN_CSV,
        *,
        weights: RetrievalWeights | None = None,
        routing_model: RoutingModel | None = None,
    ):
        self.path = Path(train_csv_path)
        self.weights = weights or RetrievalWeights()
        self.routing_model = routing_model if routing_model is not None else RoutingModel()
        self.rows: list[TrainingExample] = []
        self.skipped_rows: int = 0
        self._load()
        self._tfidf = None
        if self.rows:
            self._tfidf = _TfidfIndex([row.dictation_tokens for row in self.rows])
            self._tfidf.vectors  # fit eagerly so retrieval latency is predictable

    # -------------------------------------------------------------- index
    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"train.csv not found for retrieval index: {self.path}")
        df = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        missing = _TRAIN_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"train.csv missing required columns for retrieval: {sorted(missing)}"
            )
        rows: list[TrainingExample] = []
        skipped = 0
        for record in df.to_dict("records"):
            example = TrainingExample.from_record(record)
            if example is None:
                skipped += 1
                continue
            rows.append(example)
        self.rows = rows
        self.skipped_rows = skipped

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def available(self) -> bool:
        return bool(self.rows)

    def train_case_ids(self) -> set[str]:
        return {row.case_id for row in self.rows}

    # -------------------------------------------------------------- query
    def _query_features(self, test_case: TestCase) -> _QueryFeatures:
        try:
            parsed = parse_template(test_case.template_content)
            labels = tuple(parsed.section_labels)
        except TemplateError:
            labels = ()
        dictation_tokens = _text_tokens(test_case.dictation)
        candidates = self.routing_model.candidate_sections(test_case)
        return _QueryFeatures(
            case_id=test_case.case_id,
            modality=test_case.modality,
            body_part=test_case.body_part,
            section_labels=labels,
            norm_tokens=_text_tokens(test_case.template_content),
            candidate_sections=tuple(candidates),
            dictation_tokens=dictation_tokens,
            dictation_tfidf=(
                self._tfidf.query_vector(dictation_tokens) if self._tfidf else {}
            ),
        )

    def _combine(self, signals: dict[str, float]) -> float:
        """Weighted mean of the signals — a *display* score, not the sort key."""
        w = self.weights.components
        total = sum(w.get(name, 0.0) * value for name, value in signals.items())
        return round(total / self.weights.total, 6) if self.weights.total else 0.0

    def retrieve(
        self, test_case: TestCase, *, k: int = DEFAULT_MAX_EXAMPLES, exclude_case_id: str | None = None
    ) -> list[RetrievalResult]:
        """Return the top ``k`` training examples for *test_case*.

        Ranking is lexicographic and deterministic:

            (template tier, -modality/body part, -dictation similarity,
             -edited-section relevance, case_id)

        so a same/exact-template example always outranks a merely-similar one
        (template-edit fidelity is the mitigator RES is built on), and within
        the same template tier the example whose dictation resembles the test
        is preferred. Retail: the ranking is identical regardless of train.csv
        row order; ties resolve to the smaller training case_id.
        """
        if not self.rows or not self._tfidf:
            return []
        k = max(0, int(k))
        if k == 0:
            return []

        q = self._query_features(test_case)
        exclude = exclude_case_id if exclude_case_id is not None else test_case.case_id
        vectors = self._tfidf.vectors

        scored: list[tuple[float, str, TrainingExample, dict[str, float]]] = []
        for index, example in enumerate(self.rows):
            if example.case_id == exclude:
                continue
            signals = {
                "modality_bodypart": modality_bodypart_score(
                    q.modality, q.body_part, example.modality, example.body_part
                ),
                "template_similarity": template_similarity_score(q, example),
                "dictation_similarity": dictation_similarity_score(
                    q, example, self._tfidf, vectors[index]
                ),
                "edited_section_relevance": edited_section_relevance_score(q, example),
            }
            score = self._combine(signals)
            tier = template_tier(signals["template_similarity"])
            scored.append((tier, score, example.case_id, example, signals))

        # Lexicographic priority: template structure -> modality/body part ->
        # dictation -> edited sections -> case_id (deterministic).
        scored.sort(
            key=lambda item: (
                item[0],
                -item[4]["modality_bodypart"],
                -item[4]["dictation_similarity"],
                -item[4]["edited_section_relevance"],
                item[2],
            )
        )
        return [
            RetrievalResult(example=example, score=score, signals=signals)
            for _tier, score, _cid, example, signals in scored[:k]
        ]

    def retrieve_all(
        self, test_cases: list[TestCase] | TestDataset, *, k: int = DEFAULT_MAX_EXAMPLES
    ) -> dict[str, list[RetrievalResult]]:
        """Convenience: retrieve examples for many test cases at once."""
        cases = test_cases.cases if not isinstance(test_cases, (list, tuple)) else test_cases
        return {case.case_id: self.retrieve(case, k=k) for case in cases}


# --------------------------------------------------------------------------- #
# Inspection CLI
# --------------------------------------------------------------------------- #


def _print_result(test_case: TestCase, results: list[RetrievalResult]) -> None:
    print("TEST:")
    print(f"{test_case.case_id} {test_case.modality} / {test_case.body_part}")
    print()
    for rank, result in enumerate(results, start=1):
        example = result.example
        print(f"TOP {rank}:")
        print(example.case_id)
        print(f"score={result.score:.3f}")
        tier = template_tier(result.signals.get("template_similarity", 0.0))
        tier_name = ("exact template", "near template", "moderate",
                     "unrelated")[min(tier, 3)]
        print(f"tier={tier_name} (template_similarity={result.signals.get('template_similarity', 0.0):.3f})")
        print("reason/signals:")
        for name in ("modality_bodypart", "template_similarity",
                     "dictation_similarity", "edited_section_relevance"):
            print(f"  {name}={result.signals.get(name, 0.0):.3f}")
        print(
            f"  train: {example.modality} / {example.body_part} | "
            f"edited: {', '.join(example.edited_sections) if example.edited_sections else '(none)'}"
        )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the deterministic training-example retrieval."
    )
    parser.add_argument("--test-csv", default=str(TEST_CSV), help="path to test.csv")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV), help="path to train.csv")
    parser.add_argument("--case-id", default=None, help="inspect one test case")
    parser.add_argument(
        "--inspect",
        default=None,
        help="inspect several test cases: an integer N -> first N cases, or a "
             "comma-separated list of case_ids (e.g. 'a,b,c')",
    )
    parser.add_argument("--inspect-all", action="store_true", help="inspect every test case")
    parser.add_argument("--k", type=int, default=DEFAULT_MAX_EXAMPLES, help="examples per case")
    args = parser.parse_args(argv)

    retriever = ExampleRetriever(Path(args.train_csv))
    dataset = load_test_cases(Path(args.test_csv))
    print(f"indexed {retriever.row_count} training rows "
          f"(skipped {retriever.skipped_rows}) from {retriever.path}\n")

    cases: list[TestCase] = []
    if args.inspect_all:
        cases = list(dataset.cases)
    elif args.inspect is not None:
        try:
            n = int(args.inspect)
            cases = list(dataset.cases)[: max(0, n)]
        except ValueError:
            wanted = {cid.strip() for cid in args.inspect.split(",") if cid.strip()}
            cases = [case for case in dataset.cases if case.case_id in wanted]
    elif args.case_id is not None:
        matches = [case for case in dataset.cases if case.case_id == args.case_id]
        cases = matches
    else:
        parser.error("provide --case-id, --inspect, or --inspect-all")

    if not cases:
        print(f"No test cases matched the requested selector in {args.test_csv}")
        return 1

    for index, case in enumerate(cases):
        if index:
            print("-" * 60 + "\n")
        results = retriever.retrieve(case, k=args.k)
        _print_result(case, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())