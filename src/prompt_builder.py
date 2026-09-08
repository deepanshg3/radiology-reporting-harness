"""Prompt construction: system instruction + few-shot turns + the real case.

The system prompt lives in ``prompts/system_prompt.txt`` (editable without
touching code); the few-shot demonstrations come from ONE of two sources:

* **Dynamic** (primary during real inference): ``ExampleRetriever`` selects up
  to ``max_examples`` REAL rows from ``train.csv`` that are most relevant to
  the current case (template similarity, modality/body_part, dictation, edited
  sections). ``retrieval_prompt_builder`` wires this into the Gemini client.
* **Static** (fallback): ``prompts/few_shot_examples.json`` is used for tests or
  whenever the retrieval index is unavailable. The two sources are never
  combined, so the model never sees unrelated extra context.

Section labels are derived from the template itself via
:func:`template_editor.parse_template`, never from train.csv or guesswork.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import FEW_SHOT_EXAMPLES_PATH, SYSTEM_PROMPT_PATH
from src.data_loader import TestCase
from src.section_routing import RoutingModel
from src.template_editor import WHOLE_BLOCK_KEY, parse_template

USER_ROLE = "user"
MODEL_ROLE = "model"

DEFAULT_MAX_EXAMPLES = 3


@dataclass(frozen=True)
class BuiltPrompt:
    """A fully assembled prompt, ready to map onto Gemini content parts."""

    system_instruction: str
    turns: tuple[tuple[str, str], ...]

    @property
    def user_roles(self) -> list[str]:
        return [role for role, _ in self.turns]


def _read_text(path: Path, what: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{what} is empty: {path}")
    return text


def load_system_prompt(path: Path | str = SYSTEM_PROMPT_PATH) -> str:
    return _read_text(Path(path), "system prompt")


def load_few_shot_examples(path: Path | str = FEW_SHOT_EXAMPLES_PATH) -> list[dict]:
    """Load validated few-shot examples.

    Returns an empty list when the file is absent so a missing (or intentionally
    removed) few-shot file silently disables examples rather than crashing.
    """
    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"few_shot_examples.json is not valid JSON: {exc}") from exc
    examples = data.get("examples")
    if not isinstance(examples, list):
        raise ValueError("few_shot_examples.json must contain an 'examples' list")

    validated: list[dict] = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"example {index} is not an object")
        for key in ("template", "dictation", "output"):
            if key not in example:
                raise ValueError(f"example {index} is missing '{key}'")
        if not isinstance(example["output"], dict):
            raise ValueError(f"example {index} 'output' must be an object")
        # Parse the template now so a mis-formed example fails loudly at load.
        parse_template(example["template"])
        validated.append(example)
    return validated


def _case_user_text(test_case: TestCase, routing: RoutingModel | None = None) -> str:
    parsed = parse_template(test_case.template_content)
    labels = parsed.section_labels if parsed.is_sectioned else [WHOLE_BLOCK_KEY]
    labels_block = "\n".join(f"- {label}" for label in labels)

    hint_block = ""
    if routing is not None and routing.available and parsed.is_sectioned:
        candidates = routing.candidate_sections(test_case)
        if candidates:
            hint_block = (
                "\nCANDIDATE SECTIONS TO CONSIDER (suggested by training data — "
                "a HINT only, not a restriction):\n"
                + "\n".join(f"- {label}" for label in candidates)
            )

    return (
        "TEMPLATE SECTION LABELS (use exactly these keys in findings_edits):\n"
        f"{labels_block}\n\n"
        "TEMPLATE_CONTENT:\n"
        f"{test_case.template_content}\n\n"
        "DICTATION:\n"
        f"{test_case.dictation}"
        + hint_block
    )


def _example_turns(example: dict, *, header: str | None = None) -> tuple[tuple[str, str], ...]:
    template = example["template"]
    parsed = parse_template(template)
    labels = parsed.section_labels if parsed.is_sectioned else [WHOLE_BLOCK_KEY]
    labels_block = "\n".join(f"- {label}" for label in labels)
    header_block = f"{header}\n\n" if header else ""
    user_text = (
        header_block
        + "TEMPLATE SECTION LABELS (use exactly these keys in findings_edits):\n"
        f"{labels_block}\n\n"
        "TEMPLATE_CONTENT:\n"
        f"{template}\n\n"
        "DICTATION:\n"
        f"{example['dictation']}"
    )
    model_text = json.dumps(example["output"], ensure_ascii=False, indent=2)
    return (USER_ROLE, user_text), (MODEL_ROLE, model_text)


_DEFAULT_ROUTING: RoutingModel | None = None
_DEFAULT_RETRIEVER: object | None = None  # ExampleRetriever | None (lazy import)


def _default_routing() -> RoutingModel | None:
    """Lazily shared routing model (degraded to None if the artifact is absent)."""
    global _DEFAULT_ROUTING
    if _DEFAULT_ROUTING is None:
        model = RoutingModel()
        _DEFAULT_ROUTING = model if model.available else None
    return _DEFAULT_ROUTING


def default_retriever():
    """The lazily-shared dynamic retrieval index (None when train.csv is absent)."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        from src.example_retriever import ExampleRetriever  # lazy: needs train.csv

        try:
            _DEFAULT_RETRIEVER = ExampleRetriever()
        except (OSError, ValueError):
            # train.csv missing/malformed -> index unavailable -> static fallback.
            _DEFAULT_RETRIEVER = None
    return _DEFAULT_RETRIEVER


def build_prompt(
    test_case: TestCase,
    *,
    system_prompt: str | None = None,
    few_shot: list[dict] | None = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    routing_model: RoutingModel | None = None,
    example_retriever=None,
) -> BuiltPrompt:
    """Assemble the system instruction + few-shot turns + the real case.

    Few-shot source resolution (never both sources at once):

    * ``example_retriever`` provided  -> dynamic retrieval (top ``max_examples``
      REAL examples from train.csv for THIS case).
    * else ``few_shot`` provided      -> the given static examples.
    * else                            -> static ``few_shot_examples.json``.

    ``routing_model`` (optional) supplies candidate-section HINTS derived from
    training data; they are suggestions only and never restrict the model. When
    ``None``, a shared default routing model is used if the artifact exists,
    otherwise hints are omitted.
    """
    system = system_prompt if system_prompt is not None else load_system_prompt()

    dynamic = example_retriever is not None
    if dynamic:
        retrieved = example_retriever.retrieve(test_case, k=max_examples)
        examples = [result.to_prompt_example() for result in retrieved]
    elif few_shot is not None:
        examples = list(few_shot)
    else:
        examples = load_few_shot_examples()
    examples = examples[: max(0, max_examples)]

    routing = routing_model if routing_model is not None else _default_routing()

    turns: list[tuple[str, str]] = []
    for index, example in enumerate(examples):
        header = f"REAL TRAINING EXAMPLE {index + 1}" if dynamic else None
        turns.extend(_example_turns(example, header=header))
    turns.append((USER_ROLE, _case_user_text(test_case, routing=routing)))

    return BuiltPrompt(system_instruction=system, turns=tuple(turns))


def retrieval_prompt_builder(
    test_case: TestCase,
    *,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    routing_model: RoutingModel | None = None,
) -> BuiltPrompt:
    """Prompt builder wired to dynamic retrieval (used for real inference).

    Falls back to the static few-shot prompt whenever the retrieval index is
    unavailable (e.g. train.csv missing).
    """
    retriever = default_retriever()
    if retriever is None or not retriever.available:
        return build_prompt(test_case, routing_model=routing_model)
    return build_prompt(
        test_case,
        max_examples=max_examples,
        routing_model=routing_model,
        example_retriever=retriever,
    )