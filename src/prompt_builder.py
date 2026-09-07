"""Prompt construction: system instruction + few-shot turns + the real case.

The system prompt lives in ``prompts/system_prompt.txt`` (editable without
touching code); the few-shot demonstrations live in
``prompts/few_shot_examples.json`` (0..N, capped by ``max_examples``).
Section labels are derived from the template itself via
:func:`template_editor.parse_template`, never from train.csv or guesswork.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import FEW_SHOT_EXAMPLES_PATH, SYSTEM_PROMPT_PATH
from src.data_loader import TestCase
from src.template_editor import WHOLE_BLOCK_KEY, parse_template

USER_ROLE = "user"
MODEL_ROLE = "model"


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


def _case_user_text(test_case: TestCase) -> str:
    parsed = parse_template(test_case.template_content)
    labels = parsed.section_labels if parsed.is_sectioned else [WHOLE_BLOCK_KEY]
    labels_block = "\n".join(f"- {label}" for label in labels)
    return (
        "TEMPLATE SECTION LABELS (use exactly these keys in findings_edits):\n"
        f"{labels_block}\n\n"
        "TEMPLATE_CONTENT:\n"
        f"{test_case.template_content}\n\n"
        "DICTATION:\n"
        f"{test_case.dictation}"
    )


def _example_turns(example: dict) -> tuple[tuple[str, str], ...]:
    template = example["template"]
    parsed = parse_template(template)
    labels = parsed.section_labels if parsed.is_sectioned else [WHOLE_BLOCK_KEY]
    labels_block = "\n".join(f"- {label}" for label in labels)
    user_text = (
        "TEMPLATE SECTION LABELS (use exactly these keys in findings_edits):\n"
        f"{labels_block}\n\n"
        "TEMPLATE_CONTENT:\n"
        f"{template}\n\n"
        "DICTATION:\n"
        f"{example['dictation']}"
    )
    model_text = json.dumps(example["output"], ensure_ascii=False, indent=2)
    return (USER_ROLE, user_text), (MODEL_ROLE, model_text)


def build_prompt(
    test_case: TestCase,
    *,
    system_prompt: str | None = None,
    few_shot: list[dict] | None = None,
    max_examples: int = 5,
) -> BuiltPrompt:
    """Assemble the system instruction + few-shot turns + the real case."""
    system = system_prompt if system_prompt is not None else load_system_prompt()
    examples = few_shot if few_shot is not None else load_few_shot_examples()
    examples = list(examples[: max(0, max_examples)])

    turns: list[tuple[str, str]] = []
    for example in examples:
        turns.extend(_example_turns(example))
    turns.append((USER_ROLE, _case_user_text(test_case)))

    return BuiltPrompt(system_instruction=system, turns=tuple(turns))