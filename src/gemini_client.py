"""Report-generation interface: real Gemini client and deterministic mock.

The pipeline contract changed for the structured-edit architecture:

- :meth:`ReportGenerator.generate` returns an :class:`~src.response_parser.Edits`
  payload (``findings_edits`` + optional ``impression``), **not** a finished
  report. Converting those edits into the final report text is owned by
  ``template_editor`` inside the pipeline.
- The mock therefore exercises the *same* path as the real model (generate ->
  parse -> apply -> validate) instead of returning a ready-made string.
- The real client asks the model for strict JSON via ``response_mime_type`` +
  ``response_json_schema`` (the SDK ``response_schema`` variant does not support
  ``additionalProperties`` for the dynamic label->body map, so we pass the
  schema through ``response_json_schema`` instead).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from src.data_loader import TestCase
from src.errors import InvalidModelOutput, MockFailure
from src.prompt_builder import retrieval_prompt_builder
from src.response_parser import Edits, parse_edits, parse_model_text
from src.template_editor import parse_template

# Strict JSON-object schema mirrored in both the API request and the prompt.
EDITS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "findings_edits": {
            "type": "object",
            "description": (
                "Map from an EXACT template section label to the replacement "
                "body text for that section. Only changed sections are listed."
            ),
            "additionalProperties": {"type": "string"},
        },
        "impression": {
            "type": ["string", "null"],
            "description": (
                "Complete replacement IMPRESSION text without the 'IMPRESSION:' "
                "header, or null to keep the template impression unchanged."
            ),
        },
    },
    "required": ["findings_edits"],
    "additionalProperties": False,
}


class ReportGenerator(ABC):
    """Anything that turns one test case into structured edit instructions."""

    model_name: str = "unknown"

    @abstractmethod
    def generate(self, test_case: TestCase, *, attempt: int) -> Edits:
        """Return the proposed edits for *test_case*.

        Raises an exception on failure; the pipeline classifies, retries, and
        records the outcome. ``attempt`` is 1-based.
        """
        raise NotImplementedError


class GeminiReportGenerator(ReportGenerator):
    """Real Gemini client using strict JSON schema output.

    The google-genai SDK is imported lazily so ``import src.gemini_client``
    never requires the SDK to be installed (mock-only environments).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        prompt_builder=retrieval_prompt_builder,
    ):
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_builder = prompt_builder

    def generate(self, test_case: TestCase, *, attempt: int) -> Edits:
        prompt = self.prompt_builder(test_case)
        contents = [
            {"role": role, "parts": [{"text": text}]}
            for role, text in prompt.turns
        ]
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self._genai.types.GenerateContentConfig(
                system_instruction=prompt.system_instruction,
                response_mime_type="application/json",
                response_json_schema=EDITS_JSON_SCHEMA,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        return _response_to_edits(response)


def _response_to_edits(response) -> Edits:
    """Extract and validate the structured payload from a model response."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parse_edits(parsed)

    candidates = getattr(response, "candidates", None) or []
    if candidates and getattr(candidates[0], "content", None):
        parts = getattr(candidates[0].content, "parts", None) or []
        text = "".join(getattr(part, "text", None) or "" for part in parts)
        if text.strip():
            return parse_model_text(text)
    raise InvalidModelOutput("model response contained no usable JSON")


def _mock_edits(test_case: TestCase) -> Edits:
    """Deterministic, structure-exercising edit payload (never the network)."""
    snippet = " ".join(
        (test_case.dictation.strip() or "No significant abnormality.").split()
    )
    if len(snippet) > 120:
        snippet = snippet[:120].rstrip(" ,.;") + " ..."
    impression = (
        f"Mock impression ({test_case.case_id}) - not clinically meaningful."
    )
    parsed = parse_template(test_case.template_content)

    if not parsed.is_sectioned:
        return Edits(findings_edits={"FINDINGS": snippet}, impression=impression)

    edits: dict[str, str] = {}
    for section in parsed.sections:
        edits[section.label] = f"Mockized review for {section.label}: {snippet}"
    return Edits(findings_edits=edits, impression=impression)


class MockReportGenerator(ReportGenerator):
    """Deterministic stand-in that never touches the network.

    - Produces a synthetic :class:`Edits` payload (parser/editor/validator are
      exercised the same way as with the real model).
    - Can be told to fail specific case_ids (to exercise retries/failures).
    - Optionally appends every call to a JSONL log so tests can prove that
      successful cases are never re-queried on resume.
    """

    model_name = "mock"

    def __init__(
        self,
        *,
        fail_case_ids: set[str] | None = None,
        log_path: Path | str | None = None,
        delay_seconds: float = 0.0,
    ):
        self.fail_case_ids = set(fail_case_ids or set())
        self.log_path = Path(log_path) if log_path else None
        self.delay_seconds = delay_seconds

    def generate(self, test_case: TestCase, *, attempt: int) -> Edits:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.log_path:
            self._log_call(test_case.case_id, attempt)
        if test_case.case_id in self.fail_case_ids:
            raise MockFailure(
                f"mock failure injected for case {test_case.case_id} (attempt {attempt})"
            )
        return _mock_edits(test_case)

    def _log_call(self, case_id: str, attempt: int) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"case_id": case_id, "attempt": attempt}) + "\n")
            handle.flush()


def create_report_generator(
    settings=None, *, mock: bool = False, fail_case_ids=None, log_path=None
) -> ReportGenerator:
    """Factory. ``mock=True`` for network-free runs, otherwise the real Gemini."""
    if mock:
        return MockReportGenerator(fail_case_ids=fail_case_ids, log_path=log_path)
    if settings is None:
        raise ValueError("settings are required to create the real Gemini generator")
    return GeminiReportGenerator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
    )