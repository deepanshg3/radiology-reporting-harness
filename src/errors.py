"""Error taxonomy and retry classification for the harness.

Kept deliberately small: the pipeline and the checkpoint manager both need to
agree on error categories, and a dedicated module avoids circular imports.

Categories
----------
- AUTH / CONFIG  -> permanent, never retried (bad credentials / bad request).
- Everything else (TIMEOUT, RATE_LIMIT, TRANSIENT, INVALID_OUTPUT, GENERIC,
  MOCK) -> retryable. Retrying an invalid LLM output is deliberate: the model
  may produce a better sample on the next attempt.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for expected, recoverable pipeline failures."""


class MockFailure(PipelineError):
    """Injected failure used by the mock generator for local scenario tests."""


class InvalidModelOutput(PipelineError):
    """The model returned something that does not satisfy the contract.

    Raised by the response parser (malformed JSON / unknown keys) and by the
    report validator (unusable final report). Retryable so the model gets a
    chance to fix its output.
    """


class SubmissionError(ValueError):
    """Raised when the final submission would violate the Kaggle contract."""


class TemplateError(ValueError):
    """Raised when a template cannot be parsed into a report structure."""


class ErrorCategory:
    """Stable string categories recorded in the failure log."""

    GENERIC = "GENERIC"
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    TRANSIENT = "TRANSIENT"
    AUTH = "AUTH"
    CONFIG = "CONFIG"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    MOCK = "MOCK"


# Categories that indicate a permanent end-state (do NOT retry).
NON_RETRYABLE_CATEGORIES = frozenset({ErrorCategory.AUTH, ErrorCategory.CONFIG})

# Exception type names from the Google API client libraries that map to
# transient backend conditions. Matched by name so we do not need to import
# google.api_core / google.genai before a client exists.
_TIMEOUT_EXC_NAMES = frozenset({"DeadlineExceeded", "Aborted"})
_RATE_LIMIT_EXC_NAMES = frozenset({"ResourceExhausted", "TooManyRequests", "RateLimitError"})
_TRANSIENT_EXC_NAMES = frozenset(
    {"ServiceUnavailable", "InternalServerError", "GatewayTimeout", "BadGateway", "Unavailable"}
)

# google.genai.errors.(APIError|ClientError|ServerError) all expose ``.code``.
_GOOGLE_GENAI_ERROR_NAMES = frozenset({"APIError", "ClientError", "ServerError"})

# HTTP status -> category mapping for google-genai ``APIError`` objects.
# 429 is a concrete RATE_LIMIT signal; 401/403 credentials; 400/404 bad
# request/model; 5xx transient backend conditions.
_HTTP_CODE_TO_CATEGORY = {
    401: ErrorCategory.AUTH,
    403: ErrorCategory.AUTH,
    400: ErrorCategory.CONFIG,
    404: ErrorCategory.CONFIG,
    429: ErrorCategory.RATE_LIMIT,
    500: ErrorCategory.TRANSIENT,
    502: ErrorCategory.TRANSIENT,
    503: ErrorCategory.TRANSIENT,
    504: ErrorCategory.TRANSIENT,
}


def classify_error(exc: BaseException) -> str:
    """Map an exception to a stable ErrorCategory value."""
    name = type(exc).__name__
    if isinstance(exc, MockFailure):
        return ErrorCategory.MOCK
    if isinstance(exc, InvalidModelOutput):
        return ErrorCategory.INVALID_OUTPUT
    if isinstance(exc, TimeoutError) or name in _TIMEOUT_EXC_NAMES:
        return ErrorCategory.TIMEOUT
    if name in _RATE_LIMIT_EXC_NAMES:
        return ErrorCategory.RATE_LIMIT
    if name in _GOOGLE_GENAI_ERROR_NAMES and hasattr(exc, "code"):
        category = _HTTP_CODE_TO_CATEGORY.get(exc.code)
        if category:
            return category
    if isinstance(exc, (ConnectionError, OSError)) or name in _TRANSIENT_EXC_NAMES:
        return ErrorCategory.TRANSIENT
    return ErrorCategory.GENERIC


def is_retryable(category: str) -> bool:
    """Return True when a failure with this category may be retried."""
    return category not in NON_RETRYABLE_CATEGORIES