"""Environment configuration for the radiology-reporting-harness project.

All credentials are read from the process environment or the local ``.env``
file. Secrets are never committed and never hard-coded in source. Importing
this module is side-effect free with respect to Gemini: nothing here connects
to any external service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load optional local .env (absent/ignored in git). Precedence: real env wins.
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(DOTENV_PATH)

# --- Data paths -------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# --- Output paths -----------------------------------------------------------
PROMPTS_DIR = PROJECT_ROOT / "prompts"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
FEW_SHOT_EXAMPLES_PATH = PROMPTS_DIR / "few_shot_examples.json"

# --- Incremental-processing artifacts (never hard-code these elsewhere) ------
# Append-only JSONL checkpoint (source of truth for resume). One record per
# attempted case; the latest record for a case_id wins.
CHECKPOINT_FILE = "checkpoints.jsonl"
CHECKPOINT_PATH = CHECKPOINTS_DIR / CHECKPOINT_FILE

# Working output CSVs, regenerated from checkpoint state after each case.
PREDICTIONS_CSV = OUTPUTS_DIR / "test_predictions.csv"
FAILURES_CSV = OUTPUTS_DIR / "failures.csv"

# Final submission (written only when every case has SUCCEEDED).
SUBMISSION_CSV = OUTPUTS_DIR / "submission.csv"
# Marker that appears only when a run has processed every test case to SUCCESS.
COMPLETE_MARKER = OUTPUTS_DIR / "test_predictions.complete"


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Credentials are intentionally not defaulted."""

    gemini_api_key: str
    gemini_model: str
    gemini_temperature: float
    gemini_max_output_tokens: int
    request_delay_seconds: float
    max_retries: int
    retry_backoff_base_seconds: float


def _env_float(env: dict, name: str, default: float, minimum: float) -> float:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_int(env: dict, name: str, default: int, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def load_settings(env: dict | None = None) -> Settings:
    """Build a Settings object from environment variables (or ``.env``)."""
    env = dict(os.environ if env is None else env)
    return Settings(
        gemini_api_key=str(env.get("GEMINI_API_KEY", "")).strip(),
        gemini_model=str(
            env.get("GEMINI_MODEL") or "gemini-2.5-flash"
        ).strip(),
        gemini_temperature=_env_float(env, "GEMINI_TEMPERATURE", 0.2, 0.0),
        gemini_max_output_tokens=_env_int(
            env, "GEMINI_MAX_OUTPUT_TOKENS", 8192, 1
        ),
        request_delay_seconds=_env_float(env, "REQUEST_DELAY_SECONDS", 2.0, 0.0),
        max_retries=_env_int(env, "MAX_RETRIES", 3, 1),
        retry_backoff_base_seconds=_env_float(
            env, "RETRY_BACKOFF_BASE_SECONDS", 2.0, 0.0
        ),
    )


def require_gemini_api_key(settings: Settings) -> str:
    """Return the API key, raising a clear error if it is not configured.

    Called only by the inference layer before the first Gemini request, so
    data loading / validation never depends on credentials.
    """
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a .env file from .env.example "
            "and add your key."
        )
    return settings.gemini_api_key