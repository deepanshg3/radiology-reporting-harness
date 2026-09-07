# Radiology Reporting Harness

NLP benchmark solution: transform a radiology **dictation** + supplied
**template** into a final structured **FINDINGS + IMPRESSION** report.

## Core design principle

The model is **not** asked to write a report from scratch. The pipeline is:

```
template_content + dictation
        │
        ▼
 LLM identifies the necessary changes (structured JSON edit instructions)
        │
        ▼
 Python deterministically applies those edits to the original template
        │
        ▼
            final FINDINGS + IMPRESSION report
```

- **LLM**: language understanding only — proposes textual replacements for
  template sections, plus a replacement IMPRESSION. Output is strictly
  machine-readable JSON (`findings_edits` + `impression`), not prose.
- **Python** owns source data, `case_id`, template preservation, applying
  edits, placeholder resolution, output formatting, validation, checkpointing,
  and CSV generation.
- The LLM must never produce `case_id`, row identifiers, or the submission
  structure. `case_id` is always copied verbatim from `row["case_id"]`.

### The structured-edit contract

The model is asked (via `system_prompt.txt`, the few-shot examples, and a
`response_json_schema`) to return exactly one JSON object:

```jsonc
{
  "findings_edits": {           // required; exact template section label -> body text
    "VERTEBRAE": "Mild lower lumbar facet degenerative changes. No acute fracture."
  },
  "impression": "…"             // full replacement IMPRESSION, or null to keep template
}
```

- Keys must **exactly** match the template's section labels (listed for the
  model in the prompt). Unknown keys are rejected (`InvalidModelOutput`).
- On *prose* templates (4 cervical-spine X-ray cases with no section headers)
  the only allowed key is `FINDINGS`, which replaces the whole findings block.
- Placeholders (`[generic]`, `[left/right]`, `[left/right/bilateral]`,
  `[_laterality_]`) are passed through the prompt; as a safety net the editor
  also resolves them from the body part / dictation and *warns* (never errors)
  when unresolvable.
- Deterministic bracket cleanup: whatever survives resolution — an unresolvable
  placeholder or bracketed template prose such as `[ and demonstrate normal
  post-contrast enhancement]` — has its `[`/`]` characters stripped at apply
  time, so a final report never contains brackets. Finalization rejects any
  report that still does (safety net, e.g. for legacy records).

## Dataset contract (verified against the real files)

| File                     | Shape | Columns |
|--------------------------|-------|---------|
| `data/test.csv`          | 132 x 8 | `case_id, modality, body_part, study_description, patient_age_band, patient_sex, template_content, dictation` |
| `data/sample_submission.csv` | 132 x 2 | `case_id, report` |
| `data/train.csv`         | 636 x 9 | (test columns) + `report` — ground truth, **reference only** |

- `test.csv` and `sample_submission.csv` contain the same 132 `case_id`s in the
  same order. No missing values, no duplicate `case_id`s.
- **train.csv is never used at inference time.** No training loop, no
  Gemini evaluation over train, no embeddings or fine-tuning. One Gemini
  request per test case (~132 per full run).

### Template structure discovered

- Every template is a single string containing a `FINDINGS:` block and an
  `IMPRESSION:` block.
- 128 / 132 templates are sectioned: lines of `SECTION: body`. Headers are
  uppercase (`VERTEBRAE:`, `BONES:`) or title case (`Spinal Cord:`,
  `Medial meniscus:`); grouped headers appear bare (`Menisci:`) with their
  children on following lines. Bodies can span multiple lines.
- 4 / 132 templates (cervical spine X-ray) are prose with **no section
  headers**; their FINDINGS block is edited as one unit.
- Stable per modality/body-part section sets (CT abdomen → LIVER, PANCREAS,
  SPLEEN…; spine X-ray → VERTEBRAE, DISC SPACES, SOFT TISSUES…). No duplicate
  section labels within a single template.
- Placeholders appear in some templates and must be resolved by the edit:
  `[generic]` (17), `[left/right]` (11), `[left/right/bilateral]` (3),
  `[_laterality_]` (2).
- Empty `OTHER FINDINGS:` sections are common (28 templates) and are a valid
  target for additions.

### Dictation edge cases to design for

Empty/whitespace dictation, very short dictation (`"normal"`,
`"degen cnhge"`), telegraphic with typos, structured (US reports with measured
values), multi-finding prose, and dictations that already repeat the template
verbatim. Measurements (e.g. `121.3 mL`, `~9.6 mm`, `16 cm`) must be preserved
exactly and never invented — the report validator warns if a measured value from
the dictation is missing from the final report.

## Architecture

```
src/
    config.py            # env config, paths, runtime knobs (no secrets in code)
    data_loader.py       # load + validate test.csv; expose clean TestCase rows
    errors.py            # error taxonomy + retry classification (API code mapping)
    file_utils.py        # crash-safe writes (append-only JSONL, atomic CSV/text)
    checkpoint_manager.py# append-only JSONL checkpoint; latest-record-wins state
    validator.py         # submission contract + per-report quality checks
    gemini_client.py     # ReportGenerator: real Gemini (strict JSON) + mock
    prompt_builder.py    # builds prompts from template + dictation + few-shots
    response_parser.py   # parse/validate the LLM's structured JSON (Edits)
    template_editor.py   # deterministically apply Edits to the template
    pipeline.py          # sequential per-case orchestration, resume planning
run.py                    # CLI entry point -> final submission CSV
tests/
    checkpoint_scenarios.py  # network-free A/B/C/D checkpoint/resume checks
    test_core_modules.py     # unit + e2e checks for the Step-3 core modules
prompts/
    system_prompt.txt        # the radiologist-editor system instruction
    few_shot_examples.json   # curated editor demonstrations (0..5, capped)
evaluation/              # scoring / analysis scripts (reference only)
outputs/                 # generated artifacts (final submission, logs)
checkpoints/             # per-case resumable state (generated, ignored in git)
data/                    # competition CSVs (never modified)
```

## Configuration

Environment variables are read from the process environment or `.env`
(created from `.env.example`, never committed). See `src/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(none — required)* | Gemini credentials, loaded via `require_gemini_api_key()` only when the inference layer starts |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model id used for requests (2.5+ required for `response_json_schema`) |
| `GEMINI_TEMPERATURE` | `0.2` | Sampling temperature for the edit step |
| `GEMINI_MAX_OUTPUT_TOKENS` | `8192` | Max output tokens per request |
| `REQUEST_DELAY_SECONDS` | `2` | Sleep between normal Gemini requests; `0` also disables retry-backoff sleeps (local test mode) |
| `MAX_RETRIES` | `3` | Max attempts per case |
| `RETRY_BACKOFF_BASE_SECONDS` | `2` | Exponential backoff base |

## Checkpointing & incremental processing

The pipeline is **sequential** (one case at a time; no concurrency), with
**immediate, durable persistence after every case**. Progress never lives only
in memory. Each case runs the full generate → parse → apply → validate chain
before a record is written.

### Status model

| Status | Meaning | On resume |
|---|---|---|
| `SUCCESS` | report generated and persisted | skipped — never re-queried |
| `FAILED` | retries exhausted (or permanent error) | retried |
| *(no record)* | never attempted | processed |

There is no `PENDING` state; "pending" is the absence of a record. Successful
records also carry a `warnings` field with non-blocking quality notes (e.g.
unresolvable placeholders, dropped measured values).

### Retry classification

| Category | Example | Retry? |
|---|---|---|
| `AUTH` | API key invalid (401/403) | no |
| `CONFIG` | bad request / model not found (400/404) | no |
| `RATE_LIMIT` | HTTP 429 | yes (backoff) |
| `TIMEOUT` / `TRANSIENT` | 5xx / timeouts / network | yes (backoff) |
| `INVALID_OUTPUT` | unparseable model response, unknown section labels, unusable report | yes (fresh sample) |
| `GENERIC` / `MOCK` | anything else / injected mock failures | yes |

### Checkpoint: `checkpoints/checkpoints.jsonl` (append-only)

- Each attempted case appends one JSON record (case_id, row_index, status,
  report, attempts, latency, error category/message, model, timestamp,
  warnings) that is flushed and fsynced immediately.
- Append-only means an interrupted write leaves at worst a trailing partial
  line, which readers tolerate and skip. There is no full-file rewrite, so a
  crash cannot corrupt prior records.
- The in-memory view is **latest record per case_id**: a later `SUCCESS`
  supersedes an earlier `FAILED`. This makes retries and idempotency natural.
- Resumability: stop after any case (Ctrl-C, crash, power loss) → run again
  with `--resume`; already-successful cases are never called again.

### Working outputs (regenerated from checkpoint state after each case)

| File | Contents |
|---|---|
| `outputs/test_predictions.csv` | `case_id,report` for every SUCCESS so far — usable mid-run, but **not** a valid final submission |
| `outputs/failures.csv` | `case_id,row_index,timestamp,error_category,error_message,attempts` for every outstanding FAILED case; auto-cleans once a case succeeds |
| `outputs/submission.csv` | written **only** when every test case is SUCCESS and the full contract validates |
| `outputs/test_predictions.complete` | marker separated from the working CSV: its presence signals a run that fully finished |

All CSV/text writes go through a temp file + atomic `os.replace`, so a reader
never observes a half-written file.

## CLI

```
# Local, network-free (deterministic mock, exercises the full edit pipeline):
python run.py --mode test --mock --limit 2
python run.py --mode test --mock --limit 2 --resume
python run.py --mode test --mock --limit 2 --fresh

# Real Gemini inference (requires GEMINI_API_KEY in .env):
python run.py --mode test --limit 5                  # first 5 real cases
python run.py --mode test --resume                   # continue incrementally
python run.py --mode test                            # complete the run
python run.py --mode test --export-submission        # validate + write final CSV
```

- `--resume`: skip SUCCESS, retry FAILED, process the rest (up to `--limit N`).
- `--fresh`: wipe the checkpoint + working outputs before starting over
  (used for a clean first real run).
- Running without `--resume`/`--fresh` while a checkpoint exists is **refused**
  (exit 2) to prevent accidental full re-processing (and quota burn).
- `--export-submission` never sends requests; it validates checkpoint state and
  refuses to write a partial/duplicate/empty submission.
- `--mock` runs use the deterministic `MockReportGenerator` — zero API calls,
  zero quota. The mock returns a synthetic `Edits` payload so parser, editor,
  and validator are exercised exactly like a real run.

Local verification (no network, no input-file mutation):

```
.venv/bin/python tests/checkpoint_scenarios.py   # 47 checks (persistence stack)
.venv/bin/python tests/test_core_modules.py      # 71 checks (Step-3 core modules)
```

`checkpoint_scenarios.py` covers: successes persisted; one FAILED recorded with
retry + error metadata; resume skipping successes (proven via a call log) while
retrying the failure; idempotent third run; submission-export rejection until
complete. `test_core_modules.py` covers: template parsing (sectioned, grouped,
prose), surgical edit application, edit rejection, IMPRESSION handling,
placeholder resolution, response-parser strictness, prompt construction +
few-shot capping, report validation, error-code classification, and an
end-to-end mock CLI run with `--fresh`.

## Output contract

Final submission is exactly `case_id,report` with every test `case_id` once,
no extras, no generated ids, `report` containing the complete generated
report. `sample_submission.csv` is the structural contract and is validated
against the produced output before writing.

## Current status

Implemented:
- data foundation (`data_loader`, `config`);
- durable checkpointing / resume / failure log / real-time outputs
  (`errors`, `file_utils`, `checkpoint_manager`, `validator`, `pipeline`,
  `run.py`);
- the structured-edit design: `prompt_builder`, `response_parser`,
  `template_editor`, `prompts/system_prompt.txt`, `prompts/few_shot_examples.json`,
  and the real `GeminiReportGenerator` (strict JSON via `response_json_schema`
  on the google-genai SDK), plus a mock that exercises the same path.
- Verified locally by `tests/checkpoint_scenarios.py` (47) and
  `tests/test_core_modules.py` (71), both fully network-free.

No real Gemini API calls have been made; the first real run is:

```
python run.py --mode test --limit 5 --fresh
```