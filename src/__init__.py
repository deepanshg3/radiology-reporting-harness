"""Radiology Reporting Harness — source package.

Responsibilities are separated into independent modules:

- config:             environment config, paths, runtime knobs (no secrets in code).
- data_loader:        load test rows safely; validate the input dataset contract.
- errors:             error taxonomy + retry classification.
- file_utils:         crash-safe writes (append-only JSONL, atomic CSV/text).
- checkpoint_manager: append-only JSONL checkpointing; latest-record-wins state.
- validator:          final-submission contract validation + per-report quality checks.
- gemini_client:      ReportGenerator interface (real Gemini + deterministic mock).
- prompt_builder:     build prompts from template + dictation (+ few-shot examples).
- response_parser:    parse + validate structured LLM output (Edits payload).
- template_editor:    deterministically apply edits to the template.
- pipeline:           sequential per-case orchestration, resume planning, sync.

The pipeline contract: the model proposes structured edits
(``src.response_parser.Edits``); Python (``src.template_editor``) applies them
deterministically to the original template and never lets the model fabricate
``case_id`` or submission structure.
"""