# DataLeakDetector Project Spec

DataLeakDetector is a local data-leak evidence pipeline. It combines monitor
logs, optional visual observations, deterministic correlation rules, symbolic
taint reasoning, and an optional Neo4j graph store.

## What This Directory Contains

```text
spec/
  ARCHITECTURE.md       system design and data flow
  config/               policy and noise-profile configuration
  fixtures/             small, stable examples for tests and demos
  data/                 large local datasets; not edited by code cleanup
  data.zip              dataset archive; not edited by code cleanup
```

## Core Idea

The system treats data leakage as an evidence-chain problem:

1. A sensitive file or data object is observed.
2. The object may be transformed, copied, renamed, exported, compressed, pasted,
   or moved across applications.
3. A sink action may expose the object through upload, mail, chat, cloud sync,
   removable media, screen sharing, or another channel.
4. The reasoner searches for a connected path from source to sink.

Neo4j is used as an optional graph view of the evidence, not as a hard runtime
dependency for detection.

## JSON Files Outside `spec/data`

JSON files here are small, human-reviewable contracts. They are kept outside
`spec/data` because tests, smoke runs, and future parser work need stable sample
inputs that are easy to diff.

| File | Role | Why it is kept | Used by |
| --- | --- | --- | --- |
| `spec/config/system_noise_profile.json` | Reference profile for benign system activity and common noise sources. | Keeps noise assumptions visible instead of burying them in code; useful when tuning `policy.py` and correlation filters. | Architecture docs and future correlation tuning. |
| `spec/fixtures/sample_leak.json` | Minimal end-to-end monitor log with a sensitive file, derived artifact, and upload. | Provides the smallest runnable fixture for CLI, smoke tests, and Neo4j verification. | `main/run_e2e.py`, `tools/smoke_pipeline.py`, manual Neo4j checks. |
| `spec/fixtures/realistic_log_cases.json` | Scenario collection with realistic logs and expected sensitive-file context. | Preserves representative product behavior beyond the tiny smoke fixture. | Regression expansion for `tests/test_pipeline.py`. |
| `spec/fixtures/qwen_vlm_response_cases.json` | Sample VLM response payloads, including fenced JSON and repeated events. | Documents the expected shape for future OCR/VLM parsing without requiring a live model now. | Future FrameAnalyzer parser tests. |
| `spec/fixtures/currently_unrecognized_violation_cases.json` | Known cases that the current deterministic rules may miss. | Keeps blind spots explicit as product requirements instead of change-history notes. | Future parser/correlation improvements and regression tests. |

These fixture files should remain data-only. If a fixture needs explanation,
update this table or `spec/ARCHITECTURE.md` rather than adding non-log records
that would change test input semantics.
