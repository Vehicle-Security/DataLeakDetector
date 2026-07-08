# DataLeakDetector

DataLeakDetector is a single-package data-leak evidence pipeline for the real
sample layout under `spec/data`.

```text
spec/data case
  -> logs/logs.json or logs/keyevents.json
  -> video/*.mp4
  -> groundtruth.json initial sensitive sources
  -> FrameAnalyzer: log windows + non-uniform keyframes + OCR prefilter + VLM
  -> EventCorrelator: lineage + frontend app + sink candidates
  -> LeakReasoner: symbolic taint propagation
  -> optional Neo4j evidence graph
  -> JSON report
```

Initial sensitive files come from `groundtruth.json` or explicit
`--sensitive-file` arguments. Files created after an operation on a sensitive
file are treated as derived artifacts and are inferred through lineage and
reasoning; they are not inserted into the initial sensitive-file table.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Optional local frame/OCR dependencies:

```powershell
python -m pip install -e ".[dev,vision]"
```

## Run Real Data

Run a case directory from `spec/data`:

```powershell
python main/run_e2e.py --case spec\data\nas_samples\stage1\0-normal-ai-chatgpt-1
```

Enable visual analysis:

```powershell
python main/run_e2e.py --case spec\data\nas_samples\stage2\2-filestruct-pdfconvert-2 --vision --max-vlm-frames 8
```

With `--case`, the runner auto-discovers:

- `logs/logs.json`, falling back to `logs/keyevents.json`
- `video/*.mp4`
- `groundtruth.json`
- initial sensitive files declared by groundtruth

## OCR/VLM Flow

The visual path is disabled by default to avoid model cost. When enabled:

1. Suspicious time windows are mined from logs around groundtruth sensitive files
   and sink/transfer activity.
2. `frames.py` keeps visually changed keyframes instead of uniform sampling.
3. `ocr.py` runs OCR over keyframes.
4. High-confidence OCR becomes local evidence.
5. Low-confidence or suspicious OCR frames are sent to VLM, capped by
   `DLD_MAX_VLM_FRAMES`.
6. `parser.py` normalizes Qwen/OpenAI-compatible JSON responses.
7. Visual observations enter `EventCorrelator` and can produce Datalog facts
   even when logs do not contain the leaked file path.

## Config

Secrets stay in local `.env`, never in the repository.

```text
DLD_VISION_ENABLED=1
DLD_OCR_PROVIDER=tesseract
DLD_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DLD_VLM_MODEL=qwen-vl-max-latest
DLD_VLM_API_KEY=...
DLD_MAX_VLM_FRAMES=8
```

Dataset-specific policy terms can be appended without code changes:

```text
DLD_SENSITIVE_TOKENS=prototype,pricing
DLD_TRANSFER_TOKENS=watermark,print
DLD_SINK_TOKENS=slack,github issue
```

## Neo4j

```powershell
tools\start_neo4j.ps1
python main/run_e2e.py --case spec\data\nas_samples\stage1\0-normal-ai-chatgpt-1 --neo4j --neo4j-strict
tools\stop_neo4j.ps1
```

## Important Files

| Path | Role |
| --- | --- |
| `main/run_e2e.py` | CLI for `--case` and direct `--log` runs. |
| `main/data_leak_detector/datasets.py` | Discovers real `spec/data` sample inputs. |
| `main/data_leak_detector/pipeline.py` | Orchestrates analysis and optional graph writing. |
| `main/data_leak_detector/policy.py` | Default policy vocabulary plus env-based extensions. |
| `main/data_leak_detector/frame_analyzer/*` | Keyframe, OCR, VLM, parser, and app-recognition layer. |
| `main/data_leak_detector/event_correlator/*` | Lineage, candidate, and Datalog fact generation. |
| `main/data_leak_detector/leak_reasoner/*` | Symbolic taint propagation. |
| `tools/smoke_pipeline.py` | Quick smoke test against a real `spec/data` case. |

## Test

```powershell
python -m pytest
```

The tests use generated temporary logs for unit coverage; `spec/data` is the
source of real samples.
