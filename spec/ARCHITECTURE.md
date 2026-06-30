# DataLeakDetector Architecture

The workspace is physically organized around the canonical three-module system:

```text
01-FrameAnalyzer/       video/OCR/VLM observations and frame-analysis adapters
02-EventCorrelator/     logs + frontend windows + file lineage + evidence windows
03-LeakReasoner/        Datalog facts + leak-path reasoning + evidence reports
main/                   E2E entry point and canonical Python package
spec/                   architecture notes, docs, reports, and experiment records
tests/                  regression tests
tools/ScreenMonitor/    log collection tooling
```

The canonical Python package lives under `main/data_leak_detector`:

```text
main/data_leak_detector/
  event_correlator/
  frame_analyzer/
  leak_reasoner/
  legacy_paths.py
```

All new code should import through the canonical package:

```python
from data_leak_detector.event_correlator import EventCorrelator
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.leak_reasoner import DatalogEngine
```

Implementation paths are centralized in `main/data_leak_detector/legacy_paths.py`.
Do not add new hard-coded references to implementation directories in new code.

## Pipeline

```text
logs/video metadata
  -> FrameAnalyzer
       produces frame/VLM behavior observations
  -> EventCorrelator
       binds logs, frontend apps, windows, file lineage, and observations
  -> LeakReasoner
       injects Datalog facts and queries leak paths
  -> evidence report
```

## Workspace Hygiene

Tracked source belongs in:

- `main/`
- `01-FrameAnalyzer/`
- `02-EventCorrelator/`
- `03-LeakReasoner/`
- `tools/`
- `tests/`
- `spec/`

Generated or bulky runtime material belongs outside source review:

- `spec/output/`
- `spec/data/`
- `spec/vlm_debug_frames/`
- `__pycache__/`
