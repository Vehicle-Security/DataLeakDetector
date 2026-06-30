from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

FRAME_ANALYZER_IMPL = REPO_ROOT / "01-FrameAnalyzer"
FILE_TRACKER_IMPL = FRAME_ANALYZER_IMPL / "file_tracker"
EVENT_CORRELATOR_IMPL = REPO_ROOT / "02-EventCorrelator"
RISK_HUNTER_IMPL = FRAME_ANALYZER_IMPL / "risk_hunter"
LEAK_REASONER_IMPL = REPO_ROOT / "03-LeakReasoner"


def add_legacy_import_paths(*paths: Path) -> None:
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def add_pipeline_legacy_paths() -> None:
    add_legacy_import_paths(
        FRAME_ANALYZER_IMPL,
        FILE_TRACKER_IMPL,
        RISK_HUNTER_IMPL,
        LEAK_REASONER_IMPL,
    )
