from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from .config import EventCorrelatorConfig
from .correlator import EventCorrelator
from .schema import CorrelationBundle, EventCorrelatorInput, FileLineage


class InMemoryCorrelationService:
    def __init__(self, config: EventCorrelatorConfig | None = None):
        self.config = config or EventCorrelatorConfig()
        self.correlator = EventCorrelator(self.config)
        self._snapshots: dict[str, EventCorrelatorInput] = {}
        self._session_to_analysis: dict[str, str] = {}
        self._results: dict[str, CorrelationBundle] = {}

    def submit_correlation(self, payload: EventCorrelatorInput) -> dict[str, object]:
        session_id = str(payload.get("session_id", "") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")

        analysis_id = f"corr_{uuid4().hex}"
        self._snapshots[analysis_id] = deepcopy(payload)
        self._session_to_analysis[session_id] = analysis_id
        return {
            "accepted": True,
            "session_id": session_id,
            "analysis_id": analysis_id,
        }

    def run_correlation(self, analysis_id: str) -> CorrelationBundle:
        if analysis_id not in self._snapshots:
            raise KeyError(f"Unknown analysis_id: {analysis_id}")

        result = self.correlator.run(deepcopy(self._snapshots[analysis_id]))
        self._results[analysis_id] = result
        return result

    def get_file_lineage(self, session_id: str) -> FileLineage:
        analysis_id = self._session_to_analysis.get(session_id)
        if not analysis_id:
            raise KeyError(f"Unknown session_id: {session_id}")

        result = self._results.get(analysis_id)
        if result is None:
            result = self.run_correlation(analysis_id)

        return result["file_lineage"]

    def append_correlation_evidence(self, session_id: str, delta_payload: dict) -> dict[str, object]:
        analysis_id = self._session_to_analysis.get(session_id)
        if not analysis_id:
            raise KeyError(f"Unknown session_id: {session_id}")

        snapshot = deepcopy(self._snapshots[analysis_id])
        snapshot["log_events"] = list(snapshot.get("log_events", []) or []) + list(
            delta_payload.get("log_events", []) or []
        )
        snapshot["frame_segments"] = list(snapshot.get("frame_segments", []) or []) + list(
            delta_payload.get("frame_segments", []) or []
        )
        snapshot["session_metadata"] = {
            **dict(snapshot.get("session_metadata", {}) or {}),
            **dict(delta_payload.get("session_metadata", {}) or {}),
        }

        return self.submit_correlation(snapshot)
