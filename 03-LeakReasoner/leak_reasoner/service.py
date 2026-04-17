from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from .config import LeakReasonerConfig
from .reasoner import LeakReasoner
from .schema import LeakReasonerInput, LeakReasonerOutput


class InMemoryLeakReasonerService:
    def __init__(self, config: LeakReasonerConfig | None = None):
        self.config = config or LeakReasonerConfig()
        self.reasoner = LeakReasoner(self.config)
        self._snapshots: dict[str, LeakReasonerInput] = {}
        self._results: dict[str, LeakReasonerOutput] = {}

    def submit_analysis(self, payload: LeakReasonerInput) -> dict[str, object]:
        session_id = str(payload.get("session_id", "") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")

        analysis_id = f"reason_{uuid4().hex}"
        self._snapshots[analysis_id] = deepcopy(payload)
        return {
            "analysis_id": analysis_id,
            "session_id": session_id,
            "accepted": True,
        }

    def run_analysis(self, analysis_id: str) -> LeakReasonerOutput:
        if analysis_id not in self._snapshots:
            raise KeyError(f"Unknown analysis_id: {analysis_id}")

        result = self.reasoner.run(deepcopy(self._snapshots[analysis_id]))
        self._results[analysis_id] = result
        return result
