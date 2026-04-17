from __future__ import annotations

from typing import Any


class CorrelationFactBuilder:
    def build(self, correlation_bundle: dict[str, Any]) -> list[dict[str, Any]]:
        correlated_events = correlation_bundle.get("correlated_events", []) or []
        upload_candidates = correlation_bundle.get("upload_candidates", []) or []
        facts: list[dict[str, Any]] = []

        for event in correlated_events:
            facts.append(
                {
                    "fact_type": "correlated_event",
                    "event_type": str(event.get("event_type", "") or ""),
                    "original_file": event.get("original_file", ""),
                    "current_file": event.get("current_file", ""),
                    "app_name": event.get("app_name", ""),
                    "operation_type": event.get("operation_type", ""),
                    "behavior_category": event.get("behavior_category", ""),
                    "evidence_refs": list(event.get("evidence_refs", []) or []),
                    "confidence": float(event.get("confidence", 0.0) or 0.0),
                    "timestamp": event.get("timestamp", ""),
                }
            )

        for candidate in upload_candidates:
            facts.append(
                {
                    "fact_type": "upload_candidate",
                    "original_file": candidate.get("original_file", ""),
                    "current_files": list(candidate.get("current_files", []) or []),
                    "sink_type": candidate.get("sink_type", ""),
                    "app_name": candidate.get("app_name", ""),
                    "mapping_links": list(candidate.get("mapping_links", []) or []),
                    "evidence_refs": list(candidate.get("evidence_refs", []) or []),
                    "confidence": float(candidate.get("confidence", 0.0) or 0.0),
                    "timestamp": candidate.get("timestamp", ""),
                }
            )

        return facts
