"""用于绑定日志、观察、谱系和汇聚点证据的工作流对象。

EventCorrelator 是流水线的中间阶段：它把规范化后的原始活动转换成已关联事件、上传候选项、
谱系记录和 Datalog 事实。其依赖被拆分到兄弟模块中，这样本文件就能保持为可读的工作流，
而不是一个大一统的单体。
"""

from __future__ import annotations

from typing import Any

from ..io import flatten_text, looks_sensitive, normalize_logs, normalize_path, same_file
from ..models import CorrelatedEvent
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any
from .candidates import build_upload_candidates
from .classification import behavior_category, guess_source_by_stem, original_file_from_metadata, operation_from_text
from .config import EventCorrelatorConfig
from .facts import build_datalog_facts
from .lineage import Lineage
from .observations import nearest_observation, normalize_observations
from .output import lineage_payload, operation_record


class EventCorrelator:
    """绑定日志、视觉观察、文件谱系和上传候选项。"""

    def __init__(self, config: EventCorrelatorConfig | None = None):
        self.config = config or EventCorrelatorConfig()

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        logs = normalize_logs([item for item in payload.get("log_events") or [] if isinstance(item, dict)])
        observations = normalize_observations(payload.get("frame_segments") or [])
        sensitive_files = self._collect_sensitive_files(logs, payload.get("sensitive_files") or [])
        lineage = self._build_lineage(logs, sensitive_files)
        correlated = self._correlate(logs, observations, sensitive_files, lineage)
        uploads = build_upload_candidates(correlated, default_confidence=self.config.upload_confidence)
        facts = build_datalog_facts(correlated, uploads, lineage)

        return {
            "session_id": str(payload.get("session_id") or payload.get("record_id") or "session"),
            "analysis_status": "success" if correlated or uploads else "no_match",
            "analysis_windows": self._analysis_windows(logs, sensitive_files),
            "correlated_events": [item.to_dict() for item in correlated],
            "operation_records": [operation_record(item) for item in correlated],
            "upload_candidates": [item.to_dict() for item in uploads],
            "file_lineage": lineage_payload(lineage),
            "datalog_facts": [item.to_dict() for item in facts],
            "statistics": {
                "log_events_input": len(logs),
                "frame_segments_input": len(observations),
                "sensitive_files": len(sensitive_files),
                "correlated_events_output": len(correlated),
                "upload_candidates_output": len(uploads),
                "lineage_direct_mappings": len(lineage.direct),
                "datalog_facts_output": len(facts),
            },
            "errors": [],
        }

    def _collect_sensitive_files(self, logs, explicit: list[Any]) -> list[str]:
        sensitive: list[str] = []
        for item in explicit:
            path = normalize_path(item)
            if path and not any(same_file(path, existing) for existing in sensitive):
                sensitive.append(path)

        source_events = {"file_open", "open", "opened", "read", "file_read", "access", "file_access"}
        for event in logs:
            text = flatten_text(event.raw)
            candidate = original_file_from_metadata(event.raw) or event.file_path
            is_source_event = event.event_type in source_events
            has_explicit_original = bool(original_file_from_metadata(event.raw))
            if candidate and (has_explicit_original or is_source_event) and (looks_sensitive(candidate) or looks_sensitive(text)):
                if not any(same_file(candidate, existing) for existing in sensitive):
                    sensitive.append(normalize_path(candidate))
        return sensitive

    def _build_lineage(self, logs, sensitive_files: list[str]) -> Lineage:
        lineage = Lineage()
        known = list(sensitive_files)
        last_sensitive_by_process: dict[str, str] = {}

        for event in sorted(logs, key=lambda item: item.timestamp_ms):
            process_key = (event.process_name or event.app_name or "").lower()
            original = original_file_from_metadata(event.raw)
            if original:
                lineage.add(event.file_path, original)
                known.append(event.file_path)

            resolved = self._resolve_original(event.file_path, sensitive_files, lineage)
            if resolved and process_key:
                last_sensitive_by_process[process_key] = resolved

            text = flatten_text(event.raw)
            if event.file_path and contains_any(text, TRANSFER_TOKENS):
                source = original or guess_source_by_stem(event.file_path, known) or last_sensitive_by_process.get(process_key, "")
                if source:
                    lineage.add(event.file_path, source)
                    known.append(event.file_path)
        return lineage

    def _correlate(self, logs, observations, sensitive_files: list[str], lineage: Lineage) -> list[CorrelatedEvent]:
        correlated: list[CorrelatedEvent] = []
        recent_sensitive: tuple[str, int] | None = None

        for log in sorted(logs, key=lambda item: item.timestamp_ms):
            original = self._resolve_original(log.file_path, sensitive_files, lineage)
            if original and log.timestamp_ms:
                recent_sensitive = (original, log.timestamp_ms)

            if not original and recent_sensitive and log.timestamp_ms:
                source, timestamp_ms = recent_sensitive
                if 0 <= log.timestamp_ms - timestamp_ms <= self.config.nearby_window_ms:
                    if contains_any(flatten_text(log.raw), SINK_TOKENS):
                        original = source

            if not original:
                continue

            observation = nearest_observation(log.timestamp_ms, observations, self.config.nearby_window_ms)
            text = f"{flatten_text(log.raw)} {observation.description if observation else ''}"
            behavior = behavior_category(text)
            confidence = self.config.upload_confidence if behavior == "data_exfiltration_candidate" else 0.68
            if observation:
                confidence = max(confidence, observation.confidence)

            correlated.append(
                CorrelatedEvent(
                    event_id=f"corr_{len(correlated)}",
                    timestamp=log.timestamp,
                    event_type=log.event_type,
                    app_name=(observation.app_name if observation and observation.app_name else log.app_name or log.process_name),
                    original_file=original,
                    current_file=log.file_path or original,
                    operation_type=(observation.operation_type if observation else operation_from_text(text, log.event_type)),
                    behavior_category=behavior,
                    confidence=round(min(confidence, 1.0), 3),
                    evidence_refs=tuple(
                        [f"log:{log.event_id}"] + ([f"frame:{observation.observation_id}"] if observation else [])
                    ),
                )
            )
        return correlated

    def _resolve_original(self, file_path: str, sensitive_files: list[str], lineage: Lineage) -> str:
        if not file_path:
            return ""
        for sensitive in sensitive_files:
            if same_file(file_path, sensitive):
                return sensitive
        root = lineage.root(file_path)
        for sensitive in sensitive_files:
            if same_file(root, sensitive):
                return sensitive
        return ""

    def _analysis_windows(self, logs, sensitive_files: list[str]) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for sensitive in sensitive_files:
            times = [event.timestamp_ms for event in logs if event.timestamp_ms and same_file(event.file_path, sensitive)]
            if times:
                windows.append(
                    {
                        "sensitive_file": sensitive,
                        "start_ms": min(times) - self.config.nearby_window_ms,
                        "end_ms": max(times) + self.config.nearby_window_ms,
                    }
                )
        return windows
