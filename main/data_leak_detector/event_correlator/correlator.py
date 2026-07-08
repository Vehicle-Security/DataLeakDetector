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
from .classification import behavior_category, guess_source_by_stem, original_file_from_metadata, operation_from_text, target_file_from_metadata
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

        if not self.config.infer_sensitive_from_logs:
            return sensitive

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
            target = target_file_from_metadata(event.raw) or event.file_path
            if original and self._resolve_original(original, sensitive_files, lineage):
                lineage.add(target, original)
                known.append(target)

            resolved = self._resolve_original(event.file_path, sensitive_files, lineage)
            if resolved and process_key:
                last_sensitive_by_process[process_key] = resolved

            text = flatten_text(event.raw)
            if event.file_path and contains_any(text, TRANSFER_TOKENS):
                source = original or guess_source_by_stem(target, known) or last_sensitive_by_process.get(process_key, "")
                if source and not self._resolve_original(source, sensitive_files, lineage):
                    source = ""
                if source:
                    lineage.add(target, source)
                    known.append(target)
        return lineage

    def _correlate(self, logs, observations, sensitive_files: list[str], lineage: Lineage) -> list[CorrelatedEvent]:
        correlated: list[CorrelatedEvent] = []
        recent_sensitive: tuple[str, int] | None = None
        observation_time_mode = self._observation_time_mode(observations)

        for log in sorted(logs, key=lambda item: item.timestamp_ms):
            original = self._resolve_original(log.file_path, sensitive_files, lineage)
            observation = nearest_observation(self._log_observation_time(log, observation_time_mode), observations, self.config.nearby_window_ms)
            if original and log.timestamp_ms:
                recent_sensitive = (original, log.timestamp_ms)

            if not original and recent_sensitive and log.timestamp_ms:
                source, timestamp_ms = recent_sensitive
                if 0 <= log.timestamp_ms - timestamp_ms <= self.config.nearby_window_ms:
                    nearby_text = f"{flatten_text(log.raw)} {observation.description if observation else ''} {observation.operation_type if observation else ''}"
                    if contains_any(nearby_text, SINK_TOKENS) or contains_any(nearby_text, TRANSFER_TOKENS):
                        original = source

            if not original and observation:
                original = self._resolve_observation_original(observation, sensitive_files, lineage)
            if not original:
                continue
            text = " ".join(
                [
                    flatten_text(log.raw),
                    observation.description if observation else "",
                    observation.operation_type if observation else "",
                    observation.resource if observation else "",
                    " ".join(observation.related_resources) if observation else "",
                ]
            )
            behavior = behavior_category(text)
            confidence = self.config.upload_confidence if behavior == "data_exfiltration_candidate" else 0.68
            if observation:
                confidence = max(confidence, observation.confidence)
            current_file = target_file_from_metadata(log.raw) or log.file_path or (observation.resource if observation and observation.resource else original)

            correlated.append(
                CorrelatedEvent(
                    event_id=f"corr_{len(correlated)}",
                    timestamp=log.timestamp,
                    event_type=log.event_type,
                    app_name=(observation.app_name if observation and observation.app_name else log.app_name or log.process_name),
                    original_file=original,
                    current_file=current_file,
                    operation_type=(observation.operation_type if observation else operation_from_text(text, log.event_type)),
                    behavior_category=behavior,
                    confidence=round(min(confidence, 1.0), 3),
                    evidence_refs=tuple(
                        [f"log:{log.event_id}"] + ([f"frame:{observation.observation_id}"] if observation else [])
                    ),
                )
            )
        correlated.extend(self._correlate_visual_only(observations, sensitive_files, lineage, start_index=len(correlated)))
        return correlated

    def _observation_time_mode(self, observations) -> str:
        # OCR/keyframe evidence uses video-relative milliseconds. Some imported
        # or test VLM observations may already contain absolute epoch millis.
        return "absolute" if any(item.start_ms > 10_000_000_000 for item in observations) else "video"

    def _log_observation_time(self, log, mode: str) -> int:
        if mode == "absolute":
            return log.timestamp_ms
        return log.video_time_ms if log.video_time_ms >= 0 else log.timestamp_ms

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

    def _resolve_observation_original(self, observation, sensitive_files: list[str], lineage: Lineage) -> str:
        for candidate in [observation.resource, *observation.related_resources]:
            resolved = self._resolve_original(candidate, sensitive_files, lineage)
            if resolved:
                return resolved
        description = observation.description.lower()
        for sensitive in sensitive_files:
            if sensitive and (sensitive.lower() in description or looks_sensitive(description)):
                return sensitive
        return ""

    def _correlate_visual_only(self, observations, sensitive_files: list[str], lineage: Lineage, start_index: int) -> list[CorrelatedEvent]:
        visual_events: list[CorrelatedEvent] = []
        for observation in observations:
            if observation.source == "log_anchored":
                continue
            original = self._resolve_observation_original(observation, sensitive_files, lineage)
            if not original:
                continue
            text = f"{observation.description} {observation.operation_type} {observation.resource} {' '.join(observation.related_resources)}"
            if not (contains_any(text, SINK_TOKENS) or contains_any(text, TRANSFER_TOKENS)):
                continue
            behavior = behavior_category(text)
            current_file = observation.resource or original
            visual_events.append(
                CorrelatedEvent(
                    event_id=f"corr_{start_index + len(visual_events)}",
                    timestamp="",
                    event_type="visual_observation",
                    app_name=observation.app_name,
                    original_file=original,
                    current_file=current_file,
                    operation_type=observation.operation_type,
                    behavior_category=behavior,
                    confidence=round(min(max(observation.confidence, 0.70), 1.0), 3),
                    evidence_refs=(f"frame:{observation.observation_id}",),
                )
            )
        return visual_events

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
