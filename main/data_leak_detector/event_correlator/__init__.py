from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..io import flatten_text, looks_sensitive, normalize_logs, normalize_path, same_file
from ..models import CorrelatedEvent, DatalogFact, FrameObservation, UploadCandidate


EXTERNAL_SINK_TOKENS = (
    "upload",
    "send",
    "share",
    "mail",
    "email",
    "attach",
    "attachment",
    "cloud",
    "drive",
    "chatgpt",
    "claude",
    "gemini",
    "kimi",
    "wechat",
    "qq",
    "feishu",
    "lark",
    "dingtalk",
    "teams",
    "zoom",
    "上传",
    "发送",
    "分享",
    "外发",
    "附件",
    "邮箱",
    "邮件",
    "云盘",
    "网盘",
    "共享",
)

TRANSFER_TOKENS = (
    "created",
    "modified",
    "rename",
    "copy",
    "paste",
    "compress",
    "convert",
    "export",
    "split",
    "clipboard",
    "创建",
    "修改",
    "重命名",
    "复制",
    "粘贴",
    "压缩",
    "转换",
    "导出",
    "剪贴板",
)

TRUSTED_LOCAL_TOKENS = ("excel", "word", "wps", "explorer", "finder", "notepad")


@dataclass(frozen=True)
class EventCorrelatorConfig:
    """Knobs kept small because scoring rules are expected to change."""

    nearby_window_ms: int = 5 * 60 * 1000
    upload_confidence: float = 0.85
    transfer_confidence: float = 0.7
    trusted_local_apps: tuple[str, ...] = TRUSTED_LOCAL_TOKENS
    external_sink_tokens: tuple[str, ...] = EXTERNAL_SINK_TOKENS
    transfer_tokens: tuple[str, ...] = TRANSFER_TOKENS

    def as_dict(self) -> dict[str, Any]:
        return {
            "nearby_window_ms": self.nearby_window_ms,
            "upload_confidence": self.upload_confidence,
            "transfer_confidence": self.transfer_confidence,
            "trusted_local_apps": list(self.trusted_local_apps),
            "external_sink_tokens": list(self.external_sink_tokens),
            "transfer_tokens": list(self.transfer_tokens),
        }


@dataclass
class Lineage:
    """Small source-to-derived map used before symbolic reasoning."""

    direct: dict[str, str] = field(default_factory=dict)

    def add(self, derived: str, source: str) -> None:
        derived = normalize_path(derived)
        source = normalize_path(source)
        if derived and source and not same_file(derived, source):
            self.direct[derived] = source

    def root(self, file_path: str) -> str:
        current = normalize_path(file_path)
        seen: set[str] = set()
        while current and current in self.direct and current not in seen:
            seen.add(current)
            current = self.direct[current]
        return current

    def chain(self, file_path: str) -> str:
        current = normalize_path(file_path)
        parts = [current] if current else []
        seen: set[str] = set()
        while current and current in self.direct and current not in seen:
            seen.add(current)
            current = self.direct[current]
            parts.append(current)
        return " <- ".join(parts)


class EventCorrelator:
    """Bind logs, visual observations, file lineage, and external sink events."""

    def __init__(self, config: EventCorrelatorConfig | None = None):
        self.config = config or EventCorrelatorConfig()

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = payload.get("log_events") or []
        logs = normalize_logs([item for item in records if isinstance(item, dict)])
        observations = _normalize_observations(payload.get("frame_segments") or [])
        explicit_sensitive = [normalize_path(item) for item in payload.get("sensitive_files") or []]
        sensitive_files = self._collect_sensitive_files(logs, explicit_sensitive)
        lineage = self._build_lineage(logs, sensitive_files)

        correlated_events = self._correlate(logs, observations, sensitive_files, lineage)
        upload_candidates = self._build_upload_candidates(correlated_events, lineage)
        datalog_facts = self._facts_from_events(correlated_events, upload_candidates, lineage)

        return {
            "session_id": str(payload.get("session_id") or payload.get("record_id") or "session"),
            "analysis_status": "success" if correlated_events or upload_candidates else "no_match",
            "analysis_windows": self._analysis_windows(logs, sensitive_files),
            "correlated_events": [item.to_dict() for item in correlated_events],
            "operation_records": [_operation_record(item) for item in correlated_events],
            "upload_candidates": [item.to_dict() for item in upload_candidates],
            "file_lineage": {
                "direct_file_mappings": dict(lineage.direct),
                "full_file_mapping_chains": {
                    file_path: lineage.chain(file_path) for file_path in lineage.direct
                },
                "artifact_instances": [
                    {"artifact_id": Path(dst).name, "current_file": dst, "source_file": src}
                    for dst, src in lineage.direct.items()
                ],
            },
            "datalog_facts": [item.to_dict() for item in datalog_facts],
            "statistics": {
                "log_events_input": len(logs),
                "frame_segments_input": len(observations),
                "sensitive_files": len(sensitive_files),
                "correlated_events_output": len(correlated_events),
                "upload_candidates_output": len(upload_candidates),
                "lineage_direct_mappings": len(lineage.direct),
            },
            "errors": [],
        }

    def _collect_sensitive_files(self, logs, explicit_sensitive: list[str]) -> list[str]:
        sensitive = list(dict.fromkeys(item for item in explicit_sensitive if item))
        for event in logs:
            if event.file_path and (looks_sensitive(event.file_path) or looks_sensitive(event.window_title)):
                if not any(same_file(event.file_path, existing) for existing in sensitive):
                    sensitive.append(event.file_path)
        return sensitive

    def _build_lineage(self, logs, sensitive_files: list[str]) -> Lineage:
        lineage = Lineage()
        known = list(sensitive_files)
        for event in sorted(logs, key=lambda item: item.timestamp_ms):
            raw = event.raw
            source = normalize_path(raw.get("source_file") or raw.get("original_file") or "")
            if not source:
                source = self._guess_source_from_name(event.file_path, known)
            if source and event.file_path:
                lineage.add(event.file_path, source)
                if event.file_path not in known:
                    known.append(event.file_path)
        return lineage

    def _guess_source_from_name(self, file_path: str, known_files: list[str]) -> str:
        name = Path(normalize_path(file_path)).stem.lower()
        if not name:
            return ""
        for known in known_files:
            known_stem = Path(normalize_path(known)).stem.lower()
            if known_stem and (name.startswith(known_stem) or known_stem.startswith(name)):
                return known
        return ""

    def _correlate(self, logs, observations, sensitive_files: list[str], lineage: Lineage) -> list[CorrelatedEvent]:
        events: list[CorrelatedEvent] = []
        for log in logs:
            original = self._resolve_original(log.file_path, sensitive_files, lineage)
            if not original:
                continue
            observation = _nearest_observation(log.timestamp_ms, observations, self.config.nearby_window_ms)
            operation = observation.operation_type if observation else log.event_type
            app_name = observation.app_name if observation and observation.app_name else log.app_name
            confidence = max(0.6, observation.confidence if observation else 0.0)
            events.append(
                CorrelatedEvent(
                    event_id=f"corr_{len(events)}",
                    timestamp=log.timestamp,
                    event_type=log.event_type,
                    original_file=original,
                    current_file=log.file_path or original,
                    app_name=app_name or log.process_name,
                    operation_type=operation,
                    behavior_category=self._behavior_category(log, operation),
                    confidence=round(min(confidence, 1.0), 3),
                    evidence_refs=[f"log:{log.event_id}"]
                    + ([f"frame:{observation.observation_id}"] if observation else []),
                    status="linked",
                )
            )
        return events

    def _resolve_original(self, file_path: str, sensitive_files: list[str], lineage: Lineage) -> str:
        if not file_path:
            return ""
        for sensitive_file in sensitive_files:
            if same_file(file_path, sensitive_file):
                return sensitive_file
        root = lineage.root(file_path)
        for sensitive_file in sensitive_files:
            if same_file(root, sensitive_file):
                return sensitive_file
        return ""

    def _behavior_category(self, log, operation: str) -> str:
        text = f"{log.event_type} {operation} {log.app_name} {log.window_title}".lower()
        if _contains_any(text, self.config.external_sink_tokens):
            return "data_exfiltration_candidate"
        if _contains_any(text, self.config.transfer_tokens):
            return "hidden_transformation_candidate"
        return "sensitive_access"

    def _build_upload_candidates(self, correlated_events: list[CorrelatedEvent], lineage: Lineage) -> list[UploadCandidate]:
        candidates: list[UploadCandidate] = []
        for event in correlated_events:
            text = f"{event.event_type} {event.operation_type} {event.app_name}".lower()
            if not _contains_any(text, self.config.external_sink_tokens):
                continue
            candidates.append(
                UploadCandidate(
                    candidate_id=f"upload_{len(candidates)}",
                    timestamp=event.timestamp,
                    original_file=event.original_file,
                    current_files=[event.current_file or event.original_file],
                    app_name=event.app_name,
                    operation_type=event.operation_type,
                    sink_type=_sink_type(text),
                    confidence=max(event.confidence, self.config.upload_confidence),
                    evidence_refs=list(event.evidence_refs),
                    status="candidate",
                )
            )
        return candidates

    def _facts_from_events(
        self,
        correlated_events: list[CorrelatedEvent],
        upload_candidates: list[UploadCandidate],
        lineage: Lineage,
    ) -> list[DatalogFact]:
        facts: list[DatalogFact] = []
        opened: set[tuple[str, str]] = set()
        transferred: set[tuple[str, str]] = set()

        for event in correlated_events:
            proc = event.app_name or "unknown"
            source_key = (proc, event.original_file)
            if source_key not in opened:
                facts.append(DatalogFact("OpenFile", (f"{event.event_id}:open", proc, event.original_file, 0)))
                opened.add(source_key)

            if not same_file(event.original_file, event.current_file):
                transfer_key = (event.original_file, event.current_file)
                if transfer_key not in transferred:
                    facts.append(
                        DatalogFact(
                            "TransferFile",
                            (f"{event.event_id}:transfer", proc, event.original_file, event.current_file, 0),
                        )
                    )
                    transferred.add(transfer_key)

        for dst, src in lineage.direct.items():
            transfer_key = (src, dst)
            if transfer_key not in transferred:
                facts.append(DatalogFact("TransferFile", (f"lineage:{len(facts)}", "system", src, dst, 0)))
                transferred.add(transfer_key)

        for candidate in upload_candidates:
            proc = candidate.app_name or "unknown"
            for file_path in candidate.current_files:
                facts.append(
                    DatalogFact(
                        "LeakFile",
                        (f"{candidate.candidate_id}:leak", proc, file_path, candidate.sink_type, 0),
                    )
                )
                if not same_file(candidate.original_file, file_path):
                    facts.append(
                        DatalogFact(
                            "CrossProcessTransfer",
                            (f"{candidate.candidate_id}:bind", "system", proc, file_path, 0),
                        )
                    )
        return facts

    def _analysis_windows(self, logs, sensitive_files: list[str]) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for sensitive_file in sensitive_files:
            timestamps = [
                event.timestamp_ms
                for event in logs
                if event.timestamp_ms and same_file(event.file_path, sensitive_file)
            ]
            if timestamps:
                windows.append(
                    {
                        "sensitive_file": sensitive_file,
                        "start_ms": min(timestamps) - self.config.nearby_window_ms,
                        "end_ms": max(timestamps) + self.config.nearby_window_ms,
                    }
                )
        return windows


def classify_frontend_app(app_name: str, window_title: str = "") -> str:
    """Classify an app into a coarse policy bucket."""

    text = f"{app_name} {window_title}".lower()
    if _contains_any(text, EXTERNAL_SINK_TOKENS):
        return "external_sink"
    if _contains_any(text, TRUSTED_LOCAL_TOKENS):
        return "trusted_local"
    return "unknown"


def _normalize_observations(items: list[Any]) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    for index, item in enumerate(items):
        if isinstance(item, FrameObservation):
            observations.append(item)
            continue
        if not isinstance(item, dict):
            continue
        observations.append(
            FrameObservation(
                observation_id=str(item.get("observation_id") or item.get("segment_id") or f"obs_{index}"),
                start_ms=int(item.get("start_ms") or 0),
                end_ms=int(item.get("end_ms") or item.get("start_ms") or 0),
                app_name=str(item.get("app_name") or ""),
                operation_type=str(item.get("operation_type") or item.get("operation") or ""),
                resource=normalize_path(item.get("resource") or item.get("primary_resource") or ""),
                related_resources=[normalize_path(value) for value in item.get("related_resources") or []],
                description=str(item.get("description") or item.get("action_description") or ""),
                confidence=float(item.get("confidence") or 0.0),
                source=str(item.get("source") or "frame_analyzer"),
            )
        )
    return observations


def _nearest_observation(timestamp_ms: int, observations: list[FrameObservation], tolerance_ms: int) -> FrameObservation | None:
    if not timestamp_ms:
        return None
    best: tuple[int, FrameObservation] | None = None
    for observation in observations:
        center = observation.start_ms if not observation.end_ms else (observation.start_ms + observation.end_ms) // 2
        distance = abs(timestamp_ms - center)
        if distance <= tolerance_ms and (best is None or distance < best[0]):
            best = (distance, observation)
    return best[1] if best else None


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token.lower() in text for token in tokens)


def _sink_type(text: str) -> str:
    if any(token in text for token in ("mail", "email", "邮箱", "邮件", "attachment", "附件")):
        return "mail_attachment"
    if any(token in text for token in ("share", "screen", "meeting", "共享", "屏幕", "会议")):
        return "screen_share"
    if any(token in text for token in ("cloud", "drive", "云盘", "网盘")):
        return "cloud_sync"
    if any(token in text for token in ("chat", "qq", "wechat", "feishu", "lark", "微信", "飞书")):
        return "chat_upload"
    return "web_upload"


def _operation_record(event: CorrelatedEvent) -> dict[str, Any]:
    return {
        "operation_time": event.timestamp,
        "sensitive_file_path": event.original_file,
        "current_file": event.current_file,
        "app_name": event.app_name,
        "operation": event.operation_type,
        "behavior_category": event.behavior_category,
        "evidence_refs": list(event.evidence_refs),
        "status": event.status,
    }


__all__ = ["EventCorrelator", "EventCorrelatorConfig", "classify_frontend_app"]
