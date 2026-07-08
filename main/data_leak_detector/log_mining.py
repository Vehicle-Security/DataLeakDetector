"""Log mining entry point for frame analysis windows.

The in-memory miner keeps local development fast. When Neo4j log mining is
enabled, this module delegates to the Neo4j-backed implementation while keeping
the fallback contract in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .frame_analyzer.config import VisionConfig
from .frame_analyzer.frames import AnalysisWindow, merge_analysis_windows
from .io import flatten_text, looks_sensitive, normalize_path
from .models import LogEvent
from .neo4j.config import Neo4jConfig
from .neo4j.importer import Neo4jLogImporter
from .neo4j.queries import Neo4jLogQueries
from .policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any


@dataclass(frozen=True)
class LogMiningResult:
    windows: list[AnalysisWindow]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def mine_analysis_windows(
    *,
    case_id: str,
    log_file: str | Path,
    records: list[dict[str, Any]],
    logs: list[LogEvent],
    sensitive_files: list[str],
    vision_config: VisionConfig,
    neo4j_log_miner: bool | None = None,
    reuse_import: bool | None = None,
) -> LogMiningResult:
    config = Neo4jConfig.from_env()
    enabled = config.log_miner_enabled if neo4j_log_miner is None else neo4j_log_miner
    if not enabled:
        return InMemoryLogMiner().mine(logs=logs, sensitive_files=sensitive_files, vision_config=vision_config)

    if reuse_import is not None:
        config = config.with_overrides(reuse_import=reuse_import)

    try:
        return Neo4jLogMiner(config).mine(
            case_id=case_id,
            log_file=log_file,
            records=records,
            logs=logs,
            sensitive_files=sensitive_files,
            vision_config=vision_config,
        )
    except Exception as exc:
        if config.log_miner_strict:
            raise
        fallback = InMemoryLogMiner().mine(logs=logs, sensitive_files=sensitive_files, vision_config=vision_config)
        metadata = dict(fallback.metadata)
        metadata.update(
            {
                "neo4j_enabled": True,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return LogMiningResult(windows=fallback.windows, source="in_memory_fallback", metadata=metadata)


class InMemoryLogMiner:
    def mine(
        self,
        *,
        logs: list[LogEvent],
        sensitive_files: list[str],
        vision_config: VisionConfig,
    ) -> LogMiningResult:
        windows = build_analysis_windows(logs, sensitive_files, vision_config)
        return LogMiningResult(
            windows=windows,
            source="in_memory",
            metadata={"status": "ready", "windows": len(windows), "neo4j_enabled": False},
        )


class Neo4jLogMiner:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.importer = Neo4jLogImporter(config)
        self.queries = Neo4jLogQueries()

    def mine(
        self,
        *,
        case_id: str,
        log_file: str | Path,
        records: list[dict[str, Any]],
        logs: list[LogEvent],
        sensitive_files: list[str],
        vision_config: VisionConfig,
    ) -> LogMiningResult:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            connection_timeout=2.0,
        )
        try:
            with driver.session(database=self.config.database) as session:
                summary = self.importer.ensure_import(
                    session,
                    case_id=case_id,
                    log_file=log_file,
                    records=records,
                    logs=logs,
                    sensitive_files=sensitive_files,
                )
                candidate_ids = self.queries.candidate_event_ids(session, case_id)
                app_context = self.queries.active_apps_for_events(
                    session,
                    case_id,
                    candidate_ids,
                    max(vision_config.frame_window_after_ms, vision_config.strong_window_after_ms),
                )
        finally:
            driver.close()

        windows = self._windows_from_candidates(
            logs=logs,
            sensitive_files=sensitive_files,
            vision_config=vision_config,
            candidate_ids=candidate_ids,
            app_context=app_context,
        )
        if not windows:
            windows = build_analysis_windows(logs, sensitive_files, vision_config)

        merged = merge_analysis_windows(windows)
        return LogMiningResult(
            windows=merged,
            source="neo4j",
            metadata={
                "status": "ready",
                "neo4j_enabled": True,
                "case_id": case_id,
                "log_hash": summary.log_hash,
                "records": len(records),
                "schema_version": self.config.log_miner_schema_version,
                "batch_size": self.config.log_miner_batch_size,
                "imported": summary.imported,
                "reused_import": summary.reused,
                "imported_events": summary.imported_events,
                "import_batches": summary.import_batches,
                "candidate_events": len(candidate_ids),
                "windows": len(merged),
            },
        )

    @staticmethod
    def _windows_from_candidates(
        *,
        logs: list[LogEvent],
        sensitive_files: list[str],
        vision_config: VisionConfig,
        candidate_ids: list[str],
        app_context: dict[str, tuple[str, ...]],
    ) -> list[AnalysisWindow]:
        events_by_id = {event.event_id: event for event in logs}
        windows: list[AnalysisWindow] = []
        for event_id in candidate_ids:
            event = events_by_id.get(event_id)
            if event is None:
                continue
            window = build_analysis_window_for_event(
                event,
                logs,
                sensitive_files,
                vision_config,
                active_apps=tuple(app_context.get(event_id, ())),
            )
            if window is not None:
                windows.append(window)
        return windows


def build_analysis_windows(
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
) -> list[AnalysisWindow]:
    windows: list[AnalysisWindow] = []
    for event in logs:
        window = build_analysis_window_for_event(event, logs, sensitive_files, config)
        if window is not None:
            windows.append(window)
    return merge_analysis_windows(windows)


def build_analysis_window_for_event(
    event: LogEvent,
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
    *,
    active_apps: tuple[str, ...] | None = None,
) -> AnalysisWindow | None:
    """Convert one suspicious log event into a video analysis window."""

    sensitive = tuple(normalize_path(item).lower() for item in sensitive_files)
    text = flatten_text(event.raw)
    file_text = normalize_path(event.file_path).lower()
    sensitive_hit = any(item and item in file_text for item in sensitive) or looks_sensitive(file_text) or looks_sensitive(text)
    action_hit = contains_any(text, TRANSFER_TOKENS) or contains_any(text, SINK_TOKENS)
    priority = _window_priority(event, text, sensitive_hit, action_hit)
    if priority == "none" or event.video_time_ms < 0:
        return None

    before_ms, after_ms, step_ms, max_keyframes, diff_threshold = _window_profile(priority, config)
    return AnalysisWindow(
        start_ms=max(event.video_time_ms - before_ms, 0),
        end_ms=event.video_time_ms + after_ms,
        reason=_window_reason(event, priority),
        priority=priority,
        step_ms=step_ms,
        max_keyframes=max_keyframes,
        diff_threshold=diff_threshold,
        anchor_ms=_event_anchors(event, sensitive_hit),
        active_apps=active_apps if active_apps is not None else _active_apps_near(logs, event.video_time_ms, after_ms),
    )


def _active_apps_near(logs: list[LogEvent], center_ms: int, radius_ms: int) -> tuple[str, ...]:
    apps: list[str] = []
    for event in logs:
        if event.video_time_ms < 0 or abs(event.video_time_ms - center_ms) > radius_ms:
            continue
        app = event.app_name or event.process_name
        if app and app not in apps:
            apps.append(app)
    return tuple(apps)


def _window_priority(event: LogEvent, text: str, sensitive_hit: bool, action_hit: bool) -> str:
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    upload = event.raw.get("upload_detection") if isinstance(event.raw.get("upload_detection"), dict) else {}
    event_type = event.event_type.lower()
    category = str(extra.get("category") or "")
    raw_operation = str(extra.get("raw_operation") or "").lower()
    operation_detail = str(extra.get("operation_detail") or "")
    upload_type = str(upload.get("upload_type") or "")
    upload_status = str(upload.get("upload_status") or "").lower()
    process_name = (event.process_name or "").lower()
    window_title = event.window_title or ""

    strong_event_types = {"file_selected", "file_upload", "upload", "uploaded", "upload_complete"}
    strong_raw_ops = {"file_selected", "file_upload", "upload", "send_click"}
    if event_type in strong_event_types or raw_operation in strong_raw_ops:
        return "strong"
    if process_name == "fsquirt.exe" or "蓝牙文件传送" in window_title:
        return "strong"
    if process_name == "fsquirt.exe" and window_title == "浏览":
        return "strong"
    if any(token in category for token in ("文件上传", "直接外发")):
        return "strong"
    if any(token in operation_detail for token in ("附件", "文件选择", "已发送", "外发", "上传")):
        return "strong"
    if upload_status in {"success", "completed", "complete"} or ("upload" in upload_type.lower() and "download" not in upload_type.lower()):
        return "strong"

    if sensitive_hit or action_hit:
        return "medium"

    if str(extra.get("risk_level") or "").lower() in {"高", "high"}:
        return "weak"
    return "none"


def _event_anchors(event: LogEvent, sensitive_hit: bool) -> tuple[int, ...]:
    process_name = (event.process_name or "").lower()
    window_title = event.window_title or ""
    if event.event_type == "app_switch" and (process_name == "fsquirt.exe" or window_title in {"蓝牙文件传送", "浏览"}):
        return (event.video_time_ms,)
    if process_name == "fsquirt.exe" and event.file_path and sensitive_hit:
        return (event.video_time_ms, event.video_time_ms + 3_000, event.video_time_ms + 8_000)
    return ()


def _window_profile(priority: str, config: VisionConfig) -> tuple[int, int, int, int, float]:
    if priority == "strong":
        return (
            config.strong_window_before_ms,
            config.strong_window_after_ms,
            config.strong_frame_step_ms,
            config.max_keyframes_per_strong_window,
            config.strong_frame_diff_threshold,
        )
    if priority == "weak":
        return (
            config.frame_window_before_ms,
            config.frame_window_after_ms,
            config.weak_frame_step_ms,
            max(1, config.max_keyframes_per_window // 2),
            config.frame_diff_threshold,
        )
    return (
        config.frame_window_before_ms,
        config.frame_window_after_ms,
        config.frame_step_ms,
        config.max_keyframes_per_window,
        config.frame_diff_threshold,
    )


def _window_reason(event: LogEvent, priority: str) -> str:
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    source = str(extra.get("source") or "")
    category = str(extra.get("category") or "")
    parts = [priority, event.event_type or "activity", source, category]
    return ":".join(item for item in parts if item)
