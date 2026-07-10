"""Log mining entry point for frame analysis windows.

The in-memory miner keeps local development fast. When Neo4j log mining is
enabled, this module delegates to the Neo4j-backed implementation while keeping
the fallback contract in one place.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .frame_analyzer.config import VisionConfig
from .frame_analyzer.frames import AnalysisWindow, merge_analysis_windows
from .io import looks_sensitive, normalize_path
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
            connection_timeout=1.0,
            connection_acquisition_timeout=1.0,
            max_transaction_retry_time=0.0,
        )
        try:
            driver.verify_connectivity()
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

        merged = _thin_dense_window_anchors(_merge_windows_by_case_segment(windows, vision_config), vision_config)
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
    sensitive = tuple(normalize_path(item).lower() for item in sensitive_files)
    app_index = _ActiveAppIndex.from_logs(logs)
    for event in logs:
        window = build_analysis_window_for_event(
            event,
            logs,
            sensitive,
            config,
            active_apps=(),
            normalized_sensitive=True,
        )
        if window is not None:
            windows.append(window)
    segmented = _thin_dense_window_anchors(_filter_visual_context_windows(_merge_windows_by_case_segment(windows, config), config), config)
    return _attach_active_apps_to_windows(segmented, app_index)


def build_analysis_window_for_event(
    event: LogEvent,
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
    *,
    active_apps: tuple[str, ...] | None = None,
    active_app_index: "_ActiveAppIndex | None" = None,
    normalized_sensitive: bool = False,
) -> AnalysisWindow | None:
    """Convert one suspicious log event into a video analysis window."""

    sensitive = tuple(sensitive_files) if normalized_sensitive else tuple(normalize_path(item).lower() for item in sensitive_files)
    text = _event_search_text(event)
    context_text = _nearby_event_search_text(logs, event.video_time_ms, 4_000) if _needs_nearby_context(event) else ""
    combined_text = f"{text} {context_text}".strip()
    file_text = normalize_path(event.file_path).lower()
    sensitive_hit = _matches_sensitive_source(file_text, combined_text, sensitive) or looks_sensitive(file_text)
    transfer_hit = contains_any(combined_text, TRANSFER_TOKENS)
    sink_hit = contains_any(combined_text, SINK_TOKENS) or _is_cloud_drive_context(combined_text)
    action_hit = transfer_hit or sink_hit
    priority = _window_priority(event, combined_text, sensitive_hit, transfer_hit, sink_hit)
    if priority == "none" or event.video_time_ms < 0:
        return None
    if priority == "weak" and not config.include_weak_windows:
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
        anchor_ms=_event_anchors(event, sensitive_hit, action_hit, combined_text),
        active_apps=active_apps
        if active_apps is not None
        else _active_apps_near(logs, event.video_time_ms, after_ms, index=active_app_index),
    )


@dataclass(frozen=True)
class _ActiveAppIndex:
    times: tuple[int, ...]
    apps: tuple[str, ...]

    @classmethod
    def from_logs(cls, logs: list[LogEvent]) -> "_ActiveAppIndex":
        pairs = sorted(
            (event.video_time_ms, event.app_name or event.process_name)
            for event in logs
            if event.video_time_ms >= 0 and (event.app_name or event.process_name)
        )
        return cls(
            times=tuple(item[0] for item in pairs),
            apps=tuple(item[1] for item in pairs),
        )

    def near(self, center_ms: int, radius_ms: int) -> tuple[str, ...]:
        if center_ms < 0:
            return ()
        return self.between(center_ms - radius_ms, center_ms + radius_ms)

    def between(self, start_ms: int, end_ms: int) -> tuple[str, ...]:
        start = bisect_left(self.times, start_ms)
        end = bisect_right(self.times, end_ms)
        apps: list[str] = []
        seen: set[str] = set()
        for app in self.apps[start:end]:
            if app in seen:
                continue
            seen.add(app)
            apps.append(app)
        return tuple(apps)


def _attach_active_apps_to_windows(windows: list[AnalysisWindow], app_index: _ActiveAppIndex) -> list[AnalysisWindow]:
    if not windows:
        return windows

    return [
        AnalysisWindow(
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            reason=window.reason,
            priority=window.priority,
            step_ms=window.step_ms,
            max_keyframes=window.max_keyframes,
            diff_threshold=window.diff_threshold,
            anchor_ms=window.anchor_ms,
            active_apps=app_index.between(window.start_ms, window.end_ms),
        )
        for window in windows
    ]


def _merge_windows_by_case_segment(windows: list[AnalysisWindow], config: VisionConfig) -> list[AnalysisWindow]:
    medium_by_segment: dict[int, list[AnalysisWindow]] = {}
    non_medium: list[AnalysisWindow] = []

    for window in windows:
        if window.priority != "medium":
            non_medium.append(window)
            continue
        event_ms = _window_source_event_ms(window, config)
        segment = event_ms // config.case_segment_ms
        medium_by_segment.setdefault(segment, []).append(window)

    merged_medium: list[AnalysisWindow] = []
    for segment in sorted(medium_by_segment):
        merged_medium.extend(merge_analysis_windows(medium_by_segment[segment]))

    return sorted(
        [*merge_analysis_windows(non_medium), *merged_medium],
        key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms),
    )


def _filter_visual_context_windows(windows: list[AnalysisWindow], config: VisionConfig) -> list[AnalysisWindow]:
    if config.include_unanchored_medium_windows:
        return windows

    kept = [window for window in windows if window.priority != "medium" or window.anchor_ms]
    has_direct_evidence = any(window.priority == "strong" or window.anchor_ms for window in windows)
    if has_direct_evidence:
        return sorted(kept, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))

    medium_windows = [window for window in windows if window.priority == "medium"]
    if medium_windows:
        kept.append(min(medium_windows, key=lambda item: (item.start_ms, item.end_ms)))
    return sorted(kept, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))


def _thin_dense_window_anchors(windows: list[AnalysisWindow], config: VisionConfig) -> list[AnalysisWindow]:
    return [_with_thinned_anchors(window, config) for window in windows]


def _with_thinned_anchors(window: AnalysisWindow, config: VisionConfig) -> AnalysisWindow:
    base_budget = _base_keyframe_budget(window.priority, config)
    anchors = (
        _thin_anchors(window.anchor_ms, _anchor_min_gap_ms(window.priority, config))
        if len(window.anchor_ms) > base_budget
        else window.anchor_ms
    )
    return AnalysisWindow(
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        reason=window.reason,
        priority=window.priority,
        step_ms=window.step_ms,
        max_keyframes=max(base_budget, len(anchors)),
        diff_threshold=window.diff_threshold,
        anchor_ms=anchors,
        active_apps=window.active_apps,
    )


def _thin_anchors(anchors: tuple[int, ...], min_gap_ms: int) -> tuple[int, ...]:
    if min_gap_ms <= 0 or len(anchors) <= 1:
        return anchors
    thinned: list[int] = []
    for anchor in sorted(set(anchors)):
        if thinned and anchor - thinned[-1] < min_gap_ms:
            continue
        thinned.append(anchor)
    return tuple(thinned)


def _anchor_min_gap_ms(priority: str, config: VisionConfig) -> int:
    if priority == "strong":
        return max(config.strong_frame_step_ms * 12, 3_000)
    if priority == "weak":
        return max(config.weak_frame_step_ms, 2_000)
    return max(config.frame_step_ms, 1_000)


def _base_keyframe_budget(priority: str, config: VisionConfig) -> int:
    if priority == "strong":
        return config.max_keyframes_per_strong_window
    if priority == "weak":
        return config.max_keyframes_per_weak_window
    return config.max_keyframes_per_window


def _window_source_event_ms(window: AnalysisWindow, config: VisionConfig) -> int:
    if window.priority == "strong":
        return max(0, window.end_ms - config.strong_window_after_ms)
    if window.priority == "weak":
        return max(0, window.end_ms - config.frame_window_after_ms)
    return max(0, window.end_ms - config.frame_window_after_ms)


def _priority_sort_key(priority: str) -> int:
    return {"strong": 0, "medium": 1, "weak": 2}.get(priority, 1)


def _active_apps_near(
    logs: list[LogEvent],
    center_ms: int,
    radius_ms: int,
    *,
    index: _ActiveAppIndex | None = None,
) -> tuple[str, ...]:
    if index is not None:
        return index.near(center_ms, radius_ms)

    apps: list[str] = []
    for event in logs:
        if event.video_time_ms < 0 or abs(event.video_time_ms - center_ms) > radius_ms:
            continue
        app = event.app_name or event.process_name
        if app and app not in apps:
            apps.append(app)
    return tuple(apps)


def _event_search_text(event: LogEvent) -> str:
    raw = event.raw
    parts: list[str] = [
        event.event_type,
        event.file_path,
        event.process_name,
        event.app_name,
        event.window_title,
        event.description,
    ]
    for key in ("file_name", "file_extension", "content_preview", "operation", "description", "type", "path", "destination_path"):
        value = raw.get(key)
        if value is not None:
            parts.append(str(value))
    for key in ("extra", "upload_detection", "process_info", "window_info"):
        parts.extend(_flatten_search_parts(raw.get(key)))
    return " ".join(item.strip() for item in parts if item and item.strip())


def _nearby_event_search_text(logs: list[LogEvent], center_ms: int, radius_ms: int) -> str:
    if center_ms < 0 or radius_ms <= 0:
        return ""
    parts: list[str] = []
    for item in logs:
        if item.video_time_ms < 0 or abs(item.video_time_ms - center_ms) > radius_ms:
            continue
        parts.append(_event_search_text(item))
    return " ".join(parts)


def _needs_nearby_context(event: LogEvent) -> bool:
    event_type = event.event_type.lower()
    if event_type in {"app_switch", "window_changed", "window_closed"}:
        return True
    return _looks_like_file_selection_dialog(event.window_title)


def _flatten_search_parts(value: Any) -> list[str]:
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_flatten_search_parts(item))
        return parts
    if isinstance(value, list | tuple):
        parts = []
        for item in value:
            parts.extend(_flatten_search_parts(item))
        return parts
    text = str(value or "").strip()
    return [text] if text else []


def _matches_sensitive_source(file_text: str, text: str, sensitive_files: tuple[str, ...]) -> bool:
    if any(item and item in file_text for item in sensitive_files):
        return True

    search_text = normalize_path(text).lower()
    for item in sensitive_files:
        if not item:
            continue
        filename = item.rsplit("/", 1)[-1]
        stem = filename.rsplit(".", 1)[0]
        if filename and filename in search_text:
            return True
        if len(stem) >= 4 and stem in search_text:
            return True
    return False


def _window_priority(event: LogEvent, text: str, sensitive_hit: bool, transfer_hit: bool, sink_hit: bool) -> str:
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
    sink_context = sink_hit or _is_sink_context(process_name, text)

    strong_event_types = {"file_selected", "file_upload", "upload", "uploaded", "upload_complete"}
    strong_raw_ops = {"file_selected", "file_upload", "upload", "send_click"}
    if event_type in strong_event_types or raw_operation in strong_raw_ops:
        return "strong"
    if process_name == "fsquirt.exe":
        return "strong"
    if any(token in category for token in ("文件上传", "直接外发")):
        return "strong"
    if any(token in operation_detail for token in ("附件", "文件选择", "已发送", "外发", "上传")):
        return "strong"
    is_upload_detection = upload_status in {"success", "completed", "complete"} or (
        "upload" in upload_type.lower() and "download" not in upload_type.lower()
    )
    if is_upload_detection and sensitive_hit:
        return "strong"
    if _looks_like_file_selection_dialog(window_title) and sink_context:
        return "strong"
    if event_type in {"app_switch", "window_changed", "window_closed"} and sink_context and _looks_like_upload_progress(text):
        return "strong"

    if event_type in {"clipboard_image", "screenshot", "screen_capture"}:
        return "medium"
    if event_type in {"app_switch", "window_changed", "window_closed"} and _is_sink_app_process(process_name):
        return "medium"

    if sensitive_hit:
        return "medium"
    if sink_context and event_type in {"clipboard_text", "app_switch", "window_changed", "window_closed"}:
        return "medium"

    if str(extra.get("risk_level") or "").lower() in {"高", "high"}:
        return "weak"
    return "none"


def _event_anchors(event: LogEvent, sensitive_hit: bool, action_hit: bool, text: str = "") -> tuple[int, ...]:
    process_name = (event.process_name or "").lower()
    window_title = event.window_title or ""
    event_type = event.event_type.lower()
    if event.event_type == "app_switch" and process_name == "fsquirt.exe":
        return (event.video_time_ms,)
    if process_name == "fsquirt.exe" and event.file_path and sensitive_hit:
        return (event.video_time_ms, event.video_time_ms + 3_000, event.video_time_ms + 8_000)
    if event_type in {"app_switch", "window_changed"} and sensitive_hit:
        return (event.video_time_ms,)
    if _looks_like_file_selection_dialog(window_title) and _is_sink_context(process_name, text):
        if event_type in {"app_switch", "window_changed", "window_closed"}:
            return (event.video_time_ms, event.video_time_ms + 3_000)
        return (event.video_time_ms,)
    if _looks_like_print_or_save_dialog(window_title):
        return (event.video_time_ms,)
    if event.event_type.lower() in {"app_switch", "window_changed", "window_closed"} and _is_sink_app_process(process_name):
        return (event.video_time_ms,)
    if event_type in {"app_switch", "window_changed", "window_closed"} and _is_sink_context(process_name, text) and _looks_like_upload_progress(text):
        return (event.video_time_ms,)
    if sensitive_hit and _looks_like_screenshot_path(event.file_path):
        return (event.video_time_ms,)
    if event_type in {"clipboard_image", "screenshot", "screen_capture", "file_selected", "file_upload", "upload", "uploaded", "upload_complete"}:
        return (event.video_time_ms,)
    return ()


def _is_sink_app_process(process_name: str) -> bool:
    return process_name.lower() in {
        "qq.exe",
        "wechat.exe",
        "weixin.exe",
        "tim.exe",
        "dingtalk.exe",
        "feishu.exe",
        "lark.exe",
        "baidunetdisk.exe",
        "baidunetdiskunite.exe",
    }


def _is_sink_context(process_name: str, text: str) -> bool:
    return _is_sink_app_process(process_name) or _is_cloud_drive_context(text)


def _is_cloud_drive_context(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in (
            "cloud drive",
            "netdisk",
            "onedrive",
            "dropbox",
            "google drive",
            "pan.baidu",
            "pan.quark",
            "baidunetdisk",
            "baidu netdisk",
            "quark",
            "夸克网盘",
            "百度网盘",
            "网盘",
            "云盘",
            "缃戠洏",
            "浜戠洏",
        )
    )


def _looks_like_upload_progress(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in (
            "uploading",
            "upload in progress",
            "file upload",
            "上传中",
            "正在上传",
            "上传至",
            "等待扫描",
            "文件已选择",
            "文件上传中",
            "涓婁紶",
            "鏂囦欢涓婁紶",
        )
    )


def _looks_like_screenshot_path(path: str) -> bool:
    text = normalize_path(path).lower()
    if not text:
        return False
    return (
        "/screenshots/" in text
        or "screenshot" in text
        or text.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    )


def _looks_like_print_or_save_dialog(window_title: str) -> bool:
    title = (window_title or "").lower()
    if not title:
        return False
    return any(token in title for token in ("print", "save as", "save print output", "打印", "另存", "保存"))


def _looks_like_file_selection_dialog(window_title: str) -> bool:
    title = (window_title or "").lower()
    if not title:
        return False
    if title.strip() in {"open", "打开", "请选择"}:
        return True
    return any(
        token in title
        for token in (
            "open file",
            "choose file",
            "select file",
            "browse",
            "打开",
            "请选择",
            "选择文件",
            "文件选择",
        )
    )


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
            config.max_keyframes_per_weak_window,
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
