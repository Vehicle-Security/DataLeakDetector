"""Turn normalized audit events into small, evidence-oriented video windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

from .frame_analyzer.apps import identify_frontend_app
from .frame_analyzer.config import VisionConfig
from .frame_analyzer.frames import AnalysisWindow
from .io import flatten_text, normalize_path, parse_timestamp_ms, same_file
from .models import LogEvent
from .neo4j.config import Neo4jConfig
from .neo4j.importer import Neo4jLogImporter
from .neo4j.queries import Neo4jLogQueries
from .policy import normalize_text


_FOREGROUND_EVENTS = {"app_switch", "window_changed", "window_closed"}
_OPEN_EVENTS = {"file_open", "opened", "open", "read"}
_CLOSE_EVENTS = {"file_close", "file_closed", "closed", "close"}
_UPLOAD_EVENTS = {"file_selected", "file_upload", "upload", "uploaded", "upload_complete"}
_SEND_EVENTS = {"send", "sent", "send_click"}
_CAPTURE_EVENTS = {"screenshot", "screen_capture", "screen_recording"}
_SCREEN_SHARE_EVENTS = {"screen_share_start", "screen_sharing_started", "start_screen_share"}
_DERIVATION_EVENTS = {
    "copied",
    "export",
    "save_as",
    "print_to_pdf",
    "compressed",
    "compress",
}
_FILE_CREATION_EVENTS = {"created", "file_created", "new_file"}
_USER_DOCUMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".sql",
    ".txt",
    ".xls",
    ".xlsx",
    ".zip",
}
_HIDDEN_DERIVATION_PATH_MARKERS = (
    "/appdata/",
    "/programdata/",
    "/program files/",
    "/windows/",
    "/cache/",
    "/code cache/",
    "/temp/",
    "/backup/",
    "/recordings/",
    "/logs/",
    "/video/",
    "/wps cloud files/.",
)
_CLIPBOARD_MARKERS = ("clipboard", "copy", "paste", "pasted")


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
        metadata = {**fallback.metadata, "neo4j_enabled": True, "fallback_reason": f"{type(exc).__name__}: {exc}"}
        return LogMiningResult(fallback.windows, "in_memory_fallback", metadata)


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
            windows,
            "in_memory",
            {"status": "ready", "windows": len(windows), "neo4j_enabled": False},
        )


class Neo4jLogMiner:
    """Use Neo4j only to shortlist events; window semantics remain identical."""

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
                activity_rows = self.queries.sensitive_activity_intervals(
                    session,
                    case_id,
                    tuple(sorted(_noise_profile()["app_names"])),
                )
        finally:
            driver.close()

        timeline = _SensitiveTimeline.from_logs(logs, _normalize_sensitive_files(sensitive_files))
        by_id = {event.event_id: event for event in logs}
        action_windows = []
        for event_id in candidate_ids:
            event = by_id.get(event_id)
            if event is None:
                continue
            window = build_analysis_window_for_event(
                event,
                logs,
                sensitive_files,
                vision_config,
                active_apps=tuple(app_context.get(event_id, ())),
                sensitive_timeline=timeline,
            )
            if window is not None:
                action_windows.append(window)
        activity_windows = self._activity_windows_from_graph(activity_rows, vision_config)
        if not activity_windows:
            activity_windows = build_sensitive_activity_windows(logs, sensitive_files, vision_config)
        windows = _finalize_windows(action_windows, activity_windows, logs)
        if not windows:
            windows = build_analysis_windows(logs, sensitive_files, vision_config)
        return LogMiningResult(
            windows,
            "neo4j",
            {
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
                "sensitive_activity_intervals": len(activity_windows),
                "windows": len(windows),
            },
        )

    @staticmethod
    def _activity_windows_from_graph(rows: list[dict[str, object]], config: VisionConfig) -> list[AnalysisWindow]:
        windows = []
        for row in rows:
            start_ms = int(row.get("start_ms") or 0)
            end_ms = int(row.get("end_ms") or start_ms)
            if end_ms < start_ms:
                continue
            windows.append(
                AnalysisWindow(
                    start_ms,
                    end_ms,
                    f"sensitive_activity:{Path(str(row.get('sensitive_file') or 'context')).name}",
                    priority="activity",
                    step_ms=config.frame_step_ms,
                    max_keyframes=config.max_keyframes_per_window,
                    diff_threshold=config.frame_diff_threshold,
                    anchor_ms=tuple(sorted({int(item) for item in row.get("anchors", []) if item is not None})),
                    active_apps=tuple(str(item) for item in row.get("active_apps", []) if str(item or "").strip()),
                )
            )
        return windows


def build_analysis_windows(
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
) -> list[AnalysisWindow]:
    ordered = sorted((event for event in logs if event.video_time_ms >= 0), key=lambda event: event.video_time_ms)
    sensitive = _normalize_sensitive_files(sensitive_files)
    timeline = _SensitiveTimeline.from_logs(ordered, sensitive)
    candidates = [
        event
        for event in ordered
        if _may_need_analysis_window(event, sensitive) or _may_be_derived_file_event(event, sensitive)
    ]
    candidates = _select_semantic_candidates(candidates, sensitive)
    event_view = _compact_event_view(ordered, candidates)
    actions = []
    for event in candidates:
        window = build_analysis_window_for_event(
            event,
            event_view,
            sensitive,
            config,
            sensitive_timeline=timeline,
            normalized_sensitive=True,
        )
        if window is not None:
            actions.append(window)
    outbound_context = _build_outbound_context_windows(event_view, actions, config)
    visible_actions = [
        window
        for window in actions
        if not any(action == "transfer_anchor" for _, action in window.action_phases)
    ]
    return _finalize_windows([*visible_actions, *outbound_context], [], event_view)


def build_analysis_window_for_event(
    event: LogEvent,
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
    *,
    active_apps: tuple[str, ...] | None = None,
    active_app_index: object | None = None,
    time_index: object | None = None,
    sensitive_context_index: object | None = None,
    sensitive_timeline: "_SensitiveTimeline | None" = None,
    normalized_sensitive: bool = False,
) -> AnalysisWindow | None:
    del active_app_index, time_index, sensitive_context_index
    if event.video_time_ms < 0 or _is_noise_only_event(event):
        return None
    sensitive = tuple(sensitive_files) if normalized_sensitive else _normalize_sensitive_files(sensitive_files)
    timeline = sensitive_timeline or _SensitiveTimeline.from_logs(logs, sensitive)
    direct_sensitive = _event_matches_sensitive(event, sensitive)
    path_sensitive = bool(event.file_path) and any(same_file(event.file_path, source) for source in sensitive)
    direct_derivation = _may_be_derived_file_event(event, sensitive)
    sensitive_context = (
        direct_sensitive
        or direct_derivation
        or timeline.active_at(event.video_time_ms)
        or timeline.recent_at(event.video_time_ms)
    )
    action = _action_kind(event)
    if action == "transfer_anchor" and not direct_sensitive:
        action = ""
    if not action and direct_derivation:
        action = "derive"
    priority = "none"
    if action in {"upload", "send", "file_selected", "transfer_anchor"}:
        priority = "strong"
    elif action == "removable":
        priority = "strong"
    elif action in {"paste", "clipboard", "capture_start", "capture", "screen_share", "derive"} and sensitive_context:
        if action != "clipboard" or path_sensitive or not _is_browser_event(event):
            priority = "strong"
    elif event.event_type not in _FOREGROUND_EVENTS and _is_high_risk_hint(event):
        priority = "weak"
        action = action or "unknown_risk"

    if priority == "none" or (priority == "weak" and not config.include_weak_windows):
        return None
    if priority == "strong":
        before_ms = config.strong_window_before_ms
        after_ms = config.strong_window_after_ms
        if action == "capture":
            before_ms = max(before_ms, 15_000)
        if action in {"file_selected", "upload", "send", "removable", "screen_share"}:
            after_ms = max(after_ms, 30_000)
        step_ms = config.strong_frame_step_ms
        budget = config.max_keyframes_per_strong_window
        threshold = config.strong_frame_diff_threshold
    else:
        before_ms = config.frame_window_before_ms
        after_ms = config.frame_window_after_ms
        step_ms = config.weak_frame_step_ms
        budget = config.max_keyframes_per_weak_window
        threshold = config.frame_diff_threshold
    source = _extra(event).get("source") or "log"
    timestamp = event.video_time_ms
    return AnalysisWindow(
        max(0, timestamp - before_ms),
        timestamp + after_ms,
        f"{priority}:{event.event_type}:{source}:{action or 'review'}",
        priority=priority,
        step_ms=step_ms,
        max_keyframes=budget,
        diff_threshold=threshold,
        anchor_ms=(timestamp,),
        action_anchor_ms=(timestamp,),
        action_phases=((timestamp, action),),
        requires_post_action_state=action == "paste",
        active_apps=active_apps or (),
    )


def build_sensitive_activity_windows(
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
    *,
    normalized_sensitive: bool = False,
) -> list[AnalysisWindow]:
    sensitive = tuple(sensitive_files) if normalized_sensitive else _normalize_sensitive_files(sensitive_files)
    timeline = _SensitiveTimeline.from_logs(logs, sensitive)
    windows = []
    for source, start_ms, end_ms, anchors in timeline.activities:
        windows.append(
            AnalysisWindow(
                start_ms,
                end_ms,
                f"sensitive_activity:{Path(source).name}",
                priority="activity",
                step_ms=config.frame_step_ms,
                max_keyframes=config.max_keyframes_per_window,
                diff_threshold=config.frame_diff_threshold,
                anchor_ms=anchors,
            )
        )
    return windows


@dataclass(frozen=True)
class _SensitiveTimeline:
    activities: tuple[tuple[str, int, int, tuple[int, ...]], ...]
    signals: tuple[int, ...]

    @classmethod
    def from_logs(cls, logs: list[LogEvent], sensitive_files: tuple[str, ...]) -> "_SensitiveTimeline":
        if not sensitive_files:
            return cls((), ())
        ordered = sorted((event for event in logs if event.video_time_ms >= 0), key=lambda event: event.video_time_ms)
        session_end = max((event.video_time_ms for event in ordered), default=0)
        signals: list[int] = []
        activities = []
        for source in sensitive_files:
            related = [event for event in ordered if _event_matches_source(event, source)]
            if not related:
                continue
            signals.extend(event.video_time_ms for event in related)
            opens = [event for event in related if _is_open_event(event)]
            closes = [event for event in related if _is_close_event(event)]
            if not opens:
                continue
            for index, opened in enumerate(opens):
                next_open = opens[index + 1].video_time_ms if index + 1 < len(opens) else None
                explicit_end = parse_timestamp_ms(opened.raw.get("end_time"))
                if explicit_end and opened.timestamp_ms:
                    closed = opened.video_time_ms + max(explicit_end - opened.timestamp_ms, 0)
                else:
                    closed = next(
                        (event.video_time_ms for event in closes if event.video_time_ms >= opened.video_time_ms),
                        session_end,
                    )
                end_ms = min(closed, next_open) if next_open is not None else closed
                anchors = tuple(
                    sorted(
                        {
                            event.video_time_ms
                            for event in related
                            if opened.video_time_ms <= event.video_time_ms <= end_ms
                        }
                    )
                )
                activities.append((source, opened.video_time_ms, end_ms, anchors))
        return cls(tuple(_merge_activities(activities)), tuple(sorted(set(signals))))

    def active_at(self, timestamp_ms: int) -> bool:
        return any(start <= timestamp_ms <= end for _, start, end, _ in self.activities)

    def recent_at(self, timestamp_ms: int, radius_ms: int = 30_000) -> bool:
        return any(0 <= timestamp_ms - signal <= radius_ms for signal in self.signals)


def _merge_activities(
    activities: list[tuple[str, int, int, tuple[int, ...]]],
) -> list[tuple[str, int, int, tuple[int, ...]]]:
    merged = []
    for source, start, end, anchors in sorted(activities, key=lambda item: (item[0], item[1], item[2])):
        if merged and same_file(merged[-1][0], source) and start <= merged[-1][2] + 1:
            previous = merged[-1]
            merged[-1] = (source, previous[1], max(previous[2], end), tuple(sorted({*previous[3], *anchors})))
        else:
            merged.append((source, start, end, anchors))
    return merged


def _build_outbound_context_windows(
    logs: list[LogEvent],
    action_windows: list[AnalysisWindow],
    config: VisionConfig,
    *,
    evidence_horizon_ms: int = 600_000,
    session_gap_ms: int = 180_000,
) -> list[AnalysisWindow]:
    """Link source-side evidence to the next foreground business session."""

    evidence = sorted(
        {
            phase
            for window in action_windows
            for phase in (
                window.action_phases
                or tuple((anchor, "action") for anchor in window.action_anchor_ms)
            )
            if phase[1] in {"capture", "clipboard", "derive"}
        }
    )
    if not evidence:
        return []
    foreground = [event for event in logs if event.event_type in _FOREGROUND_EVENTS and _foreground_app_key(event)]
    source_events = {
        event.video_time_ms: event
        for event in logs
        if event.event_type not in _FOREGROUND_EVENTS
    }
    windows = []
    seen_sessions: set[tuple[int, int, str]] = set()
    for timestamp, action in evidence:
        source_event = source_events.get(timestamp)
        source_app = _foreground_app_key(source_event) if source_event is not None else ""
        start_index = next(
            (
                index
                for index, event in enumerate(foreground)
                if 0 <= event.video_time_ms - timestamp <= evidence_horizon_ms
                and _visible_foreground(event)
                and _foreground_app_key(event) != source_app
            ),
            None,
        )
        if start_index is None:
            continue
        session = []
        app = _foreground_app_key(foreground[start_index])
        for event in foreground[start_index:]:
            if session and event.video_time_ms - session[-1].video_time_ms > session_gap_ms:
                break
            if _foreground_app_key(event) != app:
                break
            if not _visible_foreground(event):
                if session:
                    break
                continue
            session.append(event)
        if not session:
            continue
        session_start = session[0].video_time_ms
        session_end = session[-1].video_time_ms
        source_has_resource = bool(source_event and normalize_path(source_event.file_path))
        transfer_signal = any(_looks_like_file_dialog(event.window_title) for event in session) or any(
            session_start <= event.video_time_ms <= session_end
            and _action_kind(event) in {"paste", "file_selected", "upload", "send"}
            for event in logs
        )
        if action == "capture" and not transfer_signal:
            continue
        if action == "clipboard" and not source_has_resource and not transfer_signal:
            continue
        identity = (session[0].video_time_ms, session[-1].video_time_ms, app)
        if identity in seen_sessions:
            continue
        seen_sessions.add(identity)
        anchor = session[-1].video_time_ms
        windows.append(
            AnalysisWindow(
                max(0, anchor - 1_000),
                anchor + max(config.strong_window_after_ms, 30_000),
                "strong:outbound_context",
                priority="strong",
                step_ms=config.strong_frame_step_ms,
                max_keyframes=6,
                diff_threshold=config.strong_frame_diff_threshold,
                anchor_ms=(anchor,),
                action_anchor_ms=(anchor,),
                action_phases=((anchor, "outbound_context"),),
            )
        )
    return windows


def _finalize_windows(
    actions: list[AnalysisWindow],
    activities: list[AnalysisWindow],
    logs: list[LogEvent],
) -> list[AnalysisWindow]:
    foreground = _ForegroundTimeline.from_logs(logs)
    merged_actions = _merge_action_windows(actions)
    windows = [*merged_actions, *activities]
    enriched = []
    for window in windows:
        apps, ranges = foreground.context(window.start_ms, window.end_ms)
        enriched.append(
            AnalysisWindow(
                window.start_ms,
                window.end_ms,
                window.reason,
                priority=window.priority,
                step_ms=window.step_ms,
                max_keyframes=window.max_keyframes,
                diff_threshold=window.diff_threshold,
                anchor_ms=window.anchor_ms,
                action_anchor_ms=window.action_anchor_ms,
                action_phases=window.action_phases,
                requires_post_action_state=window.requires_post_action_state,
                active_apps=window.active_apps or apps,
                active_ranges=ranges,
            )
        )
    order = {"strong": 0, "activity": 1, "weak": 2}
    return sorted(enriched, key=lambda item: (order.get(item.priority, 3), item.start_ms, item.end_ms))


def _merge_action_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    merged = []
    for window in sorted(windows, key=lambda item: (item.start_ms, item.end_ms, item.reason)):
        if not merged or window.priority != merged[-1].priority or window.start_ms > merged[-1].end_ms:
            merged.append(window)
            continue
        previous = merged[-1]
        merged[-1] = AnalysisWindow(
            previous.start_ms,
            max(previous.end_ms, window.end_ms),
            f"{previous.priority}:action_cluster",
            priority=previous.priority,
            step_ms=min(previous.step_ms, window.step_ms),
            max_keyframes=max(previous.max_keyframes, window.max_keyframes),
            diff_threshold=min(previous.diff_threshold, window.diff_threshold),
            anchor_ms=tuple(sorted({*previous.anchor_ms, *window.anchor_ms})),
            action_anchor_ms=tuple(sorted({*previous.action_anchor_ms, *window.action_anchor_ms})),
            action_phases=tuple(sorted({*previous.action_phases, *window.action_phases})),
            requires_post_action_state=previous.requires_post_action_state or window.requires_post_action_state,
            active_apps=tuple(dict.fromkeys((*previous.active_apps, *window.active_apps))),
        )
    return merged


@dataclass(frozen=True)
class _ForegroundTimeline:
    ranges: tuple[tuple[int, int, str], ...]

    @classmethod
    def from_logs(cls, logs: list[LogEvent]) -> "_ForegroundTimeline":
        events = sorted(
            (event for event in logs if event.video_time_ms >= 0 and event.event_type in _FOREGROUND_EVENTS),
            key=lambda event: event.video_time_ms,
        )
        session_end = max((event.video_time_ms for event in logs if event.video_time_ms >= 0), default=0)
        ranges = []
        for index, event in enumerate(events):
            if not _visible_foreground(event):
                continue
            end = events[index + 1].video_time_ms - 1 if index + 1 < len(events) else session_end
            if end >= event.video_time_ms:
                ranges.append((event.video_time_ms, end, event.app_name or event.process_name))
        return cls(tuple(ranges))

    def context(self, start_ms: int, end_ms: int) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
        matches = [item for item in self.ranges if item[1] >= start_ms and item[0] <= end_ms]
        apps = tuple(dict.fromkeys(app for _, _, app in matches if app))
        ranges = _merge_ranges(
            [(max(start, start_ms), min(end, end_ms)) for start, end, _ in matches]
        )
        return apps, ranges


def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _compact_event_view(logs: list[LogEvent], candidate_events: list[LogEvent]) -> list[LogEvent]:
    """Keep semantic events and foreground transitions; discard file-system churn."""

    candidate_ids = {event.event_id for event in candidate_events}
    kept = [
        event
        for event in logs
        if event.event_id in candidate_ids
        or event.event_type in _FOREGROUND_EVENTS
        or _action_kind(event)
        if not _is_noise_only_event(event)
    ]
    return sorted({event.event_id: event for event in kept}.values(), key=lambda event: (event.video_time_ms, event.event_id))


def _may_need_analysis_window(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    if event.video_time_ms < 0 or _is_noise_only_event(event):
        return False
    if _event_matches_sensitive(event, sensitive_files) or _action_kind(event):
        return True
    return event.event_type not in _FOREGROUND_EVENTS and _is_high_risk_hint(event)


def _select_semantic_candidates(
    events: list[LogEvent],
    sensitive_files: tuple[str, ...],
) -> list[LogEvent]:
    """Collapse monitor echoes while preserving distinct behavior phases."""

    inferred = [event for event in events if not _action_kind(event) and _may_be_derived_file_event(event, sensitive_files)]
    selections = [
        event
        for event in events
        if _action_kind(event) in {"file_selected", "upload", "send"} and Path(normalize_path(event.file_path)).name
    ]
    chosen_derivations: set[str] = set()
    if selections:
        for selection in selections:
            selected_name = Path(normalize_path(selection.file_path)).name.lower()
            matches = [
                event
                for event in inferred
                if event.video_time_ms <= selection.video_time_ms
                and Path(normalize_path(event.file_path)).name.lower() == selected_name
            ]
            if matches:
                chosen_derivations.add(min(matches, key=lambda event: event.video_time_ms).event_id)
    else:
        first_by_path: dict[str, LogEvent] = {}
        for event in inferred:
            first_by_path.setdefault(normalize_path(event.file_path).lower(), event)
        chosen_derivations.update(event.event_id for event in first_by_path.values())

    downstream = [
        event.video_time_ms
        for event in events
        if _action_kind(event) in {"paste", "capture", "file_selected", "upload", "send", "removable"}
    ]
    capture_downstream = [
        event.video_time_ms
        for event in events
        if _action_kind(event) in {"capture_start", "capture"}
    ]
    selected = []
    last_capture_ms = -10**9
    for event in events:
        action = _action_kind(event)
        is_inferred = not action and _may_be_derived_file_event(event, sensitive_files)
        if is_inferred and event.event_id not in chosen_derivations:
            continue
        if action == "clipboard":
            if any(0 <= capture_ms - event.video_time_ms <= 15_000 for capture_ms in capture_downstream):
                continue
            if not any(
                0 <= downstream_ms - event.video_time_ms <= 60_000
                for downstream_ms in downstream
            ):
                continue
        if action in {"capture_start", "capture"}:
            if event.video_time_ms - last_capture_ms <= 1_000:
                continue
            last_capture_ms = event.video_time_ms
        elif action == "clipboard" and 0 <= event.video_time_ms - last_capture_ms <= 8_000:
            continue
        selected.append(event)
    return selected


def _action_kind(event: LogEvent) -> str:
    text = _event_text(event)
    event_type = event.event_type.lower()
    operation = str(_extra(event).get("raw_operation") or event.raw.get("operation") or "").lower()
    combined = f"{event_type} {operation} {text}".lower()
    if event_type == "file_selected" or operation == "file_selected":
        return "file_selected"
    if event_type in {"opened", "read"} and "browser_file_access" in operation and _is_user_document_path(
        normalize_path(event.file_path).lower()
    ):
        return "transfer_anchor"
    if event_type in _UPLOAD_EVENTS or operation in _UPLOAD_EVENTS or _structured_upload(event):
        return "upload"
    if event_type in _SEND_EVENTS or operation in _SEND_EVENTS:
        return "send"
    if "paste" in combined:
        return "paste"
    if event_type in _SCREEN_SHARE_EVENTS or operation in _SCREEN_SHARE_EVENTS:
        return "screen_share"
    if _is_capture_start_event(event):
        return "capture_start"
    if _is_file_selection_event(event):
        return "file_selected"
    if _is_capture_event(event):
        return "capture"
    if event_type.startswith("clipboard") or any(token in operation for token in _CLIPBOARD_MARKERS):
        return "clipboard"
    if event_type in _DERIVATION_EVENTS or any(
        token in operation for token in ("export", "save_as", "print", "compress", "encode", "decode")
    ):
        return "derive"
    if any(token in combined for token in ("bluetooth", "removable", "usb", "u盘", "蓝牙", "fsquirt")):
        return "removable"
    return ""


def _is_open_event(event: LogEvent) -> bool:
    operation = str(_extra(event).get("raw_operation") or "").lower()
    return event.event_type in _OPEN_EVENTS or operation in _OPEN_EVENTS


def _is_close_event(event: LogEvent) -> bool:
    operation = str(_extra(event).get("raw_operation") or "").lower()
    return event.event_type in _CLOSE_EVENTS or operation in _CLOSE_EVENTS


def _structured_upload(event: LogEvent) -> bool:
    upload = event.raw.get("upload_detection")
    if not isinstance(upload, dict):
        return False
    upload_type = str(upload.get("upload_type") or "").lower()
    status = str(upload.get("upload_status") or upload.get("status") or "").lower()
    explicit_context = "upload" in normalize_text(f"{event.window_title} {event.description}")
    return status in {"success", "completed", "complete"} or (
        "upload" in upload_type
        and "download" not in upload_type
        and "file access" not in upload_type
        and not _is_noise_path(event.file_path)
    ) or (
        bool(upload.get("is_upload"))
        and explicit_context
        and not _is_noise_path(event.file_path)
    )


def _is_capture_event(event: LogEvent) -> bool:
    event_type = event.event_type.lower()
    operation = str(_extra(event).get("raw_operation") or "").lower()
    explicit_operation = str(_extra(event).get("operation") or event.raw.get("operation") or "").lower()
    if (
        event_type in _CAPTURE_EVENTS
        or operation in _CAPTURE_EVENTS
        or explicit_operation in _CAPTURE_EVENTS
        or "screen_capture" in explicit_operation
    ):
        return True
    process = (event.process_name or "").lower()
    path = normalize_path(event.file_path).lower()
    image = Path(path).suffix in {".png", ".jpg", ".jpeg", ".bmp"}
    screenshot_path = "/screenshots/" in path or "screenshot" in Path(path).name or "屏幕截图" in Path(path).name
    created = event_type in _FILE_CREATION_EVENTS or operation in _FILE_CREATION_EVENTS
    return image and "/appdata/" not in path and (
        "snippingtool" in process or (created and screenshot_path)
    )


def _is_capture_start_event(event: LogEvent) -> bool:
    if event.event_type.lower() not in _FOREGROUND_EVENTS:
        return False
    text = normalize_text(
        f"{event.app_name} {event.process_name} {event.window_title}"
    )
    return any(
        marker in text
        for marker in ("snippingtool", "snipping tool", "snip & sketch", "截图工具", "截屏工具")
    )


def _is_file_selection_event(event: LogEvent) -> bool:
    if event.event_type.lower() in _FOREGROUND_EVENTS or not _looks_like_file_dialog(event.window_title):
        return False
    path = normalize_path(event.file_path).lower()
    if not _is_user_document_path(path):
        return False
    source = str(_extra(event).get("source") or "").lower()
    identity = identify_frontend_app(
        app_name=event.app_name or event.process_name,
        window_title=event.window_title,
    )
    return "file_dialog" in source or identity.risk_hint.startswith("external_capable")


def _looks_like_file_dialog(title: str) -> bool:
    normalized = normalize_text(title).strip()
    return normalized in {"open", "select", "choose", "打开", "选择"} or any(
        marker in normalized for marker in ("请选择", "选择文件", "select file", "choose file")
    )


def _event_matches_sensitive(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    return any(_event_matches_source(event, source) for source in sensitive_files)


def _event_matches_source(event: LogEvent, source: str) -> bool:
    if event.file_path and same_file(event.file_path, source):
        return True
    filename = Path(normalize_path(source)).name.lower()
    return bool(filename and len(filename) >= 4 and filename in normalize_path(_event_text(event)).lower())


def _may_be_derived_file_event(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    operation = str(_extra(event).get("raw_operation") or event.raw.get("operation") or "").lower()
    if event.event_type.lower() not in _FILE_CREATION_EVENTS and operation not in _FILE_CREATION_EVENTS:
        return False
    target = normalize_path(event.file_path).lower()
    if not _is_user_document_path(target):
        return False
    return any(_target_path_references_source(target, source) for source in sensitive_files)


def _is_user_document_path(path: str) -> bool:
    if not path or Path(path).suffix.lower() not in _USER_DOCUMENT_EXTENSIONS:
        return False
    return not any(marker in path for marker in _HIDDEN_DERIVATION_PATH_MARKERS)


def _target_path_references_source(target: str, source: str) -> bool:
    normalized_source = normalize_path(source).lower()
    if not normalized_source or same_file(target, normalized_source):
        return False
    stem = Path(normalized_source).stem.strip()
    return len(stem) >= 3 and stem in target


def _normalize_sensitive_files(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for value in values if (path := normalize_path(value).lower())))


def _event_text(event: LogEvent) -> str:
    return normalize_text(
        " ".join(
            part
            for part in (
                event.event_type,
                event.file_path,
                event.process_name,
                event.app_name,
                event.window_title,
                event.description,
                flatten_text(event.raw),
            )
            if part
        )
    )


def _foreground_app_key(event: LogEvent) -> str:
    return (event.app_name or event.process_name).strip().lower().removesuffix(".exe")


def _is_browser_event(event: LogEvent) -> bool:
    app = _foreground_app_key(event)
    return any(marker in app for marker in ("browser", "chrome", "edge", "firefox"))


def _is_high_risk_hint(event: LogEvent) -> bool:
    risk = normalize_text(_extra(event).get("risk_level") or "")
    return risk in {"high", "高"}


def _visible_foreground(event: LogEvent) -> bool:
    app = (event.app_name or event.process_name).strip()
    title = event.window_title.strip()
    profile = _noise_profile()
    if not app or app.lower().removesuffix(".exe") in profile["app_names"]:
        return False
    normalized_app = app.lower().removesuffix(".exe")
    if not title and normalized_app in {
        "explorer",
        "file explorer",
        "windows explorer",
        "kwallpaper",
        "applicationframehost",
    }:
        return False
    normalized_title = normalize_text(title)
    if any(marker in normalized_title for marker in profile["window_title_markers"]):
        return False
    return normalized_title not in {"desktop", "program manager", "桌面"}


def _is_noise_only_event(event: LogEvent) -> bool:
    path = normalize_path(event.file_path).lower()
    if not path:
        return False
    if _action_kind_without_text(event):
        return False
    return _is_noise_path(path)


def _is_noise_path(value: str) -> bool:
    path = normalize_path(value).lower()
    profile = _noise_profile()
    basename = Path(path).name
    return basename in profile["basenames"] or any(marker in path for marker in profile["path_markers"])


def _action_kind_without_text(event: LogEvent) -> bool:
    event_type = event.event_type.lower()
    operation = str(_extra(event).get("raw_operation") or "").lower()
    return bool(
        event_type in _UPLOAD_EVENTS | _SEND_EVENTS | _CAPTURE_EVENTS
        or event_type.startswith("clipboard")
        or operation in _UPLOAD_EVENTS | _SEND_EVENTS
        or any(marker in operation for marker in ("paste", "clipboard", "screenshot", "screen_capture"))
    )


def _extra(event: LogEvent) -> dict[str, Any]:
    value = event.raw.get("extra")
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=1)
def _noise_profile() -> dict[str, frozenset[str] | tuple[str, ...]]:
    path = Path(__file__).resolve().parents[2] / "spec" / "config" / "system_noise_profile.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    return {
        "path_markers": tuple(str(item).lower() for item in payload.get("path_markers", [])),
        "basenames": frozenset(str(item).lower() for item in payload.get("basenames", [])),
        "app_names": frozenset(str(item).lower().removesuffix(".exe") for item in payload.get("app_names", [])),
        "window_title_markers": tuple(normalize_text(item) for item in payload.get("window_title_markers", [])),
    }


__all__ = [
    "InMemoryLogMiner",
    "LogMiningResult",
    "Neo4jLogMiner",
    "build_analysis_window_for_event",
    "build_analysis_windows",
    "build_sensitive_activity_windows",
    "mine_analysis_windows",
]
