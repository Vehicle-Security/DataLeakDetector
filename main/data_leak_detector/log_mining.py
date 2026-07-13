"""Log mining entry point for frame analysis windows.

The in-memory miner keeps local development fast. When Neo4j log mining is
enabled, this module delegates to the Neo4j-backed implementation while keeping
the fallback contract in one place.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

from .frame_analyzer.config import VisionConfig
from .frame_analyzer.frames import AnalysisWindow, merge_analysis_windows
from .io import looks_sensitive, normalize_path, parse_timestamp_ms, same_file
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
                activity_rows = self.queries.sensitive_activity_intervals(
                    session,
                    case_id,
                    tuple(sorted(_whitelisted_apps())),
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
        activity_windows = self._activity_windows_from_graph(activity_rows, vision_config)
        windows.extend(activity_windows or build_sensitive_activity_windows(logs, sensitive_files, vision_config))
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
                "sensitive_activity_intervals": len(activity_windows),
                "windows": len(merged),
            },
        )

    @staticmethod
    def _activity_windows_from_graph(rows: list[dict[str, object]], vision_config: VisionConfig) -> list[AnalysisWindow]:
        windows: list[AnalysisWindow] = []
        for row in rows:
            start_ms = int(row.get("start_ms") or 0)
            end_ms = int(row.get("end_ms") or start_ms)
            if end_ms < start_ms:
                continue
            anchors = tuple(sorted({int(item) for item in row.get("anchors", []) if item is not None}))
            apps = tuple(str(item) for item in row.get("active_apps", []) if str(item or "").strip())
            windows.append(
                AnalysisWindow(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    reason=f"sensitive_activity:{Path(str(row.get('sensitive_file') or 'context')).name}",
                    priority="activity",
                    step_ms=vision_config.frame_step_ms,
                    max_keyframes=vision_config.max_keyframes_per_window,
                    diff_threshold=vision_config.frame_diff_threshold,
                    anchor_ms=anchors,
                    active_apps=apps,
                )
            )
        return windows

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
    candidate_events = [event for event in logs if _may_need_analysis_window(event, sensitive)]
    event_view = _compact_event_view(logs, candidate_events)
    app_index = _ActiveAppIndex.from_logs(event_view)
    foreground_range_index = _ForegroundRangeIndex.from_logs(event_view)
    time_index = _LogTimeIndex.from_logs(event_view)
    sensitive_context_index = _SensitiveContextIndex.from_logs(candidate_events, event_view, sensitive)
    for event in candidate_events:
        window = build_analysis_window_for_event(
            event,
            event_view,
            sensitive,
            config,
            active_apps=(),
            time_index=time_index,
            sensitive_context_index=sensitive_context_index,
            normalized_sensitive=True,
        )
        if window is not None:
            windows.append(window)
    activity_windows = build_sensitive_activity_windows(event_view, sensitive, config)
    windows.extend(activity_windows)
    if activity_windows:
        windows = [window for window in windows if window.priority != "medium"]
    segmented = _filter_visual_context_windows(_merge_windows_by_case_segment(windows, config), config)
    segmented = _add_sink_followup_anchors(segmented, event_view)
    segmented = _thin_dense_window_anchors(segmented, config)
    return _attach_active_apps_to_windows(segmented, app_index, foreground_range_index)


def _compact_event_view(logs: list[LogEvent], candidate_events: list[LogEvent]) -> list[LogEvent]:
    """Keep only events that can influence analysis-window construction.

    The raw log remains the authority for discovering candidates. Once that
    conservative single-pass scan has kept uploads/sends, file selections,
    clipboard/capture actions, sensitive-source hits, foreground transitions,
    and derivation candidates, later context lookups should not repeatedly
    stringify unrelated opened/closed/modified filesystem noise.
    """

    if len(candidate_events) == len(logs):
        return logs

    candidate_ids = {id(event) for event in candidate_events}
    return [event for event in logs if id(event) in candidate_ids or _is_compact_context_event(event)]


def _is_compact_context_event(event: LogEvent) -> bool:
    if event.video_time_ms < 0:
        return False
    return _is_foreground_transition(event)


def _add_sink_followup_anchors(windows: list[AnalysisWindow], logs: list[LogEvent]) -> list[AnalysisWindow]:
    """Sample paste/send states after sensitive clipboard activity enters a sink app."""

    result: list[AnalysisWindow] = []
    foreground = [
        event
        for event in logs
        if event.video_time_ms >= 0
        and event.event_type.lower() in {"app_switch", "window_changed"}
        and _is_sink_app_process(event.process_name or "")
    ]
    for window in windows:
        if window.priority != "strong" or "clipboard" not in window.reason.lower() or not window.action_anchor_ms:
            result.append(window)
            continue
        clipboard_events = [
            event
            for event in logs
            if event.event_type.lower() in {"clipboard_text", "clipboard_image"}
            and window.start_ms <= event.video_time_ms <= window.end_ms
        ]
        if not clipboard_events:
            result.append(window)
            continue
        clipboard_sink_pairs = [
            (clipboard, sink)
            for clipboard in clipboard_events
            for sink in [
                next(
                    (
                        event
                        for event in foreground
                        if clipboard.video_time_ms < event.video_time_ms <= window.end_ms
                    ),
                    None,
                )
            ]
            if sink is not None
        ]
        if not clipboard_sink_pairs:
            result.append(window)
            continue
        clipboard, sink_switch = max(clipboard_sink_pairs, key=lambda pair: pair[0].video_time_ms)
        action_ms = clipboard.video_time_ms
        followups = tuple(sink_switch.video_time_ms + offset for offset in (5_000, 14_000))
        anchors = tuple(sorted({*window.anchor_ms, *followups}))
        result.append(
            AnalysisWindow(
                start_ms=window.start_ms,
                end_ms=max(window.end_ms, followups[-1]),
                reason=f"{window.reason}+strong:sink_followup",
                priority=window.priority,
                step_ms=window.step_ms,
                max_keyframes=max(window.max_keyframes, len(anchors)),
                diff_threshold=window.diff_threshold,
                anchor_ms=anchors,
                action_anchor_ms=followups,
                active_apps=window.active_apps,
                active_ranges=window.active_ranges,
            )
        )
    return result


def build_sensitive_activity_windows(
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
) -> list[AnalysisWindow]:
    """Build open-to-close video windows for configured sensitive files.

    A file remains active until its explicit close event. When logs do not
    contain a close, the interval ends at the last video-timestamped event in
    the recording rather than at an arbitrary elapsed-time cutoff.
    """

    sensitive = tuple(normalize_path(item).lower() for item in sensitive_files if normalize_path(item))
    if not sensitive:
        return []
    session_end_ms = max((event.video_time_ms for event in logs if event.video_time_ms >= 0), default=-1)
    if session_end_ms < 0:
        return []

    active: dict[str, dict[str, Any]] = {}
    completed: list[tuple[str, int, int, tuple[int, ...]]] = []
    for event in sorted(logs, key=lambda item: item.video_time_ms):
        if event.video_time_ms < 0:
            continue
        matches = _matched_sensitive_files(event, sensitive)
        if not matches:
            continue
        for sensitive_file in matches:
            state = active.get(sensitive_file)
            if _is_sensitive_close_event(event) and state is not None:
                anchors = tuple(sorted({*state["anchors"], event.video_time_ms}))
                completed.append((sensitive_file, state["start_ms"], event.video_time_ms, anchors))
                del active[sensitive_file]
                continue
            if state is None:
                active[sensitive_file] = {"start_ms": event.video_time_ms, "anchors": {event.video_time_ms}}
            else:
                state["anchors"].add(event.video_time_ms)

    for sensitive_file, state in active.items():
        completed.append((sensitive_file, state["start_ms"], session_end_ms, tuple(sorted(state["anchors"]))))

    windows: list[AnalysisWindow] = []
    for sensitive_file, start_ms, end_ms, anchors in completed:
        if end_ms < start_ms:
            continue
        windows.append(
            AnalysisWindow(
                start_ms=start_ms,
                end_ms=end_ms,
                reason=f"sensitive_activity:{Path(sensitive_file).name}",
                priority="activity",
                step_ms=config.frame_step_ms,
                max_keyframes=config.max_keyframes_per_window,
                diff_threshold=config.frame_diff_threshold,
                anchor_ms=anchors,
            )
        )
    return windows


def build_analysis_window_for_event(
    event: LogEvent,
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
    *,
    active_apps: tuple[str, ...] | None = None,
    active_app_index: "_ActiveAppIndex | None" = None,
    time_index: "_LogTimeIndex | None" = None,
    sensitive_context_index: "_SensitiveContextIndex | None" = None,
    normalized_sensitive: bool = False,
) -> AnalysisWindow | None:
    """Convert one suspicious log event into a video analysis window."""

    sensitive = tuple(sensitive_files) if normalized_sensitive else tuple(normalize_path(item).lower() for item in sensitive_files)
    text = _event_search_text(event)
    context_text = _nearby_event_search_text(logs, event.video_time_ms, 4_000, index=time_index) if _needs_nearby_context(event) else ""
    combined_text = f"{text} {context_text}".strip()
    file_text = normalize_path(event.file_path).lower()
    sensitive_hit = _matches_sensitive_source(file_text, combined_text, sensitive) or looks_sensitive(file_text)
    open_context_hit = (
        sensitive_context_index.has_open_context(event.video_time_ms)
        if sensitive_context_index is not None
        else _has_sensitive_open_context(logs, event.video_time_ms, sensitive, time_index=time_index)
    )
    recent_signal_hit = (
        sensitive_context_index.has_recent_signal(event.video_time_ms)
        if sensitive_context_index is not None
        else _has_recent_sensitive_signal(logs, event.video_time_ms, sensitive, time_index=time_index)
    )
    sensitive_context_hit = sensitive_hit or (
        (_is_clipboard_or_capture_event(event, combined_text) or _is_derivation_candidate_event(event, combined_text))
        and (open_context_hit or recent_signal_hit)
    )
    transfer_hit = contains_any(combined_text, TRANSFER_TOKENS)
    sink_hit = contains_any(combined_text, SINK_TOKENS) or _is_cloud_drive_context(combined_text)
    action_hit = transfer_hit or sink_hit
    priority = _window_priority(event, combined_text, sensitive_hit, transfer_hit, sink_hit, sensitive_context_hit)
    if priority == "none" or event.video_time_ms < 0:
        return None
    if priority == "weak" and not config.include_weak_windows:
        return None

    before_ms, after_ms, step_ms, max_keyframes, diff_threshold = _window_profile(priority, config)
    anchors = _event_anchors(event, sensitive_hit, action_hit, combined_text, sensitive_context_hit)
    action_label = _window_action_label(event, combined_text, sensitive_context_hit)
    end_ms = event.video_time_ms + after_ms
    if anchors:
        end_ms = max(end_ms, max(anchors))
    return AnalysisWindow(
        start_ms=max(event.video_time_ms - before_ms, 0),
        end_ms=end_ms,
        reason=_window_reason(event, priority, action_label=action_label),
        priority=priority,
        step_ms=step_ms,
        max_keyframes=max_keyframes,
        diff_threshold=diff_threshold,
        anchor_ms=anchors,
        action_anchor_ms=anchors if action_label else (),
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
            if event.video_time_ms >= 0
            and (event.app_name or event.process_name)
            and _is_foreground_app_event(event)
            and not _is_whitelisted_app(event.app_name or event.process_name)
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


@dataclass(frozen=True)
class _ForegroundRangeIndex:
    ranges: tuple[tuple[int, int], ...]

    @classmethod
    def from_logs(cls, logs: list[LogEvent]) -> "_ForegroundRangeIndex":
        transitions = sorted(
            (event for event in logs if event.video_time_ms >= 0 and _is_foreground_transition(event)),
            key=lambda event: event.video_time_ms,
        )
        ranges: list[tuple[int, int]] = []
        active_start: int | None = None
        for event in transitions:
            if active_start is not None:
                ranges.append((active_start, event.video_time_ms - 1))
                active_start = None
            if _is_visible_foreground_app_event(event):
                active_start = event.video_time_ms
        if active_start is not None:
            session_end_ms = max((event.video_time_ms for event in logs if event.video_time_ms >= 0), default=active_start)
            ranges.append((active_start, session_end_ms))
        return cls(ranges=tuple(ranges))

    def between(self, start_ms: int, end_ms: int) -> tuple[tuple[int, int], ...]:
        if end_ms < start_ms:
            return ()
        return tuple(
            (max(range_start, start_ms), min(range_end, end_ms))
            for range_start, range_end in self.ranges
            if range_end >= start_ms and range_start <= end_ms
        )


@dataclass(frozen=True)
class _LogTimeIndex:
    times: tuple[int, ...]
    events: tuple[LogEvent, ...]

    @classmethod
    def from_logs(cls, logs: list[LogEvent]) -> "_LogTimeIndex":
        ordered = sorted((event for event in logs if event.video_time_ms >= 0), key=lambda event: event.video_time_ms)
        return cls(times=tuple(event.video_time_ms for event in ordered), events=tuple(ordered))

    def between(self, start_ms: int, end_ms: int) -> tuple[LogEvent, ...]:
        start = bisect_left(self.times, start_ms)
        end = bisect_right(self.times, end_ms)
        return self.events[start:end]


@dataclass(frozen=True)
class _SensitiveContextIndex:
    interval_starts: tuple[int, ...]
    interval_ends: tuple[int, ...]
    signal_times: tuple[int, ...]

    @classmethod
    def from_logs(
        cls,
        candidate_events: list[LogEvent],
        all_logs: list[LogEvent],
        sensitive_files: tuple[str, ...],
    ) -> "_SensitiveContextIndex":
        ordered = sorted(
            (event for event in candidate_events if event.video_time_ms >= 0),
            key=lambda event: event.video_time_ms,
        )
        session_end_ms = max((event.video_time_ms for event in all_logs if event.video_time_ms >= 0), default=-1)
        close_times: dict[str, list[int]] = {}
        for event in ordered:
            if not _is_sensitive_close_event(event) or not event.file_path:
                continue
            for key in _file_match_keys(event.file_path):
                close_times.setdefault(key, []).append(event.video_time_ms)

        intervals: list[tuple[int, int]] = []
        signal_times: list[int] = []
        for event in ordered:
            file_text = normalize_path(event.file_path).lower()
            if _matched_sensitive_files(event, sensitive_files) or looks_sensitive(file_text):
                signal_times.append(event.video_time_ms)
            if not _is_open_context_event(event) or not _event_has_sensitive_context_signal(event, sensitive_files):
                continue
            explicit_end = parse_timestamp_ms(event.raw.get("end_time"))
            if explicit_end and event.timestamp_ms:
                end_ms = event.video_time_ms + max(explicit_end - event.timestamp_ms, 0)
                intervals.append((event.video_time_ms, end_ms))
                continue
            if event.event_type.lower() not in {"file_open", "opened", "open", "read"} or not event.file_path:
                continue
            next_close = _next_close_ms(close_times, event.file_path, event.video_time_ms)
            intervals.append((event.video_time_ms, next_close if next_close is not None else session_end_ms))

        merged = _merge_time_intervals(intervals)
        return cls(
            interval_starts=tuple(start for start, _ in merged),
            interval_ends=tuple(end for _, end in merged),
            signal_times=tuple(sorted(set(signal_times))),
        )

    def has_open_context(self, center_ms: int) -> bool:
        if center_ms < 0 or not self.interval_starts:
            return False
        index = bisect_right(self.interval_starts, center_ms) - 1
        return index >= 0 and center_ms <= self.interval_ends[index]

    def has_recent_signal(self, center_ms: int, radius_ms: int = 30_000) -> bool:
        if center_ms < 0 or not self.signal_times:
            return False
        index = bisect_left(self.signal_times, max(0, center_ms - radius_ms))
        return index < len(self.signal_times) and self.signal_times[index] <= center_ms


def _file_match_keys(file_path: str) -> tuple[str, ...]:
    normalized = normalize_path(file_path).lower()
    if not normalized:
        return ()
    filename = normalized.rsplit("/", 1)[-1]
    return tuple(dict.fromkeys((normalized, filename)))


def _next_close_ms(close_times: dict[str, list[int]], file_path: str, after_ms: int) -> int | None:
    matches: list[int] = []
    for key in _file_match_keys(file_path):
        times = close_times.get(key, [])
        index = bisect_right(times, after_ms)
        if index < len(times):
            matches.append(times[index])
    return min(matches) if matches else None


def _merge_time_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start_ms, end_ms in sorted(intervals):
        if end_ms < start_ms:
            continue
        if not merged or start_ms > merged[-1][1] + 1:
            merged.append([start_ms, end_ms])
        else:
            merged[-1][1] = max(merged[-1][1], end_ms)
    return [(start_ms, end_ms) for start_ms, end_ms in merged]


def _attach_active_apps_to_windows(
    windows: list[AnalysisWindow],
    app_index: _ActiveAppIndex,
    foreground_range_index: _ForegroundRangeIndex,
) -> list[AnalysisWindow]:
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
            action_anchor_ms=window.action_anchor_ms,
            active_apps=app_index.between(window.start_ms, window.end_ms),
            active_ranges=foreground_range_index.between(window.start_ms, window.end_ms),
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
    has_strong_evidence = any(window.priority == "strong" for window in windows)
    has_direct_evidence = has_strong_evidence or any(window.anchor_ms for window in windows)
    if has_direct_evidence:
        if has_strong_evidence:
            kept = [_with_context_medium_budget(window, config) for window in kept]
        return sorted(kept, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))

    medium_windows = [window for window in windows if window.priority == "medium"]
    if medium_windows:
        kept.append(min(medium_windows, key=lambda item: (item.start_ms, item.end_ms)))
    return sorted(kept, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))


def _with_context_medium_budget(window: AnalysisWindow, config: VisionConfig) -> AnalysisWindow:
    if window.priority != "medium" or window.reason.startswith("sensitive_activity:"):
        return window
    budget = config.max_keyframes_per_medium_window
    return AnalysisWindow(
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        reason=window.reason,
        priority=window.priority,
        step_ms=window.step_ms,
        max_keyframes=budget,
        diff_threshold=window.diff_threshold,
        anchor_ms=window.anchor_ms,
        action_anchor_ms=window.action_anchor_ms,
        active_apps=window.active_apps,
        active_ranges=window.active_ranges,
    )


def _thin_dense_window_anchors(windows: list[AnalysisWindow], config: VisionConfig) -> list[AnalysisWindow]:
    return [_with_thinned_anchors(window, config) for window in windows]


def _with_thinned_anchors(window: AnalysisWindow, config: VisionConfig) -> AnalysisWindow:
    base_budget = min(window.max_keyframes, _base_keyframe_budget(window.priority, config))
    anchors = _prioritize_action_anchors(window, base_budget, config)
    return AnalysisWindow(
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        reason=window.reason,
        priority=window.priority,
        step_ms=window.step_ms,
        max_keyframes=max(base_budget, len(anchors)),
        diff_threshold=window.diff_threshold,
        anchor_ms=anchors,
        action_anchor_ms=tuple(anchor for anchor in window.action_anchor_ms if anchor in anchors),
        active_apps=window.active_apps,
        active_ranges=window.active_ranges,
    )


def _prioritize_action_anchors(window: AnalysisWindow, base_budget: int, config: VisionConfig) -> tuple[int, ...]:
    if "file_selected" in window.reason.lower() or "file_dialog" in window.reason.lower():
        return _thin_anchors(
            window.anchor_ms,
            _anchor_min_gap_ms(window.priority, config),
            limit=base_budget,
        )
    actions = tuple(sorted(set(window.action_anchor_ms)))
    others = tuple(anchor for anchor in window.anchor_ms if anchor not in actions)
    action_limit = len(actions) if base_budget <= 0 else min(len(actions), base_budget)
    selected_actions = _pick_evenly_spaced(actions, action_limit)
    remaining = max(0, base_budget - len(selected_actions))
    selected_others = _thin_anchors(
        others,
        _anchor_min_gap_ms(window.priority, config),
        limit=remaining,
    )
    return tuple(sorted({*selected_actions, *selected_others}))


def _thin_anchors(anchors: tuple[int, ...], min_gap_ms: int, *, limit: int | None = None) -> tuple[int, ...]:
    if min_gap_ms <= 0 or len(anchors) <= 1:
        thinned = tuple(sorted(set(anchors)))
        return _pick_evenly_spaced(thinned, limit) if limit is not None else thinned
    thinned: list[int] = []
    for anchor in sorted(set(anchors)):
        if thinned and anchor - thinned[-1] < min_gap_ms:
            continue
        thinned.append(anchor)
    result = tuple(thinned)
    return _pick_evenly_spaced(result, limit) if limit is not None else result


def _pick_evenly_spaced(values: tuple[int, ...], limit: int | None) -> tuple[int, ...]:
    if limit is None or limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return (values[0],)
    last = len(values) - 1
    picked = [values[round(index * last / (limit - 1))] for index in range(limit)]
    return tuple(dict.fromkeys(picked))


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
    if window.priority == "activity":
        return window.start_ms
    if window.priority == "strong":
        return max(0, window.end_ms - config.strong_window_after_ms)
    if window.priority == "weak":
        return max(0, window.end_ms - config.frame_window_after_ms)
    return max(0, window.end_ms - config.frame_window_after_ms)


def _priority_sort_key(priority: str) -> int:
    return {"strong": 0, "activity": 1, "medium": 2, "weak": 3}.get(priority, 2)


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


def _matched_sensitive_files(event: LogEvent, sensitive_files: tuple[str, ...]) -> tuple[str, ...]:
    file_text = normalize_path(event.file_path).lower()
    direct_matches = _sensitive_matches_for_path(file_text, sensitive_files)
    if direct_matches or not _event_may_embed_sensitive_path(event):
        return direct_matches

    search_text = _normalize_sensitive_search_text(_event_search_text(event))
    return tuple(
        sensitive_file
        for sensitive_file, full_path, filename, stem in _sensitive_reference_index(sensitive_files)
        if full_path in search_text
        or (filename and filename in search_text)
        or (len(stem) >= 4 and stem in search_text)
    )


@lru_cache(maxsize=65_536)
def _sensitive_matches_for_path(file_text: str, sensitive_files: tuple[str, ...]) -> tuple[str, ...]:
    """Match repeated log paths once instead of once per event and phase."""

    normalized = normalize_path(file_text).lower()
    if not normalized:
        return ()
    matches: list[str] = []
    for sensitive_file in sensitive_files:
        if not sensitive_file:
            continue
        filename = sensitive_file.rsplit("/", 1)[-1]
        stem = filename.rsplit(".", 1)[0]
        if (
            sensitive_file in normalized
            or (filename and filename in normalized)
            or (len(stem) >= 4 and stem in normalized)
        ):
            matches.append(sensitive_file)
    return tuple(matches)


def _may_need_analysis_window(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    if event.video_time_ms < 0:
        return False
    if _is_system_noise_path(event.file_path):
        text = _event_search_text(event)
        if not (
            _looks_like_file_selection_dialog(event.window_title)
            and _is_sink_context((event.process_name or "").lower(), text)
        ):
            return False
    if _matched_sensitive_files(event, sensitive_files) or looks_sensitive(normalize_path(event.file_path).lower()):
        return True

    event_type = event.event_type.lower()
    if event_type in {
        "app_switch",
        "window_changed",
        "window_closed",
        "clipboard_text",
        "clipboard_image",
        "screenshot",
        "screen_capture",
        "file_selected",
        "file_upload",
        "upload",
        "uploaded",
        "upload_complete",
        "send",
        "sent",
        "print_to_pdf",
        "save_as",
        "export",
        "copied",
    }:
        return True
    if _looks_like_screenshot_path(event.file_path) or _is_sink_app_process(event.process_name or ""):
        return True

    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    return any(token in raw_operation for token in ("copy", "paste", "clipboard", "screenshot", "screen_capture", "export", "print", "save_as", "send", "compress", "base64", "encode", "decode"))


def _event_may_embed_sensitive_path(event: LogEvent) -> bool:
    event_type = event.event_type.lower()
    if event_type in {"app_switch", "window_changed", "window_closed", "clipboard_text", "clipboard_image", "screenshot", "screen_capture"}:
        return True
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    return any(token in raw_operation for token in ("copy", "paste", "clipboard", "screenshot", "screen_capture", "export", "print", "save_as", "compress", "base64", "encode", "decode"))


def _is_sensitive_close_event(event: LogEvent) -> bool:
    return event.event_type.lower() in {"closed", "close", "file_closed", "file_close"}


@lru_cache(maxsize=1)
def _whitelisted_apps() -> frozenset[str]:
    profile = Path(__file__).resolve().parents[2] / "spec" / "config" / "system_noise_profile.json"
    try:
        payload = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    configured = payload.get("app_names") if isinstance(payload, dict) else []
    defaults = {"system", "msmpeng", "svchost", "runtimebroker", "ffmpeg"}
    values = configured if isinstance(configured, list) else []
    return frozenset({*defaults, *(str(item).strip().lower() for item in values if str(item).strip())})


def _is_whitelisted_app(app_name: str) -> bool:
    normalized = str(app_name or "").strip().lower()
    return normalized in _whitelisted_apps()


@lru_cache(maxsize=1)
def _system_noise_path_markers() -> tuple[str, ...]:
    profile = Path(__file__).resolve().parents[2] / "spec" / "config" / "system_noise_profile.json"
    try:
        payload = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    configured = payload.get("path_markers") if isinstance(payload, dict) else []
    if not isinstance(configured, list):
        return ()
    return tuple(normalize_path(str(item)).lower() for item in configured if str(item).strip())


def _is_system_noise_path(path: str) -> bool:
    normalized = normalize_path(path).lower()
    return bool(normalized) and any(marker in normalized for marker in _system_noise_path_markers())


@lru_cache(maxsize=1)
def _whitelisted_window_title_markers() -> tuple[str, ...]:
    profile = Path(__file__).resolve().parents[2] / "spec" / "config" / "system_noise_profile.json"
    try:
        payload = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    configured = payload.get("window_title_markers") if isinstance(payload, dict) else []
    if not isinstance(configured, list):
        return ()
    return tuple(str(item).strip().lower() for item in configured if str(item).strip())


def _is_whitelisted_window_title(title: str) -> bool:
    normalized = str(title or "").strip().lower()
    return bool(normalized) and any(marker in normalized for marker in _whitelisted_window_title_markers())


def _is_foreground_app_event(event: LogEvent) -> bool:
    return _is_visible_foreground_app_event(event)


def _foreground_app_ranges(logs: list[LogEvent], start_ms: int, end_ms: int) -> tuple[tuple[int, int], ...]:
    """Return intervals with a known visible, non-whitelisted foreground app.

    App-switch events delimit foreground state. A blank shell switch or a
    desktop/program-manager title deliberately closes the preceding interval,
    so activity windows cannot turn idle desktop time into VLM input.
    """

    if end_ms < start_ms:
        return ()

    active_start: int | None = None
    ranges: list[tuple[int, int]] = []
    for event in sorted(logs, key=lambda item: item.video_time_ms):
        if event.video_time_ms < 0 or event.video_time_ms > end_ms or not _is_foreground_transition(event):
            continue
        if active_start is not None:
            ranges.append((active_start, event.video_time_ms - 1))
            active_start = None
        if _is_visible_foreground_app_event(event):
            active_start = event.video_time_ms

    if active_start is not None:
        ranges.append((active_start, end_ms))

    return tuple(
        (max(range_start, start_ms), min(range_end, end_ms))
        for range_start, range_end in ranges
        if range_end >= start_ms and range_start <= end_ms and range_start <= range_end
    )


def _is_foreground_transition(event: LogEvent) -> bool:
    return event.event_type.lower() in {"app_switch", "window_changed", "window_closed"}


def _is_visible_foreground_app_event(event: LogEvent) -> bool:
    app_name = str(event.app_name or event.process_name or "").strip()
    if not app_name or _is_whitelisted_app(app_name):
        return False

    title = str(event.window_title or "").strip().lower()
    if _is_whitelisted_window_title(title) or _is_desktop_title(title):
        return False
    # Known untitled shell/overlay transitions (for example wallpaper
    # helpers) do not identify a visible user-facing application.
    if not title and (_is_shell_app(app_name) or _is_noninteractive_overlay_app(app_name)):
        return False
    return bool(title) or _is_foreground_transition(event)


def _is_desktop_title(title: str) -> bool:
    return "program manager" in title or "desktop" in title or "桌面" in title


def _is_shell_app(app_name: str) -> bool:
    normalized = app_name.strip().lower().removesuffix(".exe")
    return normalized in {"explorer", "file explorer", "windows explorer"}


def _is_noninteractive_overlay_app(app_name: str) -> bool:
    normalized = app_name.strip().lower().removesuffix(".exe")
    return normalized in {"kwallpaper", "wallpaperhost", "wallpaper32"}


def _nearby_event_search_text(
    logs: list[LogEvent],
    center_ms: int,
    radius_ms: int,
    *,
    index: _LogTimeIndex | None = None,
) -> str:
    if center_ms < 0 or radius_ms <= 0:
        return ""
    parts: list[str] = []
    candidates = index.between(center_ms - radius_ms, center_ms + radius_ms) if index is not None else logs
    for item in candidates:
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
    normalized_file = normalize_path(file_text).lower()
    references = _sensitive_reference_index(sensitive_files)
    if any(full_path and full_path in normalized_file for _, full_path, _, _ in references):
        return True

    search_text = _normalize_sensitive_search_text(text)
    for _, full_path, filename, stem in references:
        if full_path and full_path in search_text:
            return True
        if filename and filename in search_text:
            return True
        if len(stem) >= 4 and stem in search_text:
            return True
    return False


@lru_cache(maxsize=128)
def _sensitive_reference_index(
    sensitive_files: tuple[str, ...],
) -> tuple[tuple[str, str, str, str], ...]:
    references: list[tuple[str, str, str, str]] = []
    for sensitive_file in sensitive_files:
        normalized = normalize_path(sensitive_file).lower()
        if not normalized:
            continue
        filename = normalized.rsplit("/", 1)[-1]
        stem = filename.rsplit(".", 1)[0]
        references.append((sensitive_file, normalized, filename, stem))
    return tuple(references)


def _normalize_sensitive_search_text(text: str) -> str:
    """Normalize event prose without treating a large JSON blob as a path."""

    value = str(text or "")
    if len(value) <= 2_048:
        return normalize_path(value).lower()
    return value.lower().replace("\\", "/")


def _window_priority(
    event: LogEvent,
    text: str,
    sensitive_hit: bool,
    transfer_hit: bool,
    sink_hit: bool,
    sensitive_context_hit: bool = False,
) -> str:
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
    removable_context = _is_removable_media_context(f"{text} {window_title}")

    if event_type in {"app_switch", "window_changed", "window_closed"} and not _is_visible_foreground_app_event(event):
        return "none"

    strong_event_types = {"file_selected", "file_upload", "upload", "uploaded", "upload_complete", "send", "sent"}
    strong_raw_ops = {"file_selected", "file_upload", "upload", "send", "send_click"}
    if event_type in strong_event_types or raw_operation in strong_raw_ops:
        return "strong"
    if _has_structured_file_upload_signal(event):
        return "strong"
    if process_name == "fsquirt.exe":
        return "strong"
    if sensitive_hit and removable_context and event_type in {
        "created",
        "modified",
        "renamed",
        "moved",
        "copied",
        "file_created",
        "file_modified",
        "file_renamed",
        "file_moved",
        "file_copied",
    }:
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

    if _is_clipboard_or_capture_event(event, text) and sensitive_context_hit:
        return "strong"
    if event_type in {"app_switch", "window_changed", "window_closed"} and _is_sink_app_process(process_name) and sensitive_context_hit:
        return "medium"
    if _is_derivation_candidate_event(event, text, sensitive_context_hit=sensitive_context_hit):
        # A conversion, export, screenshot, or copy is a reportable action in
        # its own right. Give it the same focused temporal context as an
        # upload rather than relying on broad sensitive-file activity probes.
        return "strong" if sensitive_context_hit else "medium"

    if sensitive_hit:
        return "medium"
    if sink_context and event_type == "clipboard_text":
        return "medium"

    if str(extra.get("risk_level") or "").lower() in {"高", "high"}:
        return "weak"
    return "none"


def _event_anchors(
    event: LogEvent,
    sensitive_hit: bool,
    action_hit: bool,
    text: str = "",
    sensitive_context_hit: bool = False,
) -> tuple[int, ...]:
    process_name = (event.process_name or "").lower()
    window_title = event.window_title or ""
    event_type = event.event_type.lower()
    if event.event_type == "app_switch" and process_name == "fsquirt.exe":
        return (event.video_time_ms,)
    if process_name == "fsquirt.exe" and event.file_path and sensitive_hit:
        return (event.video_time_ms, event.video_time_ms + 3_000, event.video_time_ms + 8_000)
    if sensitive_hit and _is_removable_media_context(f"{text} {window_title}"):
        return (event.video_time_ms, event.video_time_ms + 3_000, event.video_time_ms + 8_000)
    if event_type in {"app_switch", "window_changed"} and sensitive_hit:
        return (event.video_time_ms,)
    if _looks_like_file_selection_dialog(window_title) and _is_sink_context(process_name, text):
        if event_type in {"app_switch", "window_changed", "window_closed"}:
            offsets = (0, 3_000)
            if _is_workspace_upload_process(process_name):
                offsets = (0, 3_000, 8_000, 16_000, 20_000)
            return tuple(event.video_time_ms + offset for offset in offsets)
        return (event.video_time_ms,)
    if _looks_like_print_or_save_dialog(window_title):
        return (event.video_time_ms,)
    if event.event_type.lower() in {"app_switch", "window_changed", "window_closed"} and _is_sink_app_process(process_name):
        offsets = (0, 3_000, 8_000) if sensitive_context_hit else (0,)
        return tuple(event.video_time_ms + offset for offset in offsets)
    if event_type in {"app_switch", "window_changed", "window_closed"} and _is_sink_context(process_name, text) and _looks_like_upload_progress(text):
        return (event.video_time_ms,)
    if _is_clipboard_or_capture_event(event, text) and sensitive_context_hit:
        return (event.video_time_ms,)
    if _is_derivation_candidate_event(event, text, sensitive_context_hit=sensitive_context_hit):
        return (event.video_time_ms,)
    if sensitive_hit and action_hit:
        return (event.video_time_ms,)
    if sensitive_hit and _looks_like_screenshot_path(event.file_path):
        return (event.video_time_ms,)
    if event_type in {"file_selected", "file_upload", "upload", "uploaded", "upload_complete"}:
        return (event.video_time_ms,)
    if _has_structured_file_upload_signal(event):
        return (event.video_time_ms,)
    return ()


def _is_derivation_candidate_event(event: LogEvent, text: str, *, sensitive_context_hit: bool = False) -> bool:
    event_type = event.event_type.lower()
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    window_title = event.window_title or ""
    direct_derivation_events = {
        "print_to_pdf",
        "save_as",
        "export",
        "copied",
    }
    file_change_events = {
        "created",
        "modified",
        "renamed",
        "moved",
        "file_created",
        "file_modified",
        "file_renamed",
    }
    context_bound_ops = {
        "copy",
        "paste",
        "clipboard",
        "screenshot",
        "screen_capture",
    }
    derivation_ops = {
        "export",
        "print",
        "print_to_pdf",
        "save_as",
        "rename",
        "compress",
        "translate",
        "base64",
    }
    lowered_text = text.lower()
    if event_type in direct_derivation_events:
        return True
    if any(token in raw_operation for token in context_bound_ops):
        return sensitive_context_hit
    if raw_operation in derivation_ops:
        return True
    if any(token in raw_operation for token in derivation_ops):
        return True
    if event_type in file_change_events:
        return sensitive_context_hit and _looks_like_screenshot_path(event.file_path)
    if any(token in lowered_text for token in ("translation", "translate", "base64", "encode", "decode")):
        return True
    if _looks_like_print_or_save_dialog(window_title):
        return True
    if event_type in {"app_switch", "window_changed", "window_closed"} and (
        _looks_like_capture_context(text) or _looks_like_capture_context(window_title)
    ):
        return sensitive_context_hit
    if event_type in {"app_switch", "window_changed", "window_closed"} and contains_any(text, TRANSFER_TOKENS):
        return True
    return False


def _is_clipboard_or_capture_event(event: LogEvent, text: str = "") -> bool:
    event_type = event.event_type.lower()
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    foreground_event = event_type in {"app_switch", "window_changed", "window_closed"}
    return (
        event_type in {"clipboard_text", "clipboard_image", "screenshot", "screen_capture"}
        or any(token in raw_operation for token in ("copy", "paste", "clipboard", "screenshot", "screen_capture"))
        or _looks_like_screenshot_path(event.file_path)
        or (foreground_event and (_looks_like_capture_context(text) or _looks_like_capture_context(event.window_title)))
    )


def _has_sensitive_open_context(
    logs: list[LogEvent],
    center_ms: int,
    sensitive_files: tuple[str, ...],
    *,
    time_index: _LogTimeIndex | None = None,
) -> bool:
    if center_ms < 0:
        return False

    candidates = time_index.between(0, center_ms) if time_index is not None else logs
    for item in candidates:
        if not _is_open_context_event(item):
            continue
        if _event_has_sensitive_context_signal(item, sensitive_files):
            if _event_interval_contains(item, center_ms) or _event_is_open_until_closed(logs, item, center_ms):
                return True
        elif _foreground_interval_contains_sensitive_title(logs, item, center_ms, sensitive_files):
            return True
    return False


def _has_recent_sensitive_signal(
    logs: list[LogEvent],
    center_ms: int,
    sensitive_files: tuple[str, ...],
    *,
    time_index: _LogTimeIndex | None = None,
) -> bool:
    """Recover clipboard provenance when the source-open event is incomplete."""

    if center_ms < 0:
        return False
    candidates = time_index.between(max(0, center_ms - 30_000), center_ms) if time_index is not None else logs
    for item in candidates:
        if item.video_time_ms < 0 or item.video_time_ms > center_ms:
            continue
        file_text = normalize_path(item.file_path).lower()
        if _matched_sensitive_files(item, sensitive_files) or looks_sensitive(file_text):
            return True
    return False


def _is_open_context_event(event: LogEvent) -> bool:
    return event.event_type.lower() in {
        "app_switch",
        "window_changed",
        "window_closed",
        "file_open",
        "opened",
        "open",
        "read",
        "modified",
        "created",
        "renamed",
        "file_created",
        "file_modified",
        "file_renamed",
        "save_as",
        "export",
    }


def _event_has_sensitive_context_signal(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    text = _event_search_text(event)
    file_text = normalize_path(event.file_path).lower()
    return _matches_sensitive_source(file_text, text, sensitive_files) or looks_sensitive(file_text) or looks_sensitive(text)


def _event_interval_contains(event: LogEvent, center_ms: int) -> bool:
    end_time = parse_timestamp_ms(event.raw.get("end_time"))
    if not end_time or not event.timestamp_ms or event.video_time_ms < 0:
        return False
    end_video_ms = event.video_time_ms + max(end_time - event.timestamp_ms, 0)
    return event.video_time_ms <= center_ms <= end_video_ms


def _event_is_open_until_closed(logs: list[LogEvent], event: LogEvent, center_ms: int) -> bool:
    if event.video_time_ms < 0 or event.video_time_ms > center_ms:
        return False
    if parse_timestamp_ms(event.raw.get("end_time")):
        return False
    if event.event_type.lower() not in {"file_open", "opened", "open", "read"}:
        return False
    if not event.file_path:
        return False
    for item in logs:
        if item.video_time_ms <= event.video_time_ms or item.video_time_ms > center_ms:
            continue
        if item.event_type.lower() not in {"closed", "file_closed", "close"}:
            continue
        if same_file(item.file_path, event.file_path):
            return False
    return True


def _foreground_interval_contains_sensitive_title(
    logs: list[LogEvent],
    event: LogEvent,
    center_ms: int,
    sensitive_files: tuple[str, ...],
) -> bool:
    if event.video_time_ms < 0 or event.video_time_ms > center_ms:
        return False
    if event.event_type.lower() not in {"app_switch", "window_changed"}:
        return False
    if not _event_has_sensitive_context_signal(event, sensitive_files):
        return False

    next_foreground_ms = None
    for item in logs:
        if item.video_time_ms <= event.video_time_ms:
            continue
        if item.event_type.lower() not in {"app_switch", "window_changed", "window_closed"}:
            continue
        next_foreground_ms = item.video_time_ms
        break
    if next_foreground_ms is None:
        return True
    return center_ms <= next_foreground_ms


def _is_sink_app_process(process_name: str) -> bool:
    return process_name.lower() in {
        "qq.exe",
        "wechat.exe",
        "weixin.exe",
        "tim.exe",
        "dingtalk.exe",
        "feishu.exe",
        "lark.exe",
        "doubao.exe",
        "cherrystudio.exe",
        "cherry studio.exe",
        "baidunetdisk.exe",
        "baidunetdiskunite.exe",
    }


def _is_workspace_upload_process(process_name: str) -> bool:
    return process_name.lower() in {"feishu.exe", "lark.exe"}


def _is_sink_context(process_name: str, text: str) -> bool:
    return _is_sink_app_process(process_name) or _is_cloud_drive_context(text)


def _is_removable_media_context(text: str) -> bool:
    normalized = (text or "").lower()
    return any(
        token in normalized
        for token in (
            "usb",
            "removable",
            "flash drive",
            "thumb drive",
            "u disk",
            "u盘",
            "u 盘",
            "可移动",
            "移动硬盘",
        )
    )


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
        )
    )


def _looks_like_screenshot_path(path: str) -> bool:
    text = normalize_path(path).lower()
    if not text:
        return False
    if text.endswith(".svg") or "/resource/icons/" in text or "/resources/app/resource/icons/" in text:
        return False
    image_ext = text.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    filename = text.rsplit("/", 1)[-1]
    return (
        "screenshot" in filename
        or "screen shot" in filename
        or "屏幕截图" in filename
        or ("/screenshots/" in text and image_ext)
    )


def _looks_like_capture_context(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "screenshot",
            "screen capture",
            "snipping tool",
            "截图",
            "截屏",
            "屏幕截图",
        )
    )


def _looks_like_print_or_save_dialog(window_title: str) -> bool:
    title = (window_title or "").lower()
    if not title:
        return False
    return any(
        token in title
        for token in (
            "print",
            "save as",
            "save print output",
            "export to pdf",
            "output to pdf",
            "输出为pdf",
            "输出到pdf",
            "导出pdf",
            "打印",
            "另存",
            "保存",
        )
    )


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


def _window_reason(event: LogEvent, priority: str, *, action_label: str = "") -> str:
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    source = str(extra.get("source") or "")
    category = str(extra.get("category") or "")
    parts = [priority, event.event_type or "activity", source, category, action_label]
    return ":".join(item for item in parts if item)


def _window_action_label(event: LogEvent, text: str, sensitive_context_hit: bool) -> str:
    """Persist the behavior that made a visual window worth opening."""

    event_type = event.event_type.lower()
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    process_name = (event.process_name or "").lower()
    direct_text = _event_search_text(event)
    if event_type in {"file_selected", "file_upload", "upload", "uploaded", "upload_complete", "send", "sent"}:
        return "upload"
    if raw_operation in {"file_selected", "file_upload", "upload", "send", "send_click"}:
        return "upload"
    if _has_structured_file_upload_signal(event):
        return "upload"
    if _looks_like_file_selection_dialog(event.window_title) and _is_sink_context(process_name, direct_text):
        return "file_selected"
    if _looks_like_upload_progress(direct_text):
        return "upload_progress"
    if _is_clipboard_or_capture_event(event, direct_text) and sensitive_context_hit:
        return "capture" if _looks_like_capture_context(f"{direct_text} {event.window_title}") else "clipboard"
    if _has_direct_derivation_signal(event, direct_text):
        return "derivation"
    if _is_removable_media_context(f"{direct_text} {event.window_title}") and sensitive_context_hit:
        return "removable_transfer"
    return ""


def _has_structured_file_upload_signal(event: LogEvent) -> bool:
    """Recognize file-dialog selections emitted as ordinary file-open events."""

    upload = event.raw.get("upload_detection") if isinstance(event.raw.get("upload_detection"), dict) else {}
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    is_upload = upload.get("is_upload") is True or str(upload.get("is_upload") or "").lower() == "true"
    source = str(extra.get("source") or "").lower()
    return (
        is_upload
        and event.event_type.lower() in {"opened", "file_selected", "file_upload", "upload"}
        and source in {"recent_folder_monitor", "file_dialog_monitor"}
        and bool(normalize_path(event.file_path))
    )


def _has_direct_derivation_signal(event: LogEvent, text: str) -> bool:
    """Avoid turning nearby document activity into an action label."""

    event_type = event.event_type.lower()
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    derivation_operations = ("export", "print", "save_as", "rename", "compress", "translate", "base64")
    return (
        event_type in {"print_to_pdf", "save_as", "export", "copied"}
        or any(token in raw_operation for token in derivation_operations)
        or _looks_like_print_or_save_dialog(event.window_title)
        or _looks_like_screenshot_path(event.file_path)
    )
