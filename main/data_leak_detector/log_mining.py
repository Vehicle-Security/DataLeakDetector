"""Turn normalized audit events into small, evidence-oriented video windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
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
        # The graph query is a shortlist, not the source of truth for window
        # semantics. Merge local candidates so newly supported evidence (for
        # example a screenshot tool writing a sensitive clipboard image) is
        # not silently omitted until the Neo4j query is updated as well.
        local_windows = build_analysis_windows(logs, sensitive_files, vision_config)
        windows = _finalize_windows([*action_windows, *local_windows], activity_windows, logs)
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
    clipboard_sensitive_events = _sensitive_clipboard_event_ids(ordered, sensitive)
    candidates = [
        event
        for event in ordered
        if event.event_type in _FOREGROUND_EVENTS
        or _may_need_analysis_window(event, sensitive)
        or _may_be_derived_file_event(event, sensitive)
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
            sensitive_context_index=clipboard_sensitive_events,
            normalized_sensitive=True,
        )
        if window is not None:
            actions.append(window)
    outbound_context = _build_outbound_context_windows(ordered, actions, config, sensitive_files=sensitive)
    covered_phases = {
        phase
        for window in outbound_context
        for phase in window.action_phases
        if phase[1] not in {"external_session", "session_end"}
    }
    visible_actions = [
        window
        for window in actions
        if not any(action == "transfer_anchor" for _, action in window.action_phases)
        and any(phase not in covered_phases for phase in window.action_phases)
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
    del active_app_index, time_index
    if event.video_time_ms < 0:
        return None
    sensitive = tuple(sensitive_files) if normalized_sensitive else _normalize_sensitive_files(sensitive_files)
    if (
        _is_noise_only_event(event)
        and not _event_matches_sensitive(event, sensitive)
        and not (
            isinstance(sensitive_context_index, set | frozenset)
            and event.event_id in sensitive_context_index
        )
    ):
        return None
    timeline = sensitive_timeline or _SensitiveTimeline.from_logs(logs, sensitive)
    direct_sensitive = _event_matches_sensitive(event, sensitive)
    path_sensitive = bool(event.file_path) and any(same_file(event.file_path, source) for source in sensitive)
    direct_derivation = _may_be_derived_file_event(event, sensitive)
    action = _action_kind(event)
    recent_context_ms = 120_000 if action in {"clipboard", "paste"} else 30_000
    if action in {"clipboard", "paste"} and isinstance(sensitive_context_index, set | frozenset):
        sensitive_context = direct_sensitive or direct_derivation or event.event_id in sensitive_context_index
    else:
        sensitive_context = (
            direct_sensitive
            or direct_derivation
            or timeline.active_at(event.video_time_ms)
            or timeline.recent_at(event.video_time_ms, radius_ms=recent_context_ms)
        )
    if action == "transfer_anchor" and direct_sensitive:
        # A browser access to a known sensitive document is the file-selection
        # phase of an outbound action, even if the monitor labels it as IO.
        action = "file_selected"
    if action == "transfer_anchor" and not direct_sensitive:
        action = ""
    if not action and direct_derivation:
        app_identity = identify_frontend_app(
            app_name=event.app_name or event.process_name,
            window_title=event.window_title,
        )
        action = "file_selected" if app_identity.risk_hint.startswith("external_capable") else "derive"
    if not action and direct_sensitive and event.file_path:
        app_identity = identify_frontend_app(
            app_name=event.app_name or event.process_name,
            window_title=event.window_title,
        )
        if app_identity.risk_hint.startswith("external_capable"):
            action = "resource_identity" if event.event_type in _FOREGROUND_EVENTS else "file_selected"
    priority = "none"
    if action in {"upload", "send", "file_selected", "resource_identity", "transfer_anchor"}:
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
        if action == "derive":
            # Archive extraction and document export logs are often emitted only
            # after the UI operation completes. Keep the visible source action.
            before_ms = max(before_ms, 20_000)
        if action == "clipboard":
            before_ms = max(before_ms, 30_000)
        if action in {"file_selected", "upload", "send", "removable", "screen_share", "clipboard"}:
            after_ms = max(after_ms, 30_000)
        if action == "file_selected" and _is_mail_attachment_context(event):
            # Mail clients frequently emit only a picker-selection log. The
            # send button/confirmation can remain visible much later, so keep
            # a result frame across that otherwise unobserved gap.
            after_ms = max(after_ms, 90_000)
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
        requires_post_action_state=action in {
            "clipboard",
            "file_selected",
            "paste",
            "removable",
            "screen_share",
            "send",
            "upload",
        },
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


@dataclass(frozen=True)
class _FrontendSession:
    start_ms: int
    end_ms: int
    app_key: str
    app_name: str
    category: str
    state_ms: tuple[int, ...] = ()


_EVIDENCE_SESSION_CATEGORIES = frozenset(
    {
        "ai_chat",
        "chat",
        "cloud_drive",
        "code_hosting",
        "community",
        "external_sink",
        "mail",
        "meeting",
        "removable_media",
        "workplace",
    }
)


def _build_outbound_context_windows(
    logs: list[LogEvent],
    action_windows: list[AnalysisWindow],
    config: VisionConfig,
    *,
    sensitive_files: tuple[str, ...] = (),
    evidence_horizon_ms: int = 120_000,
    session_gap_ms: int = 30_000,
) -> list[AnalysisWindow]:
    """Build one evidence packet for each concrete external-application session."""

    phases = sorted(
        {
            phase
            for window in action_windows
            for phase in (window.action_phases or tuple((anchor, "action") for anchor in window.action_anchor_ms))
        }
    )
    sessions = [
        segment
        for session in _frontend_sessions(
            logs,
            session_gap_ms=session_gap_ms,
            sensitive_files=sensitive_files,
        )
        for segment in _segment_frontend_session(session, config.external_session_segment_ms)
    ]
    phase_owner: dict[tuple[int, str], int] = {}
    for phase in phases:
        source_phase = phase[1] in {"capture", "clipboard", "derive", "paste"}
        phase_horizon_ms = _phase_evidence_horizon_ms(phase[1], evidence_horizon_ms)
        candidates = [
            (
                0 if session.start_ms <= phase[0] <= session.end_ms else 1,
                max(0, session.start_ms - phase[0]),
                index,
            )
            for index, session in enumerate(sessions)
            if session.category in _EVIDENCE_SESSION_CATEGORIES
            and (
                (
                    source_phase
                    and session.end_ms >= phase[0]
                    and session.start_ms - phase[0] <= phase_horizon_ms
                )
                or (
                    not source_phase
                    and session.start_ms - 30_000 <= phase[0] <= session.end_ms
                )
            )
        ]
        if candidates:
            phase_owner[phase] = min(candidates)[2]

    sensitive_session_owners = _sensitive_evidence_session_owners(
        sessions,
        logs,
        sensitive_files,
        evidence_horizon_ms=evidence_horizon_ms,
    )

    windows: list[AnalysisWindow] = []
    for session_index, session in enumerate(sessions):
        if session.category not in _EVIDENCE_SESSION_CATEGORIES:
            continue

        nearby_phases = [
            phase
            for phase in phases
            if session.start_ms - _phase_evidence_horizon_ms(phase[1], evidence_horizon_ms)
            <= phase[0]
            <= session.end_ms
            and (phase not in phase_owner or phase_owner[phase] == session_index)
            and (
                session.start_ms - 30_000 <= phase[0]
                or phase[1] in {"capture", "clipboard", "derive", "paste"}
            )
        ]
        accesses = _outbound_sensitive_file_accesses(
            logs,
            sensitive_files,
            start_ms=max(0, session.start_ms - 5_000),
            end_ms=session.end_ms,
        )
        nearby_phases.extend((event.video_time_ms, "file_selected") for event in accesses)
        nearby_phases = list(_collapse_action_phases(tuple(nearby_phases)))
        if not nearby_phases and not _session_warrants_visual_evidence(
            session,
            logs,
            sensitive_files,
            evidence_horizon_ms=evidence_horizon_ms,
            session_index=session_index,
            sensitive_session_owners=sensitive_session_owners,
        ):
            # Merely opening an external-capable app is not evidence. A session
            # packet needs nearby sensitive context, an action, or a substantive
            # meeting session where the visual channel carries the evidence.
            continue

        phase_starts = [
            max(0, timestamp - config.strong_window_before_ms)
            for timestamp, _ in nearby_phases
            if timestamp <= session.end_ms
        ]
        start_ms = max(0, min([session.start_ms, *phase_starts]))
        phase_end = max((timestamp for timestamp, _ in nearby_phases), default=session.end_ms)
        end_ms = max(session.end_ms, phase_end + (30_000 if nearby_phases else 0))
        session_phases = _collapse_action_phases(
            (
                (session.start_ms, "external_session"),
                *((timestamp, "external_state") for timestamp in session.state_ms if timestamp != session.start_ms),
                *nearby_phases,
                (session.end_ms, "session_end"),
            )
        )
        anchors = tuple(sorted({timestamp for timestamp, _ in session_phases}))
        windows.append(
            AnalysisWindow(
                start_ms,
                end_ms,
                f"strong:external_session:{session.category}",
                priority="strong",
                step_ms=config.strong_frame_step_ms,
                max_keyframes=config.max_keyframes_per_strong_window,
                diff_threshold=config.strong_frame_diff_threshold,
                anchor_ms=anchors,
                action_anchor_ms=anchors,
                action_phases=session_phases,
                requires_post_action_state=True,
                active_apps=(session.app_name,),
            )
        )
    return windows


def _phase_evidence_horizon_ms(action: str, evidence_horizon_ms: int) -> int:
    # A captured image can be selected later by explicit file evidence. Without
    # that evidence, a distant external session is too weak to bind to the shot.
    if action == "capture":
        return min(evidence_horizon_ms, 30_000)
    return evidence_horizon_ms


def _segment_frontend_session(session: _FrontendSession, segment_ms: int) -> list[_FrontendSession]:
    if segment_ms <= 0 or session.end_ms - session.start_ms <= segment_ms:
        return [session]
    segments: list[_FrontendSession] = []
    start_ms = session.start_ms
    while start_ms <= session.end_ms:
        end_ms = min(session.end_ms, start_ms + segment_ms)
        segments.append(
            _FrontendSession(
                start_ms,
                end_ms,
                session.app_key,
                session.app_name,
                session.category,
                tuple(timestamp for timestamp in session.state_ms if start_ms <= timestamp <= end_ms),
            )
        )
        start_ms = end_ms + 1
    return segments


def _frontend_sessions(
    logs: list[LogEvent],
    *,
    session_gap_ms: int,
    sensitive_files: tuple[str, ...] = (),
) -> list[_FrontendSession]:
    foreground = sorted(
        (
            event
            for event in logs
            if event.video_time_ms >= 0 and event.event_type in _FOREGROUND_EVENTS
        ),
        key=lambda event: (event.video_time_ms, event.event_id),
    )
    if not foreground:
        return []
    timeline_end = max((event.video_time_ms for event in logs if event.video_time_ms >= 0), default=foreground[-1].video_time_ms)
    sessions: list[_FrontendSession] = []
    previous_identity = None
    previous_visible_index = -2
    for index, event in enumerate(foreground):
        if not _visible_foreground(event):
            previous_identity = None
            continue
        inherits_previous_session = False
        identity = identify_frontend_app(
            app_name=event.app_name or event.process_name,
            window_title=event.window_title,
        )
        if _looks_like_file_dialog(event.window_title) and previous_identity is not None:
            identity = previous_identity
            inherits_previous_session = True
        elif _is_sensitive_file_manager_companion(
            event,
            logs,
            sensitive_files,
            previous_identity=previous_identity,
        ):
            identity = previous_identity
            inherits_previous_session = True
        end_ms = min(
            foreground[index + 1].video_time_ms - 1
            if index + 1 < len(foreground)
            else max(timeline_end, event.video_time_ms) + 10_000,
            event.video_time_ms + 30_000,
        )
        app_key = (
            sessions[-1].app_key
            if inherits_previous_session and sessions
            else _frontend_session_app_key(event, identity)
        )
        if (
            sessions
            and sessions[-1].app_key == app_key
            and index == previous_visible_index + 1
            and event.video_time_ms - sessions[-1].end_ms <= session_gap_ms
        ):
            sessions[-1] = _FrontendSession(
                sessions[-1].start_ms,
                max(sessions[-1].end_ms, end_ms),
                app_key,
                identity.app_name,
                identity.category,
                tuple(sorted({*sessions[-1].state_ms, event.video_time_ms})),
            )
        else:
            sessions.append(
                _FrontendSession(
                    event.video_time_ms,
                    max(event.video_time_ms, end_ms),
                    app_key,
                    identity.app_name,
                    identity.category,
                    (event.video_time_ms,),
                )
            )
        previous_identity = identity
        previous_visible_index = index
    return sessions


def _frontend_session_app_key(event: LogEvent, identity) -> str:
    aliases = {
        "meet": "google meet",
        "google meet": "google meet",
        "wemeet": "tencent meeting",
        "tencent meeting": "tencent meeting",
        "腾讯会议": "tencent meeting",
        "doubao": "doubao",
        "豆包": "doubao",
        "ai 中文版": "doubao",
    }
    if identity.known:
        product = aliases.get(normalize_text(identity.app_name), normalize_text(identity.app_name))
    else:
        product = _foreground_app_key(event) or normalize_text(identity.app_name)
    return f"{identity.category}:{product}"


def _session_warrants_visual_evidence(
    session: _FrontendSession,
    logs: list[LogEvent],
    sensitive_files: tuple[str, ...],
    *,
    evidence_horizon_ms: int,
    session_index: int,
    sensitive_session_owners: set[int],
) -> bool:
    if not sensitive_files or session.end_ms - session.start_ms < 8_000:
        return False
    if session.category == "meeting":
        return True
    del logs, evidence_horizon_ms
    return session_index in sensitive_session_owners


def _sensitive_evidence_session_owners(
    sessions: list[_FrontendSession],
    logs: list[LogEvent],
    sensitive_files: tuple[str, ...],
    *,
    evidence_horizon_ms: int,
) -> set[int]:
    owners: set[int] = set()
    actionless_horizon_ms = min(evidence_horizon_ms, 30_000)
    for event in logs:
        if _is_close_event(event) or not (
            _event_has_strict_sensitive_identity(event, sensitive_files)
            or _may_be_derived_file_event(event, sensitive_files)
        ):
            continue
        candidates = [
            (
                0 if session.start_ms <= event.video_time_ms <= session.end_ms else 1,
                max(0, session.start_ms - event.video_time_ms),
                index,
            )
            for index, session in enumerate(sessions)
            if session.category in _EVIDENCE_SESSION_CATEGORIES
            and session.end_ms >= event.video_time_ms
            and session.start_ms - event.video_time_ms <= actionless_horizon_ms
        ]
        if candidates:
            owners.add(min(candidates)[2])
    return owners


def _outbound_sensitive_file_accesses(
    logs: list[LogEvent],
    sensitive_files: tuple[str, ...],
    *,
    start_ms: int,
    end_ms: int,
) -> list[LogEvent]:
    known_paths = {
        normalize_path(event.file_path).lower()
        for event in logs
        if event.video_time_ms <= start_ms
        and _is_user_document_path(normalize_path(event.file_path).lower())
        and _event_matches_sensitive_or_derived_path(event, sensitive_files)
    }
    accesses = [
        event
        for event in logs
        if start_ms < event.video_time_ms <= end_ms
        and event.event_type not in _FOREGROUND_EVENTS
        and not _is_close_event(event)
        and _action_kind(event) not in {"file_selected", "upload", "send", "removable", "screen_share"}
        and _is_user_document_path(normalize_path(event.file_path).lower())
        and _event_matches_sensitive_or_derived_path(event, sensitive_files)
        and (
            identify_frontend_app(
                app_name=event.app_name or event.process_name,
                window_title=event.window_title,
            ).risk_hint.startswith("external_capable")
            or normalize_path(event.file_path).lower() in known_paths
        )
    ]
    retained: list[LogEvent] = []
    for event in sorted(accesses, key=lambda item: item.video_time_ms):
        if (
            retained
            and same_file(retained[-1].file_path, event.file_path)
            and event.video_time_ms - retained[-1].video_time_ms <= 2_000
        ):
            continue
        retained.append(event)
    return retained


def _event_matches_sensitive_or_derived_path(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    target = normalize_path(event.file_path).lower()
    return _event_matches_sensitive(event, sensitive_files) or any(
        _target_path_references_source(target, source) for source in sensitive_files
    )


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
        separate_external_packets = bool(
            merged
            and merged[-1].reason.startswith("strong:external_session:")
            and window.reason.startswith("strong:external_session:")
        )
        if (
            not merged
            or separate_external_packets
            or window.priority != merged[-1].priority
            or window.start_ms > merged[-1].end_ms
        ):
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
            action_phases=_collapse_action_phases((*previous.action_phases, *window.action_phases)),
            requires_post_action_state=previous.requires_post_action_state or window.requires_post_action_state,
            active_apps=tuple(dict.fromkeys((*previous.active_apps, *window.active_apps))),
        )
    return merged


def _collapse_action_phases(
    phases: tuple[tuple[int, str], ...] | list[tuple[int, str]],
    *,
    echo_gap_ms: int = 1_500,
) -> tuple[tuple[int, str], ...]:
    """Collapse monitor echoes without merging distinct steps in one workflow."""

    collapsed: list[tuple[int, str]] = []
    for timestamp, action in sorted({(int(timestamp), str(action)) for timestamp, action in phases}):
        if (
            collapsed
            and collapsed[-1][1] == action
            and action not in {"external_session", "external_state", "resource_identity", "session_end"}
            and timestamp - collapsed[-1][0] <= echo_gap_ms
        ):
            collapsed[-1] = (timestamp, action)
        else:
            collapsed.append((timestamp, action))
    return tuple(collapsed)


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
    ]
    return sorted({event.event_id: event for event in kept}.values(), key=lambda event: (event.video_time_ms, event.event_id))


def _may_need_analysis_window(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    if event.video_time_ms < 0:
        return False
    if _event_matches_sensitive(event, sensitive_files) or _action_kind(event):
        return True
    if _is_noise_only_event(event):
        return False
    return event.event_type not in _FOREGROUND_EVENTS and _is_high_risk_hint(event)


def _sensitive_clipboard_event_ids(
    events: list[LogEvent],
    sensitive_files: tuple[str, ...],
) -> frozenset[str]:
    """Bind clipboard actions to the current document, not an unclosed file."""

    current_document_by_app: dict[str, tuple[int, bool]] = {}
    unbound_sensitive_context_ms: list[int] = []
    recent_sensitive_context_ms: list[int] = []
    clipboard_taint: tuple[int, bool] | None = None
    selected: set[str] = set()

    for event in events:
        action = _action_kind(event)
        timestamp = event.video_time_ms
        app_family = _clipboard_app_family(event)
        if action in {"clipboard", "paste"}:
            current = current_document_by_app.get(app_family)
            current_sensitive = bool(
                current
                and 0 <= timestamp - current[0] <= 120_000
                and current[1]
            )
            if not current_sensitive and app_family:
                current_sensitive = any(
                    0 <= timestamp - signal_ms <= 120_000
                    for signal_ms in unbound_sensitive_context_ms
                )
            if not current_sensitive and event.event_type.lower() == "clipboard_image":
                current_sensitive = any(
                    0 <= timestamp - signal_ms <= 30_000
                    for signal_ms in recent_sensitive_context_ms
                )
            if action == "paste" and clipboard_taint:
                current_sensitive = current_sensitive or (
                    0 <= timestamp - clipboard_taint[0] <= 120_000
                    and clipboard_taint[1]
                )
            if current_sensitive:
                selected.add(event.event_id)
            if action == "clipboard":
                clipboard_taint = (timestamp, current_sensitive)

        if event.window_title.strip():
            title_sensitive = _window_title_mentions_sensitive(event.window_title, sensitive_files)
            if title_sensitive:
                recent_sensitive_context_ms.append(timestamp)
            if app_family:
                current_document_by_app[app_family] = (timestamp, title_sensitive)
            elif title_sensitive:
                unbound_sensitive_context_ms.append(timestamp)
        elif _event_has_strict_sensitive_identity(event, sensitive_files):
            recent_sensitive_context_ms.append(timestamp)
            if app_family:
                current_document_by_app[app_family] = (timestamp, True)
            else:
                unbound_sensitive_context_ms.append(timestamp)

    return frozenset(selected)


def _clipboard_app_family(event: LogEvent) -> str:
    app = _foreground_app_key(event)
    if app in {"et", "wps", "wpp", "wps ppt"}:
        return "wps_office"
    if app in {"excel", "powerpnt", "winword", "word"}:
        return "microsoft_office"
    return app


def _window_title_mentions_sensitive(title: str, sensitive_files: tuple[str, ...]) -> bool:
    normalized_title = normalize_path(title).lower()
    return any(
        len(name) >= 4 and name in normalized_title
        for source in sensitive_files
        if (name := Path(normalize_path(source)).name.lower())
    )


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
            has_recent_sensitive = any(
                0 <= event.video_time_ms - signal.video_time_ms <= 120_000
                and _event_matches_sensitive(signal, sensitive_files)
                for signal in events
            )
            has_later_external_app = any(
                0 <= signal.video_time_ms - event.video_time_ms <= 120_000
                and signal.event_type in _FOREGROUND_EVENTS
                and identify_frontend_app(
                    app_name=signal.app_name or signal.process_name,
                    window_title=signal.window_title,
                ).category in _EVIDENCE_SESSION_CATEGORIES
                for signal in events
            )
            has_bound_sensitive_path = bool(event.file_path) and _event_matches_sensitive(event, sensitive_files)
            if not has_later_external_app and not has_bound_sensitive_path and not any(
                0 <= downstream_ms - event.video_time_ms <= 60_000
                for downstream_ms in downstream
            ) and not any(
                abs(event.video_time_ms - signal.video_time_ms) <= 30_000
                and _event_matches_sensitive(signal, sensitive_files)
                and _is_noise_only_event(signal)
                for signal in events
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
    if event_type in _OPEN_EVENTS | _FILE_CREATION_EVENTS and "browser_file_access" in operation and _is_user_document_path(
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
    if _has_removable_media_signal(combined):
        return "removable"
    return ""


def _has_removable_media_signal(text: str) -> bool:
    if any(token in text for token in ("bluetooth", "removable", "fsquirt", "u盘", "蓝牙")):
        return True
    return re.search(r"(?<![a-z0-9])usb(?![a-z0-9])", text) is not None


def _is_mail_attachment_context(event: LogEvent) -> bool:
    identity = identify_frontend_app(
        app_name=event.app_name or event.process_name,
        window_title=event.window_title,
    )
    if identity.category == "mail":
        return True
    text = _event_text(event).lower()
    return any(marker in text for marker in ("mail", "email", "邮件", "邮箱", "附件", "attachment"))


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


def _is_sensitive_file_manager_companion(
    event: LogEvent,
    logs: list[LogEvent],
    sensitive_files: tuple[str, ...],
    *,
    previous_identity: object | None,
) -> bool:
    """Keep an external session alive while Explorer supplies its file payload.

    Native Explorer windows are reported as foreground switches during drag/drop
    and file selection, even though the mail/chat window remains the actual sink.
    Only inherit the sink identity when the Explorer title or nearby file event
    identifies a sensitive source or derived artifact.
    """

    if previous_identity is None or getattr(previous_identity, "category", "") not in _EVIDENCE_SESSION_CATEGORIES:
        return False
    if not _is_file_manager_window(event):
        return False
    if _window_title_mentions_sensitive_stem(event.window_title, sensitive_files):
        return True
    timestamp = event.video_time_ms
    return any(
        other.event_type not in _FOREGROUND_EVENTS
        and not _is_close_event(other)
        and -5_000 <= other.video_time_ms - timestamp <= 30_000
        and (
            _event_matches_sensitive_or_derived_path(other, sensitive_files)
            or _may_be_derived_file_event(other, sensitive_files)
        )
        for other in logs
        if other.video_time_ms >= 0
    )


def _is_file_manager_window(event: LogEvent) -> bool:
    app_key = _foreground_app_key(event)
    if app_key in {"explorer", "file explorer", "windows explorer"}:
        return True
    title = normalize_text(event.window_title).lower()
    return "文件资源管理器" in title or "file explorer" in title


def _window_title_mentions_sensitive_stem(title: str, sensitive_files: tuple[str, ...]) -> bool:
    normalized_title = normalize_path(title).lower()
    return any(
        len(stem) >= 4 and stem in normalized_title
        for source in sensitive_files
        if (stem := Path(normalize_path(source)).stem.lower())
    )


def _event_matches_sensitive(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    return any(_event_matches_source(event, source) for source in sensitive_files)


def _event_has_strict_sensitive_identity(event: LogEvent, sensitive_files: tuple[str, ...]) -> bool:
    """Require file identity, not a stale title attached to unrelated IO."""

    if event.file_path:
        if any(same_file(event.file_path, source) for source in sensitive_files):
            return True
        if event.event_type not in _FOREGROUND_EVENTS:
            return False
    return _event_matches_sensitive(event, sensitive_files)


def _event_matches_source(event: LogEvent, source: str) -> bool:
    if event.file_path:
        if same_file(event.file_path, source):
            return True
        if (
            event.event_type not in _FOREGROUND_EVENTS
            and not event.event_type.lower().startswith("clipboard")
            and event.event_type.lower() not in {"modified", "accessed", "read"}
        ):
            return False
    filename = Path(normalize_path(source)).name.lower()
    semantic_context = normalize_path(
        " ".join(
            str(item or "")
            for item in (
                event.window_title,
                event.description,
                event.raw.get("content_preview"),
                _extra(event).get("content_preview"),
            )
        )
    ).lower()
    return bool(filename and len(filename) >= 4 and filename in semantic_context)


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
