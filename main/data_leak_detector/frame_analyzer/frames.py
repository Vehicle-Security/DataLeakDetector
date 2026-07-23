"""Evidence-oriented keyframe selection for log-anchored video windows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .apps import identify_frontend_app
from .config import VisionConfig


_ACTION_OFFSETS_MS = (-1_000, -250, -100, 0, 250, 500, 1_000, 2_000, 5_000)
_DERIVATION_OFFSETS_MS = (
    -20_000,
    -15_000,
    -10_000,
    -5_000,
    -2_000,
    -1_000,
    -250,
    0,
    500,
    2_000,
    5_000,
    10_000,
)
_FILE_SELECTION_OFFSETS_MS = (-2_000, -1_000, -250, -100, 0, 250, 500, 750, 1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 10_000, 20_000, 30_000)
_OUTBOUND_OFFSETS_MS = (-2_000, -1_000, -250, -100, 0, 250, 500, 1_000, 2_000, 5_000, 10_000)
_CLIPBOARD_OFFSETS_MS = (-2_000, -1_500, -1_000, -500, -250, -125, 0, 125, 250, 500, 750, 1_000, 2_000, 5_000, 10_000, 15_000, 25_000, 30_000)
_PASTE_OFFSETS_MS = (
    -2_000, -500, -250, -100, 0, 125, 250, 375, 500, 750, 1_000,
    1_500, 1_625, 1_750, 1_875, 2_000, 2_125, 2_250, 2_375, 2_500,
    2_625, 2_750, 2_875, 3_000, 3_125, 3_250, 3_375, 3_500,
    4_000, 5_000, 7_500, 10_000, 15_000,
)
_SCREEN_SHARE_OFFSETS_MS = (-2_000, -250, -100, 0, 250, 500, 1_000, 2_000, 5_000, 10_000, 15_000)
_CAPTURE_START_OFFSETS_MS = (-1_000, 0, 1_000, 2_000, 3_000, 5_000, 8_000)
_CAPTURE_OFFSETS_MS = (-15_000, -13_000, -11_000, -9_000, -7_000, -5_000, -3_000, -1_000, 0)
_CONTEXT_ACTIONS = {"external_session", "external_state", "resource_identity", "session_end"}
_RESULT_BEARING_ACTIONS = {"clipboard", "paste", "file_selected", "upload", "send", "screen_share", "removable", "derive"}


@dataclass(frozen=True)
class AnalysisWindow:
    start_ms: int
    end_ms: int
    reason: str
    priority: str = "medium"
    step_ms: int = 1_000
    max_keyframes: int = 18
    diff_threshold: float = 0.08
    anchor_ms: tuple[int, ...] = ()
    action_anchor_ms: tuple[int, ...] = ()
    action_phases: tuple[tuple[int, str], ...] = ()
    requires_post_action_state: bool = False
    active_apps: tuple[str, ...] = ()
    active_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class KeyFrame:
    frame_id: str
    timestamp_ms: int
    image_path: str
    score: float
    reason: str
    window_id: str = ""


@dataclass(frozen=True)
class KeyFrameDuplicate:
    frame: KeyFrame
    kept_frame_id: str
    reason: str
    delta: float
    hash_distance: int


@dataclass(frozen=True)
class KeyFrameSelection:
    keyframes: list[KeyFrame]
    raw_keyframes: list[KeyFrame]
    duplicates: list[KeyFrameDuplicate]
    warnings: list[str]


@dataclass(frozen=True)
class _FrameCandidate:
    frame: KeyFrame
    priority: str
    gray: object
    frame_hash: tuple[int, int]
    entropy: float = 0.0


def merge_analysis_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    """Merge overlapping windows only when they represent the same priority."""

    merged = []
    for priority in ("strong", "activity", "medium", "weak"):
        current: list[AnalysisWindow] = []
        for window in sorted((item for item in windows if item.priority == priority), key=lambda item: item.start_ms):
            if not current or window.start_ms > current[-1].end_ms:
                current.append(window)
                continue
            previous = current[-1]
            current[-1] = AnalysisWindow(
                previous.start_ms,
                max(previous.end_ms, window.end_ms),
                f"{previous.reason}+{window.reason}",
                priority=priority,
                step_ms=min(previous.step_ms, window.step_ms),
                max_keyframes=max(previous.max_keyframes, window.max_keyframes, len({*previous.anchor_ms, *window.anchor_ms})),
                diff_threshold=min(previous.diff_threshold, window.diff_threshold),
                anchor_ms=tuple(sorted({*previous.anchor_ms, *window.anchor_ms})),
                action_anchor_ms=tuple(sorted({*previous.action_anchor_ms, *window.action_anchor_ms})),
                action_phases=tuple(sorted({*previous.action_phases, *window.action_phases})),
                requires_post_action_state=previous.requires_post_action_state or window.requires_post_action_state,
                active_apps=tuple(dict.fromkeys((*previous.active_apps, *window.active_apps))),
                active_ranges=_merge_active_ranges(previous.active_ranges, window.active_ranges),
            )
        merged.extend(current)
    return sorted(merged, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))


def select_keyframes(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> tuple[list[KeyFrame], list[str]]:
    selection = select_keyframes_detailed(video_path, windows, config)
    return selection.keyframes, selection.warnings


def select_keyframes_detailed(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> KeyFrameSelection:
    if not str(video_path or "").strip():
        return KeyFrameSelection([], [], [], [])
    path = Path(video_path)
    if not path.exists():
        return KeyFrameSelection([], [], [], [f"video_not_found: {path}"])
    if not path.is_file():
        return KeyFrameSelection([], [], [], [f"video_not_file: {path}"])
    try:
        import cv2
    except ImportError:
        return KeyFrameSelection([], [], [], ["opencv_not_installed: install data-leak-detector[vision]"])

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return KeyFrameSelection([], [], [], [f"video_open_failed: {path}"])
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps <= 0:
        capture.release()
        return KeyFrameSelection([], [], [], ["video_fps_unavailable"])
    duration_ms = int(round(frame_count * 1000.0 / fps)) if frame_count > 0 else 0
    temp_dir = Path(tempfile.mkdtemp(prefix="dld_frames_"))
    candidates: list[_FrameCandidate] = []
    warnings: list[str] = []

    try:
        for window_index, original_window in enumerate(windows):
            if original_window.priority == "activity":
                contextual = _activity_context_window(original_window, windows)
                if contextual is None:
                    continue
                original_window = contextual
            window = _clamp_window_to_duration(original_window, duration_ms)
            timestamps = _probe_timestamps(window, config)
            frames = _read_frames_for_timestamps(cv2, capture, timestamps, fps, config)
            window_candidates = _select_window_candidates(
                cv2,
                frames,
                window,
                window_index,
                temp_dir,
                config,
            )
            candidates.extend(window_candidates)
    finally:
        capture.release()

    raw_keyframes = [candidate.frame for candidate in candidates]
    keyframes, duplicates = _dedupe_keyframes_globally(candidates, config, windows=windows)
    if not keyframes and windows:
        warnings.append("no_keyframes_selected")
    return KeyFrameSelection(keyframes, raw_keyframes, duplicates, warnings)


def build_video_coverage_windows(video_path: str | Path, config: VisionConfig) -> list[AnalysisWindow]:
    """Create sparse evidence anchors when log mining cannot locate an action."""

    path = Path(video_path)
    if not path.is_file():
        return []
    try:
        import cv2
    except ImportError:
        return []
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return []
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        return []
    duration_ms = max(1, int(round(frame_count * 1000.0 / fps)) - 1)
    anchors = tuple(
        sorted(
            {
                min(duration_ms, max(0, round(duration_ms * fraction)))
                for fraction in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
            }
        )
    )
    return [
        AnalysisWindow(
            0,
            duration_ms,
            "medium:video_coverage_fallback",
            priority="medium",
            step_ms=config.frame_step_ms,
            max_keyframes=max(18, len(anchors)),
            diff_threshold=min(config.frame_diff_threshold, 0.02),
            anchor_ms=anchors,
        )
    ]


def augment_with_video_coverage(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> list[AnalysisWindow]:
    """Add sparse full-video evidence only when logs have no result action."""

    result_actions = {
        action.split(":", 1)[0]
        for window in windows
        for _, action in window.action_phases
    } & _RESULT_BEARING_ACTIONS
    if result_actions or any("video_coverage_fallback" in window.reason for window in windows):
        return windows
    coverage = build_video_coverage_windows(video_path, config)
    return [*windows, *coverage]


def _activity_context_window(
    activity: AnalysisWindow,
    windows: list[AnalysisWindow],
) -> AnalysisWindow | None:
    actions = [
        window
        for window in windows
        if window.priority == "strong"
        and window.end_ms >= activity.start_ms
        and window.start_ms <= activity.end_ms
        and window.action_anchor_ms
    ]
    if not actions:
        return None
    anchors = activity.anchor_ms
    if actions and anchors:
        first_action = min(anchor for window in actions for anchor in window.action_anchor_ms)
        before = [anchor for anchor in anchors if anchor <= first_action]
        chosen = before[-1] if before else min(anchors, key=lambda anchor: abs(anchor - first_action))
        anchors = (chosen,)
    return AnalysisWindow(
        activity.start_ms,
        activity.end_ms,
        activity.reason,
        priority=activity.priority,
        step_ms=activity.step_ms,
        max_keyframes=min(activity.max_keyframes, 3),
        diff_threshold=activity.diff_threshold,
        anchor_ms=anchors,
        active_apps=activity.active_apps,
        active_ranges=activity.active_ranges,
    )


def _select_window_candidates(
    cv2,
    frames: dict[int, object],
    window: AnalysisWindow,
    window_index: int,
    temp_dir: Path,
    config: VisionConfig,
) -> list[_FrameCandidate]:
    if not frames:
        return []
    anchors = set(window.anchor_ms)
    retained: list[_FrameCandidate] = []
    previous_gray = None
    retained_grays = []
    retained_hashes: list[tuple[int, int]] = []
    previous_entropy = None
    last_kept_ms = -10**9
    for timestamp, frame in sorted(frames.items()):
        if window.active_ranges and window.priority == "activity" and not _timestamp_in_ranges(timestamp, window.active_ranges):
            continue
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        pixel_delta = 1.0 if previous_gray is None else _frame_delta(cv2, previous_gray, gray)
        entropy = _frame_entropy(cv2, gray)
        entropy_delta = 1.0 if previous_entropy is None else abs(entropy - previous_entropy) / 8.0
        score = max(pixel_delta, entropy_delta)
        frame_hash = _average_hash(cv2, gray)
        force_anchor = _near_any(timestamp, anchors, max(window.step_ms // 2, 125))
        action_state = _action_state_at(timestamp, window, max(window.step_ms // 2, 125))
        force_state = bool(action_state)
        # Anchors and planned action states are evidence positions. Keep them in
        # raw_all and let the global, traceable dedupe choose the better frame.
        exact_duplicate = not (force_anchor or force_state) and _is_exact_duplicate(
            cv2,
            gray,
            retained_grays,
            config.frame_exact_duplicate_threshold,
        )
        keep = _should_keep_frame(
            timestamp_ms=timestamp,
            score=score,
            diff_threshold=window.diff_threshold,
            force_keep=force_anchor or force_state,
            exact_duplicate=exact_duplicate,
            frame_hash=frame_hash,
            retained_hashes=retained_hashes,
            previous_small=previous_gray,
            last_kept_ms=last_kept_ms,
            config=config,
            entropy_delta=entropy_delta,
        )
        previous_gray = gray
        previous_entropy = entropy
        if not keep:
            continue
        frame_id = f"frame_{window_index}_{len(retained)}"
        image_path = temp_dir / f"{frame_id}_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        if force_state:
            reason = f"{window.priority}:action_state:{action_state}"
        elif force_anchor:
            reason = f"{window.priority}:anchor"
        else:
            reason = f"{window.priority}:visual_change"
        keyframe = KeyFrame(
            frame_id,
            timestamp,
            str(image_path),
            round(float(score), 4),
            reason,
            window_id=f"window_{window_index}",
        )
        candidate = _FrameCandidate(keyframe, window.priority, gray, frame_hash, entropy)
        retained.append(candidate)
        retained_grays.append(gray)
        retained_hashes.append(frame_hash)
        last_kept_ms = timestamp

    retained = _focus_semantic_action_phases(retained, window)
    limit = max(window.max_keyframes, len(window.anchor_ms))
    if len(retained) <= limit:
        return retained
    return _budget_window_candidates(retained, window, limit)


def _probe_timestamps(window: AnalysisWindow, config: VisionConfig) -> list[int]:
    timestamps = set(_action_state_timestamps(window))
    for anchor in window.anchor_ms:
        for timestamp in (anchor, anchor - window.step_ms, anchor + window.step_ms):
            if window.start_ms <= timestamp <= window.end_ms:
                timestamps.add(timestamp)
    if window.priority == "activity":
        timestamps.update(_activity_probe_timestamps(window))
    limit = max(
        window.max_keyframes * max(1, config.frame_probe_multiplier),
        len(timestamps),
        len(window.anchor_ms) * 3,
    )
    if window.priority != "activity":
        timestamps.update(_dense_probe_timestamps(window, max(0, limit - len(timestamps))))
    if not timestamps and window.start_ms <= window.end_ms:
        timestamps.add(window.start_ms)
    ordered = sorted(timestamp for timestamp in timestamps if window.start_ms <= timestamp <= window.end_ms)
    if len(ordered) <= limit:
        return ordered
    mandatory = set(_action_state_timestamps(window)) | set(window.anchor_ms)
    planned = [timestamp for timestamp in ordered if timestamp in mandatory]
    optional = [timestamp for timestamp in ordered if timestamp not in mandatory]
    return sorted([*planned, *_evenly_spaced_values(optional, max(0, limit - len(planned)))])


def _action_state_timestamps(window: AnalysisWindow) -> tuple[int, ...]:
    if not window.action_anchor_ms:
        return ()
    phases = window.action_phases or (
        (max(window.action_anchor_ms), "paste" if window.requires_post_action_state else "action"),
    )
    timestamps = set()
    for anchor, action in phases:
        timestamps.update(_phase_timestamps(window, anchor, action))
    return tuple(
        sorted(timestamps)
    )


def _action_state_at(timestamp_ms: int, window: AnalysisWindow, tolerance_ms: int) -> str:
    phases = window.action_phases or tuple(
        (anchor, "paste" if window.requires_post_action_state else "action")
        for anchor in window.action_anchor_ms
    )
    matches = [
        (abs(timestamp_ms - target), action, anchor, target)
        for anchor, action in phases
        for target in _phase_timestamps(window, anchor, action)
        if abs(timestamp_ms - target) <= tolerance_ms
    ]
    if not matches:
        return ""
    best_by_action: dict[str, tuple[int, int, int]] = {}
    for distance, action, anchor, target in matches:
        previous = best_by_action.get(action)
        candidate = (distance, anchor, target)
        if previous is None or candidate < previous:
            best_by_action[action] = candidate
    return "|".join(
        f"{action}:{_phase_role(target - anchor)}"
        for action, (_, anchor, target) in sorted(best_by_action.items())
    )


def _phase_timestamps(window: AnalysisWindow, anchor: int, action: str) -> tuple[int, ...]:
    action = action.split(":", 1)[0]
    if action == "capture_start":
        offsets = _CAPTURE_START_OFFSETS_MS
    elif action == "capture":
        offsets = _CAPTURE_OFFSETS_MS
    elif action == "file_selected":
        offsets = _FILE_SELECTION_OFFSETS_MS
    elif action in {"external_session", "external_state", "resource_identity", "session_end"}:
        offsets = (0,)
    elif action == "clipboard":
        offsets = _CLIPBOARD_OFFSETS_MS
    elif action == "paste":
        offsets = _PASTE_OFFSETS_MS
    elif action == "screen_share":
        offsets = _SCREEN_SHARE_OFFSETS_MS
    elif action in {"upload", "send", "removable"}:
        offsets = _OUTBOUND_OFFSETS_MS
    elif action == "derive":
        offsets = _DERIVATION_OFFSETS_MS
    else:
        offsets = _ACTION_OFFSETS_MS
    return tuple(
        anchor + offset
        for offset in offsets
        if window.start_ms <= anchor + offset <= window.end_ms
    )


def _focus_semantic_action_phases(
    candidates: list[_FrameCandidate],
    window: AnalysisWindow,
) -> list[_FrameCandidate]:
    if not window.action_phases:
        return candidates
    chosen: list[_FrameCandidate] = []
    claimed_ids: set[str] = set()
    phases = _collapse_frame_action_phases(window.action_phases)
    for anchor, action in phases:
        base_action = action.split(":", 1)[0]
        planned = _phase_timestamps(window, anchor, base_action) or (anchor,)
        tolerance = max(window.step_ms // 2, 125)
        start = min(planned) - tolerance
        end = max(planned) + tolerance
        phase_region = [item for item in candidates if start <= item.frame.timestamp_ms <= end]
        phase = [
            item
            for item in phase_region
            if _candidate_matches_action(item, base_action, single_phase=len(phases) == 1)
        ]
        claimed_ids.update(item.frame.frame_id for item in phase)
        if not phase:
            continue
        ordered = sorted(phase, key=lambda item: item.frame.timestamp_ms)
        if base_action in {"external_session", "external_state", "resource_identity", "session_end"}:
            chosen_items = [min(ordered, key=lambda item: abs(item.frame.timestamp_ms - anchor))]
        else:
            pre = [item for item in ordered if item.frame.timestamp_ms < anchor]
            immediate = [item for item in ordered if anchor <= item.frame.timestamp_ms <= anchor + 1_000]
            post = [item for item in ordered if anchor + 1_000 < item.frame.timestamp_ms <= anchor + 5_000]
            late = [item for item in ordered if item.frame.timestamp_ms > anchor + 5_000]
            pre_items = [*pre[:1], *pre[-1:]] if base_action in {"file_selected", "clipboard"} else pre[-1:]
            if base_action == "capture" and pre:
                pre_items = [max(pre, key=lambda item: (float(item.gray.std()), item.frame.score, item.frame.timestamp_ms))]
            if base_action == "paste":
                # Paste previews can appear for only a few hundred
                # milliseconds.  Always retain the approximately +2.25s state
                # (the common paste-render point), plus the strongest visual
                # transitions, instead of only the first empty-composer frames.
                target = min(post, key=lambda item: abs(item.frame.timestamp_ms - (anchor + 2_250))) if post else None
                strongest = sorted(post, key=lambda item: (item.frame.score, item.entropy), reverse=True)
                post_items = [target] if target else []
                post_items.extend(item for item in strongest if item not in post_items)
                post_items = sorted(post_items[:3], key=lambda item: item.frame.timestamp_ms)
            elif base_action == "file_selected":
                post_items = post[:3]
            elif base_action == "clipboard":
                post_items = post[:3]
            else:
                post_items = post[:1]
            chosen_items = [
                *pre_items,
                *([max(immediate, key=lambda item: (item.frame.score, item.frame.timestamp_ms))] if immediate else []),
                *post_items,
                *late[-1:],
            ]
        chosen.extend(chosen_items)
    optional = [item for item in candidates if item.frame.frame_id not in claimed_ids]
    return sorted(
        {item.frame.frame_id: item for item in [*optional, *chosen]}.values(),
        key=lambda item: item.frame.timestamp_ms,
    )


def _budget_window_candidates(
    candidates: list[_FrameCandidate],
    window: AnalysisWindow,
    limit: int,
) -> list[_FrameCandidate]:
    if limit <= 0:
        return []
    if len(candidates) <= limit:
        return candidates

    action_states = [
        item
        for item in candidates
        if "action_state" in item.frame.reason and not _is_context_only_candidate(item)
    ]
    context_states = [
        item
        for item in candidates
        if "action_state" in item.frame.reason and _is_context_only_candidate(item)
    ]
    anchors = [item for item in candidates if "anchor" in item.frame.reason and item not in action_states and item not in context_states]
    optional = [item for item in candidates if item not in action_states and item not in context_states and item not in anchors]

    result_limit = min(4, max(0, limit // 3))
    result_states = _post_action_visual_evidence(optional, window, result_limit)
    result_ids = {item.frame.frame_id for item in result_states}
    optional = [item for item in optional if item.frame.frame_id not in result_ids]

    mandatory = action_states or anchors
    minimum_context = min(len(context_states), 2 if mandatory or result_states else max(1, limit // 2))
    mandatory_limit = max(0, limit - len(result_states) - minimum_context)
    mandatory = _trim_mandatory_evidence(mandatory, window, mandatory_limit)

    remaining = max(0, limit - len(mandatory) - len(result_states))
    # Context action states are explicit log-planned evidence positions. Keep all
    # that fit the remaining budget so a later send/result state is not dropped
    # merely because an earlier external session already supplied context.
    context_limit = min(len(context_states), remaining)
    context = _select_context_evidence(context_states, context_limit)

    selected = [*mandatory, *result_states, *context]
    selected_ids = {item.frame.frame_id for item in selected}
    remaining_optional = [item for item in optional if item.frame.frame_id not in selected_ids]
    available = max(0, limit - len(selected))
    selected.extend(_select_optional_evidence(remaining_optional, available))
    return sorted({item.frame.frame_id: item for item in selected}.values(), key=lambda item: item.frame.timestamp_ms)


def _post_action_visual_evidence(
    candidates: list[_FrameCandidate],
    window: AnalysisWindow,
    limit: int,
) -> list[_FrameCandidate]:
    if limit <= 0 or not candidates:
        return []
    result_groups: list[list[_FrameCandidate]] = []
    context_groups: list[list[_FrameCandidate]] = []
    for anchor, action in _collapse_frame_action_phases(window.action_phases):
        base_action = action.split(":", 1)[0]
        if base_action == "external_state":
            targets = [anchor]
            groups = context_groups
        elif base_action in _RESULT_BEARING_ACTIONS:
            targets = [
                timestamp
                for timestamp in _phase_timestamps(window, anchor, base_action)
                if timestamp - anchor >= 5_000
            ]
            groups = result_groups
        else:
            continue
        for target in targets:
            group = [
                item
                for item in candidates
                if "visual_change" in item.frame.reason and target < item.frame.timestamp_ms <= min(window.end_ms, target + 5_000)
            ]
            if not group:
                continue
            groups.append(
                sorted(
                    group,
                    key=lambda item: (item.frame.score, item.entropy, item.frame.timestamp_ms),
                    reverse=True,
                )[:2]
            )

    selected: list[_FrameCandidate] = []
    selected_ids: set[str] = set()
    for group_set in (result_groups, context_groups):
        queues = [list(group) for group in group_set]
        while len(selected) < limit and any(queues):
            progressed = False
            for queue in queues:
                while queue and queue[0].frame.frame_id in selected_ids:
                    queue.pop(0)
                if not queue or len(selected) >= limit:
                    continue
                item = queue.pop(0)
                selected.append(item)
                selected_ids.add(item.frame.frame_id)
                progressed = True
            if not progressed:
                break
    return selected


def _is_context_only_candidate(candidate: _FrameCandidate) -> bool:
    actions = _candidate_actions(candidate)
    return bool(actions) and actions <= _CONTEXT_ACTIONS


def _candidate_actions(candidate: _FrameCandidate) -> set[str]:
    reason = candidate.frame.reason
    if "action_state" not in reason:
        return set()
    suffix = reason.split("action_state", 1)[1].lstrip(":")
    return {state.split(":", 1)[0] for state in suffix.split("|") if state}


def _select_context_evidence(candidates: list[_FrameCandidate], limit: int) -> list[_FrameCandidate]:
    if limit <= 0:
        return []
    ordered = sorted(candidates, key=lambda item: item.frame.timestamp_ms)
    if len(ordered) <= limit:
        return ordered

    selected: list[_FrameCandidate] = []
    starts = [item for item in ordered if "external_session" in _candidate_actions(item) or "resource_identity" in _candidate_actions(item)]
    states = [item for item in ordered if "external_state" in _candidate_actions(item)]
    ends = [item for item in ordered if "session_end" in _candidate_actions(item)]
    if starts:
        selected.append(starts[0])
    # A late external-state anchor commonly captures the actual send/upload
    # result. Prefer it over a generic session-end frame when space is tight.
    if states and len(selected) < limit and states[-1].frame.frame_id not in {item.frame.frame_id for item in selected}:
        selected.append(states[-1])
    if ends and len(selected) < limit and ends[-1].frame.frame_id not in {item.frame.frame_id for item in selected}:
        selected.append(ends[-1])
    remaining = [item for item in ordered if item.frame.frame_id not in {chosen.frame.frame_id for chosen in selected}]
    selected.extend(_evenly_spaced(remaining, limit - len(selected)))
    return sorted(selected, key=lambda item: item.frame.timestamp_ms)


def _collapse_frame_action_phases(
    phases: tuple[tuple[int, str], ...],
    *,
    echo_gap_ms: int = 1_500,
) -> tuple[tuple[int, str], ...]:
    collapsed: list[tuple[int, str]] = []
    for timestamp, action in sorted(phases):
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


def _candidate_matches_action(item: _FrameCandidate, action: str, *, single_phase: bool) -> bool:
    reason = item.frame.reason
    if "action_state" not in reason:
        return False
    suffix = reason.split("action_state", 1)[1].lstrip(":")
    if not suffix:
        return single_phase
    return any(state.split(":", 1)[0] == action for state in suffix.split("|"))


def _phase_role(offset_ms: int) -> str:
    if offset_ms < 0:
        return "pre"
    if offset_ms == 0:
        return "at"
    if offset_ms <= 1_000:
        return "immediate"
    if offset_ms <= 5_000:
        return "post"
    return "result"


def _dense_probe_timestamps(window: AnalysisWindow, slots: int) -> tuple[int, ...]:
    if slots <= 0 or window.end_ms < window.start_ms:
        return ()
    if slots == 1 or window.end_ms == window.start_ms:
        return (window.start_ms + (window.end_ms - window.start_ms) // 2,)
    span = window.end_ms - window.start_ms
    return tuple(
        window.start_ms + round(span * index / (slots - 1))
        for index in range(slots)
    )


def _trim_mandatory_evidence(
    candidates: list[_FrameCandidate],
    window: AnalysisWindow,
    limit: int,
) -> list[_FrameCandidate]:
    if limit <= 0:
        return []
    if len(candidates) <= limit:
        return candidates
    phase_groups: list[tuple[_FrameCandidate, list[_FrameCandidate]]] = []
    phases = _collapse_frame_action_phases(window.action_phases)
    phase_anchors = sorted({timestamp for timestamp, _ in phases})
    for anchor, action in phases:
        base_action = action.split(":", 1)[0]
        planned = _phase_timestamps(window, anchor, base_action) or (anchor,)
        previous = [timestamp for timestamp in phase_anchors if timestamp < anchor]
        following = [timestamp for timestamp in phase_anchors if timestamp > anchor]
        left_boundary = (max(previous) + anchor) // 2 if previous else window.start_ms
        right_boundary = (anchor + min(following)) // 2 if following else window.end_ms
        region_start = max(min(planned) - max(window.step_ms // 2, 125), left_boundary)
        region_end = min(max(planned) + max(window.step_ms // 2, 125), right_boundary)
        group = [
            item
            for item in candidates
            if region_start <= item.frame.timestamp_ms <= region_end
            and _candidate_matches_action(item, base_action, single_phase=len(phases) == 1)
        ]
        if not group:
            continue
        # For paste, the immediate state usually still shows the empty
        # composer.  Under a tight strong-window budget, preserve the later
        # post-action state where the pasted payload is actually visible.
        if base_action == "paste":
            post = [item for item in group if anchor + 1_000 < item.frame.timestamp_ms <= anchor + 5_000]
            core = max(post or group, key=lambda item: (item.frame.timestamp_ms, item.frame.score))
        elif base_action == "clipboard":
            pre = [item for item in group if item.frame.timestamp_ms < anchor]
            core = min(pre, key=lambda item: (abs(item.frame.timestamp_ms - (anchor - 2_000)), -item.frame.score)) if pre else min(group, key=lambda item: (abs(item.frame.timestamp_ms - anchor), -item.frame.score))
        else:
            core = min(group, key=lambda item: (abs(item.frame.timestamp_ms - anchor), -item.frame.score))
        phase_groups.append((core, _phase_evidence_preferences(group, anchor, base_action)))

    selected: list[_FrameCandidate] = []
    selected_ids: set[str] = set()
    cores = list(dict.fromkeys(core.frame.frame_id for core, _ in phase_groups))
    core_lookup = {core.frame.frame_id: core for core, _ in phase_groups}
    if len(cores) > limit:
        cores = [item.frame.frame_id for item in _evenly_spaced([core_lookup[item] for item in cores], limit)]
    for frame_id in cores:
        selected.append(core_lookup[frame_id])
        selected_ids.add(frame_id)

    queues = [list(preferences) for _, preferences in phase_groups]
    while len(selected) < limit and any(queues):
        progressed = False
        for queue in queues:
            while queue and queue[0].frame.frame_id in selected_ids:
                queue.pop(0)
            if not queue or len(selected) >= limit:
                continue
            item = queue.pop(0)
            selected.append(item)
            selected_ids.add(item.frame.frame_id)
            progressed = True
        if not progressed:
            break

    remaining = [item for item in candidates if item.frame.frame_id not in selected_ids]
    selected.extend(_select_optional_evidence(remaining, limit - len(selected)))
    return sorted(selected, key=lambda item: item.frame.timestamp_ms)


def _phase_evidence_preferences(
    candidates: list[_FrameCandidate],
    anchor: int,
    action: str,
) -> list[_FrameCandidate]:
    ordered = sorted(candidates, key=lambda item: item.frame.timestamp_ms)
    pre = [item for item in ordered if item.frame.timestamp_ms < anchor]
    immediate = [item for item in ordered if anchor <= item.frame.timestamp_ms <= anchor + 1_000]
    post = [item for item in ordered if anchor + 1_000 < item.frame.timestamp_ms <= anchor + 5_000]
    result = [item for item in ordered if item.frame.timestamp_ms > anchor + 5_000]
    if action in {"external_session", "external_state", "resource_identity", "session_end"}:
        return []
    preferences: list[_FrameCandidate] = []
    if action == "file_selected" and pre:
        preferences.extend((pre[0], pre[-1]))
    elif pre:
        preferences.append(pre[-1])
    if result:
        preferences.append(result[-1])
    if immediate:
        preferences.append(max(immediate, key=lambda item: (item.frame.score, item.frame.timestamp_ms)))
    if post:
        preferences.extend(post[:5] if action in {"paste", "file_selected"} else post[:3] if action == "clipboard" else post[:1])
    return list({item.frame.frame_id: item for item in preferences}.values())


def _select_optional_evidence(candidates: list[_FrameCandidate], limit: int) -> list[_FrameCandidate]:
    if limit <= 0:
        return []
    if len(candidates) <= limit:
        return candidates
    spread_limit = max(1, limit // 2)
    selected = _evenly_spaced(candidates, spread_limit)
    selected_ids = {item.frame.frame_id for item in selected}
    ranked = sorted(
        (item for item in candidates if item.frame.frame_id not in selected_ids),
        key=lambda item: (item.frame.score, item.entropy, -item.frame.timestamp_ms),
        reverse=True,
    )
    selected.extend(ranked[: limit - len(selected)])
    return sorted(selected, key=lambda item: item.frame.timestamp_ms)


def _representative_anchors(anchors: tuple[int, ...], *, limit: int) -> tuple[int, ...]:
    ordered = sorted(set(anchors))
    if len(ordered) <= limit:
        return tuple(ordered)
    last = len(ordered) - 1
    return tuple(dict.fromkeys(ordered[round(index * last / (limit - 1))] for index in range(limit)))


def _activity_probe_timestamps(window: AnalysisWindow) -> tuple[int, ...]:
    timestamps = set(window.anchor_ms)
    ranges = window.active_ranges or ((window.start_ms, window.end_ms),)
    external = any(
        identify_frontend_app(app_name=app).risk_hint.startswith("external_capable")
        for app in window.active_apps
    )
    for start, end in ranges:
        if end < start:
            continue
        if external:
            timestamps.update((start, start + (end - start) // 2, end))
        elif window.anchor_ms:
            nearby = [anchor for anchor in window.anchor_ms if start <= anchor <= end]
            timestamps.update(nearby[:1])
            timestamps.update(nearby[-1:])
    return tuple(sorted(timestamp for timestamp in timestamps if window.start_ms <= timestamp <= window.end_ms))


def _coverage_timestamps(window: AnalysisWindow) -> tuple[int, ...]:
    if window.priority == "activity" or window.anchor_ms or window.end_ms <= window.start_ms:
        return ()
    span = window.end_ms - window.start_ms
    return tuple(window.start_ms + round(span * fraction) for fraction in (0.0, 0.5, 1.0))


def _should_keep_frame(
    *,
    timestamp_ms: int,
    score: float,
    diff_threshold: float,
    force_keep: bool,
    exact_duplicate: bool,
    frame_hash: tuple[int, int],
    retained_hashes: list[tuple[int, int]],
    previous_small,
    last_kept_ms: int,
    config: VisionConfig,
    entropy_delta: float = 0.0,
) -> bool:
    del previous_small
    if force_keep:
        return not exact_duplicate
    if exact_duplicate:
        return False
    del frame_hash, retained_hashes
    if score < diff_threshold and entropy_delta < config.frame_entropy_change_threshold:
        return False
    return config.frame_min_keep_gap_ms <= 0 or last_kept_ms < 0 or timestamp_ms - last_kept_ms >= config.frame_min_keep_gap_ms


def _dedupe_keyframes_globally(
    candidates: list[_FrameCandidate],
    config: VisionConfig,
    *,
    windows: list[AnalysisWindow] | None = None,
) -> tuple[list[KeyFrame], list[KeyFrameDuplicate]]:
    del windows
    retained: list[_FrameCandidate] = []
    duplicates: list[KeyFrameDuplicate] = []
    for candidate in sorted(candidates, key=lambda item: (item.frame.timestamp_ms, _priority_sort_key(item.priority))):
        duplicate_index = None
        duplicate_delta = 0.0
        duplicate_hash = 0
        for index, kept in enumerate(retained):
            gap = abs(candidate.frame.timestamp_ms - kept.frame.timestamp_ms)
            candidate_actions = _candidate_actions(candidate)
            kept_actions = _candidate_actions(kept)
            if (
                gap >= 500
                and candidate_actions
                and kept_actions
                and candidate_actions & kept_actions & {"clipboard", "paste", "file_selected"}
            ):
                # Distinct source/action/result states are useful lineage
                # evidence even when most of the screen remains unchanged.
                continue
            if gap < max(config.frame_anchor_duplicate_gap_ms, 500) and (
                ":pre" in candidate.frame.reason or ":pre" in kept.frame.reason
            ):
                # The payload immediately before a click is distinct evidence,
                # even when the clicked-state pixels are nearly unchanged.
                continue
            if (
                gap >= 1_000
                and "action_state" in candidate.frame.reason
                and "action_state" in kept.frame.reason
            ):
                continue
            has_anchor = "anchor" in candidate.frame.reason or "anchor" in kept.frame.reason
            same_action_phase = gap <= (max(config.frame_anchor_duplicate_gap_ms, 500) if has_anchor else 2_000)
            if not same_action_phase:
                continue
            delta = _array_delta(candidate.gray, kept.gray)
            hash_distance = _hamming(candidate.frame_hash, kept.frame_hash)
            entropy_delta = abs(candidate.entropy - kept.entropy) / 8.0
            exact = delta <= config.frame_exact_duplicate_threshold
            visual_change = "visual_change" in candidate.frame.reason or "visual_change" in kept.frame.reason
            pixel_close = delta <= config.frame_near_duplicate_threshold
            hash_close = hash_distance <= config.frame_hash_distance_threshold
            entropy_close = entropy_delta <= config.frame_entropy_duplicate_threshold
            near = (
                same_action_phase
                and not visual_change
                and sum((pixel_close, hash_close, entropy_close)) >= 2
            )
            if exact or near:
                duplicate_index = index
                duplicate_delta = delta
                duplicate_hash = hash_distance
                break
        if duplicate_index is None:
            retained.append(candidate)
            continue
        kept = retained[duplicate_index]
        if _candidate_rank(candidate) > _candidate_rank(kept):
            duplicates.append(
                KeyFrameDuplicate(kept.frame, candidate.frame.frame_id, "lower_evidence_priority", duplicate_delta, duplicate_hash)
            )
            retained[duplicate_index] = candidate
        else:
            duplicates.append(
                KeyFrameDuplicate(candidate.frame, kept.frame.frame_id, "visual_duplicate", duplicate_delta, duplicate_hash)
            )
    return [item.frame for item in sorted(retained, key=lambda item: item.frame.timestamp_ms)], duplicates


def _candidate_rank(candidate: _FrameCandidate) -> tuple[int, int, float, int]:
    reason = candidate.frame.reason
    return (
        1 if candidate.priority == "strong" else 0,
        1 if "anchor" in reason or "action_state" in reason else 0,
        candidate.frame.score,
        -candidate.frame.timestamp_ms,
    )


def _focus_actionable_keyframes(
    keyframes: list[KeyFrame],
    candidates: list[_FrameCandidate],
    windows: list[AnalysisWindow],
) -> list[KeyFrame]:
    """Compatibility boundary: selection now happens before global deduplication."""

    del candidates, windows
    return sorted(keyframes, key=lambda frame: frame.timestamp_ms)


def _focus_file_dialog_flows(
    keyframes: list[KeyFrame],
    candidates: list[_FrameCandidate],
    windows: list[AnalysisWindow],
) -> list[KeyFrame]:
    del candidates, windows
    return sorted(keyframes, key=lambda frame: frame.timestamp_ms)


def _clamp_window_to_duration(window: AnalysisWindow, duration_ms: int) -> AnalysisWindow:
    end_ms = min(window.end_ms, duration_ms) if duration_ms > 0 else window.end_ms
    start_ms = min(max(window.start_ms, 0), end_ms)
    return AnalysisWindow(
        start_ms,
        end_ms,
        window.reason,
        priority=window.priority,
        step_ms=window.step_ms,
        max_keyframes=window.max_keyframes,
        diff_threshold=window.diff_threshold,
        anchor_ms=tuple(anchor for anchor in window.anchor_ms if start_ms <= anchor <= end_ms),
        action_anchor_ms=tuple(anchor for anchor in window.action_anchor_ms if start_ms <= anchor <= end_ms),
        action_phases=tuple(phase for phase in window.action_phases if start_ms <= phase[0] <= end_ms),
        requires_post_action_state=window.requires_post_action_state,
        active_apps=window.active_apps,
        active_ranges=_clip_active_ranges(window.active_ranges, start_ms, end_ms),
    )


def _read_frames_for_timestamps(cv2, capture, timestamps: list[int], fps: float, config: VisionConfig) -> dict[int, object]:
    frames = {}
    for group in _timestamp_groups(sorted(set(timestamps)), config.frame_sequential_gap_ms):
        if len(group) == 1:
            frame = _seek_read_frame(cv2, capture, group[0])
            if frame is not None:
                frames[group[0]] = frame
        else:
            frames.update(_read_timestamp_group_sequentially(cv2, capture, group, fps))
    # Some codecs report unreliable positions during sequential reads. Retry
    # only missing evidence timestamps with direct seeks before giving up.
    for timestamp in timestamps:
        if timestamp in frames:
            continue
        frame = _seek_read_frame(cv2, capture, timestamp)
        if frame is not None:
            frames[timestamp] = frame
    return frames


def _timestamp_groups(timestamps: list[int], max_gap_ms: int) -> list[list[int]]:
    if not timestamps:
        return []
    if max_gap_ms <= 0:
        return [[timestamp] for timestamp in timestamps]
    groups = [[timestamps[0]]]
    for timestamp in timestamps[1:]:
        if timestamp - groups[-1][-1] <= max_gap_ms:
            groups[-1].append(timestamp)
        else:
            groups.append([timestamp])
    return groups


def _seek_read_frame(cv2, capture, timestamp_ms: int):
    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = capture.read()
    return frame.copy() if ok else None


def _read_timestamp_group_sequentially(cv2, capture, timestamps: list[int], fps: float) -> dict[int, object]:
    frames = {}
    if not timestamps:
        return frames
    frame_interval_ms = max(1, int(round(1000.0 / max(fps, 1.0))))
    tolerance_ms = max(frame_interval_ms, 50)
    target_index = 0
    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamps[0]))
    while target_index < len(timestamps):
        ok, frame = capture.read()
        if not ok:
            break
        position_ms = int(round(capture.get(cv2.CAP_PROP_POS_MSEC) or timestamps[target_index]))
        if position_ms > timestamps[-1] + tolerance_ms:
            break
        while target_index < len(timestamps) and position_ms + tolerance_ms >= timestamps[target_index]:
            frames[timestamps[target_index]] = frame.copy()
            target_index += 1
    return frames


def _frame_delta(cv2, left, right) -> float:
    return float(cv2.absdiff(left, right).mean() / 255.0)


def _frame_entropy(cv2, gray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    return -sum(
        probability * math.log2(probability)
        for count in histogram
        if count > 0
        for probability in (float(count) / total,)
    )


def _array_delta(left, right) -> float:
    try:
        import numpy as np
    except ImportError:
        return 1.0
    return float(np.mean(np.abs(left.astype("float32") - right.astype("float32"))) / 255.0)


def _average_hash(cv2, gray) -> tuple[int, int]:
    small = cv2.resize(gray, (8, 8))
    mean = float(small.mean())
    value = 0
    for bit, pixel in enumerate(small.flatten()):
        if float(pixel) >= mean:
            value |= 1 << bit
    return value, 64


def _hamming(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] != right[1]:
        return max(left[1], right[1])
    return int((left[0] ^ right[0]).bit_count())


def _is_exact_duplicate(cv2, gray, retained_grays: list[object], threshold: float) -> bool:
    return any(_frame_delta(cv2, gray, retained) <= threshold for retained in retained_grays)


def _near_any(timestamp_ms: int, targets: set[int], tolerance_ms: int) -> bool:
    return any(abs(timestamp_ms - target) <= tolerance_ms for target in targets)


def _evenly_spaced(items: list[_FrameCandidate], limit: int) -> list[_FrameCandidate]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return items
    last = len(items) - 1
    return [items[round(index * last / max(1, limit - 1))] for index in range(limit)]


def _evenly_spaced_values(values: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(values) <= limit:
        return values
    last = len(values) - 1
    return sorted({values[round(index * last / max(1, limit - 1))] for index in range(limit)})


def _timestamp_in_ranges(timestamp_ms: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= timestamp_ms <= end for start, end in ranges)


def _clip_active_ranges(
    ranges: tuple[tuple[int, int], ...],
    start_ms: int,
    end_ms: int,
) -> tuple[tuple[int, int], ...]:
    return _merge_active_ranges(
        tuple((max(start, start_ms), min(end, end_ms)) for start, end in ranges if start <= end_ms and end >= start_ms)
    )


def _merge_active_ranges(*groups: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((start, end) for group in groups for start, end in group if start <= end):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _priority_sort_key(priority: str) -> int:
    return {"strong": 0, "activity": 1, "medium": 2, "weak": 3}.get(priority, 2)


def _ffmpeg_cuda_frame_command(executable: str, video_path: Path, timestamp_ms: int, decoder: str) -> list[str]:
    return [
        executable,
        "-v",
        "error",
        "-hwaccel",
        "cuda",
        "-c:v",
        decoder,
        "-ss",
        f"{max(timestamp_ms, 0) / 1000.0:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]


@lru_cache(maxsize=1)
def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    if os.name != "nt":
        return None
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(packages.glob("BtbN.FFmpeg.*/*/bin/ffmpeg.exe"), reverse=True)
    return str(candidates[0]) if candidates else None


__all__ = [
    "AnalysisWindow",
    "KeyFrame",
    "KeyFrameDuplicate",
    "KeyFrameSelection",
    "build_video_coverage_windows",
    "augment_with_video_coverage",
    "merge_analysis_windows",
    "select_keyframes",
    "select_keyframes_detailed",
]
