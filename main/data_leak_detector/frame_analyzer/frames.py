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


_ACTION_OFFSETS_MS = (-2_000, 0)
_DERIVATION_OFFSETS_MS = (-2_000, 0, 2_000, 5_000)
_FILE_SELECTION_OFFSETS_MS = (-2_000, 0, 2_000, 5_000)
_OUTBOUND_OFFSETS_MS = (-2_000, 0, 2_000, 5_000, 10_000, 15_000)
_OUTBOUND_CONTEXT_OFFSETS_MS = (0, 5_000, 15_000, 25_000, 28_000, 30_000)
_CAPTURE_START_OFFSETS_MS = (-1_000, 0, 1_000, 2_000, 3_000, 5_000, 8_000)
_CAPTURE_OFFSETS_MS = (-15_000, -13_000, -11_000, -9_000, -7_000, -5_000, -3_000, -1_000, 0)


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
        exact_duplicate = _is_exact_duplicate(cv2, gray, retained_grays, config.frame_exact_duplicate_threshold)
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
    mandatory = [item for item in retained if "anchor" in item.frame.reason or "action_state" in item.frame.reason]
    optional = [item for item in retained if item not in mandatory]
    available = max(0, limit - len(mandatory))
    return sorted([*mandatory, *_evenly_spaced(optional, available)], key=lambda item: item.frame.timestamp_ms)


def _probe_timestamps(window: AnalysisWindow, config: VisionConfig) -> list[int]:
    timestamps = set(_action_state_timestamps(window))
    for anchor in _representative_anchors(window.anchor_ms, limit=4):
        for timestamp in (anchor, anchor - window.step_ms, anchor + window.step_ms):
            if window.start_ms <= timestamp <= window.end_ms:
                timestamps.add(timestamp)
    if window.priority == "activity":
        timestamps.update(_activity_probe_timestamps(window))
    elif not timestamps:
        timestamps.update(_coverage_timestamps(window))
    if not timestamps and window.start_ms <= window.end_ms:
        timestamps.add(window.start_ms)
    limit = max(window.max_keyframes * max(1, config.frame_probe_multiplier), len(window.anchor_ms))
    ordered = sorted(timestamp for timestamp in timestamps if window.start_ms <= timestamp <= window.end_ms)
    return _evenly_spaced_values(ordered, limit)


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
        (abs(timestamp_ms - target), action)
        for anchor, action in phases
        for target in _phase_timestamps(window, anchor, action)
        if abs(timestamp_ms - target) <= tolerance_ms
    ]
    return min(matches, default=(0, ""))[1]


def _phase_timestamps(window: AnalysisWindow, anchor: int, action: str) -> tuple[int, ...]:
    if action == "capture_start":
        offsets = _CAPTURE_START_OFFSETS_MS
    elif action == "capture":
        offsets = _CAPTURE_OFFSETS_MS
    elif action == "file_selected":
        offsets = _FILE_SELECTION_OFFSETS_MS
    elif action == "outbound_context":
        offsets = _OUTBOUND_CONTEXT_OFFSETS_MS
    elif action in {"upload", "send", "removable", "screen_share", "paste"}:
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
    focused = list(candidates)
    for anchor, action in sorted(window.action_phases):
        if action == "capture_start":
            start, end = anchor - 1_500, anchor + 8_500
        elif action == "capture":
            start, end = anchor - 15_500, anchor + 500
        elif action == "paste":
            start, end = anchor - 2_500, anchor + 11_000
        elif action in {"clipboard", "derive"}:
            start, end = anchor - 2_500, anchor + 500
        else:
            continue
        phase = [item for item in focused if start <= item.frame.timestamp_ms <= end]
        if not phase:
            continue
        before_or_at = [item for item in phase if item.frame.timestamp_ms <= anchor]
        pool = before_or_at if action in {"capture", "clipboard"} and before_or_at else phase
        if action == "capture_start":
            after_start = [item for item in pool if item.frame.timestamp_ms >= anchor]
            chosen_items = [max(after_start or pool, key=lambda item: (item.frame.score, float(item.gray.std())))]
        elif action == "capture":
            chosen_items = [min(pool, key=lambda item: (float(item.gray.mean()), -item.frame.score))]
        elif action == "derive":
            ordered = sorted(pool, key=lambda item: item.frame.timestamp_ms)
            chosen_items = [ordered[0], ordered[-1]] if ordered[0] is not ordered[-1] else [ordered[0]]
        else:
            chosen_items = [max(pool, key=lambda item: item.frame.timestamp_ms)]
        focused = [item for item in focused if item not in phase]
        focused.extend(chosen_items)
    return sorted({item.frame.frame_id: item for item in focused}.values(), key=lambda item: item.frame.timestamp_ms)


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
    "merge_analysis_windows",
    "select_keyframes",
    "select_keyframes_detailed",
]
