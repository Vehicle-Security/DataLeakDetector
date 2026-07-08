"""Non-uniform keyframe selection for long screen recordings.

This module only deals with video windows and pixels. The decision of which log
events deserve a window lives in `data_leak_detector.log_mining`.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import VisionConfig


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
    active_apps: tuple[str, ...] = ()


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


def merge_analysis_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    """Merge overlapping windows with the same priority."""

    merged: list[AnalysisWindow] = []
    for priority in ("strong", "medium", "weak"):
        same_priority = [window for window in windows if window.priority == priority]
        merged.extend(_merge_same_priority_windows(same_priority))
    return sorted(merged, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))


def select_keyframes(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> tuple[list[KeyFrame], list[str]]:
    """Select keyframes by visual change instead of uniform sampling."""

    selection = select_keyframes_detailed(video_path, windows, config)
    return selection.keyframes, selection.warnings


def select_keyframes_detailed(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> KeyFrameSelection:
    """Select keyframes and preserve pre-global-dedup debug frames."""

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
        return KeyFrameSelection([], [], [], ["opencv_not_installed: install data-leak-detector[vision] to enable keyframe extraction"])

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return KeyFrameSelection([], [], [], [f"video_open_failed: {path}"])

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        capture.release()
        return KeyFrameSelection([], [], [], ["video_fps_unavailable"])

    temp_dir = Path(tempfile.mkdtemp(prefix="dld_frames_"))
    candidates: list[_FrameCandidate] = []
    warnings: list[str] = []

    try:
        for window_index, window in enumerate(windows):
            retained_for_window = 0
            previous_small = None
            retained_hashes: list[tuple[int, int]] = []
            retained_small_frames = []
            last_kept_ms = -10**9
            timestamp = window.start_ms
            while timestamp <= window.end_ms and retained_for_window < window.max_keyframes:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
                ok, frame = capture.read()
                if not ok:
                    timestamp += window.step_ms
                    continue

                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                score = 1.0 if previous_small is None else _frame_delta(cv2, previous_small, gray)
                frame_hash = _average_hash(cv2, gray)
                force_keep = _is_near_anchor(timestamp, window.anchor_ms, window.step_ms)
                exact_duplicate = _is_exact_duplicate(cv2, gray, retained_small_frames, config.frame_exact_duplicate_threshold)
                keep = _should_keep_frame(
                    timestamp_ms=timestamp,
                    score=score,
                    diff_threshold=window.diff_threshold,
                    force_keep=force_keep,
                    exact_duplicate=exact_duplicate,
                    frame_hash=frame_hash,
                    retained_hashes=retained_hashes,
                    previous_small=previous_small,
                    last_kept_ms=last_kept_ms,
                    config=config,
                )
                if keep:
                    frame_id = f"frame_{window_index}_{retained_for_window}"
                    image_path = temp_dir / f"{frame_id}_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    reason = f"{window.priority}:window_start" if previous_small is None else f"{window.priority}:visual_change"
                    keyframe = KeyFrame(frame_id, timestamp, str(image_path), round(float(score), 4), reason, window_id=f"window_{window_index}")
                    candidates.append(_FrameCandidate(frame=keyframe, priority=window.priority, gray=gray, frame_hash=frame_hash))
                    previous_small = gray
                    retained_hashes.append(frame_hash)
                    retained_small_frames.append(gray)
                    last_kept_ms = timestamp
                    retained_for_window += 1
                timestamp += window.step_ms
    finally:
        capture.release()

    raw_keyframes = [candidate.frame for candidate in candidates]
    keyframes, duplicates = _dedupe_keyframes_globally(candidates, config)
    if not keyframes and windows:
        warnings.append("no_keyframes_selected")
    return KeyFrameSelection(keyframes=keyframes, raw_keyframes=raw_keyframes, duplicates=duplicates, warnings=warnings)


@dataclass(frozen=True)
class _FrameCandidate:
    frame: KeyFrame
    priority: str
    gray: object
    frame_hash: tuple[int, int]


def _frame_delta(cv2, previous, current) -> float:
    diff = cv2.absdiff(previous, current)
    return float(diff.mean() / 255.0)


def _average_hash(cv2, gray) -> tuple[int, int]:
    tiny = cv2.resize(gray, (16, 16))
    mean = float(tiny.mean())
    bits = 0
    for index, value in enumerate(tiny.flatten()):
        if value >= mean:
            bits |= 1 << index
    return bits, 64


def _hamming(left: tuple[int, int], right: tuple[int, int]) -> int:
    return int((left[0] ^ right[0]).bit_count())


def _is_exact_duplicate(cv2, current, retained_frames: list, threshold: float) -> bool:
    return any(_frame_delta(cv2, retained, current) <= threshold for retained in retained_frames)


def _dedupe_keyframes_globally(
    candidates: list[_FrameCandidate],
    config: VisionConfig,
) -> tuple[list[KeyFrame], list[KeyFrameDuplicate]]:
    if not candidates:
        return [], []

    try:
        import cv2
    except ImportError:
        return [candidate.frame for candidate in candidates], []

    kept: list[_FrameCandidate] = []
    duplicates: list[KeyFrameDuplicate] = []
    for candidate in sorted(candidates, key=lambda item: (_priority_sort_key(item.priority), item.frame.timestamp_ms)):
        duplicate_of = _find_global_duplicate(cv2, candidate, kept, config.frame_exact_duplicate_threshold)
        if duplicate_of is None:
            kept.append(candidate)
            continue
        kept_candidate, delta, hash_distance = duplicate_of
        duplicates.append(
            KeyFrameDuplicate(
                frame=candidate.frame,
                kept_frame_id=kept_candidate.frame.frame_id,
                reason="same_timestamp" if candidate.frame.timestamp_ms == kept_candidate.frame.timestamp_ms else "near_exact_visual_duplicate",
                delta=round(float(delta), 6),
                hash_distance=hash_distance,
            )
        )
    return [candidate.frame for candidate in sorted(kept, key=lambda item: item.frame.timestamp_ms)], duplicates


def _find_global_duplicate(
    cv2,
    candidate: _FrameCandidate,
    kept: list[_FrameCandidate],
    exact_threshold: float,
) -> tuple[_FrameCandidate, float, int] | None:
    for retained in kept:
        delta = _frame_delta(cv2, retained.gray, candidate.gray)
        hash_distance = _hamming(candidate.frame_hash, retained.frame_hash)
        if candidate.frame.timestamp_ms == retained.frame.timestamp_ms or delta <= exact_threshold:
            return retained, delta, hash_distance
    return None


def _should_keep_frame(
    *,
    timestamp_ms: int,
    score: float,
    frame_hash: tuple[int, int],
    diff_threshold: float,
    force_keep: bool,
    exact_duplicate: bool,
    retained_hashes: list[tuple[int, int]],
    previous_small,
    last_kept_ms: int,
    config: VisionConfig,
) -> bool:
    if previous_small is None:
        return True
    if exact_duplicate:
        return False
    if force_keep:
        return True
    if score < diff_threshold:
        return False
    if config.frame_min_keep_gap_ms > 0 and timestamp_ms - last_kept_ms < config.frame_min_keep_gap_ms:
        return False
    return all(_hamming(frame_hash, retained) > config.frame_hash_distance_threshold for retained in retained_hashes)


def _is_near_anchor(timestamp_ms: int, anchors: tuple[int, ...], step_ms: int) -> bool:
    tolerance = max(step_ms // 2, 125)
    return any(abs(timestamp_ms - anchor) <= tolerance for anchor in anchors)


def _merge_same_priority_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    merged: list[AnalysisWindow] = []
    for window in sorted(windows, key=lambda item: item.start_ms):
        if not merged or window.start_ms > merged[-1].end_ms:
            merged.append(window)
            continue
        previous = merged[-1]
        apps = tuple(dict.fromkeys([*previous.active_apps, *window.active_apps]))
        anchors = tuple(sorted({*previous.anchor_ms, *window.anchor_ms}))
        merged[-1] = AnalysisWindow(
            start_ms=previous.start_ms,
            end_ms=max(previous.end_ms, window.end_ms),
            reason=f"{previous.reason}+{window.reason}",
            priority=previous.priority,
            step_ms=min(previous.step_ms, window.step_ms),
            max_keyframes=max(previous.max_keyframes, window.max_keyframes),
            diff_threshold=min(previous.diff_threshold, window.diff_threshold),
            anchor_ms=anchors,
            active_apps=apps,
        )
    return merged


def _priority_sort_key(priority: str) -> int:
    return {"strong": 0, "medium": 1, "weak": 2}.get(priority, 1)
