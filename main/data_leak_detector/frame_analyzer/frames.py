"""Non-uniform keyframe selection for long screen recordings.

This module only deals with video windows and pixels. The decision of which log
events deserve a window lives in `data_leak_detector.log_mining`.
"""

from __future__ import annotations

from functools import lru_cache
import os
import shutil
import subprocess
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


def merge_analysis_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    """Merge overlapping windows with the same priority."""

    merged: list[AnalysisWindow] = []
    for priority in ("strong", "activity", "medium", "weak"):
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
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_ms = int(round(frame_count * 1000.0 / fps)) if frame_count > 0 else 0

    temp_dir = Path(tempfile.mkdtemp(prefix="dld_frames_"))
    candidates: list[_FrameCandidate] = []
    warnings: list[str] = []
    cuda_decoder = _cuda_decoder_for_video(path)

    try:
        for window_index, window in enumerate(windows):
            retained_for_window = 0
            previous_small = None
            retained_hashes: list[tuple[int, int]] = []
            retained_small_frames = []
            last_kept_ms = -10**9
            probe_window = _clamp_window_to_duration(window, duration_ms)
            timestamps = _probe_timestamps(probe_window, config)
            candidate_limit = window.max_keyframes
            if probe_window.active_ranges:
                candidate_limit = max(candidate_limit, window.max_keyframes * config.frame_probe_multiplier)
            coverage_ms = _coverage_timestamps(probe_window)
            frames_by_timestamp = _read_frames_for_timestamps(
                cv2,
                capture,
                path,
                timestamps,
                fps,
                config,
                cuda_decoder=cuda_decoder,
            )
            for timestamp in timestamps:
                if retained_for_window >= candidate_limit:
                    break
                if probe_window.active_ranges and not _timestamp_in_ranges(timestamp, probe_window.active_ranges):
                    continue
                frame = frames_by_timestamp.get(timestamp)
                if frame is None:
                    continue

                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                score = 1.0 if previous_small is None else _frame_delta(cv2, previous_small, gray)
                frame_hash = _average_hash(cv2, gray)
                force_anchor = _is_near_anchor(timestamp, window.anchor_ms, window.step_ms)
                force_activity_context = bool(probe_window.active_ranges) and _is_near_anchor(
                    timestamp,
                    window.anchor_ms,
                    window.step_ms * 2,
                )
                force_activity_gap = bool(probe_window.active_ranges) and not force_activity_context
                force_coverage = _is_near_anchor(timestamp, coverage_ms, window.step_ms)
                force_keep = force_anchor or force_activity_context or force_activity_gap or force_coverage
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
                    if force_anchor or force_activity_context:
                        reason = f"{window.priority}:anchor"
                    elif force_coverage:
                        reason = f"{window.priority}:coverage"
                    elif probe_window.active_ranges:
                        reason = f"{window.priority}:activity_gap"
                    elif previous_small is None:
                        reason = f"{window.priority}:window_start"
                    else:
                        reason = f"{window.priority}:visual_change"
                    keyframe = KeyFrame(frame_id, timestamp, str(image_path), round(float(score), 4), reason, window_id=f"window_{window_index}")
                    candidates.append(_FrameCandidate(frame=keyframe, priority=window.priority, gray=gray, frame_hash=frame_hash))
                    previous_small = gray
                    retained_hashes.append(frame_hash)
                    retained_small_frames.append(gray)
                    last_kept_ms = timestamp
                    retained_for_window += 1
    finally:
        capture.release()

    raw_keyframes = [candidate.frame for candidate in candidates]
    keyframes, duplicates = _dedupe_keyframes_globally(candidates, config)
    keyframes = _focus_activity_gap_keyframes(keyframes)
    if not keyframes and windows:
        warnings.append("no_keyframes_selected")
    return KeyFrameSelection(keyframes=keyframes, raw_keyframes=raw_keyframes, duplicates=duplicates, warnings=warnings)


@dataclass(frozen=True)
class _FrameCandidate:
    frame: KeyFrame
    priority: str
    gray: object
    frame_hash: tuple[int, int]


def _focus_activity_gap_keyframes(keyframes: list[KeyFrame]) -> list[KeyFrame]:
    """Keep compact context/action evidence for activity-gap fallback windows."""

    by_window: dict[str, list[KeyFrame]] = {}
    for frame in keyframes:
        by_window.setdefault(frame.window_id or "window_unknown", []).append(frame)

    focused: list[KeyFrame] = []
    for window_frames in by_window.values():
        activity_gaps = [frame for frame in window_frames if "activity_gap" in frame.reason.lower()]
        if not activity_gaps:
            focused.extend(window_frames)
            continue

        action = max(activity_gaps, key=lambda frame: frame.timestamp_ms)
        preceding_gap = [frame for frame in activity_gaps if frame.timestamp_ms < action.timestamp_ms]
        preceding_anchors = [
            frame
            for frame in window_frames
            if "anchor" in frame.reason.lower() and frame.timestamp_ms <= action.timestamp_ms
        ]
        if preceding_anchors:
            focused.append(max(preceding_anchors, key=lambda frame: frame.timestamp_ms))
        if preceding_gap:
            focused.append(max(preceding_gap, key=lambda frame: frame.timestamp_ms))
        focused.append(action)
    return sorted(focused, key=lambda frame: frame.timestamp_ms)


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
    for candidate in sorted(candidates, key=_dedupe_sort_key):
        duplicate_of = _find_global_duplicate(
            cv2,
            candidate,
            kept,
            config.frame_exact_duplicate_threshold,
            config.frame_near_duplicate_threshold,
            config.frame_hash_distance_threshold,
            config.frame_anchor_duplicate_gap_ms,
        )
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
    near_threshold: float,
    hash_distance_threshold: int,
    anchor_duplicate_gap_ms: int,
) -> tuple[_FrameCandidate, float, int] | None:
    candidate_is_anchor = "anchor" in candidate.frame.reason
    candidate_can_near_dedupe = _can_near_dedupe(candidate.frame)
    for retained in kept:
        retained_is_anchor = "anchor" in retained.frame.reason
        if candidate_is_anchor and not retained_is_anchor:
            continue
        delta = _frame_delta(cv2, retained.gray, candidate.gray)
        hash_distance = _hamming(candidate.frame_hash, retained.frame_hash)
        if candidate.frame.timestamp_ms == retained.frame.timestamp_ms or delta <= exact_threshold:
            return retained, delta, hash_distance
        anchor_near_duplicate = (
            candidate_is_anchor
            and retained_is_anchor
            and delta <= near_threshold
            and hash_distance <= hash_distance_threshold
        )
        close_anchor_near_duplicate = (
            anchor_near_duplicate
            and anchor_duplicate_gap_ms > 0
            and abs(candidate.frame.timestamp_ms - retained.frame.timestamp_ms) <= anchor_duplicate_gap_ms
        )
        if (candidate_can_near_dedupe or close_anchor_near_duplicate or anchor_near_duplicate) and delta <= near_threshold and hash_distance <= hash_distance_threshold:
            return retained, delta, hash_distance
    return None


def _can_near_dedupe(frame: KeyFrame) -> bool:
    reason = frame.reason.lower()
    return "coverage" in reason or "window_start" in reason


def _dedupe_evidence_priority(candidate: _FrameCandidate) -> int:
    """Prefer anchors tied to a sensitive-file activity over generic actions."""

    if candidate.priority == "activity" and "anchor" in candidate.frame.reason:
        return 0
    return _priority_sort_key(candidate.priority) + 1


def _dedupe_sort_key(candidate: _FrameCandidate) -> tuple[int, int]:
    priority = _dedupe_evidence_priority(candidate)
    timestamp = candidate.frame.timestamp_ms
    if priority == 0:
        return priority, -timestamp
    return priority, timestamp


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
    if force_keep:
        return True
    if exact_duplicate:
        return False
    if score < diff_threshold:
        return False
    if config.frame_min_keep_gap_ms > 0 and timestamp_ms - last_kept_ms < config.frame_min_keep_gap_ms:
        return False
    return all(_hamming(frame_hash, retained) > config.frame_hash_distance_threshold for retained in retained_hashes)


def _is_near_anchor(timestamp_ms: int, anchors: tuple[int, ...], step_ms: int) -> bool:
    tolerance = max(step_ms // 2, 125)
    return any(abs(timestamp_ms - anchor) <= tolerance for anchor in anchors)


def _coverage_timestamps(window: AnalysisWindow) -> tuple[int, ...]:
    if window.priority == "activity" or window.anchor_ms or window.end_ms <= window.start_ms or window.max_keyframes <= 3:
        return ()
    slots = min(window.max_keyframes, 12)
    if slots <= 1:
        return (window.start_ms,)
    span = window.end_ms - window.start_ms
    return tuple(window.start_ms + round(slot * span / (slots - 1)) for slot in range(slots))


def _clamp_window_to_duration(window: AnalysisWindow, duration_ms: int) -> AnalysisWindow:
    if duration_ms <= 0:
        return window
    end_ms = min(window.end_ms, duration_ms)
    start_ms = min(window.start_ms, end_ms)
    return AnalysisWindow(
        start_ms=start_ms,
        end_ms=end_ms,
        reason=window.reason,
        priority=window.priority,
        step_ms=window.step_ms,
        max_keyframes=window.max_keyframes,
        diff_threshold=window.diff_threshold,
        anchor_ms=tuple(anchor for anchor in window.anchor_ms if start_ms <= anchor <= end_ms),
        active_apps=window.active_apps,
        active_ranges=_clip_active_ranges(window.active_ranges, start_ms, end_ms),
    )


def _probe_timestamps(window: AnalysisWindow, config: VisionConfig) -> list[int]:
    if window.priority == "activity" or window.active_ranges:
        return _activity_probe_timestamps(window)
    if window.priority == "strong" and window.anchor_ms:
        return _strong_probe_timestamps(window)
    if window.end_ms <= window.start_ms:
        return [window.start_ms]

    exact = list(range(window.start_ms, window.end_ms + 1, max(1, window.step_ms)))
    max_probes = max(window.max_keyframes, window.max_keyframes * config.frame_probe_multiplier)
    if len(exact) <= max_probes:
        timestamps = exact
    else:
        timestamps = []
        last_index = len(exact) - 1
        for slot in range(max_probes):
            timestamps.append(exact[round(slot * last_index / max(1, max_probes - 1))])

    anchors = [anchor for anchor in window.anchor_ms if window.start_ms <= anchor <= window.end_ms]
    ordered: list[int] = []
    seen: set[int] = set()
    for timestamp in [*sorted(anchors), window.start_ms, window.end_ms, *timestamps]:
        if timestamp in seen:
            continue
        seen.add(timestamp)
        ordered.append(timestamp)
    return sorted(ordered)


def _activity_probe_timestamps(window: AnalysisWindow) -> list[int]:
    """Probe sensitive activity anchors and sparse relative positions between them."""

    anchors = sorted(anchor for anchor in window.anchor_ms if window.start_ms <= anchor <= window.end_ms)
    if not anchors:
        return []

    # A leak action can occur between two file-operation anchors.  Cover those
    # gaps by relative positions so continuous activity remains observable
    # without adopting a wall-clock sampling interval.
    gap_candidates_by_position: dict[int, list[int]] = {1: [], 2: [], 3: []}
    confirmation_candidates: list[tuple[int, int]] = []
    for range_start, range_end in _activity_probe_ranges(window):
        range_anchors = [anchor for anchor in anchors if range_start <= anchor <= range_end]
        boundaries = [range_start, *range_anchors, range_end]
        for start_ms, end_ms in zip(boundaries, boundaries[1:]):
            span_ms = end_ms - start_ms
            if span_ms <= 3 * window.step_ms:
                continue
            for fraction in (1, 2, 3):
                gap_candidates_by_position[fraction].append(start_ms + round(span_ms * fraction / 4))
            if any(abs(anchor - end_ms) <= window.step_ms for anchor in anchors):
                operation_timestamp = start_ms + round(span_ms * 3 / 4)
                confirmation_candidates.append((operation_timestamp, start_ms + round(span_ms * 7 / 8)))

    # File exfiltration normally follows the preceding file activity.  Prioritize
    # later gap positions, but retain explicit anchors and their nearby context
    # after that priority set so they cannot consume the whole frame budget.
    gap_budget = max(3, window.max_keyframes // 2)
    gap_timestamps: list[int] = []
    if confirmation_candidates:
        operation_timestamp, confirmation_timestamp = max(confirmation_candidates, key=lambda item: item[1])
        gap_timestamps.extend((operation_timestamp, confirmation_timestamp))
    for fraction in (3, 2, 1):
        remaining = gap_budget - len(gap_timestamps)
        if remaining <= 0:
            break
        gap_timestamps.extend(_evenly_spaced_timestamps(gap_candidates_by_position[fraction], remaining))

    anchor_context = [
        timestamp
        for anchor in anchors
        for timestamp in (anchor - window.step_ms, anchor + window.step_ms, anchor + 2 * window.step_ms)
        if window.start_ms <= timestamp <= window.end_ms
    ]
    ordered: list[int] = []
    seen: set[int] = set()
    for timestamp in [*gap_timestamps, *anchors, *anchor_context]:
        if timestamp not in seen:
            seen.add(timestamp)
            ordered.append(timestamp)
    return sorted(ordered)


def _activity_probe_ranges(window: AnalysisWindow) -> tuple[tuple[int, int], ...]:
    if window.active_ranges:
        return _clip_active_ranges(window.active_ranges, window.start_ms, window.end_ms)
    return ((window.start_ms, window.end_ms),)


def _evenly_spaced_timestamps(timestamps: list[int], limit: int) -> list[int]:
    ordered = sorted(set(timestamps))
    if len(ordered) <= limit:
        return ordered
    last_index = len(ordered) - 1
    return [ordered[round(slot * last_index / max(1, limit - 1))] for slot in range(limit)]


def _strong_probe_timestamps(window: AnalysisWindow) -> list[int]:
    """Keep every explicit risk anchor before adding its immediate context."""

    anchors = sorted(anchor for anchor in window.anchor_ms if window.start_ms <= anchor <= window.end_ms)
    ordered = list(dict.fromkeys(anchors))
    for anchor in anchors:
        for timestamp in (anchor - window.step_ms, anchor + window.step_ms):
            if window.start_ms <= timestamp <= window.end_ms and timestamp not in ordered:
                ordered.append(timestamp)
    return ordered


def _read_frames_for_timestamps(
    cv2,
    capture,
    video_path: Path,
    timestamps: list[int],
    fps: float,
    config: VisionConfig,
    *,
    cuda_decoder: str | None,
) -> dict[int, object]:
    if not timestamps:
        return {}

    ordered_timestamps = sorted(set(timestamps))
    frames = _read_frames_with_ffmpeg_cuda(cv2, video_path, ordered_timestamps, cuda_decoder)
    remaining = [timestamp for timestamp in ordered_timestamps if timestamp not in frames]
    for group in _timestamp_groups(remaining, config.frame_sequential_gap_ms):
        if len(group) == 1:
            frame = _seek_read_frame(cv2, capture, group[0])
            if frame is not None:
                frames[group[0]] = frame
            continue
        frames.update(_read_timestamp_group_sequentially(cv2, capture, group, fps))
    return frames


def _read_frames_with_ffmpeg_cuda(cv2, video_path: Path, timestamps: list[int], decoder: str | None) -> dict[int, object]:
    if not decoder:
        return {}

    try:
        import numpy as np
    except ImportError:
        return {}

    executable = _ffmpeg_executable()
    if not executable:
        return {}

    frames: dict[int, object] = {}
    for timestamp_ms in timestamps:
        command = _ffmpeg_cuda_frame_command(executable, video_path, timestamp_ms, decoder)
        try:
            result = subprocess.run(command, capture_output=True, timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0 or not result.stdout:
            continue
        frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frames[timestamp_ms] = frame
    return frames


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


@lru_cache(maxsize=1)
def _ffmpeg_cuda_available(executable: str) -> bool:
    try:
        result = subprocess.run([executable, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "cuda" in result.stdout.lower()


@lru_cache(maxsize=64)
def _cuda_decoder_for_video(video_path: Path) -> str | None:
    executable = _ffmpeg_executable()
    if not executable or not _ffmpeg_cuda_available(executable):
        return None

    ffprobe = str(Path(executable).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    decoders = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "av1": "av1_cuvid"}
    return decoders.get(result.stdout.strip().lower()) if result.returncode == 0 else None


def _timestamp_groups(timestamps: list[int], max_gap_ms: int) -> list[list[int]]:
    if not timestamps:
        return []
    if max_gap_ms <= 0:
        return [[timestamp] for timestamp in timestamps]

    groups: list[list[int]] = [[timestamps[0]]]
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
    frames: dict[int, object] = {}
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
            max_keyframes=max(previous.max_keyframes, window.max_keyframes, len(anchors)),
            diff_threshold=min(previous.diff_threshold, window.diff_threshold),
            anchor_ms=anchors,
            active_apps=apps,
            active_ranges=_merge_active_ranges(previous.active_ranges, window.active_ranges),
        )
    return merged


def _timestamp_in_ranges(timestamp_ms: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start_ms <= timestamp_ms <= end_ms for start_ms, end_ms in ranges)


def _clip_active_ranges(
    ranges: tuple[tuple[int, int], ...],
    start_ms: int,
    end_ms: int,
) -> tuple[tuple[int, int], ...]:
    return _merge_active_ranges(
        tuple((max(range_start, start_ms), min(range_end, end_ms)) for range_start, range_end in ranges if range_start <= end_ms and range_end >= start_ms),
    )


def _merge_active_ranges(*range_groups: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    intervals = sorted((start_ms, end_ms) for group in range_groups for start_ms, end_ms in group if start_ms <= end_ms)
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in intervals:
        if not merged or start_ms > merged[-1][1] + 1:
            merged.append((start_ms, end_ms))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
    return tuple(merged)


def _priority_sort_key(priority: str) -> int:
    return {"strong": 0, "activity": 1, "medium": 2, "weak": 3}.get(priority, 2)
