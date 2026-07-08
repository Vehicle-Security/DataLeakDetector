"""Non-uniform keyframe selection for long screen recordings."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..models import LogEvent
from ..io import flatten_text, looks_sensitive, normalize_path
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any
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


def build_analysis_windows(
    logs: list[LogEvent],
    sensitive_files: Iterable[str],
    config: VisionConfig,
) -> list[AnalysisWindow]:
    sensitive = tuple(normalize_path(item).lower() for item in sensitive_files)
    windows: list[AnalysisWindow] = []

    for event in logs:
        text = flatten_text(event.raw)
        file_text = normalize_path(event.file_path).lower()
        sensitive_hit = any(item and item in file_text for item in sensitive) or looks_sensitive(file_text) or looks_sensitive(text)
        action_hit = contains_any(text, TRANSFER_TOKENS) or contains_any(text, SINK_TOKENS)
        priority = _window_priority(event, text, sensitive_hit, action_hit)
        if priority == "none":
            continue
        center = event.video_time_ms
        if center < 0:
            continue
        before_ms, after_ms, step_ms, max_keyframes, diff_threshold = _window_profile(priority, config)
        windows.append(
            AnalysisWindow(
                start_ms=max(center - before_ms, 0),
                end_ms=center + after_ms,
                reason=_window_reason(event, priority),
                priority=priority,
                step_ms=step_ms,
                max_keyframes=max_keyframes,
                diff_threshold=diff_threshold,
                anchor_ms=_event_anchors(event, sensitive_hit),
                active_apps=_active_apps_near(logs, center, after_ms),
            )
        )

    return _merge_windows(windows)


def select_keyframes(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> tuple[list[KeyFrame], list[str]]:
    """Select keyframes by visual change instead of uniform sampling.

    The algorithm scans only suspicious windows and keeps frames that are both
    visually changed and not near-duplicates of earlier retained frames in the
    same window. It intentionally has no "keep every N seconds" fallback:
    model/OCR budget should be spent on evidence-bearing states, not elapsed
    time. OpenCV is optional; when unavailable the caller gets an explanatory
    warning and the deterministic log path still works.
    """

    if not str(video_path or "").strip():
        return [], []
    path = Path(video_path)
    if not path.exists():
        return [], [f"video_not_found: {path}"]
    if not path.is_file():
        return [], [f"video_not_file: {path}"]

    try:
        import cv2
    except ImportError:
        return [], ["opencv_not_installed: install data-leak-detector[vision] to enable keyframe extraction"]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], [f"video_open_failed: {path}"]

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        capture.release()
        return [], ["video_fps_unavailable"]

    temp_dir = Path(tempfile.mkdtemp(prefix="dld_frames_"))
    frames: list[KeyFrame] = []
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
                    frames.append(KeyFrame(frame_id, timestamp, str(image_path), round(float(score), 4), reason, window_id=f"window_{window_index}"))
                    previous_small = gray
                    retained_hashes.append(frame_hash)
                    retained_small_frames.append(gray)
                    last_kept_ms = timestamp
                    retained_for_window += 1
                timestamp += window.step_ms
    finally:
        capture.release()

    if not frames and windows:
        warnings.append("no_keyframes_selected")
    return frames, warnings


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

    if str(extra.get("risk_level") or "") in {"高", "high"}:
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


def _is_near_anchor(timestamp_ms: int, anchors: tuple[int, ...], step_ms: int) -> bool:
    tolerance = max(step_ms // 2, 125)
    return any(abs(timestamp_ms - anchor) <= tolerance for anchor in anchors)


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


def _merge_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    merged: list[AnalysisWindow] = []
    for priority in ("strong", "medium", "weak"):
        same_priority = [window for window in windows if window.priority == priority]
        merged.extend(_merge_same_priority_windows(same_priority))
    return sorted(merged, key=lambda item: (_priority_sort_key(item.priority), item.start_ms, item.end_ms))


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
