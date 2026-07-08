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
    active_apps: tuple[str, ...] = ()


@dataclass(frozen=True)
class KeyFrame:
    frame_id: str
    timestamp_ms: int
    image_path: str
    score: float
    reason: str


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
        if not (sensitive_hit or action_hit):
            continue
        center = event.timestamp_ms
        if not center:
            continue
        windows.append(
            AnalysisWindow(
                start_ms=max(center - config.frame_window_before_ms, 0),
                end_ms=center + config.frame_window_after_ms,
                reason=event.event_type or "sensitive_activity",
                active_apps=_active_apps_near(logs, center, config.frame_window_after_ms),
            )
        )

    return _merge_windows(windows)


def select_keyframes(
    video_path: str | Path,
    windows: list[AnalysisWindow],
    config: VisionConfig,
) -> tuple[list[KeyFrame], list[str]]:
    """Select keyframes by visual change instead of uniform sampling.

    The algorithm scans only suspicious windows, compares candidate frames with
    the last retained frame, and keeps application switches, page jumps, and
    large content changes. OpenCV is optional; when unavailable the caller gets
    an explanatory warning and the deterministic log path still works.
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
            timestamp = window.start_ms
            while timestamp <= window.end_ms and retained_for_window < config.max_keyframes_per_window:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
                ok, frame = capture.read()
                if not ok:
                    timestamp += config.frame_step_ms
                    continue

                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                score = 1.0 if previous_small is None else _frame_delta(cv2, previous_small, gray)
                keep = previous_small is None or score >= config.frame_diff_threshold
                if keep:
                    frame_id = f"frame_{window_index}_{retained_for_window}"
                    image_path = temp_dir / f"{frame_id}_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    reason = "window_start" if previous_small is None else "visual_change"
                    frames.append(KeyFrame(frame_id, timestamp, str(image_path), round(float(score), 4), reason))
                    previous_small = gray
                    retained_for_window += 1
                timestamp += config.frame_step_ms
    finally:
        capture.release()

    if not frames and windows:
        warnings.append("no_keyframes_selected")
    return frames, warnings


def _frame_delta(cv2, previous, current) -> float:
    diff = cv2.absdiff(previous, current)
    return float(diff.mean() / 255.0)


def _active_apps_near(logs: list[LogEvent], center_ms: int, radius_ms: int) -> tuple[str, ...]:
    apps: list[str] = []
    for event in logs:
        if not event.timestamp_ms or abs(event.timestamp_ms - center_ms) > radius_ms:
            continue
        app = event.app_name or event.process_name
        if app and app not in apps:
            apps.append(app)
    return tuple(apps)


def _merge_windows(windows: list[AnalysisWindow]) -> list[AnalysisWindow]:
    merged: list[AnalysisWindow] = []
    for window in sorted(windows, key=lambda item: item.start_ms):
        if not merged or window.start_ms > merged[-1].end_ms:
            merged.append(window)
            continue
        previous = merged[-1]
        apps = tuple(dict.fromkeys([*previous.active_apps, *window.active_apps]))
        merged[-1] = AnalysisWindow(
            start_ms=previous.start_ms,
            end_ms=max(previous.end_ms, window.end_ms),
            reason=f"{previous.reason}+{window.reason}",
            active_apps=apps,
        )
    return merged
