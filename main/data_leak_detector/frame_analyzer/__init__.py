from __future__ import annotations

from data_leak_detector.legacy_paths import FRAME_ANALYZER_IMPL, add_legacy_import_paths


def analyze_video_behavior(*args, **kwargs):
    add_legacy_import_paths(FRAME_ANALYZER_IMPL)
    from relavance_frame import analyze_video_behavior as legacy_analyze_video_behavior

    return legacy_analyze_video_behavior(*args, **kwargs)


__all__ = ["analyze_video_behavior"]

