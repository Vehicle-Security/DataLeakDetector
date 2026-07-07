"""FrameAnalyzer package boundary.

The public API exports a deterministic observation builder. Keeping the package
boundary explicit makes it possible to swap in OCR/VLM implementations later
without changing callers in the pipeline or tests.
"""

from __future__ import annotations

from .analyzer import analyze_video_behavior

__all__ = ["analyze_video_behavior"]
