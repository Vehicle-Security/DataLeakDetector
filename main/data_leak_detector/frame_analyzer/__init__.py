"""FrameAnalyzer 包边界。

公共 API 导出的是一个确定性的观察构建器。把包边界显式保留下来，可以在以后替换成 OCR/VLM 实现，
而无需修改流水线或测试中的调用方。
"""

from __future__ import annotations

from .analyzer import analyze_video_behavior

__all__ = ["analyze_video_behavior"]
