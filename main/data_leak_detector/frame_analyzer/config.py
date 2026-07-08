"""Runtime configuration for OCR/VLM-assisted frame analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False
    mode: str = "hybrid"
    frame_window_before_ms: int = 30_000
    frame_window_after_ms: int = 120_000
    frame_step_ms: int = 1_000
    frame_diff_threshold: float = 0.08
    max_keyframes_per_window: int = 18
    max_vlm_frames: int = 8
    ocr_provider: str = "none"
    ocr_min_confidence: float = 0.70
    vlm_provider: str = "openai_compatible"
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_chat_url: str = ""
    vlm_model: str = "qwen-vl-max-latest"
    vlm_api_key: str = ""
    vlm_timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "VisionConfig":
        _load_dotenv()
        return cls(
            enabled=_env_bool("DLD_VISION_ENABLED", False),
            mode=os.getenv("DLD_VISION_MODE", "hybrid"),
            frame_window_before_ms=_env_int("DLD_FRAME_WINDOW_BEFORE_MS", 30_000),
            frame_window_after_ms=_env_int("DLD_FRAME_WINDOW_AFTER_MS", 120_000),
            frame_step_ms=_env_int("DLD_FRAME_STEP_MS", 1_000),
            frame_diff_threshold=_env_float("DLD_FRAME_DIFF_THRESHOLD", 0.08),
            max_keyframes_per_window=_env_int("DLD_MAX_KEYFRAMES_PER_WINDOW", 18),
            max_vlm_frames=_env_int("DLD_MAX_VLM_FRAMES", 8),
            ocr_provider=os.getenv("DLD_OCR_PROVIDER", "none"),
            ocr_min_confidence=_env_float("DLD_OCR_MIN_CONFIDENCE", 0.70),
            vlm_provider=os.getenv("DLD_VLM_PROVIDER", "openai_compatible"),
            vlm_base_url=os.getenv("DLD_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            vlm_chat_url=os.getenv("DLD_VLM_CHAT_URL", ""),
            vlm_model=os.getenv("DLD_VLM_MODEL", "qwen-vl-max-latest"),
            vlm_api_key=os.getenv("DLD_VLM_API_KEY", ""),
            vlm_timeout_seconds=_env_int("DLD_VLM_TIMEOUT_SECONDS", 60),
        )

    def with_overrides(
        self,
        *,
        enabled: bool | None = None,
        mode: str | None = None,
        max_vlm_frames: int | None = None,
    ) -> "VisionConfig":
        return VisionConfig(
            enabled=self.enabled if enabled is None else enabled,
            mode=self.mode if mode is None else mode,
            frame_window_before_ms=self.frame_window_before_ms,
            frame_window_after_ms=self.frame_window_after_ms,
            frame_step_ms=self.frame_step_ms,
            frame_diff_threshold=self.frame_diff_threshold,
            max_keyframes_per_window=self.max_keyframes_per_window,
            max_vlm_frames=self.max_vlm_frames if max_vlm_frames is None else max_vlm_frames,
            ocr_provider=self.ocr_provider,
            ocr_min_confidence=self.ocr_min_confidence,
            vlm_provider=self.vlm_provider,
            vlm_base_url=self.vlm_base_url,
            vlm_chat_url=self.vlm_chat_url,
            vlm_model=self.vlm_model,
            vlm_api_key=self.vlm_api_key,
            vlm_timeout_seconds=self.vlm_timeout_seconds,
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env", override=False)
