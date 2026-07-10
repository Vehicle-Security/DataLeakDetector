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
    case_segment_ms: int = 300_000
    frame_step_ms: int = 1_000
    strong_frame_step_ms: int = 250
    weak_frame_step_ms: int = 2_000
    strong_window_before_ms: int = 5_000
    strong_window_after_ms: int = 15_000
    frame_diff_threshold: float = 0.08
    strong_frame_diff_threshold: float = 0.015
    frame_exact_duplicate_threshold: float = 0.001
    frame_hash_distance_threshold: int = 12
    frame_min_keep_gap_ms: int = 0
    frame_probe_multiplier: int = 6
    frame_sequential_gap_ms: int = 5_000
    frame_anchor_duplicate_gap_ms: int = 500
    ocr_text_similarity_threshold: float = 0.92
    max_keyframes_per_window: int = 18
    max_keyframes_per_strong_window: int = 24
    max_keyframes_per_medium_window: int = 2
    max_keyframes_per_weak_window: int = 2
    include_weak_windows: bool = False
    include_unanchored_medium_windows: bool = False
    max_vlm_frames: int = 8
    ocr_provider: str = "none"
    ocr_min_confidence: float = 0.70
    ocr_max_image_side: int = 1_280
    ocr_workers: int = 1
    ocr_batch_size: int = 8
    ocr_use_cuda: bool = False
    ocr_roi_enabled: bool = False
    ocr_roi_window_first: bool = True
    ocr_roi_min_text_density: float = 0.002
    ocr_roi_max_regions: int = 3
    ocr_roi_padding: int = 24
    rapidocr_use_cuda: bool = False
    vlm_provider: str = "openai_compatible"
    vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_chat_url: str = ""
    vlm_model: str = "qwen-vl-max-latest"
    vlm_api_key: str = ""
    vlm_timeout_seconds: int = 60
    vlm_dry_run: bool = False
    vlm_frame_strategy: str = "ocr_triage"
    vlm_grid_size: int = 1
    vlm_include_empty_ocr_strong_frames: bool = True
    vlm_max_frames_per_window: int = 3

    @classmethod
    def from_env(cls) -> "VisionConfig":
        _load_dotenv()
        ocr_use_cuda = _env_bool("DLD_OCR_USE_CUDA", _env_bool("DLD_RAPIDOCR_USE_CUDA", False))
        return cls(
            enabled=_env_bool("DLD_VISION_ENABLED", False),
            mode=os.getenv("DLD_VISION_MODE", "hybrid"),
            frame_window_before_ms=_env_int("DLD_FRAME_WINDOW_BEFORE_MS", 30_000),
            frame_window_after_ms=_env_int("DLD_FRAME_WINDOW_AFTER_MS", 120_000),
            case_segment_ms=max(1, _env_int("DLD_CASE_SEGMENT_MS", 300_000)),
            frame_step_ms=_env_int("DLD_FRAME_STEP_MS", 1_000),
            strong_frame_step_ms=_env_int("DLD_STRONG_FRAME_STEP_MS", 250),
            weak_frame_step_ms=_env_int("DLD_WEAK_FRAME_STEP_MS", 2_000),
            strong_window_before_ms=_env_int("DLD_STRONG_WINDOW_BEFORE_MS", 5_000),
            strong_window_after_ms=_env_int("DLD_STRONG_WINDOW_AFTER_MS", 15_000),
            frame_diff_threshold=_env_float("DLD_FRAME_DIFF_THRESHOLD", 0.08),
            strong_frame_diff_threshold=_env_float("DLD_STRONG_FRAME_DIFF_THRESHOLD", 0.015),
            frame_exact_duplicate_threshold=_env_float("DLD_FRAME_EXACT_DUPLICATE_THRESHOLD", 0.001),
            frame_hash_distance_threshold=_env_int("DLD_FRAME_HASH_DISTANCE_THRESHOLD", 12),
            frame_min_keep_gap_ms=_env_int("DLD_FRAME_MIN_KEEP_GAP_MS", 0),
            frame_probe_multiplier=max(1, _env_int("DLD_FRAME_PROBE_MULTIPLIER", 6)),
            frame_sequential_gap_ms=max(0, _env_int("DLD_FRAME_SEQUENTIAL_GAP_MS", 5_000)),
            frame_anchor_duplicate_gap_ms=max(0, _env_int("DLD_FRAME_ANCHOR_DUPLICATE_GAP_MS", 500)),
            ocr_text_similarity_threshold=_env_float("DLD_OCR_TEXT_SIMILARITY_THRESHOLD", 0.92),
            max_keyframes_per_window=_env_int("DLD_MAX_KEYFRAMES_PER_WINDOW", 18),
            max_keyframes_per_strong_window=_env_int("DLD_MAX_KEYFRAMES_PER_STRONG_WINDOW", 24),
            max_keyframes_per_medium_window=_env_int("DLD_MAX_KEYFRAMES_PER_MEDIUM_WINDOW", 2),
            max_keyframes_per_weak_window=_env_int("DLD_MAX_KEYFRAMES_PER_WEAK_WINDOW", 2),
            include_weak_windows=_env_bool("DLD_INCLUDE_WEAK_WINDOWS", False),
            include_unanchored_medium_windows=_env_bool("DLD_INCLUDE_UNANCHORED_MEDIUM_WINDOWS", False),
            max_vlm_frames=_env_int("DLD_MAX_VLM_FRAMES", 8),
            ocr_provider=os.getenv("DLD_OCR_PROVIDER", "none"),
            ocr_min_confidence=_env_float("DLD_OCR_MIN_CONFIDENCE", 0.70),
            ocr_max_image_side=_env_int("DLD_OCR_MAX_IMAGE_SIDE", 1_280),
            ocr_workers=max(1, _env_int("DLD_OCR_WORKERS", 1)),
            ocr_batch_size=max(1, _env_int("DLD_OCR_BATCH_SIZE", 8)),
            ocr_use_cuda=ocr_use_cuda,
            ocr_roi_enabled=_env_bool("DLD_OCR_ROI_ENABLED", False),
            ocr_roi_window_first=_env_bool("DLD_OCR_ROI_WINDOW_FIRST", True),
            ocr_roi_min_text_density=_env_float("DLD_OCR_ROI_MIN_TEXT_DENSITY", 0.002),
            ocr_roi_max_regions=max(1, _env_int("DLD_OCR_ROI_MAX_REGIONS", 3)),
            ocr_roi_padding=max(0, _env_int("DLD_OCR_ROI_PADDING", 24)),
            rapidocr_use_cuda=ocr_use_cuda,
            vlm_provider=os.getenv("DLD_VLM_PROVIDER", "openai_compatible"),
            vlm_base_url=os.getenv("DLD_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            vlm_chat_url=os.getenv("DLD_VLM_CHAT_URL", ""),
            vlm_model=os.getenv("DLD_VLM_MODEL", "qwen-vl-max-latest"),
            vlm_api_key=os.getenv("DLD_VLM_API_KEY", ""),
            vlm_timeout_seconds=_env_int("DLD_VLM_TIMEOUT_SECONDS", 60),
            vlm_dry_run=_env_bool("DLD_VLM_DRY_RUN", False),
            vlm_frame_strategy=os.getenv("DLD_VLM_FRAME_STRATEGY", "ocr_triage"),
            vlm_grid_size=max(1, _env_int("DLD_VLM_GRID_SIZE", 1)),
            vlm_include_empty_ocr_strong_frames=_env_bool("DLD_VLM_INCLUDE_EMPTY_OCR_STRONG_FRAMES", True),
            vlm_max_frames_per_window=max(1, _env_int("DLD_VLM_MAX_FRAMES_PER_WINDOW", 3)),
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
            case_segment_ms=self.case_segment_ms,
            frame_step_ms=self.frame_step_ms,
            strong_frame_step_ms=self.strong_frame_step_ms,
            weak_frame_step_ms=self.weak_frame_step_ms,
            strong_window_before_ms=self.strong_window_before_ms,
            strong_window_after_ms=self.strong_window_after_ms,
            frame_diff_threshold=self.frame_diff_threshold,
            strong_frame_diff_threshold=self.strong_frame_diff_threshold,
            frame_exact_duplicate_threshold=self.frame_exact_duplicate_threshold,
            frame_hash_distance_threshold=self.frame_hash_distance_threshold,
            frame_min_keep_gap_ms=self.frame_min_keep_gap_ms,
            frame_probe_multiplier=self.frame_probe_multiplier,
            frame_sequential_gap_ms=self.frame_sequential_gap_ms,
            frame_anchor_duplicate_gap_ms=self.frame_anchor_duplicate_gap_ms,
            ocr_text_similarity_threshold=self.ocr_text_similarity_threshold,
            max_keyframes_per_window=self.max_keyframes_per_window,
            max_keyframes_per_strong_window=self.max_keyframes_per_strong_window,
            max_keyframes_per_medium_window=self.max_keyframes_per_medium_window,
            max_keyframes_per_weak_window=self.max_keyframes_per_weak_window,
            include_weak_windows=self.include_weak_windows,
            include_unanchored_medium_windows=self.include_unanchored_medium_windows,
            max_vlm_frames=self.max_vlm_frames if max_vlm_frames is None else max_vlm_frames,
            ocr_provider=self.ocr_provider,
            ocr_min_confidence=self.ocr_min_confidence,
            ocr_max_image_side=self.ocr_max_image_side,
            ocr_workers=self.ocr_workers,
            ocr_batch_size=self.ocr_batch_size,
            ocr_use_cuda=self.ocr_use_cuda,
            ocr_roi_enabled=self.ocr_roi_enabled,
            ocr_roi_window_first=self.ocr_roi_window_first,
            ocr_roi_min_text_density=self.ocr_roi_min_text_density,
            ocr_roi_max_regions=self.ocr_roi_max_regions,
            ocr_roi_padding=self.ocr_roi_padding,
            rapidocr_use_cuda=self.ocr_use_cuda,
            vlm_provider=self.vlm_provider,
            vlm_base_url=self.vlm_base_url,
            vlm_chat_url=self.vlm_chat_url,
            vlm_model=self.vlm_model,
            vlm_api_key=self.vlm_api_key,
            vlm_timeout_seconds=self.vlm_timeout_seconds,
            vlm_dry_run=self.vlm_dry_run,
            vlm_frame_strategy=self.vlm_frame_strategy,
            vlm_grid_size=self.vlm_grid_size,
            vlm_include_empty_ocr_strong_frames=self.vlm_include_empty_ocr_strong_frames,
            vlm_max_frames_per_window=self.vlm_max_frames_per_window,
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
