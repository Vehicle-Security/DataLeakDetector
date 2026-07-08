"""OCR provider abstraction for reading every selected keyframe."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .config import VisionConfig
from .frames import KeyFrame

_RAPIDOCR_ENGINE_CACHE: dict[tuple[bool, int], object] = {}


@dataclass(frozen=True)
class OcrResult:
    frame: KeyFrame
    text: str
    confidence: float
    provider: str


class OcrProvider(Protocol):
    def read(self, frame: KeyFrame) -> OcrResult:
        ...


class NoopOcrProvider:
    def read(self, frame: KeyFrame) -> OcrResult:
        return OcrResult(frame=frame, text="", confidence=0.0, provider="none")


class TesseractOcrProvider:
    """Local OCR provider backed by pytesseract when installed."""

    def read(self, frame: KeyFrame) -> OcrResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return OcrResult(frame=frame, text="", confidence=0.0, provider="tesseract_missing")

        image = Image.open(frame.image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = str(text or "").strip()
            if not text:
                continue
            try:
                score = float(conf)
            except (TypeError, ValueError):
                score = -1.0
            if score >= 0:
                confidences.append(score / 100.0)
            words.append(text)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OcrResult(frame=frame, text=" ".join(words), confidence=round(confidence, 3), provider="tesseract")


class RapidOcrProvider:
    """Fast local OCR provider backed by RapidOCR/ONNX Runtime.

    RapidOCR works well on Chinese Windows UI text and can use CUDA through
    onnxruntime-gpu when `DLD_RAPIDOCR_USE_CUDA=1` and CUDAExecutionProvider is
    available. It still falls back to CPU through ONNX Runtime if CUDA is not
    active.
    """

    def __init__(self, *, use_cuda: bool = False, max_image_side: int = 1_280):
        self.use_cuda = use_cuda
        self.max_image_side = max_image_side
        self._engine = None

    def read(self, frame: KeyFrame) -> OcrResult:
        try:
            engine = self._get_engine()
            output = engine(self._load_image(frame.image_path))
        except ImportError:
            return OcrResult(frame=frame, text="", confidence=0.0, provider="rapidocr_missing")
        except Exception as exc:
            return OcrResult(frame=frame, text=f"[rapidocr_error:{type(exc).__name__}:{exc}]", confidence=0.0, provider="rapidocr_error")

        texts = [str(item).strip() for item in getattr(output, "txts", ()) if str(item).strip()]
        scores = []
        for score in getattr(output, "scores", ()):
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue
        confidence = sum(scores) / len(scores) if scores else 0.0
        return OcrResult(
            frame=frame,
            text=" ".join(texts),
            confidence=round(confidence, 3),
            provider=_rapidocr_provider_name(engine),
        )

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        cache_key = (self.use_cuda, threading.get_ident())
        cached = _RAPIDOCR_ENGINE_CACHE.get(cache_key)
        if cached is not None:
            self._engine = cached
            return self._engine

        params = {"Global.log_level": "warning"}
        if self.use_cuda:
            _preload_onnxruntime_cuda_dlls()
            params["EngineConfig.onnxruntime.use_cuda"] = True
        from rapidocr import RapidOCR

        self._engine = RapidOCR(params=params)
        _RAPIDOCR_ENGINE_CACHE[cache_key] = self._engine
        return self._engine

    def _load_image(self, image_path: str):
        if self.max_image_side <= 0:
            return image_path
        try:
            import cv2
        except ImportError:
            return image_path

        image = cv2.imread(image_path)
        if image is None:
            return image_path
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_image_side:
            return image

        scale = self.max_image_side / float(longest)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def build_ocr_provider(config: VisionConfig) -> OcrProvider:
    provider = config.ocr_provider.lower()
    if provider == "rapidocr":
        return RapidOcrProvider(use_cuda=config.rapidocr_use_cuda, max_image_side=config.ocr_max_image_side)
    if provider == "tesseract":
        return TesseractOcrProvider()
    return NoopOcrProvider()


def run_ocr(frames: list[KeyFrame], config: VisionConfig) -> list[OcrResult]:
    if not frames:
        return []
    workers = max(1, int(config.ocr_workers or 1))
    if workers == 1:
        provider = build_ocr_provider(config)
        return [provider.read(frame) for frame in frames]

    thread_local = threading.local()

    def read(frame: KeyFrame) -> OcrResult:
        provider = getattr(thread_local, "provider", None)
        if provider is None:
            provider = build_ocr_provider(config)
            thread_local.provider = provider
        return provider.read(frame)

    with ThreadPoolExecutor(max_workers=min(workers, len(frames)), thread_name_prefix="dld_ocr") as executor:
        return list(executor.map(read, frames))


def _preload_onnxruntime_cuda_dlls() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    try:
        import onnxruntime as ort
    except ImportError:
        return

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
    except TypeError:
        preload()


def _rapidocr_provider_name(engine: object) -> str:
    for part_name in ("text_det", "text_cls", "text_rec"):
        part = getattr(engine, part_name, None)
        if part is None:
            continue
        for value in getattr(part, "__dict__", {}).values():
            session = getattr(value, "session", value)
            get_providers = getattr(session, "get_providers", None)
            if get_providers is None:
                continue
            try:
                providers = list(get_providers())
            except Exception:
                continue
            if providers and providers[0] == "CUDAExecutionProvider":
                return "rapidocr_cuda"
    return "rapidocr_cpu"
