"""OCR provider abstraction for reading every selected keyframe."""

from __future__ import annotations

import os
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import VisionConfig
from .frames import KeyFrame

_RAPIDOCR_ENGINE_CACHE: dict[tuple[bool, int], object] = {}
_PADDLEOCR_ENGINE_CACHE: dict[tuple[bool, int], object] = {}
_DLL_DIRECTORY_HANDLES: list[object] = []
_NVIDIA_DLL_DIRS_READY = False


@dataclass(frozen=True)
class OcrResult:
    frame: KeyFrame
    text: str
    confidence: float
    provider: str


class OcrProvider(Protocol):
    def read(self, frame: KeyFrame) -> OcrResult:
        ...

    def read_batch(self, frames: list[KeyFrame]) -> list[OcrResult]:
        ...


class NoopOcrProvider:
    def read(self, frame: KeyFrame) -> OcrResult:
        return OcrResult(frame=frame, text="", confidence=0.0, provider="none")

    def read_batch(self, frames: list[KeyFrame]) -> list[OcrResult]:
        return [self.read(frame) for frame in frames]


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

    def read_batch(self, frames: list[KeyFrame]) -> list[OcrResult]:
        return [self.read(frame) for frame in frames]


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

    def read_batch(self, frames: list[KeyFrame]) -> list[OcrResult]:
        return [self.read(frame) for frame in frames]

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
            _add_nvidia_cuda_dll_directories()
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


class PaddleOcrProvider:
    """GPU-oriented OCR provider backed by PaddleOCR 3.x."""

    def __init__(self, *, use_cuda: bool = False, max_image_side: int = 1_280):
        self.use_cuda = use_cuda
        self.max_image_side = max_image_side
        self._engine = None

    def read(self, frame: KeyFrame) -> OcrResult:
        return self.read_batch([frame])[0]

    def read_batch(self, frames: list[KeyFrame]) -> list[OcrResult]:
        if not frames:
            return []
        try:
            engine = self._get_engine()
            inputs = [self._load_image(frame.image_path) for frame in frames]
            outputs = _run_paddle_engine(engine, inputs)
        except ImportError:
            return [OcrResult(frame=frame, text="", confidence=0.0, provider="paddleocr_missing") for frame in frames]
        except Exception as exc:
            return [
                OcrResult(frame=frame, text=f"[paddleocr_error:{type(exc).__name__}:{exc}]", confidence=0.0, provider="paddleocr_error")
                for frame in frames
            ]

        results: list[OcrResult] = []
        for frame, output in zip(frames, outputs, strict=False):
            text, confidence = _parse_paddle_output(output)
            results.append(
                OcrResult(
                    frame=frame,
                    text=text,
                    confidence=confidence,
                    provider="paddleocr_gpu" if self.use_cuda else "paddleocr_cpu",
                )
            )
        for frame in frames[len(results) :]:
            results.append(OcrResult(frame=frame, text="", confidence=0.0, provider="paddleocr_empty"))
        return results

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        cache_key = (self.use_cuda, threading.get_ident())
        cached = _PADDLEOCR_ENGINE_CACHE.get(cache_key)
        if cached is not None:
            self._engine = cached
            return self._engine
        if self.use_cuda:
            _add_nvidia_cuda_dll_directories()
        from paddleocr import PaddleOCR

        device = "gpu:0" if self.use_cuda else "cpu"
        try:
            self._engine = PaddleOCR(
                lang="ch",
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            self._engine = PaddleOCR(lang="ch", use_gpu=self.use_cuda, show_log=False)
        _PADDLEOCR_ENGINE_CACHE[cache_key] = self._engine
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
            return image_path

        scale = self.max_image_side / float(longest)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def build_ocr_provider(config: VisionConfig) -> OcrProvider:
    provider = config.ocr_provider.lower()
    use_cuda = getattr(config, "ocr_use_cuda", config.rapidocr_use_cuda)
    if provider == "paddleocr":
        return PaddleOcrProvider(use_cuda=use_cuda, max_image_side=config.ocr_max_image_side)
    if provider == "rapidocr":
        return RapidOcrProvider(use_cuda=use_cuda, max_image_side=config.ocr_max_image_side)
    if provider == "tesseract":
        return TesseractOcrProvider()
    return NoopOcrProvider()


def run_ocr(frames: list[KeyFrame], config: VisionConfig) -> list[OcrResult]:
    if not frames:
        return []
    workers = max(1, int(config.ocr_workers or 1))
    if workers == 1:
        provider = build_ocr_provider(config)
        return _read_provider_batches(provider, frames, config.ocr_batch_size)

    thread_local = threading.local()

    def read(frame: KeyFrame) -> OcrResult:
        provider = getattr(thread_local, "provider", None)
        if provider is None:
            provider = build_ocr_provider(config)
            thread_local.provider = provider
        return provider.read(frame)

    with ThreadPoolExecutor(max_workers=min(workers, len(frames)), thread_name_prefix="dld_ocr") as executor:
        return list(executor.map(read, frames))


def _read_provider_batches(provider: OcrProvider, frames: list[KeyFrame], batch_size: int) -> list[OcrResult]:
    results: list[OcrResult] = []
    for start in range(0, len(frames), max(1, batch_size)):
        results.extend(provider.read_batch(frames[start : start + batch_size]))
    return results


def _run_paddle_engine(engine: object, inputs: list[object]) -> list[object]:
    predict = getattr(engine, "predict", None)
    if callable(predict):
        return list(predict(inputs))

    ocr = getattr(engine, "ocr", None)
    if callable(ocr):
        return [ocr(image, cls=False) for image in inputs]

    raise RuntimeError("PaddleOCR engine has neither predict nor ocr")


def _parse_paddle_output(output: object) -> tuple[str, float]:
    payload = _paddle_payload(output)
    texts = _first_list(payload, ("rec_texts", "texts", "text"))
    scores = _first_list(payload, ("rec_scores", "scores", "confidence"))
    if texts:
        clean_texts = [str(item).strip() for item in texts if str(item).strip()]
        clean_scores = [_score(item) for item in scores]
        valid_scores = [item for item in clean_scores if item >= 0]
        confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        return " ".join(clean_texts), round(confidence, 3)

    lines = list(_iter_legacy_paddle_lines(output))
    clean_texts: list[str] = []
    clean_scores: list[float] = []
    for text, score in lines:
        if text:
            clean_texts.append(text)
        if score >= 0:
            clean_scores.append(score)
    confidence = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
    return " ".join(clean_texts), round(confidence, 3)


def _paddle_payload(output: object) -> dict:
    if isinstance(output, dict):
        payload = output
    else:
        payload = getattr(output, "res", None)
        if payload is None:
            payload = getattr(output, "__dict__", {})
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]
    return payload if isinstance(payload, dict) else {}


def _first_list(payload: dict, keys: tuple[str, ...]) -> list:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list | tuple):
            return list(value)
        if isinstance(value, str):
            return [value]
    return []


def _iter_legacy_paddle_lines(value: object):
    if not isinstance(value, list | tuple):
        return
    if len(value) >= 2 and isinstance(value[1], tuple | list) and len(value[1]) >= 2:
        text = str(value[1][0]).strip()
        yield text, _score(value[1][1])
        return
    for item in value:
        yield from _iter_legacy_paddle_lines(item)


def _score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _add_nvidia_cuda_dll_directories() -> None:
    global _NVIDIA_DLL_DIRS_READY
    if _NVIDIA_DLL_DIRS_READY or os.name != "nt":
        return

    site_packages = Path(sysconfig.get_paths().get("purelib", ""))
    dll_dirs = [
        site_packages / "nvidia" / package / "bin"
        for package in (
            "cuda_runtime",
            "cublas",
            "cudnn",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
        )
    ]

    existing_path = os.environ.get("PATH", "")
    existing_parts = [part for part in existing_path.split(os.pathsep) if part]
    prepended: list[str] = []
    add_dll_directory = getattr(os, "add_dll_directory", None)
    for dll_dir in dll_dirs:
        if not dll_dir.is_dir():
            continue
        path = str(dll_dir)
        if add_dll_directory is not None:
            _DLL_DIRECTORY_HANDLES.append(add_dll_directory(path))
        if path not in existing_parts and path not in prepended:
            prepended.append(path)

    if prepended:
        os.environ["PATH"] = os.pathsep.join([*prepended, existing_path])
    _NVIDIA_DLL_DIRS_READY = True


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
