"""OCR provider abstraction and prefiltering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import VisionConfig
from .frames import KeyFrame


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


def build_ocr_provider(config: VisionConfig) -> OcrProvider:
    if config.ocr_provider.lower() == "tesseract":
        return TesseractOcrProvider()
    return NoopOcrProvider()


def run_ocr(frames: list[KeyFrame], config: VisionConfig) -> list[OcrResult]:
    provider = build_ocr_provider(config)
    return [provider.read(frame) for frame in frames]
