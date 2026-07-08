"""OpenAI-compatible VLM client used for Qwen and similar providers."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import VisionConfig
from .frames import KeyFrame
from .ocr import OcrResult


@dataclass(frozen=True)
class VlmRequestFrame:
    frame: KeyFrame
    ocr_text: str
    ocr_confidence: float


@dataclass(frozen=True)
class VlmResponse:
    text: str
    provider: str
    model: str


class OpenAICompatibleVlmClient:
    def __init__(self, config: VisionConfig):
        self.config = config

    def analyze(
        self,
        frames: list[VlmRequestFrame],
        *,
        sensitive_files: list[str],
        active_apps: list[str],
    ) -> VlmResponse:
        if not frames:
            return VlmResponse(text='{"events":[]}', provider=self.config.vlm_provider, model=self.config.vlm_model)
        if not self.config.vlm_api_key:
            raise RuntimeError("DLD_VLM_API_KEY is not set")

        content: list[dict[str, object]] = [{"type": "text", "text": _prompt(frames, sensitive_files, active_apps)}]
        for item in frames:
            image_url = f"data:image/jpeg;base64,{_image_b64(item.frame.image_path)}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        body = {
            "model": self.config.vlm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }
        request = urllib.request.Request(
            _chat_url(self.config),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.vlm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.vlm_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vlm_http_error: {exc.code} {detail}") from exc

        text = payload["choices"][0]["message"]["content"]
        return VlmResponse(text=str(text), provider=self.config.vlm_provider, model=self.config.vlm_model)


def choose_vlm_frames(
    ocr_results: list[OcrResult],
    *,
    min_confidence: float,
    max_frames: int,
) -> list[VlmRequestFrame]:
    candidates: list[VlmRequestFrame] = []
    for result in ocr_results:
        text = result.text.strip()
        low_confidence = result.confidence < min_confidence
        suspicious_text = any(term in text.lower() for term in ("upload", "send", "share", "password", "secret", "粘贴", "上传", "发送", "共享"))
        if low_confidence or suspicious_text:
            candidates.append(VlmRequestFrame(result.frame, text, result.confidence))
    return candidates[:max_frames]


def _prompt(frames: list[VlmRequestFrame], sensitive_files: list[str], active_apps: list[str]) -> str:
    frame_lines = [
        f"- frame_id={item.frame.frame_id}, timestamp_ms={item.frame.timestamp_ms}, "
        f"ocr_confidence={item.ocr_confidence}, ocr_text={item.ocr_text[:500]}"
        for item in frames
    ]
    return (
        "You are analyzing screen-recording keyframes for enterprise data-leak evidence.\n"
        "Return strict JSON only: {\"events\":[{\"time_range\":\"YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS\","
        "\"app_name\":\"...\",\"behavior_category\":\"normal|direct_leak|hidden_transfer|unknown_risk\","
        "\"operation_type\":\"...\",\"original_filename\":\"...\",\"modified_filename\":\"...\","
        "\"description\":\"...\",\"confidence\":0.0}]}\n"
        "Classify direct leaks such as upload, email send, chat paste, AI prompt paste, screen share, or cloud sync.\n"
        "Classify hidden transfers such as screenshot, screen recording, copy, export, rename, split, compression, or derived files.\n"
        "Use unknown_risk when the foreground app or behavior cannot be identified but appears near sensitive activity.\n"
        f"Sensitive files or keywords: {sensitive_files}\n"
        f"Non-whitelisted active apps from logs: {active_apps}\n"
        "OCR context:\n" + "\n".join(frame_lines)
    )


def _image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _chat_url(config: VisionConfig) -> str:
    if config.vlm_chat_url:
        return config.vlm_chat_url
    return config.vlm_base_url.rstrip("/") + "/chat/completions"
