"""OpenAI-compatible VLM client used for Qwen and similar providers."""

from __future__ import annotations

import base64
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..io import basename, normalize_path
from ..policy import SENSITIVE_TOKENS, SINK_TOKENS, TRANSFER_TOKENS, UNKNOWN_RISK_TOKENS, contains_any
from .config import VisionConfig
from .frames import KeyFrame
from .ocr import OcrResult


@dataclass(frozen=True)
class VlmRequestFrame:
    frame: KeyFrame
    ocr_text: str
    ocr_confidence: float
    selection_reason: str = ""
    selection_score: int = 0
    source_frames: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class VlmResponse:
    text: str
    provider: str
    model: str
    raw_payload: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    dry_run: bool = False


class OpenAICompatibleVlmClient:
    def __init__(self, config: VisionConfig):
        self.config = config

    def request_summary(
        self,
        frames: list[VlmRequestFrame],
        *,
        sensitive_files: list[str],
        active_apps: list[str],
    ) -> dict[str, Any]:
        """Build a replayable request summary without embedding large base64 images."""

        prompt = _prompt(frames, sensitive_files, active_apps)
        return {
            "provider": self.config.vlm_provider,
            "model": self.config.vlm_model,
            "chat_url": _chat_url(self.config),
            "dry_run": self.config.vlm_dry_run,
            "frame_strategy": self.config.vlm_frame_strategy,
            "grid_size": self.config.vlm_grid_size,
            "temperature": 0,
            "prompt": prompt,
            "sensitive_context": _sensitive_context(sensitive_files),
            "active_apps": active_apps,
            "frames": [_request_frame_to_dict(item) for item in frames],
            "request_metrics": record_vlm_request_metrics(frames, prompt),
        }

    def analyze(
        self,
        frames: list[VlmRequestFrame],
        *,
        sensitive_files: list[str],
        active_apps: list[str],
    ) -> VlmResponse:
        if not frames:
            return VlmResponse(text='{"events":[]}', provider=self.config.vlm_provider, model=self.config.vlm_model)
        if self.config.vlm_dry_run:
            return VlmResponse(text='{"events":[]}', provider=self.config.vlm_provider, model=self.config.vlm_model, dry_run=True)
        if not self.config.vlm_api_key:
            raise RuntimeError("DLD_VLM_API_KEY is not set")

        body = self._request_body(frames, sensitive_files=sensitive_files, active_apps=active_apps)
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
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        return VlmResponse(text=str(text), provider=self.config.vlm_provider, model=self.config.vlm_model, raw_payload=payload, usage=usage)

    def _request_body(
        self,
        frames: list[VlmRequestFrame],
        *,
        sensitive_files: list[str],
        active_apps: list[str],
    ) -> dict[str, Any]:
        content: list[dict[str, object]] = [{"type": "text", "text": _prompt(frames, sensitive_files, active_apps)}]
        for item in frames:
            image_url = f"data:image/jpeg;base64,{_image_b64(item.frame.image_path)}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        return {
            "model": self.config.vlm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }


def choose_vlm_frames(
    ocr_results: list[OcrResult],
    *,
    min_confidence: float,
    max_frames: int,
    strategy: str = "ocr_triage",
    include_empty_ocr_strong_frames: bool = False,
    max_frames_per_window: int | None = None,
) -> list[VlmRequestFrame]:
    if max_frames == 0:
        return []
    unlimited = max_frames < 0
    if max_frames_per_window is not None and max_frames_per_window <= 0:
        max_frames_per_window = None

    normalized_strategy = _normalize_frame_strategy(strategy)
    candidates: list[tuple[int, int, VlmRequestFrame]] = []
    for result in ocr_results:
        text = result.text.strip()
        ocr_available = result.provider not in {"none", "tesseract_missing"}
        low_confidence = ocr_available and result.confidence < min_confidence
        suspicious_text = contains_any(text, SINK_TOKENS + TRANSFER_TOKENS + SENSITIVE_TOKENS + UNKNOWN_RISK_TOKENS)
        strong_or_anchor = _is_strong_or_anchor(result.frame)
        empty_strong = include_empty_ocr_strong_frames and not text and strong_or_anchor
        force_vlm = normalized_strategy == "ocr_all"
        if force_vlm or low_confidence or suspicious_text or empty_strong:
            score = _vlm_frame_score(
                result,
                suspicious_text=suspicious_text,
                low_confidence=low_confidence,
                empty_strong=empty_strong,
            )
            candidates.append(
                (
                    score,
                    result.frame.timestamp_ms,
                    VlmRequestFrame(
                        result.frame,
                        text,
                        result.confidence,
                        selection_reason=_selection_reason(
                            force_vlm=force_vlm,
                            suspicious_text=suspicious_text,
                            low_confidence=low_confidence,
                            empty_strong=empty_strong,
                        ),
                        selection_score=score,
                    ),
                )
            )
    selected: list[VlmRequestFrame] = []
    per_window: dict[str, int] = {}
    for _, _, frame in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if max_frames_per_window is not None:
            window_id = frame.frame.window_id or "window_unknown"
            if per_window.get(window_id, 0) >= max_frames_per_window:
                continue
            per_window[window_id] = per_window.get(window_id, 0) + 1
        selected.append(frame)
        if not unlimited and len(selected) >= max_frames:
            break
    return sorted(selected, key=lambda item: item.frame.timestamp_ms)


def choose_keyframes_for_vlm(
    frames: list[KeyFrame],
    *,
    max_frames: int,
    max_frames_per_window: int | None = None,
) -> list[VlmRequestFrame]:
    if max_frames == 0:
        return []
    unlimited = max_frames < 0
    if max_frames_per_window is not None and max_frames_per_window <= 0:
        max_frames_per_window = None

    candidates = [
        (
            _keyframe_score(frame),
            frame.timestamp_ms,
            VlmRequestFrame(
                frame=frame,
                ocr_text="",
                ocr_confidence=0.0,
                selection_reason="direct_keyframe",
                selection_score=_keyframe_score(frame),
            ),
        )
        for frame in frames
    ]
    if max_frames_per_window is not None:
        candidates_by_window: dict[str, list[tuple[int, int, VlmRequestFrame]]] = {}
        for candidate in candidates:
            window_id = candidate[2].frame.window_id or "window_unknown"
            candidates_by_window.setdefault(window_id, []).append(candidate)
        limited_candidates: list[tuple[int, int, VlmRequestFrame]] = []
        for window_candidates in candidates_by_window.values():
            limited_candidates.extend(_select_temporally_diverse_candidates(window_candidates, max_frames_per_window))
        candidates = limited_candidates
    elif not unlimited and len(candidates) > max_frames:
        candidates = _select_temporally_diverse_candidates(candidates, max_frames)

    selected: list[VlmRequestFrame] = []
    for _, _, frame in sorted(candidates, key=lambda item: (-item[0], item[1])):
        selected.append(frame)
        if not unlimited and len(selected) >= max_frames:
            break
    return sorted(selected, key=lambda item: item.frame.timestamp_ms)


def _select_temporally_diverse_candidates(
    candidates: list[tuple[int, int, VlmRequestFrame]],
    limit: int,
) -> list[tuple[int, int, VlmRequestFrame]]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    max_score = max(score for score, _, _ in candidates)
    preferred = [candidate for candidate in candidates if candidate[0] >= max_score - 10]
    selected = _pick_evenly_by_time(preferred, min(limit, len(preferred)))
    if len(selected) < limit:
        selected_keys = {(candidate[2].frame.frame_id, candidate[1]) for candidate in selected}
        remainder = [
            candidate
            for candidate in sorted(candidates, key=lambda item: (-item[0], item[1]))
            if (candidate[2].frame.frame_id, candidate[1]) not in selected_keys
        ]
        selected.extend(remainder[: limit - len(selected)])
    return selected


def _pick_evenly_by_time(
    candidates: list[tuple[int, int, VlmRequestFrame]],
    limit: int,
) -> list[tuple[int, int, VlmRequestFrame]]:
    if not candidates or limit <= 0:
        return []
    if limit == 1:
        return [max(candidates, key=lambda item: (item[0], -item[1]))]

    ordered = sorted(candidates, key=lambda item: item[1])
    start = ordered[0][1]
    end = ordered[-1][1]
    if start == end:
        return sorted(ordered, key=lambda item: (-item[0], item[1]))[:limit]

    selected: list[tuple[int, int, VlmRequestFrame]] = []
    selected_keys: set[tuple[str, int]] = set()
    for index in range(limit):
        target = start + round((end - start) * index / (limit - 1))
        available = [
            candidate
            for candidate in ordered
            if (candidate[2].frame.frame_id, candidate[1]) not in selected_keys
        ]
        if not available:
            break
        chosen = min(available, key=lambda item: (abs(item[1] - target), -item[0], item[1]))
        selected.append(chosen)
        selected_keys.add((chosen[2].frame.frame_id, chosen[1]))
    return selected


def build_vlm_frame_grids(
    frames: list[VlmRequestFrame],
    *,
    grid_size: int,
    output_dir: str | Path | None,
) -> list[VlmRequestFrame]:
    if grid_size <= 1 or len(frames) <= 1:
        return frames

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("pillow_not_installed: install data-leak-detector[vision] to enable VLM grid images") from exc

    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="dld_vlm_grid_"))
    root.mkdir(parents=True, exist_ok=True)
    cells_per_grid = grid_size * grid_size
    grid_frames: list[VlmRequestFrame] = []
    for grid_index, group in enumerate(_chunks(frames, cells_per_grid)):
        source_images = [(item, Image.open(item.frame.image_path).convert("RGB")) for item in group]
        columns = min(grid_size, len(source_images))
        rows = (len(source_images) + columns - 1) // columns
        cell_width, cell_height = _grid_cell_size([image for _, image in source_images])
        label_height = 32
        canvas = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        source_payload: list[dict[str, Any]] = []

        for index, (item, image) in enumerate(source_images):
            row = index // columns
            column = index % columns
            cell_id = f"{chr(ord('A') + row)}{column + 1}"
            x = column * cell_width
            y = row * (cell_height + label_height)
            draw.rectangle((x, y, x + cell_width, y + label_height), fill=(20, 20, 20))
            draw.text((x + 8, y + 8), f"{cell_id} {item.frame.frame_id}", fill=(255, 255, 255))
            fitted = ImageOps.contain(image, (cell_width, cell_height))
            canvas.paste(fitted, (x + (cell_width - fitted.width) // 2, y + label_height + (cell_height - fitted.height) // 2))
            source_payload.append({**_request_frame_to_dict(item), "cell_id": cell_id})

        target = root / f"vlm_grid_{grid_index:03d}.jpg"
        canvas.save(target, quality=90)
        timestamps = [item.frame.timestamp_ms for item in group]
        score = max((item.selection_score for item in group), default=0)
        grid_keyframe = KeyFrame(
            frame_id=f"vlm_grid_{grid_index}",
            timestamp_ms=min(timestamps) if timestamps else 0,
            image_path=str(target),
            score=float(score),
            reason=f"vlm_grid:{grid_size}x{grid_size}",
            window_id="vlm_grid",
        )
        grid_frames.append(
            VlmRequestFrame(
                frame=grid_keyframe,
                ocr_text="",
                ocr_confidence=0.0,
                selection_reason=f"vlm_grid_{grid_size}x{grid_size}",
                selection_score=score,
                source_frames=tuple(source_payload),
            )
        )
    return grid_frames


def prepare_vlm_frame_images(
    frames: list[VlmRequestFrame],
    *,
    max_image_side: int,
    output_dir: str | Path | None,
) -> list[VlmRequestFrame]:
    if max_image_side <= 0 or not frames:
        return frames

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("pillow_not_installed: install data-leak-detector[vision] to resize VLM images") from exc

    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="dld_vlm_input_"))
    root.mkdir(parents=True, exist_ok=True)
    prepared: list[VlmRequestFrame] = []
    for index, item in enumerate(frames):
        source = Path(item.frame.image_path)
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            fitted = ImageOps.contain(rgb, (max_image_side, max_image_side))
            if fitted.size == rgb.size:
                prepared.append(item)
                continue
            target = root / f"{index:03d}_{item.frame.timestamp_ms}ms_{source.stem}.jpg"
            fitted.save(target, quality=88, optimize=True)
        prepared.append(replace(item, frame=replace(item.frame, image_path=str(target))))
    return prepared


def record_vlm_request_metrics(frames: list[VlmRequestFrame], prompt: str) -> dict[str, Any]:
    image_sizes = [_image_size(item.frame.image_path) for item in frames]
    image_pixels = sum(width * height for width, height in image_sizes)
    prompt_chars = len(prompt)
    return {
        "prompt_chars": prompt_chars,
        "image_count": len(frames),
        "image_pixels": image_pixels,
        "image_megapixels": round(image_pixels / 1_000_000, 3),
        "image_sizes": [{"width": width, "height": height} for width, height in image_sizes],
    }


def _vlm_frame_score(result: OcrResult, *, suspicious_text: bool, low_confidence: bool, empty_strong: bool = False) -> int:
    reason = result.frame.reason.lower()
    text = result.text
    score = 0
    if reason.startswith("strong"):
        score += 100
    elif reason.startswith("weak"):
        score += 40
    if contains_any(text, SINK_TOKENS):
        score += 80
    if contains_any(text, TRANSFER_TOKENS):
        score += 60
    if contains_any(text, SENSITIVE_TOKENS):
        score += 40
    if suspicious_text:
        score += 20
    if low_confidence:
        score += 10
    if "anchor" in reason:
        score += 35
    if empty_strong:
        score += 25
    return score


def _keyframe_score(frame: KeyFrame) -> int:
    reason = frame.reason.lower()
    score = 0
    if reason.startswith("strong"):
        score += 100
    elif reason.startswith("medium"):
        score += 60
    elif reason.startswith("weak"):
        score += 40
    if "anchor" in reason:
        score += 35
    if "window_start" in reason:
        score += 5
    score += min(int(frame.score * 10), 10)
    return score


def _prompt(frames: list[VlmRequestFrame], sensitive_files: list[str], active_apps: list[str]) -> str:
    frame_lines = [
        f"- frame_id={item.frame.frame_id}, timestamp_ms={item.frame.timestamp_ms}, "
        f"window_id={item.frame.window_id}, reason={item.frame.reason}, "
        f"selection_reason={item.selection_reason}, ocr_confidence={item.ocr_confidence}, ocr_text={item.ocr_text[:500]}"
        for item in frames
    ]
    grid_lines = [
        _grid_source_line(item, source)
        for item in frames
        for source in item.source_frames
    ]
    sensitive_lines = [
        f"- path={item['path']}, basename={item['basename']}, stem={item['stem']}"
        for item in _sensitive_context(sensitive_files)
    ]
    return (
        "You are analyzing screen-recording keyframes for enterprise data-leak evidence. "
        "Use the images, OCR text, frame metadata, and active app hints only. "
        "Do not infer labels from any groundtruth annotation; groundtruth is evaluation-only.\n"
        "Return strict JSON only with this schema: "
        "{\"events\":[{\"evidence_frame_ids\":[\"frame_0_0\"],"
        "\"timestamp_ms\":0,\"time_range\":\"YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS\","
        "\"app_name\":\"...\",\"behavior_category\":\"normal|direct_leak|hidden_transfer|unknown_risk\","
        "\"operation_type\":\"...\",\"original_filename\":\"...\",\"modified_filename\":\"...\","
        "\"sink_type\":\"ai_chat|mail_attachment|cloud_sync|chat_upload|screen_share|removable_media|network_upload|unknown\","
        "\"description\":\"...\",\"confidence\":0.0}]}\n"
        "Every non-empty event must include evidence_frame_ids from the provided frame_id values. "
        "When an image is a grid, cite original source_frame_id values from the grid mapping whenever possible. "
        "Prefer timestamp_ms from the frame metadata when an absolute time range is not visually available. "
        "Classify direct_leak for upload, email send, chat paste, AI prompt paste, screen share, cloud sync, "
        "or moving/copying a sensitive file to USB/removable media/external drive. "
        "Classify hidden_transfer for screenshot, screen recording, copy, export, rename, split, compression, or derived files. "
        "Use unknown_risk when the foreground app or behavior cannot be identified but appears near sensitive activity. "
        "Drop purely normal reading/opening events unless they explain a later risky operation.\n"
        "Frontend application and sink cues:\n"
        "- mail_attachment: compose/send screens, attachment chips, paperclip icons, recipient fields, Outlook/Gmail/QQ mail/163 mail.\n"
        "- cloud_sync: upload dialogs/progress, drag-drop upload areas, sync status, Baidu Netdisk/Quark/OneDrive/Google Drive/Dropbox.\n"
        "- chat_upload: IM file-send panels or attachment cards in WeChat/QQ/Feishu/DingTalk/Lark/Teams chat.\n"
        "- ai_chat: ChatGPT/Kimi/DeepSeek/Claude/Gemini/Qwen pages, prompt box, attached file chips, generated answer containing sensitive content.\n"
        "- screen_share: meeting toolbar, share-screen banner, Zoom/Tencent Meeting/Teams sharing controls.\n"
        "- removable_media: Windows Explorer copy/move progress, target drive named USB/removable/U disk/flash drive/external drive, "
        "or destination on a removable drive letter. Treat this as direct_leak, not merely hidden_transfer.\n"
        "When a frame shows a sensitive or derived sensitive file being attached, uploaded, sent, synced, shared, or copied to removable media, "
        "emit behavior_category=direct_leak and the most specific sink_type.\n"
        "Sensitive source files and aliases:\n" + "\n".join(sensitive_lines) + "\n"
        f"Non-whitelisted active apps from logs: {active_apps}\n"
        "Frame/OCR context:\n" + "\n".join(frame_lines) + "\n"
        "Grid cell mapping:\n" + ("\n".join(grid_lines) if grid_lines else "- none")
    )


def _request_frame_to_dict(item: VlmRequestFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_id": item.frame.frame_id,
        "timestamp_ms": item.frame.timestamp_ms,
        "image_path": item.frame.image_path,
        "reason": item.frame.reason,
        "window_id": item.frame.window_id,
        "ocr_text": item.ocr_text,
        "ocr_confidence": item.ocr_confidence,
        "selection_reason": item.selection_reason,
        "selection_score": item.selection_score,
    }
    if item.source_frames:
        payload["source_frames"] = list(item.source_frames)
    return payload


def _sensitive_context(sensitive_files: list[str]) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []
    for item in sensitive_files:
        path = normalize_path(item)
        if not path:
            continue
        name = basename(path)
        context.append({"path": path, "basename": name, "stem": Path(name).stem})
    return context


def _selection_reason(*, force_vlm: bool, suspicious_text: bool, low_confidence: bool, empty_strong: bool) -> str:
    reasons = []
    if force_vlm:
        reasons.append("ocr_all_to_vlm")
    if suspicious_text:
        reasons.append("suspicious_ocr")
    if low_confidence:
        reasons.append("low_confidence_ocr")
    if empty_strong:
        reasons.append("empty_ocr_strong_anchor")
    return "+".join(reasons) or "candidate"


def _is_strong_or_anchor(frame: KeyFrame) -> bool:
    reason = frame.reason.lower()
    return reason.startswith("strong") or "anchor" in reason


def _normalize_frame_strategy(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "ocr_prefilter": "ocr_triage",
        "ocr_filtered": "ocr_triage",
        "triage": "ocr_triage",
        "all_ocr": "ocr_all",
        "ocr_all_to_vlm": "ocr_all",
        "direct": "direct_keyframes",
        "all_keyframes": "direct_keyframes",
        "keyframes": "direct_keyframes",
    }
    return aliases.get(normalized, normalized or "ocr_triage")


def _chunks(items: list[VlmRequestFrame], size: int) -> list[list[VlmRequestFrame]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _grid_cell_size(images: list[object]) -> tuple[int, int]:
    widths = [max(1, int(getattr(image, "width", 1))) for image in images]
    heights = [max(1, int(getattr(image, "height", 1))) for image in images]
    cell_width = max(1, min(max(widths), 720))
    aspects = sorted(width / height for width, height in zip(widths, heights, strict=False) if height > 0)
    aspect = aspects[len(aspects) // 2] if aspects else 16 / 9
    cell_height = max(1, int(round(cell_width / max(aspect, 0.1))))
    return cell_width, min(cell_height, 720)


def _image_size(path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _grid_source_line(grid: VlmRequestFrame, source: dict[str, Any]) -> str:
    return (
        f"- grid_frame_id={grid.frame.frame_id}, cell={source.get('cell_id', '')}, "
        f"source_frame_id={source.get('frame_id', '')}, timestamp_ms={source.get('timestamp_ms', 0)}, "
        f"reason={source.get('reason', '')}, selection_reason={source.get('selection_reason', '')}, "
        f"ocr_confidence={source.get('ocr_confidence', 0.0)}, ocr_text={str(source.get('ocr_text', ''))[:500]}"
    )


def _image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _chat_url(config: VisionConfig) -> str:
    if config.vlm_chat_url:
        return config.vlm_chat_url
    return config.vlm_base_url.rstrip("/") + "/chat/completions"
