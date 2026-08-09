"""VLM request frames, prompt construction, and OpenAI-compatible client."""

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
from .config import VisionConfig
from .frames import KeyFrame


@dataclass(frozen=True)
class VlmRequestFrame:
    frame: KeyFrame
    visual_note: str
    visual_confidence: float
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
            "frame_source": "direct_keyframes",
            "grid_size": self.config.vlm_grid_size,
            "grid_layout": self.config.vlm_grid_layout,
            "temperature": 0,
            "enable_thinking": self.config.vlm_enable_thinking,
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

        body: dict[str, Any] = {
            "model": self.config.vlm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
        }
        if self.config.vlm_enable_thinking is not None:
            body["enable_thinking"] = self.config.vlm_enable_thinking
        return body


def choose_keyframes_for_vlm(
    frames: list[KeyFrame],
    *,
    max_frames: int,
    max_frames_per_window: int | None = None,
) -> list[VlmRequestFrame]:
    if max_frames == 0:
        return []
    selected = [
        VlmRequestFrame(
            frame=frame,
            visual_note="",
            visual_confidence=0.0,
            selection_score=_keyframe_score(frame),
        )
        for frame in sorted(frames, key=lambda item: item.timestamp_ms)
    ]
    if max_frames_per_window is not None and max_frames_per_window > 0:
        by_window: dict[str, list[VlmRequestFrame]] = {}
        for item in selected:
            by_window.setdefault(item.frame.window_id or "window_unknown", []).append(item)
        selected = [
            item
            for group in by_window.values()
            for item in _evenly_spread_frames(group, max_frames_per_window)
        ]
        selected.sort(key=lambda item: item.frame.timestamp_ms)
    if max_frames > 0:
        selected = _evenly_spread_frames(selected, max_frames)
    return selected


def _evenly_spread_frames(frames: list[VlmRequestFrame], limit: int) -> list[VlmRequestFrame]:
    if limit <= 0 or len(frames) <= limit:
        return frames
    if limit == 1:
        return [frames[-1]]
    ordered = sorted(frames, key=lambda item: item.frame.timestamp_ms)
    start = ordered[0].frame.timestamp_ms
    end = ordered[-1].frame.timestamp_ms
    selected: list[VlmRequestFrame] = []
    for index in range(limit):
        target = start + round((end - start) * index / (limit - 1))
        available = [item for item in ordered if item not in selected]
        selected.append(min(available, key=lambda item: (abs(item.frame.timestamp_ms - target), -item.frame.timestamp_ms)))
    return sorted(selected, key=lambda item: item.frame.timestamp_ms)


def build_vlm_frame_grids(
    frames: list[VlmRequestFrame],
    *,
    grid_size: int,
    grid_layout: str = "",
    output_dir: str | Path | None,
) -> list[VlmRequestFrame]:
    if (grid_size <= 1 and not grid_layout) or len(frames) <= 1:
        return frames

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("pillow_not_installed: install data-leak-detector[vision] to enable VLM grid images") from exc

    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="dld_vlm_grid_"))
    root.mkdir(parents=True, exist_ok=True)
    rows_per_grid, columns_per_grid = _grid_dimensions(grid_size, grid_layout)
    cells_per_grid = rows_per_grid * columns_per_grid
    by_window: dict[str, list[VlmRequestFrame]] = {}
    for item in frames:
        by_window.setdefault(item.frame.window_id or "window_unknown", []).append(item)

    grid_frames: list[VlmRequestFrame] = []
    grid_groups = [
        (window_id, group)
        for window_id, window_frames in by_window.items()
        for group in _chunks(window_frames, cells_per_grid)
    ]
    for grid_index, (window_id, group) in enumerate(grid_groups):
        source_images = [(item, Image.open(item.frame.image_path).convert("RGB")) for item in group]
        columns = min(columns_per_grid, len(source_images))
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
            reason=f"vlm_grid:{rows_per_grid}x{columns_per_grid}",
            window_id=window_id,
        )
        grid_frames.append(
            VlmRequestFrame(
                frame=grid_keyframe,
                visual_note="",
                visual_confidence=0.0,
                selection_score=score,
                source_frames=tuple(source_payload),
            )
        )
    return grid_frames


def _grid_dimensions(grid_size: int, grid_layout: str) -> tuple[int, int]:
    layout = grid_layout.strip().lower().replace("*", "x").replace("\u00d7", "x")
    if not layout:
        size = max(1, grid_size)
        return size, size
    parts = layout.split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid_vlm_grid_layout: {grid_layout!r}; expected rowsxcolumns such as 2x1")
    try:
        rows, columns = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid_vlm_grid_layout: {grid_layout!r}; expected rowsxcolumns such as 2x1") from exc
    if rows < 1 or columns < 1:
        raise ValueError(f"invalid_vlm_grid_layout: {grid_layout!r}; dimensions must be positive")
    return rows, columns


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
    if "activity_gap" in reason:
        score += 50
    if "window_start" in reason:
        score += 5
    score += min(int(frame.score * 10), 10)
    return score


def _prompt(frames: list[VlmRequestFrame], sensitive_files: list[str], active_apps: list[str]) -> str:
    frame_lines = [
        f"- frame_id={item.frame.frame_id}, timestamp_ms={item.frame.timestamp_ms}, "
        f"window_id={item.frame.window_id}, reason={item.frame.reason}, "
        f"visual_confidence={item.visual_confidence}, visual_note={item.visual_note[:500]}"
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
        "Use the images, frame metadata, and active app hints only. "
        "Do not infer labels from any groundtruth annotation; groundtruth is evaluation-only.\n"
        "Return strict JSON only with this schema: "
        "{\"events\":[{\"evidence_frame_ids\":[\"frame_0_0\"],"
        "\"timestamp_ms\":0,\"time_range\":\"YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS\","
        "\"app_name\":\"...\",\"behavior_category\":\"normal|direct_leak|hidden_transfer|unknown_risk\","
        "\"operation_type\":\"...\",\"original_filename\":\"...\",\"modified_filename\":\"...\"," 
        "\"sink_type\":\"ai_chat|mail_attachment|cloud_sync|chat_upload|screen_share|removable_media|network_upload|unknown\"," 
        "\"action_status\":\"selected|submitted|in_progress|completed|failed|unknown\"," 
        "\"description\":\"...\",\"confidence\":0.0}]}\n"
        "Every non-empty event must include evidence_frame_ids from the provided frame_id values. "
        "When an image is a grid, cite original source_frame_id values from the grid mapping whenever possible. "
        "Prefer timestamp_ms from the frame metadata when an absolute time range is not visually available. "
        "Classify direct_leak for upload, email send, chat paste, AI prompt paste, screen share, cloud sync, "
        "or moving/copying a sensitive file to USB/removable media/external drive. "
        "Classify hidden_transfer for screenshot, screen recording, copy, export, rename, split, compression, or derived files. "
        "Use unknown_risk when the foreground app or behavior cannot be identified but appears near sensitive activity. "
        "Drop purely normal reading/opening events unless they explain a later risky operation.\n"
        "Frames with the same window_id are one chronological evidence packet. Correlate them before emitting events: "
        "a filename may be readable in an earlier file picker or attachment frame while the submit, progress, success, or failure state "
        "appears in a later frame. For the same resource and the same action, emit one combined event and cite every source frame needed for that conclusion. "
        "Emit separate events for different resources or independent actions, even if they share a timestamp. Use the strongest terminal-state frame "
        "for timestamp_ms when completion, failure, or progress appears later than the identity frame. "
        "Do not require the filename and the result state to be visible in the same image. "
        "If an exact sensitive filename is visible but no later action is proven, emit a preparation/identity event with "
        "behavior_category=unknown_risk and action_status=unknown so downstream evidence can bind it to a later proven action.\n"
        "Use original_filename and modified_filename only for exact filenames visible in the images or listed in the supplied sensitive-file context. "
        "Never translate, paraphrase, semantically substitute, or invent a filename. Use unknown when the exact filename is unreadable.\n"
        "Frontend application and sink cues:\n"
        "- mail_attachment: compose/send screens, attachment chips, paperclip icons, recipient fields, Outlook/Gmail/QQ mail/163 mail.\n"
        "- cloud_sync: upload dialogs/progress, drag-drop upload areas, sync status, Baidu Netdisk/Quark/OneDrive/Google Drive/Dropbox.\n"
        "A file merely being located in or browsed under a OneDrive/cloud-synced folder is not enough to prove cloud_sync for that file. "
        "A static cloud/check icon only proves historical sync, not a new transfer during this recording. Require current upload/sync progress, "
        "a newly completed result, or a chronological create/move into the sync folder.\n"
        "A real drag operation is proven when the same frame shows a selected local file being carried under the pointer, a drag-copy/move badge, "
        "and a destination-app drop target or tooltip such as 'drag to upload to the current folder' / '拖拽上传至当前文件夹'. "
        "This is not a static empty-state illustration: emit direct_leak with sink_type=cloud_sync and action_status=selected or submitted, "
        "and bind the filename from the visible source Explorer selection or an earlier frame in the same window.\n"
        "- chat_upload: IM file-send panels or attachment cards in WeChat/QQ/Feishu/DingTalk/Lark/Teams chat, "
        "plus Tencent Meeting/Zoom/Teams meeting chat file sends and meeting-document imports.\n"
        "- ai_chat: ChatGPT/Kimi/DeepSeek/Claude/Gemini/Qwen pages, prompt box, attached file chips, generated answer containing sensitive content.\n"
        "- screen_share: meeting toolbar, share-screen banner, Zoom/Tencent Meeting/Teams sharing controls, or sensitive content "
        "visibly exposed during an active share. Importing a file into meeting documents or chat is chat_upload, not screen_share.\n"
        "- removable_media: Windows Explorer copy/move progress, target drive named USB/removable/U disk/flash drive/external drive, "
        "or destination on a removable drive letter. Treat this as direct_leak, not merely hidden_transfer.\n"
        "When a frame shows a sensitive or derived sensitive file being attached, uploaded, sent, synced, shared, or copied to removable media, "
        "emit behavior_category=direct_leak and the most specific sink_type.\n"
        "Treat an email send confirmation as an external-send action: when a compose screen shows a sensitive attachment and an explicit Send button "
        "or a confirmation dialog with Send/Cancel choices, emit direct_leak with sink_type=mail_attachment. "
        "Do not require an inbox update, a sent-mail receipt, or a server-side success message.\n"
        "Do not treat a transfer capability that is merely visible as an executed leak: an unselected context-menu or toolbar action such as "
        "'Send to my phone', a generic share sheet, or a 'drag file to send' panel is not enough by itself. "
        "Require evidence that the transfer was selected or submitted, such as an addressee/conversation/destination, a file in a transfer queue, "
        "upload/send progress, or an explicit Send/Confirm control. A generic 'Send File' panel that only offers copy file, drag-to-send, or open-location "
        "actions without such evidence is a copy/preparation action: classify it as hidden_transfer rather than direct_leak.\n"
        "For a two-step website upload form, selecting a local file into a staging area is preparation when a separate Upload/Submit button remains "
        "unclicked and there is no transfer progress. This differs from an attachment already inserted into a chat or mail composer.\n"
        "Do not infer an upload from a local File Explorer/WPS preview pane or from AI/OCR toolbar capabilities alone. Require an explicit invocation, "
        "submission, progress, or external service result. Pasting sensitive text into a third-party online encoder/converter is network_upload even "
        "when the transformation itself is Base64 encoding, translation, or conversion.\n"
        "Text shown in a monitoring dashboard, terminal, PowerShell window, or monitoring log is not proof that an external action occurred. "
        "Do not emit direct_leak from a log line naming an AI/chat application unless the destination UI, attachment, submission, progress, or result "
        "is independently visible.\n"
        "A chat application icon, notification, background window, or ambiguous dark thumbnail does not prove that a local screenshot or screen "
        "recording was attached. Require a visible chat composer attachment card or a clearly identified sent item; otherwise keep the recording "
        "as hidden_transfer only.\n"
        "A screenshot preview inserted into a chat composer, or an exact sensitive-file card staged in a chat/file-transfer assistant, is selected/attached "
        "outbound evidence even before the final Send click. The preview/card must be visibly present in the composer itself; a sensitive document shown "
        "in another window, tab, or earlier frame does not establish that the composer contains it. If the composer is empty, do not emit chat_upload "
        "merely because a QQ/WeChat window, Send button, or prior recording/screenshot is visible. If a leave-page warning says entered information may be lost, inspect the background form: "
        "a visible sensitive-file card there remains staged attachment evidence and must retain the filename identity from earlier frames.\n"
        "For nested virtual-machine or remote-desktop scenes, cross-check the filename in the inner file card against any readable host desktop icon, "
        "file picker, or earlier frame. Do not replace it with a merely similar name from sensitive-file context; use unknown if the exact text is unreadable.\n"
        "A local screen recorder, MP4 creation/playback, or QQ recording UI is hidden_transfer, not screen_share, unless an independent meeting/share banner, "
        "sharing toolbar state, remote participant, or explicit active-share indicator is also visible.\n"
        "For QQ recording windows such as '录屏生成视频' or '制作MP4', do not infer a chat attachment from the QQ app name, a dark preview, or a visible Send "
        "control. Require an actual chat composer or message thread plus a clearly identified attachment card or sent item before emitting direct_leak.\n"
        "The primary decision is whether an outbound action exists, not whether it succeeded. Selection/attachment, submit, progress, success, rejection, "
        "unsupported-file, cancellation, timeout, and error screens can all prove that an upload/send was attempted and must emit direct_leak. "
        "Fill action_status only when obvious; it is audit metadata and does not change the leak verdict.\n"
        "Sensitive source files and aliases:\n" + "\n".join(sensitive_lines) + "\n"
        f"Non-whitelisted active apps from logs: {active_apps}\n"
        "Frame context:\n" + "\n".join(frame_lines) + "\n"
        "Grid cell mapping:\n" + ("\n".join(grid_lines) if grid_lines else "- none")
    )


def _request_frame_to_dict(item: VlmRequestFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_id": item.frame.frame_id,
        "timestamp_ms": item.frame.timestamp_ms,
        "image_path": item.frame.image_path,
        "reason": item.frame.reason,
        "window_id": item.frame.window_id,
        "visual_note": item.visual_note,
        "visual_confidence": item.visual_confidence,
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


def _chunks(items: list[VlmRequestFrame], size: int) -> list[list[VlmRequestFrame]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _grid_cell_size(images: list[object]) -> tuple[int, int]:
    widths = [max(1, int(getattr(image, "width", 1))) for image in images]
    heights = [max(1, int(getattr(image, "height", 1))) for image in images]
    cell_width = max(1, min(max(widths), 1_280))
    aspects = sorted(width / height for width, height in zip(widths, heights, strict=False) if height > 0)
    aspect = aspects[len(aspects) // 2] if aspects else 16 / 9
    cell_height = max(1, int(round(cell_width / max(aspect, 0.1))))
    return cell_width, min(cell_height, 1_280)


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
        f"reason={source.get('reason', '')}, "
        f"visual_confidence={source.get('visual_confidence', 0.0)}, "
        f"visual_note={str(source.get('visual_note', ''))[:500]}"
    )


def _image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _chat_url(config: VisionConfig) -> str:
    if config.vlm_chat_url:
        return config.vlm_chat_url
    return config.vlm_base_url.rstrip("/") + "/chat/completions"
