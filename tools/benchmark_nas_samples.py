"""Evaluate log-first triage over downloaded NAS samples.

The runner is intentionally data-shape driven rather than sample-name driven:
it discovers cases from ``groundtruth.json`` plus logs, prefers key event logs,
and reports whether the current detector can either resolve an event
deterministically or route it to VLM review.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = REPO_ROOT / "main"
DEFAULT_DATA_ROOT = REPO_ROOT / "spec" / "data" / "nas_samples"
LOG_FILE_PRIORITY = ("keyevents.json", "logs.json")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

RISK_LABEL_PREFIXES = (
    "\u76f4\u63a5\u5916\u53d1",
    "\u76f4\u63a5\u4e0a\u4f20",
    "\u6f5c\u5728\u9690\u85cf\u884c\u4e3a",
    "\u9690\u85cf\u884c\u4e3a",
    "\u8fdd\u89c4",
    "\u654f\u611f\u64cd\u4f5c",
    "\u526a\u8d34\u677f\u64cd\u4f5c",
    "\u5e94\u7528\u5207\u6362",
)
BENIGN_LABEL_PREFIXES = ("\u6b63\u5e38\u64cd\u4f5c",)

DEFAULT_BLACKLIST_APPS = [
    "ChatGPT",
    "Claude",
    "Gemini",
    "DeepSeek",
    "Kimi",
    "Poe",
    "Doubao",
    "Tongyi",
    "Yiyan",
    "Feishu",
    "Lark",
    "DingTalk",
    "WeChat",
    "QQ",
    "TIM",
    "Discord",
    "Slack",
    "Gmail",
    "Outlook",
    "Proton",
    "163",
    "mail.",
    "Google Drive",
    "OneDrive",
    "Dropbox",
    "Baidu",
    "Quark",
    "Weiyun",
    "CSDN",
    "GitHub",
    "GitLab",
    "Gitee",
    "Bitbucket",
    "Medium",
    "Notion",
    "Yuque",
    "Zoom",
    "Teams",
    "Edge",
    "Chrome",
    "msedge.exe",
    "chrome.exe",
]
DEFAULT_WHITELIST_APPS = [
    "Word",
    "WINWORD",
    "Excel",
    "EXCEL",
    "WPS",
    "Explorer",
    "Finder",
    "kdesk64",
]

EXTRA_LOG_TOKENS = (
    "chatgpt",
    "claude",
    "gemini",
    "deepseek",
    "kimi",
    "poe",
    "doubao",
    "tongyi",
    "teams",
    "wechat",
    "qq",
    "dingtalk",
    "feishu",
    "lark",
    "gmail",
    "outlook",
    "mail",
    "drive",
    "upload",
    "attach",
    "send",
    "share",
    "meeting",
    "vmware",
    "virtualbox",
    "hyper-v",
    "virtual machine",
    "ubuntu - vmware",
    "openeuler",
    "remote desktop",
    "mstsc",
    "anydesk",
    "todesk",
    "sunlogin",
    "\u4e0a\u4f20",
    "\u9644\u4ef6",
    "\u53d1\u9001",
    "\u5206\u4eab",
    "\u4f1a\u8bae",
    "\u90ae\u7bb1",
    "\u865a\u62df\u673a",
    "\u8fdc\u7a0b\u684c\u9762",
)


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, expected: bool, predicted: bool) -> str:
        if expected and predicted:
            self.tp += 1
            return "tp"
        if expected and not predicted:
            self.fn += 1
            return "fn"
        if not expected and predicted:
            self.fp += 1
            return "fp"
        self.tn += 1
        return "tn"

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class BenchmarkSummary:
    triage: Metrics = field(default_factory=Metrics)
    deterministic: Metrics = field(default_factory=Metrics)
    final: Metrics = field(default_factory=Metrics)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    deterministic_hits: int = 0
    vlm_reviews: int = 0
    live_vlm_reviews: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "triage": self.triage.to_dict(),
                "deterministic": self.deterministic.to_dict(),
                "final": self.final.to_dict(),
                "deterministic_hits": self.deterministic_hits,
                "vlm_reviews": self.vlm_reviews,
                "live_vlm_reviews": self.live_vlm_reviews,
                "skipped_cases": len(self.skipped),
            },
            "cases": self.cases,
            "skipped": self.skipped,
        }


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json_lenient(path: Path) -> Any:
    text = _read_text(path).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        items = []
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            item, end = decoder.raw_decode(text[pos:])
            items.append(item)
            pos += end
        if items:
            return items
        raise


def _choose_log_file(case_dir: Path) -> Optional[Path]:
    logs_dir = case_dir / "logs"
    if not logs_dir.exists():
        return None
    for name in LOG_FILE_PRIORITY:
        candidate = logs_dir / name
        if candidate.exists():
            return candidate
    files = sorted(logs_dir.glob("*.json"), key=lambda item: item.stat().st_size)
    return files[0] if files else None


def _log_key(log: Dict[str, Any]) -> tuple:
    return (
        str(log.get("timestamp", "")),
        str(log.get("event_type", "")),
        str(log.get("file_path", "")),
        str(log.get("window_info", {}).get("window_title", "")),
    )


def _is_extra_log_relevant(log: Dict[str, Any], log_first: Any) -> bool:
    text = " ".join(
        str(part or "")
        for part in (
            log.get("event_type", ""),
            log.get("file_path", ""),
            log.get("file_name", ""),
            log.get("content_preview", ""),
            log.get("app_name", ""),
            log.get("process_info", {}).get("process_name", ""),
            log.get("window_info", {}).get("window_title", ""),
        )
    )
    lowered = text.lower()
    return log_first.is_sensitive_name(text) or any(token in lowered for token in EXTRA_LOG_TOKENS)


def _load_case_logs(case_dir: Path, log_first: Any) -> tuple[List[Dict[str, Any]], str]:
    primary = _choose_log_file(case_dir)
    if not primary:
        return [], ""
    logs = _read_json_lenient(primary)
    if not isinstance(logs, list):
        raise ValueError("primary log file is not a JSON array")

    source_name = primary.name
    full_log = case_dir / "logs" / "logs.json"
    if primary.name == "keyevents.json" and full_log.exists():
        try:
            full_logs = _read_json_lenient(full_log)
        except Exception:
            full_logs = []
        if isinstance(full_logs, list):
            seen = {_log_key(item) for item in logs if isinstance(item, dict)}
            added = 0
            max_extra = int(os.getenv("DLD_NAS_MAX_EXTRA_LOG_EVENTS", "300"))
            for item in full_logs:
                if not isinstance(item, dict) or not _is_extra_log_relevant(item, log_first):
                    continue
                key = _log_key(item)
                if key in seen:
                    continue
                seen.add(key)
                logs.append(item)
                added += 1
                if added >= max_extra:
                    break
            if added:
                source_name = f"{primary.name}+logs.json:{added}"

    return logs, source_name


def _choose_groundtruth(case_dir: Path) -> Optional[Path]:
    for name in ("groundtruth.json", "groudtruth.json", "groungtruth.json"):
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def _operation_items(groundtruth: Any) -> List[Dict[str, Any]]:
    if isinstance(groundtruth, dict):
        ops = groundtruth.get("operations", [])
        return [item for item in ops if isinstance(item, dict)]
    if isinstance(groundtruth, list):
        merged: List[Dict[str, Any]] = []
        for item in groundtruth:
            if isinstance(item, dict):
                merged.extend(_operation_items(item))
        return merged
    return []


def _is_risk_label(label: str) -> bool:
    text = str(label or "").strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in BENIGN_LABEL_PREFIXES):
        return False
    return any(text.startswith(prefix) for prefix in RISK_LABEL_PREFIXES)


def _expected_positive(groundtruth: Any) -> bool:
    return any(_is_risk_label(item.get("operation", "")) for item in _operation_items(groundtruth))


def _sensitive_files_from_groundtruth(groundtruth: Any) -> List[str]:
    seen = set()
    files = []
    for item in _operation_items(groundtruth):
        path = str(item.get("sensitive_file_path", "") or "").strip()
        if path and path.lower() not in seen:
            seen.add(path.lower())
            files.append(path)
    return files


def _looks_like_file_reference(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 260:
        return False
    if "/" in text or "\\" in text:
        return True
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", text))


def _sensitive_files_from_logs(logs: List[Dict[str, Any]], log_first: Any) -> List[str]:
    seen = set()
    files = []
    for log in logs:
        hint = log_first.file_hint_from_log(log)
        path = log_first.normalize_path(log.get("file_path", "") or hint)
        if hasattr(log_first, "is_system_noise_path") and log_first.is_system_noise_path(path):
            continue
        if not _looks_like_file_reference(path):
            continue
        name = log.get("file_name") or log_first.basename(path)
        content_preview = str(log.get("content_preview", "") or "")
        if path and log_first.is_sensitive_name(f"{name} {content_preview}"):
            key = log_first.file_key(path)
            if key not in seen:
                seen.add(key)
                files.append(path)
    return files


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _parse_dt(value: str) -> Optional[datetime]:
    text = str(value or "").strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _recording_start(groundtruth: Any, logs: List[Dict[str, Any]]) -> Optional[datetime]:
    if isinstance(groundtruth, dict):
        dt = _parse_dt(groundtruth.get("recording_start_time", ""))
        if dt:
            return dt
    for log in logs:
        dt = _parse_dt(log.get("timestamp", ""))
        if dt:
            return dt
    return None


def _choose_video_file(case_dir: Path) -> Optional[Path]:
    video_dir = case_dir / "video"
    if not video_dir.exists():
        return None
    candidates = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")))
    return candidates[0] if candidates else None


def _windows_from_fallback(fallback_meta: Dict[str, Any], logs: List[Dict[str, Any]]) -> List[Tuple[datetime, datetime]]:
    windows = []
    for item in fallback_meta.get("analysis_windows", []) or []:
        start = _parse_dt(item.get("start", ""))
        end = _parse_dt(item.get("end", ""))
        if start and end and end >= start:
            windows.append((start, end))
    if windows:
        return windows

    for event in fallback_meta.get("candidate_events", []) or []:
        dt = _parse_dt(event.get("timestamp", ""))
        if dt:
            windows.append((dt - timedelta(seconds=5), dt + timedelta(seconds=45)))
    if windows:
        return windows

    parsed = [_parse_dt(log.get("timestamp", "")) for log in logs]
    parsed = [dt for dt in parsed if dt]
    if not parsed:
        return []
    return [(min(parsed), max(parsed))]


def _review_log_context(logs: List[Dict[str, Any]], windows: List[Tuple[datetime, datetime]], limit: int = 24) -> List[Dict[str, Any]]:
    rows = []
    for log in logs:
        dt = _parse_dt(log.get("timestamp", ""))
        if not dt:
            continue
        if windows and not any(start - timedelta(seconds=3) <= dt <= end + timedelta(seconds=3) for start, end in windows):
            continue
        text = " ".join(
            str(part or "")
            for part in (
                log.get("event_type", ""),
                log.get("file_path", ""),
                log.get("file_name", ""),
                log.get("content_preview", ""),
                log.get("app_name", ""),
                log.get("process_info", {}).get("process_name", ""),
                log.get("window_info", {}).get("window_title", ""),
            )
        ).lower()
        if not any(
            token in text
            for token in (
                "clipboard",
                "paste",
                "copy",
                "chatgpt",
                "poe",
                "gemini",
                "upload",
                "send",
                "mail",
                "drive",
                "share",
                "\u7c98\u8d34",
                "\u590d\u5236",
                "\u4e0a\u4f20",
                "\u53d1\u9001",
                "\u90ae\u7bb1",
                "\u5206\u4eab",
            )
        ):
            continue
        rows.append(
            {
                "timestamp": log.get("timestamp", ""),
                "event_type": log.get("event_type", ""),
                "app_name": log.get("app_name") or log.get("process_info", {}).get("process_name", ""),
                "file_name": log.get("file_name", ""),
                "window_title": log.get("window_info", {}).get("window_title", ""),
                "content_preview": str(log.get("content_preview", "") or "")[:220],
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _sample_frame_times(windows: List[Tuple[datetime, datetime]], max_frames: int) -> List[datetime]:
    if max_frames <= 0:
        return []
    points: List[datetime] = []
    per_window = max(1, max_frames // max(1, len(windows)))
    for start, end in windows:
        duration = max(0.0, (end - start).total_seconds())
        count = min(per_window, max_frames - len(points))
        if count <= 0:
            break
        if count == 1 or duration == 0:
            points.append(start + timedelta(seconds=duration / 2))
        else:
            for idx in range(count):
                points.append(start + timedelta(seconds=duration * idx / (count - 1)))
    return points[:max_frames]


def _frame_time_candidates(
    windows: List[Tuple[datetime, datetime]],
    max_frames: int,
    max_candidates: int,
) -> List[datetime]:
    if max_frames <= 0:
        return []
    if not windows:
        return []

    budget = max(max_frames, max_candidates)
    durations = [max(0.0, (end - start).total_seconds()) for start, end in windows]
    total_duration = sum(durations) or float(len(windows))
    points: List[datetime] = []

    for idx, (start, end) in enumerate(windows):
        remaining_windows = len(windows) - idx
        remaining_budget = budget - len(points)
        if remaining_budget <= 0:
            break

        duration = durations[idx]
        proportional = int(round(budget * ((duration or 1.0) / total_duration)))
        count = max(3, proportional)
        count = min(count, remaining_budget - max(0, remaining_windows - 1))
        count = max(1, count)

        if count == 1 or duration <= 0:
            points.append(start + timedelta(seconds=duration / 2))
            continue
        for pos in range(count):
            ratio = pos / (count - 1)
            points.append(start + timedelta(seconds=duration * ratio))

    deduped = []
    seen = set()
    for dt in sorted(points):
        key = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dt)
    return deduped[:budget]


def _thumbnail_scene_score(frame: Any, previous_thumb: Any) -> tuple[float, Any]:
    import cv2

    thumb = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    if previous_thumb is None:
        return 1.0, gray
    score = float(cv2.absdiff(gray, previous_thumb).mean()) / 255.0
    return score, gray


def _select_representative_frames(candidates: List[Dict[str, Any]], max_frames: int) -> List[Dict[str, Any]]:
    if max_frames <= 0 or not candidates:
        return []
    if len(candidates) <= max_frames:
        for item in candidates:
            item["selection_reason"] = item.get("selection_reason") or "candidate"
        return candidates

    selected: Dict[int, Dict[str, Any]] = {}

    def add(item: Dict[str, Any], reason: str) -> None:
        if len(selected) >= max_frames:
            return
        key = int(item["frame_index"])
        if key in selected:
            selected[key]["selection_reason"] = f"{selected[key]['selection_reason']}+{reason}"
            return
        copy = dict(item)
        copy["selection_reason"] = reason
        selected[key] = copy

    add(candidates[0], "window_start")
    if len(candidates) > 2:
        add(candidates[len(candidates) // 2], "window_mid")
    add(candidates[-1], "window_end")

    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("scene_score", 0.0)), int(item.get("frame_index", 0))),
        reverse=True,
    )
    min_gap = max(1, int(os.getenv("DLD_VLM_REVIEW_MIN_FRAME_GAP", "12")))
    for item in ranked:
        if len(selected) >= max_frames:
            break
        frame_index = int(item["frame_index"])
        if any(abs(frame_index - existing) < min_gap for existing in selected):
            continue
        add(item, "scene_change")

    for item in ranked:
        if len(selected) >= max_frames:
            break
        add(item, "scene_change_fallback")

    return sorted(selected.values(), key=lambda item: int(item["frame_index"]))


def _encode_frame_image(
    frame: Any,
    item: Dict[str, Any],
    index: int,
    max_edge: int,
    jpeg_quality: int,
) -> Optional[Dict[str, Any]]:
    import cv2

    height, width = frame.shape[:2]
    largest = max(height, width)
    if largest > max_edge:
        scale = max_edge / largest
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return None
    return {
        "index": index,
        "timestamp": item["timestamp"],
        "frame_index": item["frame_index"],
        "scene_score": round(float(item.get("scene_score", 0.0)), 4),
        "selection_reason": item.get("selection_reason", ""),
        "b64": base64.b64encode(buffer).decode("ascii"),
    }


def _extract_representative_frame_images(
    video_path: Path,
    recording_start: datetime,
    windows: List[Tuple[datetime, datetime]],
    max_frames: int,
    max_edge: int = 960,
    jpeg_quality: int = 65,
) -> List[Dict[str, Any]]:
    import cv2

    candidate_budget = max(
        max_frames,
        int(os.getenv("DLD_VLM_REVIEW_CANDIDATE_FRAMES", str(max(24, max_frames * 6)))),
    )
    frame_times = _frame_time_candidates(windows, max_frames, candidate_budget)
    if not frame_times:
        frame_times = _sample_frame_times(windows, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidates = []
    previous_thumb = None
    seen_frames = set()
    try:
        for dt in frame_times:
            offset = max(0.0, (dt - recording_start).total_seconds())
            frame_index = int(round(offset * fps))
            if frame_count:
                frame_index = min(max(0, frame_index), max(0, frame_count - 1))
            if frame_index in seen_frames:
                continue
            seen_frames.add(frame_index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            scene_score, previous_thumb = _thumbnail_scene_score(frame, previous_thumb)
            candidates.append(
                {
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_index": frame_index,
                    "scene_score": scene_score,
                    "frame": frame,
                }
            )
    finally:
        cap.release()

    selected = _select_representative_frames(candidates, max_frames)
    images = []
    for idx, item in enumerate(selected, 1):
        encoded = _encode_frame_image(item.pop("frame"), item, idx, max_edge, jpeg_quality)
        if encoded:
            images.append(encoded)
    return images


def _live_vlm_review_case(
    case_dir: Path,
    groundtruth: Any,
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    fallback_meta: Dict[str, Any],
    max_frames: int,
) -> Dict[str, Any]:
    video_path = _choose_video_file(case_dir)
    rec_start = _recording_start(groundtruth, logs)
    if not video_path or not rec_start:
        return {"status": "skipped", "reason": "missing_video_or_recording_start", "is_violation": True}

    windows = _windows_from_fallback(fallback_meta, logs)
    images = _extract_representative_frame_images(
        video_path,
        rec_start,
        windows,
        max_frames,
        max_edge=int(os.getenv("DLD_VLM_REVIEW_IMAGE_MAX_EDGE", "960")),
        jpeg_quality=int(os.getenv("DLD_VLM_REVIEW_JPEG_QUALITY", "65")),
    )
    if not images:
        return {"status": "skipped", "reason": "no_frames_extracted", "is_violation": True}

    api_key = _first_env("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "VL_API_KEY")
    base_url = _first_env("OPENAI_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "VL_BASE_URL")
    model = _first_env("VL_MODEL_NAME", "OPENAI_MODEL", "QWEN_VL_MODEL", "QWEN_MODEL") or "qwen3.7-plus"
    if not api_key:
        return {"status": "skipped", "reason": "missing_vlm_api_key", "is_violation": True}

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    candidate_events = fallback_meta.get("candidate_events", [])[:12]
    review_logs = _review_log_context(logs, windows)
    prompt = f"""你是数据泄露审计员。请只根据给定日志上下文和视频关键帧，判断是否已经完成敏感数据外泄。

敏感文件/对象:
{json.dumps(sensitive_files, ensure_ascii=False, indent=2)}

候选日志事件:
{json.dumps(candidate_events, ensure_ascii=False, indent=2)}

窗口内关键日志摘要:
{json.dumps(review_logs, ensure_ascii=False, indent=2)}

判定规则:
- 只有看到或能强证据确认发送成功、上传完成、附件已提交、屏幕共享/截图/VM复制已暴露敏感内容，才判定 is_violation=true。
- 以下情况必须判定 false：仅打开敏感文件、右键/选择文件、进入上传页、附件已添加但仍在编辑页、保存草稿/草稿箱、取消/关闭、下载文件、正常编辑、复制到本地白名单应用、监控系统自己的“已完成/完成状态”提示。
- 邮件/聊天必须看到“已发送/发送成功/消息已出现在会话中/收件方可见”等完成态才为 true；“发送按钮高亮”“写信页面”“草稿”都不是完成态。
- AI 服务/网页聊天是特殊 sink：只要敏感文本已经出现在外部 AI/聊天输入框、prompt 框、对话消息或回答上下文中，即视为内容已暴露，可以判定 true；仅打开 AI 页面但未出现敏感内容则为 false。
- 网盘/Git/网页上传必须看到“上传完成/文件已出现在远端列表/提交成功”等完成态才为 true；“选择文件/上传对话框/进度未完成”都不是完成态。
- 不要把日志里的 risk_level、category、监控系统页面状态当作完成外泄证据，它们只能说明需要人工/VLM复核。
- 如果画面看不清或证据不足，判定 false，但 confidence 给低一些，reason 写明缺什么证据。
- 输出一个 JSON 对象，不要 markdown:
{{"is_violation": true/false, "confidence": 0.0-1.0, "completed_action": "send|upload|screen_share|screenshot|vm_copy|none|unknown", "evidence_frames": [1,2], "reason": "..."}}
"""
    contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        contents.append(
            {
                "type": "text",
                "text": (
                    f"Frame {image['index']} @ {image['timestamp']} "
                    f"(source_frame={image['frame_index']}, reason={image.get('selection_reason', '')}, "
                    f"scene_score={image.get('scene_score', 0.0)})"
                ),
            }
        )
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image['b64']}"}})

    llm = ChatOpenAI(model=model, base_url=base_url or None, api_key=api_key)
    response = llm.invoke([HumanMessage(content=contents)])
    text = str(response.content or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"status": "failed", "reason": "non_json_response", "raw": text, "is_violation": True}
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"status": "failed", "reason": "bad_json_response", "raw": text, "is_violation": True}
    verdict["status"] = "success"
    verdict["frames_sent"] = len(images)
    verdict["frame_selection"] = [
        {
            "index": image["index"],
            "timestamp": image["timestamp"],
            "frame_index": image["frame_index"],
            "scene_score": image.get("scene_score"),
            "selection_reason": image.get("selection_reason"),
        }
        for image in images
    ]
    verdict["model"] = model
    _postprocess_vlm_verdict(verdict)
    return verdict


def _postprocess_vlm_verdict(verdict: Dict[str, Any]) -> None:
    if not verdict.get("is_violation"):
        return
    reason = str(verdict.get("reason", "") or "").lower()
    preliminary_markers = (
        "准备发送",
        "准备上传",
        "右键点击",
        "右键",
        "选择了'上传",
        "选择了上传",
        "上传文件选项",
        "发送按钮",
        "按钮高亮",
        "附件已添加",
        "已准备发送",
        "file selected",
        "choose file",
        "selected file",
        "upload option",
    )
    if not any(marker in reason for marker in preliminary_markers):
        return
    verdict["raw_is_violation"] = verdict.get("is_violation")
    verdict["is_violation"] = False
    verdict["postprocess_reason"] = "downgraded_preliminary_action_without_completion_evidence"


def _verdict_timestamp(fallback_meta: Dict[str, Any], logs: List[Dict[str, Any]]) -> str:
    for event in fallback_meta.get("candidate_events", []) or []:
        timestamp = str(event.get("timestamp", "") or "").strip()
        if timestamp:
            dt = _parse_dt(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else timestamp.replace("T", " ").split(".")[0]
    for log in logs:
        timestamp = str(log.get("timestamp", "") or "").strip()
        if timestamp:
            dt = _parse_dt(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else timestamp.replace("T", " ").split(".")[0]
    return ""


def _frame_segments_from_vlm_verdict(
    verdict: Dict[str, Any],
    sensitive_files: List[str],
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if verdict.get("status") != "success" or not verdict.get("is_violation"):
        return []

    timestamp = _verdict_timestamp(fallback_meta, logs)
    completed_action = str(verdict.get("completed_action", "") or "unknown")
    primary_resource = sensitive_files[0] if sensitive_files else "unknown"
    candidate_events = fallback_meta.get("candidate_events", []) or []
    app_name = ""
    if candidate_events:
        app_name = str(candidate_events[0].get("app_name", "") or "")
    if not app_name:
        for log in logs:
            app_name = str(log.get("app_name") or log.get("process_info", {}).get("process_name", "") or "")
            if app_name:
                break

    return [
        {
            "segment_id": "vlm_verdict_0",
            "time_range": f"{timestamp} - {timestamp}" if timestamp else "",
            "app_name": app_name or "unknown",
            "operation_type": completed_action,
            "primary_resource": primary_resource,
            "related_resources": list(sensitive_files[1:]),
            "action_description": str(verdict.get("reason", "") or ""),
            "visible_evidence": [str(verdict.get("reason", "") or "")],
            "supporting_timestamps": [timestamp] if timestamp else [],
            "confidence": float(verdict.get("confidence", 0.0) or 0.0),
        }
    ]


def _run_event_correlator_bundle(
    case_id: str,
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    groundtruth: Any,
    frame_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from data_leak_detector.event_correlator import EventCorrelator

    recording_start = ""
    if isinstance(groundtruth, dict):
        recording_start = str(groundtruth.get("recording_start_time", "") or "")

    payload = {
        "session_id": case_id.replace("\\", "/"),
        "record_id": case_id.replace("\\", "/"),
        "log_events": logs,
        "frame_segments": frame_segments,
        "sensitive_files": sensitive_files,
        "recording_start_time": recording_start,
        "session_metadata": {"source": "nas_benchmark"},
    }
    return EventCorrelator().run(payload)


def _case_dirs(root: Path, stages: Optional[List[str]]) -> Iterable[Path]:
    stage_dirs = [root / stage for stage in stages] if stages else [path for path in root.iterdir() if path.is_dir()]
    for stage_dir in stage_dirs:
        if not stage_dir.exists():
            continue
        for gt_path in stage_dir.rglob("groundtruth.json"):
            if gt_path.parent.name.lower() in {"logs", "video"}:
                continue
            yield gt_path.parent
        for misspelled in ("groudtruth.json", "groungtruth.json"):
            for gt_path in stage_dir.rglob(misspelled):
                if gt_path.parent.name.lower() in {"logs", "video"}:
                    continue
                yield gt_path.parent


def run_benchmark(
    root: Path,
    stages: Optional[List[str]] = None,
    case_filters: Optional[List[str]] = None,
    use_vlm: bool = False,
    max_vlm_cases: int = 0,
    max_vlm_frames: int = 6,
) -> Dict[str, Any]:
    from data_leak_detector.legacy_paths import RISK_HUNTER_IMPL

    risk_dir = RISK_HUNTER_IMPL
    sys.path.insert(0, str(risk_dir))
    try:
        log_first = _load_module("nas_log_first_detector", risk_dir / "log_first_detector.py")
        run_e2e = _load_module("nas_run_e2e", REPO_ROOT / "main" / "run_e2e.py")
    finally:
        if str(risk_dir) in sys.path:
            sys.path.remove(str(risk_dir))

    result = BenchmarkSummary()
    seen_cases = set()
    wanted = {item.replace("\\", "/").strip().lower() for item in case_filters or []}
    case_dirs: List[tuple[Path, str]] = []
    for case_dir in sorted(_case_dirs(root, stages)):
        if case_dir in seen_cases:
            continue
        seen_cases.add(case_dir)
        case_id = str(case_dir.relative_to(root))
        if wanted:
            normalized_case_id = case_id.replace("\\", "/").lower()
            if normalized_case_id not in wanted and case_dir.name.lower() not in wanted:
                continue
        case_dirs.append((case_dir, case_id))

    total_cases = len(case_dirs)
    for case_index, (case_dir, case_id) in enumerate(case_dirs, 1):
        gt_path = _choose_groundtruth(case_dir)
        if not gt_path:
            result.skipped.append({"case": case_id, "reason": "missing_groundtruth_or_logs"})
            _progress(f"[CASE {case_index}/{total_cases}] SKIP case={case_id} reason=missing_groundtruth_or_logs")
            continue
        try:
            groundtruth = _read_json_lenient(gt_path)
            logs, log_source_name = _load_case_logs(case_dir, log_first)
            if not logs:
                raise ValueError("missing or empty log file")
        except Exception as exc:
            result.skipped.append({"case": case_id, "reason": f"parse_error: {exc}"})
            _progress(f"[CASE {case_index}/{total_cases}] SKIP case={case_id} reason=parse_error:{exc}")
            continue

        sensitive_files = _sensitive_files_from_groundtruth(groundtruth)
        for path in _sensitive_files_from_logs(logs, log_first):
            if path.lower() not in {item.lower() for item in sensitive_files}:
                sensitive_files.append(path)

        detector = log_first.LogFirstDetector(
            sensitive_files=sensitive_files,
            blacklist_apps=DEFAULT_BLACKLIST_APPS,
            whitelist_apps=DEFAULT_WHITELIST_APPS,
        )
        detection = detector.analyze(logs)
        should_run_vlm, fallback_meta = run_e2e._should_use_vlm_fallback(logs, detection)

        expected = _expected_positive(groundtruth)
        deterministic_positive = len(detection.get("upload_events", [])) > 0
        triage_positive = deterministic_positive or should_run_vlm
        vlm_verdict: Optional[Dict[str, Any]] = None
        correlation_bundle: Optional[Dict[str, Any]] = None
        final_positive = triage_positive
        if (
            use_vlm
            and should_run_vlm
            and not deterministic_positive
            and (max_vlm_cases <= 0 or result.live_vlm_reviews < max_vlm_cases)
        ):
            next_live_idx = result.live_vlm_reviews + 1
            live_limit = str(max_vlm_cases) if max_vlm_cases > 0 else "all"
            _progress(
                f"[VLM {next_live_idx}/{live_limit} START] "
                f"case={case_id} progress={case_index}/{total_cases} "
                f"expected={int(expected)} reasons={','.join(fallback_meta.get('reasons', []))} "
                f"max_frames={max_vlm_frames}"
            )
            result.live_vlm_reviews += 1
            vlm_verdict = _live_vlm_review_case(
                case_dir=case_dir,
                groundtruth=groundtruth,
                logs=logs,
                sensitive_files=sensitive_files,
                fallback_meta=fallback_meta,
                max_frames=max_vlm_frames,
            )
            if vlm_verdict.get("status") == "success":
                frame_segments = _frame_segments_from_vlm_verdict(
                    vlm_verdict,
                    sensitive_files,
                    fallback_meta,
                    logs,
                )
                correlation_bundle = _run_event_correlator_bundle(
                    case_id,
                    logs=[],
                    sensitive_files=sensitive_files,
                    groundtruth=groundtruth,
                    frame_segments=frame_segments,
                )
                final_positive = len(correlation_bundle.get("upload_candidates", [])) > 0
            else:
                final_positive = True
        elif deterministic_positive:
            final_positive = True

        triage_bucket = result.triage.add(expected, triage_positive)
        deterministic_bucket = result.deterministic.add(expected, deterministic_positive)
        final_bucket = result.final.add(expected, final_positive)
        if deterministic_positive:
            result.deterministic_hits += 1
        if should_run_vlm:
            result.vlm_reviews += 1

        vlm_status = "none"
        frames_sent = 0
        confidence = ""
        completed_action = ""
        if vlm_verdict:
            vlm_status = str(vlm_verdict.get("status", "unknown"))
            frames_sent = int(vlm_verdict.get("frames_sent", 0) or 0)
            confidence = str(vlm_verdict.get("confidence", ""))
            completed_action = str(vlm_verdict.get("completed_action", ""))
        elif should_run_vlm:
            vlm_status = "queued" if use_vlm else "triage_only"

        _progress(
            f"[CASE {case_index}/{total_cases}] {final_bucket.upper()} "
            f"case={case_id} expected={int(expected)} final={int(final_positive)} "
            f"det={int(deterministic_positive)} triage={int(triage_positive)} "
            f"vlm={vlm_status} frames={frames_sent} confidence={confidence} "
            f"action={completed_action} reasons={','.join(fallback_meta.get('reasons', []))}"
        )

        result.cases.append(
            {
                "case": case_id,
                "log_file": log_source_name,
                "expected_positive": expected,
                "triage_positive": triage_positive,
                "deterministic_positive": deterministic_positive,
                "triage_bucket": triage_bucket,
                "deterministic_bucket": deterministic_bucket,
                "final_positive": final_positive,
                "final_bucket": final_bucket,
                "upload_events": len(detection.get("upload_events", [])),
                "operation_records": len(detection.get("operation_records", [])),
                "sensitive_files": len(sensitive_files),
                "vlm_decision": fallback_meta.get("decision"),
                "vlm_reasons": fallback_meta.get("reasons", []),
                "live_vlm_verdict": vlm_verdict,
                "correlation_bundle": correlation_bundle,
            }
        )

    return result.to_dict()


def _print_report(report: Dict[str, Any]) -> None:
    print("\nNAS Sample Benchmark")
    print("=" * 40)
    for name in ("triage", "deterministic", "final"):
        metrics = report["summary"][name]
        print(
            f"{name:14} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"(tp={metrics['tp']}, fp={metrics['fp']}, tn={metrics['tn']}, fn={metrics['fn']})"
        )
    print(
        f"deterministic_hits={report['summary']['deterministic_hits']} "
        f"vlm_reviews={report['summary']['vlm_reviews']} "
        f"live_vlm_reviews={report['summary']['live_vlm_reviews']} "
        f"skipped={report['summary']['skipped_cases']}"
    )
    failures = [case for case in report["cases"] if case["final_bucket"] in {"fp", "fn"}]
    print(f"final_failures={len(failures)}")
    for case in failures[:30]:
        print(
            f"- {case['case']} {case['final_bucket']} "
            f"det={case['upload_events']} vlm={case['vlm_decision']} "
            f"reasons={','.join(case['vlm_reasons'])}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark downloaded NAS samples with optional live VLM verification.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage", action="append", help="Limit to a stage directory, e.g. --stage stage1")
    parser.add_argument("--case", action="append", help="Limit to a case path/name, e.g. --case stage1/2-ai-poe-1")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human report.")
    parser.add_argument("--json-output", type=Path, help="Write full JSON report to this path.")
    parser.add_argument("--use-vlm", action="store_true", help="Call a live VLM to verify triage-only cases.")
    parser.add_argument("--max-vlm-cases", type=int, default=0, help="Maximum live VLM cases to review; 0 means no limit.")
    parser.add_argument("--max-vlm-frames", type=int, default=6, help="Maximum frames sent per live VLM case.")
    args = parser.parse_args(argv)

    report = run_benchmark(
        args.data_root,
        args.stage,
        case_filters=args.case,
        use_vlm=args.use_vlm,
        max_vlm_cases=args.max_vlm_cases,
        max_vlm_frames=args.max_vlm_frames,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
