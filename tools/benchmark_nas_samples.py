"""Evaluate log-first triage over downloaded NAS samples.

The runner is intentionally data-shape driven rather than sample-name driven:
it discovers cases from ``groundtruth.json`` plus logs, prefers key event logs,
and reports whether the current detector can either resolve an event
deterministically or route it to VLM review.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
import re
import sys
from threading import Lock
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

COMPLETION_OCR_TOKENS = (
    "sent",
    "uploaded",
    "completed",
    "success",
    "delivered",
    "shared",
    "committed",
    "merged",
    "attachment",
    "attach",
    "send",
    "upload",
    "\u5df2\u53d1\u9001",
    "\u53d1\u9001\u6210\u529f",
    "\u5df2\u4e0a\u4f20",
    "\u4e0a\u4f20\u5b8c\u6210",
    "\u4e0a\u4f20\u6210\u529f",
    "\u5b8c\u6210",
    "\u6210\u529f",
    "\u9644\u4ef6",
    "\u63d0\u4ea4\u6210\u529f",
    "\u5df2\u5171\u4eab",
    "\u5206\u4eab\u6210\u529f",
    "\u590d\u5236\u5b8c\u6210",
)

PRELIMINARY_OCR_TOKENS = (
    "draft",
    "cancel",
    "choose file",
    "selected file",
    "file picker",
    "\u8349\u7a3f",
    "\u53d6\u6d88",
    "\u9009\u62e9\u6587\u4ef6",
    "\u6253\u5f00",
    "\u4fdd\u5b58",
    "\u5199\u90ae\u4ef6",
)

_OCR_READER: Any = None
_OCR_READER_FAILED = False
_OCR_READER_LOCK = Lock()
_RAPID_OCR_READER: Any = None
_RAPID_OCR_READER_FAILED = False
_RAPID_OCR_READER_LOCK = Lock()


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


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


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


def _merge_review_windows(
    windows: List[Tuple[datetime, datetime]],
    gap_seconds: int,
) -> List[Tuple[datetime, datetime]]:
    if not windows:
        return []
    merged: List[Tuple[datetime, datetime]] = []
    for start, end in sorted(windows):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + timedelta(seconds=max(0, gap_seconds)):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _split_review_window(
    start: datetime,
    end: datetime,
    segment_seconds: int,
    overlap_seconds: int,
) -> List[Tuple[datetime, datetime]]:
    duration = max(0.0, (end - start).total_seconds())
    if duration <= segment_seconds or segment_seconds <= 0:
        return [(start, end)]

    segments: List[Tuple[datetime, datetime]] = []
    step = max(1, segment_seconds - max(0, overlap_seconds))
    cursor = start
    while cursor < end:
        seg_end = min(end, cursor + timedelta(seconds=segment_seconds))
        segments.append((cursor, seg_end))
        if seg_end >= end:
            break
        cursor += timedelta(seconds=step)
    return segments


def _event_text(event: Dict[str, Any]) -> str:
    parts = [
        event.get("event_type", ""),
        event.get("app_name", ""),
        event.get("file_path", ""),
        event.get("file_name", ""),
        event.get("content_preview", ""),
    ]
    process_info = event.get("process_info")
    if isinstance(process_info, dict):
        parts.append(process_info.get("process_name", ""))
    window_info = event.get("window_info")
    if isinstance(window_info, dict):
        parts.append(window_info.get("window_title", ""))
    return " ".join(str(part or "") for part in parts)


def _segment_signal_score(
    segment: Tuple[datetime, datetime],
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    start, end = segment
    padded_start = start - timedelta(seconds=3)
    padded_end = end + timedelta(seconds=3)
    score = 0.0
    hit_events = 0
    matched_tokens = set()

    candidate_events = fallback_meta.get("candidate_events", []) or []
    for event in candidate_events:
        dt = _parse_dt(event.get("timestamp", ""))
        if not dt or not (padded_start <= dt <= padded_end):
            continue
        hit_events += 1
        score += 2.0
        text = _event_text(event).lower()
        for token in EXTRA_LOG_TOKENS + COMPLETION_OCR_TOKENS:
            if token.lower() in text:
                matched_tokens.add(token)
                score += 1.0

    log_hits = 0
    for log in logs:
        dt = _parse_dt(log.get("timestamp", ""))
        if not dt or not (padded_start <= dt <= padded_end):
            continue
        text = _event_text(log).lower()
        if any(token.lower() in text for token in EXTRA_LOG_TOKENS):
            log_hits += 1
            score += 0.5

    duration = max(0.0, (end - start).total_seconds())
    score += min(duration / 60.0, 2.0)
    return score, {
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 3),
        "candidate_events": hit_events,
        "log_hits": log_hits,
        "matched_tokens": sorted(str(token) for token in matched_tokens)[:12],
        "score": round(score, 3),
    }


def _prepare_review_segments(
    windows: List[Tuple[datetime, datetime]],
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> Tuple[List[Tuple[datetime, datetime]], Dict[str, Any]]:
    merge_gap = _int_env("DLD_VLM_WINDOW_MERGE_GAP_SECONDS", 8, minimum=0)
    segment_seconds = _int_env("DLD_VLM_SEGMENT_SECONDS", 60, minimum=5)
    overlap_seconds = min(
        segment_seconds - 1,
        _int_env("DLD_VLM_SEGMENT_OVERLAP_SECONDS", 8, minimum=0),
    )
    max_segments = _int_env("DLD_VLM_MAX_SEGMENTS_PER_CASE", 3, minimum=1)

    merged = _merge_review_windows(windows, merge_gap)
    split_segments: List[Tuple[datetime, datetime]] = []
    for start, end in merged:
        split_segments.extend(_split_review_window(start, end, segment_seconds, overlap_seconds))

    scored = []
    for segment in split_segments:
        score, meta = _segment_signal_score(segment, fallback_meta, logs)
        scored.append((score, segment, meta))
    if scored:
        kept = sorted(scored, key=lambda item: item[0], reverse=True)[:max_segments]
        kept_segments = [item[1] for item in sorted(kept, key=lambda item: item[1][0])]
        kept_meta = [item[2] for item in sorted(kept, key=lambda item: item[1][0])]
    else:
        kept_segments = merged[:max_segments]
        kept_meta = []

    return kept_segments, {
        "raw_windows": len(windows),
        "merged_windows": len(merged),
        "split_segments": len(split_segments),
        "selected_segments": len(kept_segments),
        "merge_gap_seconds": merge_gap,
        "segment_seconds": segment_seconds,
        "overlap_seconds": overlap_seconds,
        "max_segments": max_segments,
        "segments": kept_meta,
    }


def _compact_ocr_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _ocr_reader() -> Any:
    global _OCR_READER, _OCR_READER_FAILED
    if _OCR_READER or _OCR_READER_FAILED:
        return _OCR_READER
    with _OCR_READER_LOCK:
        if _OCR_READER or _OCR_READER_FAILED:
            return _OCR_READER
        try:
            import easyocr
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _OCR_READER = easyocr.Reader(["ch_sim", "en"], gpu=(device == "cuda"))
        except Exception:
            _OCR_READER_FAILED = True
            _OCR_READER = None
    return _OCR_READER


def _rapid_ocr_reader() -> Any:
    global _RAPID_OCR_READER, _RAPID_OCR_READER_FAILED
    if _RAPID_OCR_READER or _RAPID_OCR_READER_FAILED:
        return _RAPID_OCR_READER
    with _RAPID_OCR_READER_LOCK:
        if _RAPID_OCR_READER or _RAPID_OCR_READER_FAILED:
            return _RAPID_OCR_READER
        try:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_READER = RapidOCR()
        except Exception:
            _RAPID_OCR_READER_FAILED = True
            _RAPID_OCR_READER = None
    return _RAPID_OCR_READER


def _ocr_engine_name() -> str:
    value = os.getenv("DLD_VLM_OCR_ENGINE", "auto").strip().lower()
    if value in {"0", "false", "no", "off", "none", "disabled"}:
        return "none"
    if value in {"easyocr", "easy"}:
        return "easyocr"
    if value in {"rapidocr", "rapid", "onnxruntime", "onnx"}:
        return "rapidocr"
    if value == "auto":
        try:
            import torch

            return "easyocr" if torch.cuda.is_available() else "none"
        except Exception:
            return "none"
    return "auto"


def _rapid_ocr_frame_text(frame: Any) -> str:
    reader = _rapid_ocr_reader()
    if not reader:
        return ""
    try:
        result = reader(frame)
    except Exception:
        return ""
    rows = result[0] if isinstance(result, tuple) else result
    texts: List[str] = []
    for row in rows or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            texts.append(str(row[1]).strip())
    return " ".join(text for text in texts if text)


def _ocr_frame_text(frame: Any) -> str:
    if not _ocr_prefilter_enabled():
        return ""
    engine = _ocr_engine_name()
    if engine == "rapidocr":
        return _rapid_ocr_frame_text(frame)
    if engine != "easyocr":
        return ""
    reader = _ocr_reader()
    if not reader:
        return ""
    try:
        results = reader.readtext(frame, detail=0, paragraph=False)
    except Exception:
        return ""
    return " ".join(str(item).strip() for item in results if str(item).strip())


def _ocr_risk_flags(text: str, sensitive_files: List[str]) -> List[str]:
    compact = _compact_ocr_text(text)
    flags: List[str] = []
    if not compact:
        return flags
    if any(_compact_ocr_text(token) in compact for token in COMPLETION_OCR_TOKENS):
        flags.append("completion_keyword")
    if any(_compact_ocr_text(token) in compact for token in PRELIMINARY_OCR_TOKENS):
        flags.append("preliminary_keyword")
    for path in sensitive_files:
        name = Path(str(path)).name
        stem = Path(str(path)).stem
        for token in (name, stem):
            normalized = _compact_ocr_text(token)
            if normalized and normalized in compact:
                flags.append("sensitive_name_visible")
                return sorted(set(flags))
    return sorted(set(flags))


def _ocr_prefilter_enabled() -> bool:
    value = os.getenv("DLD_VLM_ENABLE_OCR_PREFILTER", "auto").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return _ocr_engine_name() != "none"
    return _ocr_engine_name() != "none"


def _adaptive_vlm_frame_budget(
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    max_frames: int,
) -> Tuple[int, Dict[str, Any]]:
    cap = max(1, int(max_frames))
    min_frames = min(cap, _int_env("DLD_VLM_REVIEW_MIN_FRAMES", min(4, cap), minimum=1))
    base_frames = min(cap, max(min_frames, _int_env("DLD_VLM_REVIEW_BASE_FRAMES", 6, minimum=1)))
    frames = base_frames
    reasons: List[str] = []

    windows = _windows_from_fallback(fallback_meta, logs)
    review_segments, segment_meta = _prepare_review_segments(windows, fallback_meta, logs)
    durations = [max(0.0, (end - start).total_seconds()) for start, end in review_segments]
    total_window_seconds = sum(durations)
    max_window_seconds = max(durations) if durations else 0.0

    def bump(amount: int, reason: str) -> None:
        nonlocal frames
        if frames >= cap:
            return
        frames = min(cap, frames + amount)
        reasons.append(reason)

    if len(review_segments) >= 2:
        bump(2, "multiple_review_windows")
    if len(review_segments) >= 4:
        bump(1, "many_review_windows")
    if total_window_seconds >= 120:
        bump(2, "long_total_review_window")
    elif total_window_seconds >= 60:
        bump(1, "medium_total_review_window")
    if max_window_seconds >= 90:
        bump(1, "long_single_review_window")

    candidate_events = fallback_meta.get("candidate_events", []) or []
    if len(candidate_events) >= 8:
        bump(2, "many_candidate_events")
    elif len(candidate_events) >= 4:
        bump(1, "several_candidate_events")

    fallback_reasons = [str(item or "") for item in fallback_meta.get("reasons", []) or []]
    if len(fallback_reasons) >= 3:
        bump(1, "multiple_fallback_reasons")

    text_parts: List[str] = list(fallback_reasons)
    for event in candidate_events[:16]:
        text_parts.extend(
            str(event.get(key, "") or "")
            for key in ("event_type", "app_name", "file_name", "content_preview")
        )
        window_info = event.get("window_info")
        if isinstance(window_info, dict):
            text_parts.append(str(window_info.get("window_title", "") or ""))
    signal_text = " ".join(text_parts).lower()

    if any(token in signal_text for token in ("vm", "virtual", "remote desktop", "mstsc", "anydesk", "todesk", "sunlogin", "\u865a\u62df\u673a", "\u8fdc\u7a0b")):
        bump(2, "remote_or_vm_context")
    if any(token in signal_text for token in ("meeting", "screen_share", "share screen", "zoom", "teams", "feishu", "lark", "\u4f1a\u8bae", "\u5c4f\u5e55\u5171\u4eab")):
        bump(2, "meeting_or_screen_share_context")
    if any(token in signal_text for token in ("clipboard", "paste", "copy", "screenshot", "\u526a\u8d34\u677f", "\u7c98\u8d34", "\u590d\u5236", "\u622a\u56fe")):
        bump(1, "clipboard_or_screenshot_context")
    if any(token in signal_text for token in ("upload", "drive", "mail", "attach", "send", "git", "\u4e0a\u4f20", "\u90ae\u7bb1", "\u9644\u4ef6", "\u53d1\u9001")):
        bump(1, "external_transfer_context")
    if any(token in signal_text for token in ("chatgpt", "claude", "gemini", "deepseek", "kimi", "poe", "ai_service", "\u5bf9\u8bdd", "\u8f93\u5165\u6846")):
        bump(1, "ai_context")

    return frames, {
        "adaptive": True,
        "min_frames": min_frames,
        "base_frames": base_frames,
        "max_frames": cap,
        "selected_frames": frames,
        "window_count": len(windows),
        "segment_count": len(review_segments),
        "total_window_seconds": round(total_window_seconds, 3),
        "max_window_seconds": round(max_window_seconds, 3),
        "candidate_events": len(candidate_events),
        "fallback_reasons": fallback_reasons,
        "complexity_reasons": reasons,
        "segment_plan": segment_meta,
    }


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
    min_gap = _int_env("DLD_VLM_REVIEW_MIN_FRAME_GAP", 12, minimum=1)
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


def _annotate_and_limit_image_frames(
    selected: List[Dict[str, Any]],
    sensitive_files: List[str],
    max_frames: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ocr_enabled = _ocr_prefilter_enabled()
    if not selected:
        return [], {
            "ocr_enabled": ocr_enabled,
            "ocr_engine": _ocr_engine_name(),
            "image_frames": 0,
            "text_only_frames": 0,
        }

    default_max_images = min(max_frames, 6 if ocr_enabled else 8)
    max_image_frames = min(
        max_frames,
        _int_env("DLD_VLM_REVIEW_MAX_IMAGE_FRAMES", default_max_images, minimum=1),
    )
    min_image_frames = min(
        max_image_frames,
        _int_env("DLD_VLM_REVIEW_MIN_IMAGE_FRAMES", min(4, max_image_frames), minimum=1),
    )
    scene_threshold = float(os.getenv("DLD_VLM_REVIEW_IMAGE_SCENE_THRESHOLD", "0.08"))
    max_ocr_frames = min(
        len(selected),
        _int_env("DLD_VLM_REVIEW_MAX_OCR_FRAMES", min(len(selected), 3), minimum=0),
    )
    ocr_ranked = sorted(
        selected,
        key=lambda item: (
            "window_start" in str(item.get("selection_reason", "")),
            "window_end" in str(item.get("selection_reason", "")),
            float(item.get("scene_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    ocr_frame_ids = {int(item["frame_index"]) for item in ocr_ranked[:max_ocr_frames]} if ocr_enabled else set()
    seen_ocr = set()

    for item in selected:
        should_ocr = int(item["frame_index"]) in ocr_frame_ids
        ocr_text = _ocr_frame_text(item["frame"]) if should_ocr else ""
        flags = _ocr_risk_flags(ocr_text, sensitive_files)
        compact = _compact_ocr_text(ocr_text)
        duplicate_ocr = bool(compact and compact in seen_ocr)
        if compact:
            seen_ocr.add(compact)
        reason = str(item.get("selection_reason", "") or "")
        scene_score = float(item.get("scene_score", 0.0) or 0.0)
        image_priority = 0.0
        image_reasons: List[str] = []
        if "window_start" in reason or "window_end" in reason:
            image_priority += 3.0
            image_reasons.append("boundary_context")
        if "completion_keyword" in flags or "sensitive_name_visible" in flags:
            image_priority += 4.0
            image_reasons.append("ocr_risk_hit")
        if scene_score >= scene_threshold:
            image_priority += 2.0 + min(scene_score, 1.0)
            image_reasons.append("scene_change")
        if duplicate_ocr and image_priority < 4.0:
            image_priority -= 2.0
            image_reasons.append("ocr_duplicate")
        if not ocr_text:
            image_priority += 0.5
            image_reasons.append("ocr_not_run" if ocr_enabled and not should_ocr else "no_ocr_text")

        item["ocr_text"] = ocr_text[:500]
        item["ocr_flags"] = flags
        item["ocr_ran"] = should_ocr
        item["ocr_duplicate"] = duplicate_ocr
        item["image_priority"] = round(image_priority, 4)
        item["image_decision_reasons"] = image_reasons

    ranked = sorted(
        selected,
        key=lambda item: (
            float(item.get("image_priority", 0.0)),
            float(item.get("scene_score", 0.0) or 0.0),
            -int(item.get("frame_index", 0)),
        ),
        reverse=True,
    )
    image_frame_ids = {
        int(item["frame_index"])
        for item in ranked
        if float(item.get("image_priority", 0.0)) > 0
    }
    if len(image_frame_ids) > max_image_frames:
        image_frame_ids = {int(item["frame_index"]) for item in ranked[:max_image_frames]}
    if len(image_frame_ids) < min_image_frames:
        for item in ranked:
            image_frame_ids.add(int(item["frame_index"]))
            if len(image_frame_ids) >= min_image_frames:
                break

    for item in selected:
        item["image_sent"] = int(item["frame_index"]) in image_frame_ids
        if not item["image_sent"]:
            item["image_decision_reasons"] = list(item.get("image_decision_reasons", [])) + ["text_only_context"]

    return selected, {
        "ocr_enabled": ocr_enabled,
        "ocr_engine": _ocr_engine_name(),
        "ocr_reader_loaded": bool(_OCR_READER),
        "ocr_reader_failed": bool(_OCR_READER_FAILED),
        "rapid_ocr_reader_loaded": bool(_RAPID_OCR_READER),
        "rapid_ocr_reader_failed": bool(_RAPID_OCR_READER_FAILED),
        "max_image_frames": max_image_frames,
        "min_image_frames": min_image_frames,
        "max_ocr_frames": max_ocr_frames,
        "ocr_frames": sum(1 for item in selected if item.get("ocr_ran")),
        "scene_threshold": scene_threshold,
        "image_frames": sum(1 for item in selected if item.get("image_sent")),
        "text_only_frames": sum(1 for item in selected if not item.get("image_sent")),
    }


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
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    max_frames: int,
    max_edge: int = 960,
    jpeg_quality: int = 65,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import cv2

    review_segments, segment_meta = _prepare_review_segments(windows, fallback_meta, logs)
    candidate_budget = max(
        max_frames,
        _int_env("DLD_VLM_REVIEW_CANDIDATE_FRAMES", max(24, max_frames * 6), minimum=1),
    )
    frame_times = _frame_time_candidates(review_segments, max_frames, candidate_budget)
    if not frame_times:
        frame_times = _sample_frame_times(review_segments or windows, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"segment_plan": segment_meta, "candidate_budget": candidate_budget, "error": "video_open_failed"}
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
    selected, prefilter_meta = _annotate_and_limit_image_frames(selected, sensitive_files, max_frames)
    images = []
    for idx, item in enumerate(selected, 1):
        frame = item.pop("frame")
        encoded: Optional[Dict[str, Any]]
        if item.get("image_sent"):
            encoded = _encode_frame_image(frame, item, idx, max_edge, jpeg_quality)
        else:
            encoded = {
                "index": idx,
                "timestamp": item["timestamp"],
                "frame_index": item["frame_index"],
                "scene_score": round(float(item.get("scene_score", 0.0)), 4),
                "selection_reason": item.get("selection_reason", ""),
            }
        if encoded:
            encoded["image_sent"] = bool(item.get("image_sent"))
            encoded["ocr_text"] = item.get("ocr_text", "")
            encoded["ocr_flags"] = item.get("ocr_flags", [])
            encoded["ocr_ran"] = bool(item.get("ocr_ran"))
            encoded["ocr_duplicate"] = bool(item.get("ocr_duplicate"))
            encoded["image_priority"] = item.get("image_priority", 0.0)
            encoded["image_decision_reasons"] = item.get("image_decision_reasons", [])
            images.append(encoded)
    return images, {
        "segment_plan": segment_meta,
        "candidate_budget": candidate_budget,
        "candidate_frames": len(candidates),
        "selected_frames": len(selected),
        **prefilter_meta,
    }


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
        return {
            "status": "skipped",
            "reason": "missing_video_or_recording_start",
            "is_violation": True,
            "max_frames_requested": max_frames,
        }

    windows = _windows_from_fallback(fallback_meta, logs)
    frame_records, frame_plan = _extract_representative_frame_images(
        video_path,
        rec_start,
        windows,
        fallback_meta,
        logs,
        sensitive_files,
        max_frames,
        max_edge=_int_env("DLD_VLM_REVIEW_IMAGE_MAX_EDGE", 960, minimum=1),
        jpeg_quality=_int_env("DLD_VLM_REVIEW_JPEG_QUALITY", 65, minimum=1),
    )
    image_records = [item for item in frame_records if item.get("image_sent")]
    if not frame_records:
        return {
            "status": "skipped",
            "reason": "no_frames_extracted",
            "is_violation": True,
            "max_frames_requested": max_frames,
            "frame_plan": frame_plan,
        }

    api_key = _first_env("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "VL_API_KEY")
    base_url = _first_env("OPENAI_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "VL_BASE_URL")
    model = _first_env("VL_MODEL_NAME", "OPENAI_MODEL", "QWEN_VL_MODEL", "QWEN_MODEL") or "qwen3.7-plus"
    if not api_key:
        return {
            "status": "skipped",
            "reason": "missing_vlm_api_key",
            "is_violation": True,
            "frames_sent": len(image_records),
            "frame_context_count": len(frame_records),
            "max_frames_requested": max_frames,
            "frame_plan": frame_plan,
        }

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

视觉帧说明:
- image_sent=true 的帧附带原图，请以视觉证据为准。
- image_sent=false 的帧只提供本地 OCR/抽帧摘要，用于补充时间线，不能单独作为完成态强证据。
- OCR 可能缺字或误识别；若 OCR 与图片冲突，以图片为准。

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
    for image in frame_records:
        contents.append(
            {
                "type": "text",
                "text": (
                    f"Frame {image['index']} @ {image['timestamp']} "
                    f"(source_frame={image['frame_index']}, reason={image.get('selection_reason', '')}, "
                    f"scene_score={image.get('scene_score', 0.0)}, "
                    f"image_sent={str(bool(image.get('image_sent'))).lower()}, "
                    f"ocr_flags={image.get('ocr_flags', [])}, "
                    f"ocr={json.dumps(str(image.get('ocr_text', '') or '')[:300], ensure_ascii=False)})"
                ),
            }
        )
        if image.get("image_sent") and image.get("b64"):
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image['b64']}"}})

    llm = ChatOpenAI(model=model, base_url=base_url or None, api_key=api_key)
    response = llm.invoke([HumanMessage(content=contents)])
    text = str(response.content or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {
            "status": "failed",
            "reason": "non_json_response",
            "raw": text,
            "is_violation": True,
            "frames_sent": len(image_records),
            "frame_context_count": len(frame_records),
            "max_frames_requested": max_frames,
            "frame_plan": frame_plan,
        }
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "reason": "bad_json_response",
            "raw": text,
            "is_violation": True,
            "frames_sent": len(image_records),
            "frame_context_count": len(frame_records),
            "max_frames_requested": max_frames,
            "frame_plan": frame_plan,
        }
    verdict["status"] = "success"
    verdict["frames_sent"] = len(image_records)
    verdict["frame_context_count"] = len(frame_records)
    verdict["max_frames_requested"] = max_frames
    verdict["frame_plan"] = frame_plan
    verdict["frame_selection"] = [
        {
            "index": image["index"],
            "timestamp": image["timestamp"],
            "frame_index": image["frame_index"],
            "scene_score": image.get("scene_score"),
            "selection_reason": image.get("selection_reason"),
            "image_sent": bool(image.get("image_sent")),
            "ocr_text": image.get("ocr_text", ""),
            "ocr_flags": image.get("ocr_flags", []),
            "ocr_ran": bool(image.get("ocr_ran")),
            "ocr_duplicate": bool(image.get("ocr_duplicate")),
            "image_priority": image.get("image_priority", 0.0),
            "image_decision_reasons": image.get("image_decision_reasons", []),
        }
        for image in frame_records
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
    max_vlm_frames: int = 12,
    vlm_workers: int = 1,
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
    case_records: List[Dict[str, Any]] = []
    pending_vlm_records: List[Dict[str, Any]] = []
    live_limit = str(max_vlm_cases) if max_vlm_cases > 0 else "all"

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

        adaptive_frames = 0
        frame_budget_meta: Dict[str, Any] = {}
        if (
            use_vlm
            and should_run_vlm
            and not deterministic_positive
            and (max_vlm_cases <= 0 or len(pending_vlm_records) < max_vlm_cases)
        ):
            adaptive_frames, frame_budget_meta = _adaptive_vlm_frame_budget(
                fallback_meta,
                logs,
                max_frames=max_vlm_frames,
            )
            next_live_idx = len(pending_vlm_records) + 1
            _progress(
                f"[VLM {next_live_idx}/{live_limit} QUEUED] "
                f"case={case_id} progress={case_index}/{total_cases} "
                f"expected={int(expected)} reasons={','.join(fallback_meta.get('reasons', []))} "
                f"frames={adaptive_frames}/{max_vlm_frames} "
                f"complexity={','.join(frame_budget_meta.get('complexity_reasons', [])) or 'base'}"
            )
        record = {
            "case_index": case_index,
            "total_cases": total_cases,
            "case_dir": case_dir,
            "case_id": case_id,
            "log_source_name": log_source_name,
            "groundtruth": groundtruth,
            "logs": logs,
            "sensitive_files": sensitive_files,
            "detection": detection,
            "fallback_meta": fallback_meta,
            "expected": expected,
            "deterministic_positive": deterministic_positive,
            "triage_positive": triage_positive,
            "should_run_vlm": should_run_vlm,
            "adaptive_vlm_frames": adaptive_frames,
            "vlm_frame_budget": frame_budget_meta,
            "vlm_live_queued": adaptive_frames > 0,
            "live_vlm_verdict": None,
        }
        case_records.append(record)
        if adaptive_frames > 0:
            pending_vlm_records.append(record)

    result.live_vlm_reviews = len(pending_vlm_records)
    workers = max(1, int(vlm_workers))
    if pending_vlm_records:
        _progress(f"[VLM] running {len(pending_vlm_records)} live reviews with workers={workers}")
        future_to_record = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for record in pending_vlm_records:
                future = executor.submit(
                    _live_vlm_review_case,
                    case_dir=record["case_dir"],
                    groundtruth=record["groundtruth"],
                    logs=record["logs"],
                    sensitive_files=record["sensitive_files"],
                    fallback_meta=record["fallback_meta"],
                    max_frames=record["adaptive_vlm_frames"],
                )
                future_to_record[future] = record

            for done_index, future in enumerate(as_completed(future_to_record), 1):
                record = future_to_record[future]
                try:
                    verdict = future.result()
                except Exception as exc:
                    verdict = {
                        "status": "failed",
                        "reason": f"exception:{type(exc).__name__}",
                        "error": str(exc),
                        "is_violation": True,
                        "max_frames_requested": record["adaptive_vlm_frames"],
                    }
                record["live_vlm_verdict"] = verdict
                _progress(
                    f"[VLM {done_index}/{len(pending_vlm_records)} DONE] "
                    f"case={record['case_id']} status={verdict.get('status', 'unknown')} "
                    f"image_frames={verdict.get('frames_sent', 0)} "
                    f"context_frames={verdict.get('frame_context_count', 0)}/{record['adaptive_vlm_frames']} "
                    f"reason={verdict.get('reason', '')}"
                )

    for record in case_records:
        case_index = record["case_index"]
        total_cases = record["total_cases"]
        case_id = record["case_id"]
        logs = record["logs"]
        sensitive_files = record["sensitive_files"]
        groundtruth = record["groundtruth"]
        detection = record["detection"]
        fallback_meta = record["fallback_meta"]
        expected = record["expected"]
        deterministic_positive = record["deterministic_positive"]
        triage_positive = record["triage_positive"]
        should_run_vlm = record["should_run_vlm"]
        vlm_verdict = record["live_vlm_verdict"]

        correlation_bundle: Optional[Dict[str, Any]] = None
        final_positive = triage_positive
        if vlm_verdict:
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
        frame_context_count = 0
        confidence = ""
        completed_action = ""
        if vlm_verdict:
            vlm_status = str(vlm_verdict.get("status", "unknown"))
            frames_sent = int(vlm_verdict.get("frames_sent", 0) or 0)
            frame_context_count = int(vlm_verdict.get("frame_context_count", 0) or 0)
            confidence = str(vlm_verdict.get("confidence", ""))
            completed_action = str(vlm_verdict.get("completed_action", ""))
        elif should_run_vlm:
            if not use_vlm:
                vlm_status = "triage_only"
            elif record["vlm_live_queued"]:
                vlm_status = "queued"
            else:
                vlm_status = "live_limit_reached"

        _progress(
            f"[CASE {case_index}/{total_cases}] {final_bucket.upper()} "
            f"case={case_id} expected={int(expected)} final={int(final_positive)} "
            f"det={int(deterministic_positive)} triage={int(triage_positive)} "
            f"vlm={vlm_status} image_frames={frames_sent} context_frames={frame_context_count} "
            f"requested_frames={record['adaptive_vlm_frames']} confidence={confidence} "
            f"action={completed_action} reasons={','.join(fallback_meta.get('reasons', []))}"
        )

        result.cases.append(
            {
                "case": case_id,
                "log_file": record["log_source_name"],
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
                "vlm_live_queued": record["vlm_live_queued"],
                "adaptive_vlm_frames": record["adaptive_vlm_frames"],
                "vlm_frame_budget": record["vlm_frame_budget"],
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
    parser.add_argument(
        "--max-vlm-frames",
        type=int,
        default=12,
        help="Hard cap for adaptive live VLM frames per case.",
    )
    parser.add_argument(
        "--vlm-workers",
        type=int,
        default=_int_env("DLD_VLM_WORKERS", 1, minimum=1),
        help="Number of concurrent live VLM requests.",
    )
    args = parser.parse_args(argv)

    report = run_benchmark(
        args.data_root,
        args.stage,
        case_filters=args.case,
        use_vlm=args.use_vlm,
        max_vlm_cases=args.max_vlm_cases,
        max_vlm_frames=args.max_vlm_frames,
        vlm_workers=args.vlm_workers,
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
