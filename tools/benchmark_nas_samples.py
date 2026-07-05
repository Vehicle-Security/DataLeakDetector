"""Evaluate log-first triage over downloaded NAS samples.

The runner is intentionally data-shape driven rather than sample-name driven:
it discovers cases from ``groundtruth.json`` plus logs, prefers key event logs,
and reports whether the current detector can either resolve an event
deterministically or route it to VLM review.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import io
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
_PROGRESS_LOG_HANDLE: Optional[Any] = None

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from data_leak_detector.evidence_semantics import (  # noqa: E402
    CONFIRMED_LOG_RULES,
    decide_evidence_outcome,
    is_confirmed_risk_level,
    log_rules_confirm_leak,
)

RISK_LABEL_PREFIXES = (
    "\u6f5c\u5728\u9690\u85cf\u884c\u4e3a",
    "\u9690\u85cf\u884c\u4e3a",
    "\u654f\u611f\u64cd\u4f5c",
    "\u526a\u8d34\u677f\u64cd\u4f5c",
    "\u5e94\u7528\u5207\u6362",
)
CONFIRMED_LABEL_PREFIXES = (
    "\u6210\u529f\u5916\u53d1",
    "\u76f4\u63a5\u5916\u53d1",
    "\u76f4\u63a5\u4e0a\u4f20",
    "\u8fdd\u89c4\u4e0a\u4f20",
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

TRANSFER_WINDOW_TOKENS = (
    "send file",
    "sending file",
    "file transfer",
    "upload file",
    "attach file",
    "\u53d1\u9001\u6587\u4ef6",
    "\u4f20\u8f93\u6587\u4ef6",
    "\u4e0a\u4f20\u6587\u4ef6",
    "\u6dfb\u52a0\u9644\u4ef6",
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

TERMINAL_SUCCESS_TOKENS = (
    "message sent",
    "mail sent",
    "email sent",
    "sent successfully",
    "send success",
    "upload complete",
    "upload completed",
    "uploaded successfully",
    "upload success",
    "share link created",
    "shared successfully",
    "publish success",
    "published successfully",
    "commit successful",
    "\u6d88\u606f\u5df2\u53d1\u9001",
    "\u90ae\u4ef6\u5df2\u53d1\u9001",
    "\u53d1\u9001\u6210\u529f",
    "\u5df2\u53d1\u9001",
    "\u4e0a\u4f20\u5b8c\u6210",
    "\u4e0a\u4f20\u6210\u529f",
    "\u5df2\u4e0a\u4f20",
    "\u5206\u4eab\u6210\u529f",
    "\u53d1\u5e03\u6210\u529f",
    "\u63d0\u4ea4\u6210\u529f",
)

TERMINAL_FAILURE_TOKENS = (
    "upload failed",
    "send failed",
    "delivery failed",
    "failed to upload",
    "failed to send",
    "network error",
    "try again",
    "\u4e0a\u4f20\u5931\u8d25",
    "\u53d1\u9001\u5931\u8d25",
    "\u7f51\u7edc\u9519\u8bef",
    "\u91cd\u8bd5",
)

TERMINAL_CANCEL_TOKENS = (
    "cancelled",
    "canceled",
    "cancel upload",
    "upload canceled",
    "upload cancelled",
    "send canceled",
    "discard draft",
    "remove attachment",
    "attachment removed",
    "\u53d6\u6d88\u4e0a\u4f20",
    "\u4e0a\u4f20\u5df2\u53d6\u6d88",
    "\u53d6\u6d88\u53d1\u9001",
    "\u5220\u9664\u9644\u4ef6",
    "\u79fb\u9664\u9644\u4ef6",
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

EXTERNAL_SINK_TOKENS = (
    "chatgpt",
    "claude",
    "gemini",
    "deepseek",
    "kimi",
    "poe",
    "doubao",
    "tongyi",
    "gmail",
    "outlook",
    "proton",
    "mail",
    "email",
    "dropbox",
    "onedrive",
    "google drive",
    "baidu",
    "weiyun",
    "quark",
    "github",
    "gitlab",
    "gitee",
    "bitbucket",
    "csdn",
    "zhihu",
    "juejin",
    "qq",
    "wechat",
    "dingtalk",
    "feishu",
    "lark",
    "teams",
    "zoom",
    "online tool",
    "web tool",
    "online converter",
    "base64",
    "toolshu",
    "\u90ae\u7bb1",
    "\u4e91\u76d8",
    "\u7f51\u76d8",
    "\u9644\u4ef6",
    "\u53d1\u9001",
    "\u4e0a\u4f20",
    "\u5206\u4eab",
    "\u5bf9\u8bdd",
    "\u8f93\u5165\u6846",
)

ATTEMPT_ACTION_TOKENS = (
    "file_selected",
    "clipboard_text",
    "clipboard_copy",
    "clipboard_image",
    "paste",
    "attach",
    "attachment",
    "send",
    "upload",
    "share",
    "commit",
    "\u9009\u62e9\u6587\u4ef6",
    "\u526a\u8d34\u677f",
    "\u590d\u5236",
    "\u7c98\u8d34",
    "\u9644\u4ef6",
    "\u53d1\u9001",
    "\u4e0a\u4f20",
    "\u5206\u4eab",
)

POSITIVE_RISK_LEVELS = {
    "attempted",
    "in_progress",
    "content_exposed",
    "completed",
}

SEGMENT_SECONDS = 45
SEGMENT_OVERLAP_SECONDS = 10
FRAMES_PER_SEGMENT = 8
CANDIDATE_FRAMES_PER_SEGMENT = 24
MAX_SEGMENTS_PER_CASE = 6
MAX_IMAGE_FRAMES_PER_SEGMENT = 4
MAX_OCR_FRAMES_PER_SEGMENT = 3
IMAGE_SCENE_THRESHOLD = 0.08
STATUS_REGION_THRESHOLD = 0.12
MIN_FRAME_GAP = 12
IMAGE_MAX_EDGE = 960
JPEG_QUALITY = 65
SINK_SESSION_IDLE_SECONDS = 90
SINK_SESSION_MAX_SECONDS = 900
SINK_SESSION_MIN_SECONDS = 60
SINK_SESSION_HEARTBEAT_SECONDS = 30
SINK_SESSION_TRACKING_OFFSETS_SECONDS = (15, 30, 60, 120, 180, 240, 300, 420, 600, 900)

MONITOR_UI_TOKENS = (
    "localhost:5000",
    "localhost 5000",
    "win monitor",
    "\u6570\u636e\u6cc4\u9732\u884c\u4e3a\u76d1\u63a7",
    "\u63a7\u5236\u9762\u677f",
    "\u4f1a\u8bdd\u5217\u8868",
)

_OCR_READER: Any = None
_OCR_READER_FAILED = False
_OCR_READER_LOCK = Lock()
_OCR_INFER_LOCK = Lock()
_RAPID_OCR_READER: Any = None
_RAPID_OCR_READER_FAILED = False
_RAPID_OCR_READER_LOCK = Lock()
_RAPID_OCR_INFER_LOCK = Lock()


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
    rules_only: Metrics = field(default_factory=Metrics)
    vlm_only: Metrics = field(default_factory=Metrics)
    risk: Metrics = field(default_factory=Metrics)
    final: Metrics = field(default_factory=Metrics)
    confirmed: Metrics = field(default_factory=Metrics)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    deterministic_hits: int = 0
    vlm_reviews: int = 0
    live_vlm_reviews: int = 0
    vlm_remote_requests: int = 0
    vlm_local_resolutions: int = 0
    vlm_cache_hits: int = 0
    datalog_cases: int = 0
    datalog_positive: int = 0
    datalog_confirmed: int = 0
    datalog_fallbacks: int = 0
    frame_coverage_cases: int = 0
    frame_coverage_completion: int = 0
    frame_coverage_content_exposed: int = 0
    frame_coverage_staging: int = 0
    frame_coverage_external_sink: int = 0
    frame_coverage_sensitive_object: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "triage": self.triage.to_dict(),
                "deterministic": self.deterministic.to_dict(),
                "rules_only": self.rules_only.to_dict(),
                "vlm_only": self.vlm_only.to_dict(),
                "risk": self.risk.to_dict(),
                "final": self.final.to_dict(),
                "confirmed": self.confirmed.to_dict(),
                "final_semantics": "groundtruth_aligned",
                "confirmed_semantics": "confirmed_leak",
                "risk_semantics": "staging_or_attempted_or_confirmed",
                "deterministic_hits": self.deterministic_hits,
                "vlm_reviews": self.vlm_reviews,
                "live_vlm_reviews": self.live_vlm_reviews,
                "vlm_remote_requests": self.vlm_remote_requests,
                "vlm_local_resolutions": self.vlm_local_resolutions,
                "vlm_cache_hits": self.vlm_cache_hits,
                "datalog_cases": self.datalog_cases,
                "datalog_positive": self.datalog_positive,
                "datalog_confirmed": self.datalog_confirmed,
                "datalog_fallbacks": self.datalog_fallbacks,
                "keyframe_coverage": {
                    "cases": self.frame_coverage_cases,
                    "completion_anchor": self.frame_coverage_completion,
                    "content_exposed_anchor": self.frame_coverage_content_exposed,
                    "staging_anchor": self.frame_coverage_staging,
                    "external_sink_anchor": self.frame_coverage_external_sink,
                    "sensitive_object_anchor": self.frame_coverage_sensitive_object,
                },
                "skipped_cases": len(self.skipped),
            },
            "cases": self.cases,
            "skipped": self.skipped,
        }


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
    if _PROGRESS_LOG_HANDLE is not None:
        print(message, file=_PROGRESS_LOG_HANDLE, flush=True)


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


def _repair_unclosed_simple_string_values(text: str) -> str:
    # Some NAS logs contain mojibake values like `"category": "xxx,` where the
    # closing quote was lost before a comma/newline. Repair only this narrow
    # object-field shape so we do not invent structure in severely truncated JSON.
    return re.sub(r'(:\s*"[^"\r\n]*?)(,)(\s*\r?\n\s*")', r'\1"\2\3', text)


def _read_json_lenient(path: Path) -> Any:
    text = _read_text(path).strip()
    candidates = [text]

    collapsed_quotes = re.sub(r'""([^"\r\n]*?)""', r'"\1"', text)
    if collapsed_quotes != text:
        candidates.append(collapsed_quotes)

    repaired_unclosed = _repair_unclosed_simple_string_values(text)
    if repaired_unclosed != text:
        candidates.append(repaired_unclosed)

    expanded: List[str] = []
    for candidate in candidates:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = _repair_unclosed_simple_string_values(repaired)
        if repaired != candidate:
            expanded.append(repaired)
    candidates.extend(expanded)

    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            last_error = exc

    for candidate in candidates:
        decoder = json.JSONDecoder(strict=False)
        items = []
        pos = 0
        try:
            while pos < len(candidate):
                while pos < len(candidate) and candidate[pos].isspace():
                    pos += 1
                if pos >= len(candidate):
                    break
                item, end = decoder.raw_decode(candidate[pos:])
                items.append(item)
                pos += end
            if items:
                return items
        except json.JSONDecodeError as exc:
            last_error = exc
            pos += 1
    if text.startswith("["):
        recovered = _recover_json_array_objects(text)
        if recovered:
            return recovered
    if last_error:
        raise last_error
    raise json.JSONDecodeError("empty JSON", text, 0)


def _recover_json_array_objects(text: str) -> List[Any]:
    decoder = json.JSONDecoder(strict=False)
    items: List[Any] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                raw = text[start:index + 1]
                repairs = [
                    raw,
                    re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw),
                    re.sub(r'""([^"\r\n]*?)""', r'"\1"', raw),
                    _repair_unclosed_simple_string_values(raw),
                ]
                for candidate in repairs:
                    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
                    candidate = _repair_unclosed_simple_string_values(candidate)
                    try:
                        items.append(decoder.decode(candidate))
                        break
                    except json.JSONDecodeError:
                        continue
                start = None
    return items


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


def _candidate_log_files(case_dir: Path) -> List[Path]:
    logs_dir = case_dir / "logs"
    if not logs_dir.exists():
        return []
    result: List[Path] = []
    seen: set[Path] = set()
    for name in LOG_FILE_PRIORITY:
        candidate = logs_dir / name
        if candidate.exists():
            result.append(candidate)
            seen.add(candidate.resolve())
    for candidate in sorted(logs_dir.glob("*.json"), key=lambda item: item.stat().st_size, reverse=True):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(candidate)
    return result


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
    candidates = _candidate_log_files(case_dir)
    if not candidates:
        return [], ""
    errors: List[str] = []
    primary = candidates[0]
    logs: Any = []
    for candidate in candidates:
        try:
            parsed = _read_json_lenient(candidate)
        except Exception as exc:
            errors.append(f"{candidate.name}:{type(exc).__name__}")
            continue
        if not isinstance(parsed, list):
            errors.append(f"{candidate.name}:not_array")
            continue
        dict_rows = [item for item in parsed if isinstance(item, dict)]
        if not dict_rows:
            errors.append(f"{candidate.name}:empty")
            continue
        primary = candidate
        logs = dict_rows
        break
    else:
        raise ValueError(f"no readable log JSON arrays ({', '.join(errors)})")

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


def _is_confirmed_label(label: str) -> bool:
    text = str(label or "").strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in BENIGN_LABEL_PREFIXES):
        return False
    return any(text.startswith(prefix) for prefix in CONFIRMED_LABEL_PREFIXES)


def _expected_level(groundtruth: Any) -> str:
    levels = []
    for item in _operation_items(groundtruth):
        label = str(item.get("operation", "") or "")
        if _is_confirmed_label(label):
            levels.append("confirmed")
        elif _is_risk_label(label):
            levels.append("risk")
    if "confirmed" in levels:
        return "confirmed"
    if "risk" in levels:
        return "risk"
    return "normal"


def _expected_positive(groundtruth: Any) -> bool:
    return _expected_level(groundtruth) != "normal"


def _final_positive_for_expected_level(expected_level: str, risk_positive: bool, confirmed_leak: bool) -> bool:
    if expected_level == "confirmed":
        return confirmed_leak
    if expected_level == "risk":
        return risk_positive
    return confirmed_leak


def _groundtruth_is_event_log(groundtruth: Any) -> bool:
    """Detect corrupted cases whose groundtruth.json actually holds monitor logs."""
    if not isinstance(groundtruth, list) or not groundtruth:
        return False
    dict_rows = [item for item in groundtruth if isinstance(item, dict)]
    if not dict_rows:
        return False
    if _operation_items(groundtruth):
        return False
    event_like = sum(1 for item in dict_rows if "event_type" in item or "timestamp" in item)
    return event_like >= max(1, len(dict_rows) // 2)


def _is_valid_sensitive_file_ref(path: str) -> bool:
    text = str(path or "").strip()
    if not _looks_like_file_reference(text):
        return False
    lowered = text.replace("\\", "/").lower()
    if lowered.endswith((".lnk", ".tmp", ".log", ".aodl", ".journal")):
        return False
    if "/appdata/" in lowered and "/desktop/" not in lowered and "/documents/" not in lowered:
        return False
    return True


def _sensitive_files_from_groundtruth(groundtruth: Any) -> List[str]:
    seen = set()
    files = []
    for item in _operation_items(groundtruth):
        path = str(item.get("sensitive_file_path", "") or "").strip()
        if path and _is_valid_sensitive_file_ref(path) and path.lower() not in seen:
            seen.add(path.lower())
            files.append(path)
    return files


def _looks_like_file_reference(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 260:
        return False
    lowered = text.replace("\\", "/").lower()
    if lowered.endswith(".lnk") or "/appdata/roaming/microsoft/windows/recent/" in lowered:
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
        if not _is_valid_sensitive_file_ref(path):
            continue
        name = log.get("file_name") or log_first.basename(path)
        content_preview = str(log.get("content_preview", "") or "")
        if path and log_first.is_sensitive_name(f"{name} {content_preview}"):
            key = log_first.file_key(path)
            if key not in seen:
                seen.add(key)
                files.append(path)
    return files


def _fallback_sensitive_files_from_logs(logs: List[Dict[str, Any]], log_first: Any, limit: int = 8) -> List[str]:
    seen = set()
    files = []
    for log in logs:
        path = log_first.normalize_path(log.get("file_path", "") or "")
        if not path or not _is_valid_sensitive_file_ref(path):
            continue
        if hasattr(log_first, "is_system_noise_path") and log_first.is_system_noise_path(path):
            continue
        name = str(log.get("file_name", "") or log_first.basename(path) or "")
        content_preview = str(log.get("content_preview", "") or "")
        window_title = (
            str((log.get("window_info", {}) or {}).get("window_title", "") or "")
            if isinstance(log.get("window_info"), dict)
            else ""
        )
        text = " ".join(
            str(part or "")
            for part in (
                name,
                path,
                content_preview,
                window_title,
            )
        )
        if not log_first.is_sensitive_name(text):
            continue
        if not log_first.is_sensitive_name(f"{name} {log_first.basename(path)}"):
            if not content_preview or not window_title:
                continue
            if log_first.basename(path).lower() not in window_title.lower().replace("\\", "/"):
                continue
        key = log_first.file_key(path)
        if key in seen:
            continue
        seen.add(key)
        files.append(path)
        if len(files) >= limit:
            break
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


def _recording_start_from_video_name(video_path: Optional[Path]) -> Optional[datetime]:
    if not video_path:
        return None
    match = re.search(r"recording_(\d{8})_(\d{6})", video_path.stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _first_log_time(logs: List[Dict[str, Any]]) -> Optional[datetime]:
    for log in logs:
        dt = _parse_dt(log.get("timestamp", ""))
        if dt:
            return dt
    return None


def _recording_start(
    groundtruth: Any,
    logs: List[Dict[str, Any]],
    video_path: Optional[Path] = None,
) -> Optional[datetime]:
    video_dt = _recording_start_from_video_name(video_path)
    log_dt = _first_log_time(logs)
    reference_dt = video_dt or log_dt
    if isinstance(groundtruth, dict):
        dt = _parse_dt(groundtruth.get("recording_start_time", ""))
        if dt and (not reference_dt or abs((dt - reference_dt).total_seconds()) <= 3600):
            return dt
    return video_dt or log_dt


def _choose_video_file(case_dir: Path) -> Optional[Path]:
    video_dir = case_dir / "video"
    if not video_dir.exists():
        return None
    candidates = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")))
    return candidates[0] if candidates else None


def _video_end_time(video_path: Optional[Path], recording_start: Optional[datetime]) -> Optional[datetime]:
    if not video_path or not recording_start:
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        finally:
            cap.release()
        if fps <= 0 or frames <= 0:
            return None
        return recording_start + timedelta(seconds=float(frames) / float(fps))
    except Exception:
        return None


def _sensitive_token_in_text(text: str, sensitive_files: List[str]) -> bool:
    lowered = str(text or "").replace("\\", "/").lower()
    if not lowered:
        return False
    for path in sensitive_files:
        normalized = str(path or "").replace("\\", "/").lower().strip()
        if not normalized:
            continue
        name = normalized.rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0] if "." in name else name
        tokens = [normalized, name]
        if len(stem) >= 2:
            tokens.append(stem)
        for token in tokens:
            if token and token in lowered:
                return True
    return False


def _event_app_label(event: Dict[str, Any]) -> str:
    window_info = event.get("window_info")
    process_info = event.get("process_info")
    parts = [event.get("app_name", "")]
    if isinstance(window_info, dict):
        parts.append(window_info.get("window_title", ""))
    if isinstance(process_info, dict):
        parts.append(process_info.get("process_name", ""))
    return " ".join(str(part or "") for part in parts)


def _is_external_sink_event(event: Dict[str, Any]) -> bool:
    text = f"{_event_text(event)} {_event_app_label(event)}".lower()
    if any(token.lower() in text for token in EXTERNAL_SINK_TOKENS):
        return True
    app_label = _event_app_label(event).lower()
    if not app_label:
        return False
    if any(token.lower() in app_label for token in DEFAULT_WHITELIST_APPS):
        return False
    return any(token.lower() in text for token in ATTEMPT_ACTION_TOKENS)


def _terminal_status_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    if any(token.lower() in lowered for token in TERMINAL_FAILURE_TOKENS):
        return "failed"
    if any(token.lower() in lowered for token in TERMINAL_CANCEL_TOKENS):
        return "canceled"
    if any(token.lower() in lowered for token in TERMINAL_SUCCESS_TOKENS):
        return "completed"
    return ""


def _build_sink_sessions(
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    recording_start: Optional[datetime] = None,
    recording_end: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source, rows in (("candidate_event", fallback_meta.get("candidate_events", []) or []), ("log", logs)):
        for row in rows:
            if not isinstance(row, dict):
                continue
            dt = _parse_dt(row.get("timestamp", ""))
            if not dt:
                continue
            if recording_start and dt < recording_start:
                continue
            if recording_end and dt > recording_end:
                continue
            key = (dt.strftime("%Y-%m-%d %H:%M:%S.%f"), str(row.get("event_type", "")), _event_text(row)[:160])
            if key in seen:
                continue
            seen.add(key)
            item = dict(row)
            item["_dt"] = dt
            item["_source"] = source
            item["_text"] = _event_text(row)
            item["_app_label"] = _event_app_label(row)
            events.append(item)
    events.sort(key=lambda item: item["_dt"])
    if not events or not sensitive_files:
        return []

    log_end = max((item["_dt"] for item in events), default=None)
    hard_recording_end = recording_end or log_end
    sessions: List[Dict[str, Any]] = []

    for event in events:
        start = event["_dt"]
        text = f"{event['_text']} {event['_app_label']}"
        if not _sensitive_token_in_text(text, sensitive_files):
            continue
        if not (_is_external_sink_event(event) or any(token.lower() in text.lower() for token in ATTEMPT_ACTION_TOKENS)):
            continue

        hard_end = start + timedelta(seconds=SINK_SESSION_MAX_SECONDS)
        if hard_recording_end:
            hard_end = min(hard_end, hard_recording_end)
        min_end = start + timedelta(seconds=SINK_SESSION_MIN_SECONDS)
        last_activity = start
        terminal_time: Optional[datetime] = None
        terminal_status = ""
        terminal_source = ""
        active_apps = {str(event.get("_app_label", "") or "").strip()}
        matched_events: List[Dict[str, Any]] = []

        for later in events:
            dt = later["_dt"]
            if dt < start or dt > hard_end:
                continue
            later_text = f"{later['_text']} {later['_app_label']}"
            external = _is_external_sink_event(later)
            sensitive = _sensitive_token_in_text(later_text, sensitive_files)
            status = _terminal_status_from_text(later_text)
            if external or sensitive or status:
                last_activity = dt
                if later.get("_app_label"):
                    active_apps.add(str(later["_app_label"]))
                matched_events.append(later)
            if status and (external or sensitive or dt <= start + timedelta(seconds=120)):
                terminal_status = status
                terminal_time = dt
                terminal_source = str(later.get("_source", "log"))
                break

        if terminal_time:
            end = min(hard_end, terminal_time + timedelta(seconds=10))
            tracking_end = end
        else:
            idle_end = last_activity + timedelta(seconds=SINK_SESSION_IDLE_SECONDS)
            end = max(min_end, min(hard_end, idle_end))
            tracking_end = hard_end
        if hard_recording_end:
            end = min(end, hard_recording_end)
            tracking_end = min(tracking_end, hard_recording_end)
        if end < start:
            end = start
        if tracking_end < end:
            tracking_end = end

        sessions.append(
            {
                "session_id": f"sink_session_{len(sessions) + 1:02d}",
                "start": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": end.strftime("%Y-%m-%d %H:%M:%S"),
                "start_event_type": str(event.get("event_type", "") or ""),
                "start_source": str(event.get("_source", "")),
                "terminal_status": terminal_status or "unknown",
                "terminal_time": terminal_time.strftime("%Y-%m-%d %H:%M:%S") if terminal_time else "",
                "terminal_source": terminal_source,
                "tracking_end": tracking_end.strftime("%Y-%m-%d %H:%M:%S"),
                "tracking_duration_seconds": round((tracking_end - start).total_seconds(), 3),
                "active_external_apps": sorted({item for item in active_apps if item})[:12],
                "matched_events": len(matched_events),
                "tracking_reason": "sensitive_object_entered_external_sink",
            }
        )

    return _merge_sink_sessions(sessions)


def _merge_sink_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed = []
    for session in sessions:
        start = _parse_dt(session.get("start", ""))
        end = _parse_dt(session.get("end", ""))
        if start and end:
            parsed.append((start, end, dict(session)))
    if not parsed:
        return []

    merged: List[Dict[str, Any]] = []
    for start, end, session in sorted(parsed, key=lambda item: item[0]):
        if not merged:
            merged.append(session)
            continue
        prev = merged[-1]
        prev_start = _parse_dt(prev.get("start", "")) or start
        prev_end = _parse_dt(prev.get("end", "")) or end
        prev_tracking_end = _parse_dt(prev.get("tracking_end", "")) or prev_end
        session_tracking_end = _parse_dt(session.get("tracking_end", "")) or end
        if start <= prev_end + timedelta(seconds=10):
            new_end = max(prev_end, end)
            new_tracking_end = max(prev_tracking_end, session_tracking_end)
            prev["end"] = new_end.strftime("%Y-%m-%d %H:%M:%S")
            prev["tracking_end"] = new_tracking_end.strftime("%Y-%m-%d %H:%M:%S")
            prev["matched_events"] = int(prev.get("matched_events", 0) or 0) + int(session.get("matched_events", 0) or 0)
            apps = set(prev.get("active_external_apps", []) or []) | set(session.get("active_external_apps", []) or [])
            prev["active_external_apps"] = sorted(apps)[:12]
            if prev.get("terminal_status") == "unknown" and session.get("terminal_status") != "unknown":
                prev["terminal_status"] = session.get("terminal_status", "unknown")
                prev["terminal_time"] = session.get("terminal_time", "")
                prev["terminal_source"] = session.get("terminal_source", "")
            prev["duration_seconds"] = round((new_end - prev_start).total_seconds(), 3)
            prev["tracking_duration_seconds"] = round((new_tracking_end - prev_start).total_seconds(), 3)
            continue
        merged.append(session)

    for index, session in enumerate(merged, 1):
        start = _parse_dt(session.get("start", ""))
        end = _parse_dt(session.get("end", ""))
        tracking_end = _parse_dt(session.get("tracking_end", "")) or end
        session["session_id"] = f"sink_session_{index:02d}"
        if start and end:
            session["duration_seconds"] = round((end - start).total_seconds(), 3)
        if start and tracking_end:
            session["tracking_duration_seconds"] = round((tracking_end - start).total_seconds(), 3)
    return merged


def _sink_session_windows(sessions: List[Dict[str, Any]]) -> List[Tuple[datetime, datetime]]:
    windows: List[Tuple[datetime, datetime]] = []
    for session in sessions:
        start = _parse_dt(session.get("start", ""))
        end = _parse_dt(session.get("tracking_end", "")) or _parse_dt(session.get("end", ""))
        if start and end and end >= start:
            windows.append((start, end))
    return windows


def _sessions_for_segment(sessions: List[Dict[str, Any]], segment: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
    start, end = segment
    result: List[Dict[str, Any]] = []
    for session in sessions:
        session_start = _parse_dt(session.get("start", ""))
        session_end = _parse_dt(session.get("tracking_end", "")) or _parse_dt(session.get("end", ""))
        if not session_start or not session_end:
            continue
        if session_start <= end and session_end >= start:
            result.append(session)
    return result


def _sink_session_anchor_times(fallback_meta: Dict[str, Any]) -> List[Tuple[datetime, str]]:
    anchors: List[Tuple[datetime, str]] = []

    def add_offsets(base: Optional[datetime], session_id: str, label: str, offsets: Tuple[int, ...]) -> None:
        if not base:
            return
        for offset in offsets:
            suffix = "" if offset == 0 else ("plus" if offset > 0 else "minus") + str(abs(offset))
            reason = f"{session_id}:{label}" if offset == 0 else f"{session_id}:{label}_{suffix}"
            anchors.append((base + timedelta(seconds=offset), reason))

    for session in fallback_meta.get("sink_sessions", []) or []:
        if not isinstance(session, dict):
            continue
        session_id = str(session.get("session_id", "sink_session"))
        start = _parse_dt(session.get("start", ""))
        end = _parse_dt(session.get("end", ""))
        tracking_end = _parse_dt(session.get("tracking_end", "")) or end
        terminal = _parse_dt(session.get("terminal_time", ""))
        terminal_status = str(session.get("terminal_status", "unknown") or "unknown")
        add_offsets(start, session_id, "session_start", (0, 3, 8))
        add_offsets(terminal, session_id, f"terminal_{terminal_status}", (-2, 0, 2, 5, 8, 12))
        if terminal_status in {"", "unknown"}:
            add_offsets(end, session_id, "session_end", (-3, 0, 2, 5, 8, 12, 20, 35, 60))
            add_offsets(tracking_end, session_id, "tracking_end", (-60, -20, -5, 0))
        else:
            add_offsets(end, session_id, "session_end", (-3, 0, 2, 5, 8, 12, 20))
        if start and tracking_end and terminal_status in {"", "unknown"}:
            for offset in SINK_SESSION_TRACKING_OFFSETS_SECONDS:
                dt = start + timedelta(seconds=offset)
                if start < dt < tracking_end:
                    anchors.append((dt, f"{session_id}:session_tracking_plus{offset}"))
        if start and end:
            cursor = start + timedelta(seconds=SINK_SESSION_HEARTBEAT_SECONDS)
            while cursor < end:
                anchors.append((cursor, f"{session_id}:session_heartbeat"))
                cursor += timedelta(seconds=SINK_SESSION_HEARTBEAT_SECONDS)
    return sorted(anchors, key=lambda item: item[0])


def _windows_from_fallback(fallback_meta: Dict[str, Any], logs: List[Dict[str, Any]]) -> List[Tuple[datetime, datetime]]:
    windows = []
    for start, end in _sink_session_windows([item for item in fallback_meta.get("sink_sessions", []) or [] if isinstance(item, dict)]):
        windows.append((start, end))
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
    strong_transfer_hits = 0
    sink_session_hits = 0
    matched_tokens = set()

    for anchor_time, anchor_reason in _sink_session_anchor_times(fallback_meta):
        if padded_start <= anchor_time <= padded_end:
            sink_session_hits += 1
            score += 32.0
            matched_tokens.add(anchor_reason)

    candidate_events = fallback_meta.get("candidate_events", []) or []
    for event in candidate_events:
        dt = _parse_dt(event.get("timestamp", ""))
        if not dt or not (padded_start <= dt <= padded_end):
            continue
        hit_events += 1
        score += 2.0
        text = _event_text(event).lower()
        if any(token.lower() in text for token in TRANSFER_WINDOW_TOKENS):
            strong_transfer_hits += 1
            score += 64.0
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
        if any(token.lower() in text for token in TRANSFER_WINDOW_TOKENS):
            strong_transfer_hits += 1
            score += 16.0
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
        "sink_session_hits": sink_session_hits,
        "strong_transfer_hits": strong_transfer_hits,
        "matched_tokens": sorted(str(token) for token in matched_tokens)[:12],
        "score": round(score, 3),
    }


def _prepare_review_segments(
    windows: List[Tuple[datetime, datetime]],
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    max_segments: Optional[int] = None,
) -> Tuple[List[Tuple[datetime, datetime]], Dict[str, Any]]:
    merge_gap = 8
    segment_seconds = SEGMENT_SECONDS
    overlap_seconds = min(segment_seconds - 1, SEGMENT_OVERLAP_SECONDS)
    max_segments = max(1, min(MAX_SEGMENTS_PER_CASE, int(max_segments or MAX_SEGMENTS_PER_CASE)))

    merged = _merge_review_windows(windows, merge_gap)
    split_segments: List[Tuple[datetime, datetime]] = []
    for start, end in merged:
        split_segments.extend(_split_review_window(start, end, segment_seconds, overlap_seconds))

    scored = []
    for segment in split_segments:
        score, meta = _segment_signal_score(segment, fallback_meta, logs)
        scored.append((score, segment, meta))
    if scored:
        protected: List[Tuple[float, Tuple[datetime, datetime], Dict[str, Any]]] = []
        session_anchors = [item[0] for item in _sink_session_anchor_times(fallback_meta)]
        for item in scored:
            _, segment, _ = item
            start, end = segment
            if any(start <= anchor <= end for anchor in session_anchors):
                protected.append(item)

        selected: List[Tuple[float, Tuple[datetime, datetime], Dict[str, Any]]] = []
        seen_segments: set[Tuple[datetime, datetime]] = set()
        if protected:
            protected_by_time = sorted(protected, key=lambda value: value[1][0])
            protected_quota = min(max_segments, len(protected_by_time))
            if protected_quota == 1:
                protected_seed = [max(protected_by_time, key=lambda value: value[0])]
            else:
                protected_seed = []
                for pos in range(protected_quota):
                    idx = round(pos * (len(protected_by_time) - 1) / max(1, protected_quota - 1))
                    protected_seed.append(protected_by_time[idx])
            for item in protected_seed:
                key = item[1]
                if key in seen_segments:
                    continue
                selected.append(item)
                seen_segments.add(key)
                if len(selected) >= max_segments:
                    break
            for item in sorted(protected, key=lambda value: value[0], reverse=True):
                if len(selected) >= max_segments:
                    break
                key = item[1]
                if key in seen_segments:
                    continue
                selected.append(item)
                seen_segments.add(key)
        for item in sorted(scored, key=lambda value: value[0], reverse=True):
            if len(selected) >= max_segments:
                break
            key = item[1]
            if key in seen_segments:
                continue
            selected.append(item)
            seen_segments.add(key)

        kept = selected[:max_segments]
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


def _ocr_text_is_monitor_ui(text: str) -> bool:
    lowered = str(text or "").lower()
    compact = _compact_ocr_text(text)
    return any(token.lower() in lowered or _compact_ocr_text(token) in compact for token in MONITOR_UI_TOKENS)


def _ocr_reader(allow_load: bool = True) -> Any:
    global _OCR_READER, _OCR_READER_FAILED
    if _OCR_READER or _OCR_READER_FAILED:
        return _OCR_READER
    if not allow_load:
        return None
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
        with _RAPID_OCR_INFER_LOCK:
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
    reader = _ocr_reader(allow_load=_ocr_prewarm_enabled())
    if not reader:
        return ""
    try:
        with _OCR_INFER_LOCK:
            results = reader.readtext(frame, detail=0, paragraph=False)
    except Exception:
        return ""
    return " ".join(str(item).strip() for item in results if str(item).strip())


def _ocr_risk_flags(text: str, sensitive_files: List[str]) -> List[str]:
    compact = _compact_ocr_text(text)
    flags: List[str] = []
    if not compact:
        return flags
    monitor_ui = _ocr_text_is_monitor_ui(text)
    if not monitor_ui and any(_compact_ocr_text(token) in compact for token in COMPLETION_OCR_TOKENS):
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


def _ocr_prewarm_enabled() -> bool:
    value = os.getenv("DLD_VLM_OCR_PREWARM", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _warm_ocr_reader() -> None:
    if not _ocr_prefilter_enabled():
        return
    engine = _ocr_engine_name()
    if engine == "easyocr":
        _ocr_reader()
    elif engine == "rapidocr":
        _rapid_ocr_reader()


def _compact_log_text_for_gate(log: Dict[str, Any]) -> str:
    window_info = log.get("window_info")
    process_info = log.get("process_info")
    return " ".join(
        str(part or "")
        for part in (
            log.get("event_type", ""),
            log.get("app_name", ""),
            log.get("file_path", ""),
            log.get("file_name", ""),
            log.get("content_preview", ""),
            process_info.get("process_name", "") if isinstance(process_info, dict) else "",
            window_info.get("window_title", "") if isinstance(window_info, dict) else "",
        )
    ).lower()


def _contains_gate_token(text: str, tokens: Iterable[str]) -> bool:
    padded = f" {text.lower()} "
    return any(token in padded for token in tokens)


def _sensitive_signal_present(text: str, detection: Dict[str, Any]) -> bool:
    compact_text = _compact_ocr_text(text)
    if not compact_text:
        return False
    for record in detection.get("operation_records", [])[:128]:
        for key in ("sensitive_file_path", "source_path", "target_path", "description"):
            value = str(record.get(key, "") or "")
            for token in (value, Path(value).name, Path(value).stem):
                compact = _compact_ocr_text(token)
                if compact and len(compact) >= 3 and compact in compact_text:
                    return True
    return False


def _vlm_gate_features(
    logs: List[Dict[str, Any]],
    detection: Dict[str, Any],
    fallback_meta: Dict[str, Any],
) -> Dict[str, Any]:
    event_types = {
        str(log.get("event_type", "") or "").lower()
        for log in logs
    }
    candidate_events = fallback_meta.get("candidate_events", []) or []
    for event in candidate_events:
        event_types.add(str(event.get("event_type", "") or "").lower())

    text_parts = [_compact_log_text_for_gate(log) for log in logs]
    for event in candidate_events[:48]:
        text_parts.extend(
            str(event.get(key, "") or "").lower()
            for key in ("event_type", "app_name", "file_name", "content_preview", "window_title")
        )
    for record in detection.get("operation_records", [])[:64]:
        text_parts.extend(
            str(record.get(key, "") or "").lower()
            for key in ("operation", "app_name", "description", "sensitive_file_path")
        )
    signal_text = " ".join(text_parts)

    explicit_transfer_events = {
        "data_upload",
        "file_send",
        "file_share",
        "screen_share_start",
        "screen_recording_started",
    }
    direct_visual_events = {
        "screenshot_capture",
        "clipboard_image",
    }
    selection_events = {
        "file_selected",
        "clipboard_text",
        "clipboard_copy",
        "clipboard_image",
    }

    features = {
        "explicit_transfer_event": bool(event_types & explicit_transfer_events),
        "direct_visual_capture_event": bool(event_types & direct_visual_events),
        "selection_or_clipboard_event": bool(event_types & selection_events),
        "completion_text": _contains_gate_token(signal_text, COMPLETION_OCR_TOKENS),
        "preliminary_text": _contains_gate_token(signal_text, PRELIMINARY_OCR_TOKENS),
        "sensitive_context": _sensitive_signal_present(signal_text, detection),
        "external_sink_context": _contains_gate_token(signal_text, EXTERNAL_SINK_TOKENS),
        "attempt_action_context": _contains_gate_token(signal_text, ATTEMPT_ACTION_TOKENS),
        "screenshot_context": _contains_gate_token(
            signal_text,
            ("screenshot", "screen capture", "snipping", "snipaste", "\u622a\u56fe", "\u622a\u5c4f"),
        ),
        "export_context": _contains_gate_token(signal_text, ("export", "\u5bfc\u51fa")),
        "vm_context": _contains_gate_token(
            signal_text,
            ("vmware", "virtualbox", "hyper-v", "virtual machine", "ubuntu - vmware", "openeuler", "\u865a\u62df\u673a"),
        ),
        "remote_context": _contains_gate_token(
            signal_text,
            ("remote desktop", "mstsc", "anydesk", "todesk", "sunlogin", "\u8fdc\u7a0b\u684c\u9762"),
        ),
        "external_cloud_or_mail_context": _contains_gate_token(
            signal_text,
            (
                "gmail",
                "outlook",
                "proton",
                "qqmail",
                "mail",
                "email",
                "dropbox",
                "onedrive",
                "google drive",
                "baidu",
                "weiyun",
                "quark",
                "\u90ae\u7bb1",
                "\u9644\u4ef6",
                "\u4e91\u76d8",
            ),
        ),
        "git_context": _contains_gate_token(signal_text, ("github", "gitlab", "gitee", "bitbucket", "commit", "merge")),
        "archive_or_convert_context": _contains_gate_token(
            signal_text,
            ("zip", "compress", "archive", "convert", "pdf", "\u538b\u7f29", "\u8f6c\u6362"),
        ),
        "candidate_events": len(candidate_events),
        "event_types": sorted(item for item in event_types if item)[:64],
    }
    return features


def _local_vlm_gate_decision(
    *,
    mode: str,
    logs: List[Dict[str, Any]],
    detection: Dict[str, Any],
    fallback_meta: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_mode = str(mode or "all").strip().lower()
    if normalized_mode not in {"all", "strict", "adaptive", "aggressive"}:
        normalized_mode = "all"

    features = _vlm_gate_features(logs, detection, fallback_meta)
    decision = {
        "mode": normalized_mode,
        "action": "queue_remote_vlm",
        "local_verdict": None,
        "reason": "remote_vlm_required",
        "features": features,
    }
    if normalized_mode == "all":
        decision["reason"] = "gate_disabled"
        return decision

    positive_reasons: List[str] = []
    sensitive_context = bool(features.get("sensitive_context"))
    external_sink_context = bool(features.get("external_sink_context"))
    if features["explicit_transfer_event"] and sensitive_context:
        positive_reasons.append("explicit_transfer_event")
    # Screenshot/visual-capture contexts are deliberately NOT local-positive
    # shortcuts anymore. Text-only contexts misfire on cancelled uploads, so
    # those cases must go to the remote VLM instead.
    if features["export_context"] and sensitive_context and external_sink_context:
        positive_reasons.append("export_context")

    if normalized_mode in {"adaptive", "aggressive"}:
        if (
            features["vm_context"]
            and sensitive_context
            and not features["preliminary_text"]
            and not features["external_cloud_or_mail_context"]
        ):
            positive_reasons.append("vm_context")

    if normalized_mode == "aggressive":
        if features["git_context"] and features["completion_text"] and sensitive_context:
            positive_reasons.append("git_context_with_completion_text")
        if features["archive_or_convert_context"] and features["completion_text"] and sensitive_context:
            positive_reasons.append("archive_or_convert_with_completion_text")

    if positive_reasons:
        decision["action"] = "queue_remote_vlm"
        decision["local_verdict"] = False
        decision["reason"] = "local_features_require_remote_vlm:" + ",".join(sorted(set(positive_reasons)))
    return decision


VLM_REVIEW_CACHE_VERSION = "v8-sink-session-tracking"


def _bool_env(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _vlm_review_cache_dir() -> Path:
    value = os.getenv("DLD_VLM_REVIEW_CACHE_DIR", "").strip()
    return Path(value) if value else REPO_ROOT / "spec" / "output" / "cache" / "vlm_reviews"


def _path_fingerprint(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    try:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError:
        return {"path": str(path), "exists": False}


def _stable_digest(payload: Dict[str, Any], length: int = 24) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _vlm_review_cache_path(
    *,
    case_dir: Path,
    video_path: Optional[Path],
    rec_start: Optional[datetime],
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    max_frames: int,
    model: str,
    base_url: str,
) -> Path:
    review_logs = _review_log_context(logs, _windows_from_fallback(fallback_meta, logs))
    cache_payload = {
        "cache_version": VLM_REVIEW_CACHE_VERSION,
        "case": str(case_dir.resolve()),
        "video": _path_fingerprint(video_path),
        "recording_start": rec_start.strftime("%Y-%m-%d %H:%M:%S") if rec_start else "",
        "fallback": {
            "reasons": fallback_meta.get("reasons", []),
            "candidate_events": fallback_meta.get("candidate_events", [])[:24],
            "analysis_windows": fallback_meta.get("analysis_windows", []),
        },
        "review_logs": review_logs,
        "sensitive_files": sorted(str(item) for item in sensitive_files),
        "max_frames": int(max_frames),
        "model": model,
        "base_url": base_url,
        "params": {
            "segment_seconds": SEGMENT_SECONDS,
            "segment_overlap_seconds": SEGMENT_OVERLAP_SECONDS,
            "frames_per_segment": FRAMES_PER_SEGMENT,
            "candidate_frames_per_segment": CANDIDATE_FRAMES_PER_SEGMENT,
            "max_segments_per_case": MAX_SEGMENTS_PER_CASE,
            "max_image_frames_per_segment": MAX_IMAGE_FRAMES_PER_SEGMENT,
            "max_ocr_frames_per_segment": MAX_OCR_FRAMES_PER_SEGMENT,
            "min_frame_gap": MIN_FRAME_GAP,
            "image_scene_threshold": IMAGE_SCENE_THRESHOLD,
            "status_region_threshold": STATUS_REGION_THRESHOLD,
            "sink_session_idle_seconds": SINK_SESSION_IDLE_SECONDS,
            "sink_session_max_seconds": SINK_SESSION_MAX_SECONDS,
            "sink_session_heartbeat_seconds": SINK_SESSION_HEARTBEAT_SECONDS,
            "image_max_edge": IMAGE_MAX_EDGE,
            "jpeg_quality": JPEG_QUALITY,
            "ocr_prefilter": os.getenv("DLD_VLM_ENABLE_OCR_PREFILTER", ""),
            "ocr_engine": _ocr_engine_name(),
        },
    }
    return _vlm_review_cache_dir() / f"{_stable_digest(cache_payload)}.json"


def _read_vlm_review_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    if not _bool_env("DLD_VLM_REVIEW_CACHE", True):
        return None
    try:
        if not cache_path.exists():
            return None
        with cache_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        verdict = cached.get("verdict") if isinstance(cached, dict) else None
        if not isinstance(verdict, dict):
            return None
        result = dict(verdict)
        result["cache_hit"] = True
        result["cache_path"] = str(cache_path)
        return result
    except Exception as exc:
        _progress(f"[VLM CACHE] read_failed path={cache_path} error={type(exc).__name__}:{exc}")
        return None


def _write_vlm_review_cache(cache_path: Path, verdict: Dict[str, Any]) -> None:
    if not _bool_env("DLD_VLM_REVIEW_CACHE", True):
        return
    if _vlm_missing_api_key(verdict):
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": VLM_REVIEW_CACHE_VERSION,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verdict": verdict,
        }
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(cache_path)
    except Exception as exc:
        _progress(f"[VLM CACHE] write_failed path={cache_path} error={type(exc).__name__}:{exc}")


def _vlm_missing_api_key(verdict: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(verdict, dict):
        return False
    if str(verdict.get("reason", "")) == "missing_vlm_api_key":
        return True
    segment_verdicts = [item for item in verdict.get("segment_verdicts", []) or [] if isinstance(item, dict)]
    return bool(segment_verdicts) and all(str(item.get("reason", "")) == "missing_vlm_api_key" for item in segment_verdicts)


def _adaptive_vlm_frame_budget(
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    max_frames: int,
) -> Tuple[int, Dict[str, Any]]:
    cap = max(1, int(max_frames))
    # max_frames is an emergency cap. The planner should converge on a small
    # evidence package instead of treating the cap as the normal target.
    min_frames = min(cap, max(4, FRAMES_PER_SEGMENT // 2))
    base_frames = min(cap, max(min_frames, FRAMES_PER_SEGMENT))
    frames = base_frames
    reasons: List[str] = []

    windows = _windows_from_fallback(fallback_meta, logs)
    planning_segments = max(1, min(MAX_SEGMENTS_PER_CASE, max(2, min(4, cap // max(1, FRAMES_PER_SEGMENT // 2)))))
    review_segments, segment_meta = _prepare_review_segments(windows, fallback_meta, logs, max_segments=planning_segments)
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
        bump(2, "multiple_review_segments")
    if len(review_segments) >= 4:
        bump(2, "many_review_segments")
    if total_window_seconds >= 300:
        bump(3, "very_long_total_review_window")
    elif total_window_seconds >= 120:
        bump(2, "long_total_review_window")
    elif total_window_seconds >= 60:
        bump(1, "medium_total_review_window")

    sink_sessions = [item for item in fallback_meta.get("sink_sessions", []) or [] if isinstance(item, dict)]
    if sink_sessions:
        bump(2, "sink_session_tracking")
    if any(float(item.get("duration_seconds", 0.0) or 0.0) >= 600 for item in sink_sessions):
        bump(3, "very_long_sink_session")
    elif any(float(item.get("duration_seconds", 0.0) or 0.0) >= 180 for item in sink_sessions):
        bump(1, "long_sink_session")
    if any(str(item.get("terminal_status", "") or "") in {"", "unknown"} for item in sink_sessions):
        bump(1, "sink_session_terminal_unknown")
    if any(str(item.get("terminal_status", "") or "") in {"completed", "failed", "canceled"} for item in sink_sessions):
        bump(1, "sink_session_terminal")

    candidate_events = fallback_meta.get("candidate_events", []) or []
    if len(candidate_events) >= 8:
        bump(1, "many_candidate_events")

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

    soft_target = min(cap, max(8, FRAMES_PER_SEGMENT + 8))
    if max_window_seconds >= 600 or any(
        float(item.get("duration_seconds", 0.0) or 0.0) >= 600
        for item in sink_sessions
    ):
        soft_target = min(cap, soft_target + 4)
    if any(reason in reasons for reason in ("remote_or_vm_context", "meeting_or_screen_share_context")):
        soft_target = min(cap, soft_target + 2)
    if frames > soft_target:
        frames = soft_target
        reasons.append("soft_target_clamped")

    return frames, {
        "adaptive": True,
        "min_frames": min_frames,
        "base_frames": base_frames,
        "max_frames": cap,
        "selected_frames": frames,
        "frames_per_segment": FRAMES_PER_SEGMENT,
        "planning_segments": planning_segments,
        "window_count": len(windows),
        "segment_count": len(review_segments),
        "total_window_seconds": round(total_window_seconds, 3),
        "max_window_seconds": round(max_window_seconds, 3),
        "sink_sessions": sink_sessions,
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


def _sample_frame_times(windows: List[Tuple[datetime, datetime]], max_frames: int) -> List[Tuple[datetime, str]]:
    if max_frames <= 0:
        return []
    points: List[Tuple[datetime, str]] = []
    per_window = max(1, max_frames // max(1, len(windows)))
    for start, end in windows:
        duration = max(0.0, (end - start).total_seconds())
        count = min(per_window, max_frames - len(points))
        if count <= 0:
            break
        if count == 1 or duration == 0:
            points.append((start + timedelta(seconds=duration / 2), "sample_mid"))
        else:
            for idx in range(count):
                points.append((start + timedelta(seconds=duration * idx / (count - 1)), "sample"))
    return points[:max_frames]


def _event_anchor_times(
    windows: List[Tuple[datetime, datetime]],
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    max_points: int,
) -> List[Tuple[datetime, str]]:
    if max_points <= 0:
        return []

    padded_windows = [
        (start - timedelta(seconds=5), end + timedelta(seconds=60))
        for start, end in windows
    ]

    def in_window(dt: datetime) -> bool:
        return not padded_windows or any(start <= dt <= end for start, end in padded_windows)

    events: List[Dict[str, Any]] = []
    events.extend(item for item in fallback_meta.get("candidate_events", []) or [] if isinstance(item, dict))
    events.extend(item for item in logs if isinstance(item, dict))

    anchors: List[Tuple[datetime, str]] = []
    for event in events:
        dt = _parse_dt(event.get("timestamp", ""))
        if not dt or not in_window(dt):
            continue
        text = _event_text(event).lower()
        if not any(token.lower() in text for token in ATTEMPT_ACTION_TOKENS + COMPLETION_OCR_TOKENS + EXTERNAL_SINK_TOKENS):
            continue

        event_type = str(event.get("event_type", "") or "").lower()
        if any(token in text for token in ("upload", "\u4e0a\u4f20", "send", "\u53d1\u9001", "attach", "\u9644\u4ef6", "commit")):
            offsets = (-3, 0, 3, 5, 8, 12, 18, 25, 35, 45)
            reason = f"event_anchor_transfer:{event_type or 'unknown'}"
        elif any(token in text for token in ("file_selected", "clipboard", "paste", "\u526a\u8d34", "\u590d\u5236", "\u7c98\u8d34")):
            offsets = (0, 3, 8, 12, 15, 19, 25, 30, 45)
            reason = f"event_anchor_precursor:{event_type or 'unknown'}"
        else:
            offsets = (0, 8, 20)
            reason = f"event_anchor_context:{event_type or 'unknown'}"

        for offset in offsets:
            anchors.append((dt + timedelta(seconds=offset), reason))
            if len(anchors) >= max_points:
                return anchors
    return anchors[:max_points]


def _frame_time_candidates(
    windows: List[Tuple[datetime, datetime]],
    max_frames: int,
    max_candidates: int,
    fallback_meta: Optional[Dict[str, Any]] = None,
    logs: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[datetime, str]]:
    if max_frames <= 0:
        return []
    if not windows:
        return []

    budget = max(max_frames, max_candidates)
    durations = [max(0.0, (end - start).total_seconds()) for start, end in windows]
    total_duration = sum(durations) or float(len(windows))
    def anchor_priority(item: Tuple[datetime, str]) -> Tuple[int, datetime]:
        reason = item[1]
        if "terminal_" in reason:
            return (0, item[0])
        if "tracking_end" in reason:
            return (1, item[0])
        if "session_tracking" in reason:
            return (2, item[0])
        if "session_end" in reason:
            return (3, item[0])
        if "session_start" in reason:
            return (4, item[0])
        if "session_heartbeat" in reason:
            return (6, item[0])
        return (5, item[0])

    session_anchors = sorted([
        (dt, reason)
        for dt, reason in _sink_session_anchor_times(fallback_meta or {})
        if any(start - timedelta(seconds=5) <= dt <= end + timedelta(seconds=75) for start, end in windows)
    ], key=anchor_priority)
    points: List[Tuple[datetime, str]] = [
        (dt, f"event_anchor_{reason}")
        for dt, reason in session_anchors[: max(1, int(budget * 0.75))]
    ]
    points.extend(
        _event_anchor_times(
            windows,
            fallback_meta or {},
            logs or [],
            max_points=max(0, min(budget, max(1, int(budget * 0.75))) - len(points)),
        )
    )

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
            points.append((start + timedelta(seconds=duration / 2), "window_mid_candidate"))
            continue
        for pos in range(count):
            ratio = pos / (count - 1)
            points.append((start + timedelta(seconds=duration * ratio), "window_candidate"))

    deduped: List[Tuple[datetime, str]] = []
    seen: Dict[str, str] = {}
    for dt, reason in points:
        key = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if key in seen:
            if reason not in seen[key]:
                seen[key] = f"{seen[key]}+{reason}"
            continue
        seen[key] = reason
        deduped.append((dt, reason))
        if len(deduped) >= budget:
            break
    return sorted(deduped, key=lambda item: item[0])


def _thumbnail_scene_score(frame: Any, previous_thumb: Any) -> tuple[float, Any]:
    import cv2

    thumb = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    if previous_thumb is None:
        return 1.0, gray
    score = float(cv2.absdiff(gray, previous_thumb).mean()) / 255.0
    return score, gray


def _status_region_visual_score(frame: Any, previous_thumb: Any) -> tuple[float, Any]:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    y1 = int(height * 0.55)
    y2 = int(height * 0.93)
    x1 = 0
    x2 = int(width * 0.62)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0, previous_thumb

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (128, 48), interpolation=cv2.INTER_AREA)
    diff_score = 0.0
    if previous_thumb is not None:
        diff_score = float(cv2.absdiff(thumb, previous_thumb).mean()) / 255.0

    edges = cv2.Canny(thumb, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)
    dark_ratio = float(np.count_nonzero(thumb < 90)) / float(thumb.size or 1)
    bright_ratio = float(np.count_nonzero(thumb > 170)) / float(thumb.size or 1)
    contrast = float(thumb.std()) / 128.0

    toast_like = 0.0
    if dark_ratio >= 0.08 and bright_ratio >= 0.02 and edge_density >= 0.035:
        toast_like = 0.08 + min(0.12, dark_ratio * 0.3) + min(0.08, bright_ratio * 0.6)

    score = min(1.0, diff_score * 3.0 + edge_density * 1.8 + contrast * 0.35 + toast_like)
    return score, thumb


def _select_representative_frames(candidates: List[Dict[str, Any]], max_frames: int) -> List[Dict[str, Any]]:
    if max_frames <= 0 or not candidates:
        return []
    if len(candidates) <= max_frames:
        for item in candidates:
            item["selection_reason"] = item.get("selection_reason") or item.get("selection_hint") or "candidate"
        return candidates

    selected: Dict[int, Dict[str, Any]] = {}

    def is_critical_anchor(item: Dict[str, Any]) -> bool:
        hint = str(item.get("selection_hint", "") or "")
        return any(
            token in hint
            for token in (
                "terminal_",
                "tracking_end",
                "session_tracking",
                "session_end",
                "session_start",
                "event_anchor_transfer",
            )
        )

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

    def critical_priority(item: Dict[str, Any]) -> Tuple[int, int]:
        hint = str(item.get("selection_hint", "") or "")
        if "terminal_" in hint:
            rank = 0
        elif "tracking_end" in hint:
            rank = 1
        elif "session_tracking" in hint:
            rank = 2
        elif "session_end" in hint:
            rank = 3
        elif "event_anchor_transfer" in hint:
            rank = 4
        else:
            rank = 5
        return (rank, int(item.get("frame_index", 0)))

    critical_candidates = sorted(
        [item for item in candidates if is_critical_anchor(item)],
        key=critical_priority,
    )
    for item in critical_candidates[: max(1, min(max_frames, max_frames // 2 + 1))]:
        add(item, str(item.get("selection_hint") or "critical_anchor"))

    anchor_limit = min(max_frames, max(1, max_frames // 2))
    anchor_candidates = [
        item
        for item in candidates
        if "event_anchor" in str(item.get("selection_hint", ""))
    ]
    ordered_anchors = sorted(anchor_candidates, key=lambda value: int(value.get("frame_index", 0)))
    if len(ordered_anchors) > anchor_limit:
        anchor_slots = []
        for pos in range(anchor_limit):
            idx = round(pos * (len(ordered_anchors) - 1) / max(anchor_limit - 1, 1))
            anchor_slots.append(ordered_anchors[idx])
        ordered_anchors = anchor_slots
    for item in ordered_anchors[:anchor_limit]:
        add(item, str(item.get("selection_hint") or "event_anchor"))

    status_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.get("status_region_score", 0.0) or 0.0),
            float(item.get("scene_score", 0.0) or 0.0),
            int(item.get("frame_index", 0)),
        ),
        reverse=True,
    )
    for item in status_candidates[: max(1, max_frames // 4)]:
        if float(item.get("status_region_score", 0.0) or 0.0) >= STATUS_REGION_THRESHOLD:
            add(item, "status_region_candidate")

    add(candidates[0], "window_start")
    if len(candidates) > 2:
        add(candidates[len(candidates) // 2], "window_mid")
    add(candidates[-1], "window_end")

    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("scene_score", 0.0)), int(item.get("frame_index", 0))),
        reverse=True,
    )
    min_gap = MIN_FRAME_GAP
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

    max_image_frames = min(max_frames, MAX_IMAGE_FRAMES_PER_SEGMENT)
    min_image_frames = min(max_image_frames, min(2, max_image_frames))
    scene_threshold = IMAGE_SCENE_THRESHOLD
    max_ocr_frames = min(len(selected), MAX_OCR_FRAMES_PER_SEGMENT if ocr_enabled else 0)
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
        if "event_anchor" in reason:
            image_priority += 3.5
            image_reasons.append("event_anchor")
        if "completion_keyword" in flags or "sensitive_name_visible" in flags:
            image_priority += 4.0
            image_reasons.append("ocr_risk_hit")
        if scene_score >= scene_threshold:
            image_priority += 2.0 + min(scene_score, 1.0)
            image_reasons.append("scene_change")
        status_score = float(item.get("status_region_score", 0.0) or 0.0)
        if status_score >= STATUS_REGION_THRESHOLD:
            image_priority += 4.5 + min(status_score, 1.0) * 4.0
            if status_score >= 0.5:
                image_priority += 4.0
            image_reasons.append("status_region_candidate")
        if duplicate_ocr and image_priority < 4.0:
            image_priority -= 2.0
            image_reasons.append("ocr_duplicate")
        if not ocr_text:
            image_priority += 0.5
            image_reasons.append("ocr_not_run" if ocr_enabled and not should_ocr else "no_ocr_text")
        elif _ocr_text_is_monitor_ui(ocr_text) and "sensitive_name_visible" not in flags:
            image_priority -= 3.0
            image_reasons.append("monitor_ui_downrank")

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
    status_ranked = [
        item
        for item in sorted(
            selected,
            key=lambda value: float(value.get("status_region_score", 0.0) or 0.0),
            reverse=True,
        )
        if float(item.get("status_region_score", 0.0) or 0.0) >= STATUS_REGION_THRESHOLD
    ]
    for item in status_ranked[: max(1, max_image_frames // 3)]:
        image_frame_ids.add(int(item["frame_index"]))
    if len(image_frame_ids) > max_image_frames:
        protected = {int(item["frame_index"]) for item in status_ranked[: max(1, max_image_frames // 3)]}
        kept: List[int] = []
        for item in ranked:
            frame_id = int(item["frame_index"])
            if frame_id in image_frame_ids and frame_id not in kept:
                kept.append(frame_id)
            if len(kept) >= max_image_frames:
                break
        for frame_id in protected:
            if frame_id not in kept:
                if len(kept) >= max_image_frames:
                    kept[-1] = frame_id
                else:
                    kept.append(frame_id)
        image_frame_ids = set(kept[:max_image_frames])
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
        "status_region_score": round(float(item.get("status_region_score", 0.0)), 4),
        "selection_reason": item.get("selection_reason", ""),
        "b64": base64.b64encode(buffer).decode("ascii"),
    }


def _extract_representative_frame_images(
    video_path: Path,
    recording_start: datetime,
    segment: Tuple[datetime, datetime],
    segment_id: str,
    fallback_meta: Dict[str, Any],
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    max_frames: int,
    max_edge: int = 960,
    jpeg_quality: int = 65,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import cv2

    windows = [segment]
    candidate_budget = max(max_frames, CANDIDATE_FRAMES_PER_SEGMENT)
    frame_times = _frame_time_candidates(windows, max_frames, candidate_budget, fallback_meta, logs)
    if not frame_times:
        frame_times = _sample_frame_times(windows, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"segment_id": segment_id, "candidate_budget": candidate_budget, "error": "video_open_failed"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidates = []
    previous_thumb = None
    previous_status_thumb = None
    seen_frames = set()
    try:
        for dt, hint in frame_times:
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
            status_score, previous_status_thumb = _status_region_visual_score(frame, previous_status_thumb)
            candidates.append(
                {
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_index": frame_index,
                    "scene_score": scene_score,
                    "status_region_score": status_score,
                    "selection_hint": hint,
                    "segment_id": segment_id,
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
                "status_region_score": round(float(item.get("status_region_score", 0.0)), 4),
                "selection_reason": item.get("selection_reason", ""),
            }
        if encoded:
            encoded["image_sent"] = bool(item.get("image_sent"))
            encoded["segment_id"] = segment_id
            encoded["ocr_text"] = item.get("ocr_text", "")
            encoded["ocr_flags"] = item.get("ocr_flags", [])
            encoded["ocr_ran"] = bool(item.get("ocr_ran"))
            encoded["ocr_duplicate"] = bool(item.get("ocr_duplicate"))
            encoded["image_priority"] = item.get("image_priority", 0.0)
            encoded["image_decision_reasons"] = item.get("image_decision_reasons", [])
            images.append(encoded)
    return images, {
        "segment_id": segment_id,
        "segment_start": segment[0].strftime("%Y-%m-%d %H:%M:%S"),
        "segment_end": segment[1].strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_budget": candidate_budget,
        "candidate_frames": len(candidates),
        "selected_frames": len(selected),
        **prefilter_meta,
    }


def _build_vlm_review_prompt(
    sensitive_files: List[str],
    candidate_events: List[Dict[str, Any]],
    review_logs: List[Dict[str, Any]],
    sink_sessions: List[Dict[str, Any]],
    segment_id: str,
    segment: Tuple[datetime, datetime],
) -> str:
    return f"""You are auditing one behavior segment from a desktop recording for possible sensitive-data leakage.
Use only the provided log context and sampled video frames. Prefer visual evidence over OCR when they disagree.

Segment:
{{
  "segment_id": {json.dumps(segment_id)},
  "start": {json.dumps(segment[0].strftime("%Y-%m-%d %H:%M:%S"))},
  "end": {json.dumps(segment[1].strftime("%Y-%m-%d %H:%M:%S"))}
}}

Sensitive files or objects:
{json.dumps(sensitive_files, ensure_ascii=False, indent=2)}

Candidate log events:
{json.dumps(candidate_events, ensure_ascii=False, indent=2)}

Relevant in-window log summary:
{json.dumps(review_logs, ensure_ascii=False, indent=2)}

Sink sessions that guide this review:
{json.dumps(sink_sessions, ensure_ascii=False, indent=2)}

Frame notes:
- Frames with image_sent=true include the real image.
- Frames with image_sent=false are text/OCR timeline context only.
- OCR can be wrong or incomplete, so do not rely on OCR alone when the image contradicts it.
- Ignore the monitor application's own status page as proof of leakage.
- Session metadata is only a locator. It does not prove completion by itself.
- For long uploads/sends, inspect session_start, heartbeat, terminal, and session_end frames for success, failure, cancel, or still-in-progress UI states.

Terminal-state rules:
- Completed: visible "Message sent", "Sent", "Upload complete", "Uploaded", success confirmation, newly-created remote listing after an outbound upload, or recipient-visible artifact.
- Failed: visible "Upload failed", "Send failed", network error, retry prompt, or delivery failure.
- Canceled: visible "Canceled", "Discard draft", attachment removed, upload canceled, or send canceled.
- In progress: visible progress bar below 100%, "Uploading...", "Sending...", spinner/progress without success.
- Do not mark failed/canceled/in-progress sessions as completed leakage.

Outbound-direction rules:
- Downloading a sensitive file from email/cloud/drive to the local computer is not leakage.
- A browser download bubble, "download attachment" URL, download bar, or local editor opening a downloaded file is not outbound leakage.
- A file already visible in a remote cloud/mail listing is not proof of a new upload unless the frames/logs show an outbound upload/save/share/send flow that reaches success.
- A selected checkbox in a remote listing only means the object is selected; it is not a completed upload/share by itself.
- Reading or editing an existing remote document is not completed leakage unless the user newly pasted/uploaded/shared sensitive local content into that external service.

CRITICAL: Distinguish between staging and completion states carefully:

❌ STAGING (selected_or_attached / preparation):
- File appears in file picker or attachment list
- Upload dialog is open but not submitted
- Compose window with attachment visible but not sent
- "Send" button highlighted but not clicked
- Form filled but not submitted

✅ COMPLETION (completed / content_exposed):
- "发送成功" / "Send Success" / "已发送" / "Sent" status visible
- "上传完成" / "Upload Complete" / "已上传" / "Uploaded" confirmation
- Progress bar shows 100% completion
- Message appears in sent messages list or conversation history
- File newly appears in remote cloud storage listing immediately after an observed outbound upload/save action
- Commit/push success in git interface
- Content visible in external AI chat history (content_exposed)
- Screenshot captured or screen share active with sensitive content visible

⚠️ IN PROGRESS (in_progress):
- Progress bar visible but < 100%
- "正在上传" / "Uploading..." / "Sending..." status
- Transfer dialog with incomplete progress

Return both a completed-action verdict and a risk-stage verdict:
- is_violation=true only when the evidence shows completed leakage or direct exposure of sensitive content in an external sink.
- risk_level must be one of: none, preparation, selected_or_attached, in_progress, content_exposed, completed.
- transfer_direction must be one of: outbound, inbound, local, unknown.
- terminal_state must be one of: completed, staged, in_progress, failed, canceled, none, unknown.
- artifact_state must be one of: new_remote_artifact, existing_remote_artifact, local_download, file_picker_only, draft_attachment, remote_input_content, active_screen_exposure, local_only, unknown.
- completion_evidence and non_completion_evidence must contain short enum-like tokens, for example: message_sent_toast, upload_success_banner, generated_share_link, sent_message_visible, existing_remote_listing, selected_checkbox_only, download_bubble, file_picker_only, draft_not_sent, no_success_confirmation.
- Identify frontend applications by category/capability, not by a fixed brand list. If the brand is unknown, infer the category from UI features.
- Use selected_or_attached when a sensitive file is selected or attached in an external sink but not submitted yet.
- Use in_progress when upload/send/share is visibly underway but completion is not shown.
- Use content_exposed when sensitive text/image content is newly entered or displayed in an external AI/chat/message input, conversation, active screen share, screenshot, or remote recipient-visible surface. Do not use content_exposed for a normal download, an existing remote listing, or merely opening an existing cloud document.
- Use completed when upload/send/share/commit/publish success, remote listing, sent message, shared screen exposure, screenshot capture, or VM/remote copy completion is visible.
- Use preparation for opening a sink page, browsing, right-clicking, or unrelated local editing without sensitive data entering the sink.
- Use none when there is no sensitive-object interaction.

Strict completion rules:
- Email/chat is completed only after sent/success, the message appears in the conversation, or the recipient-visible artifact is shown.
- Cloud/Git/community upload is completed only after success, a newly-created remote listing tied to an outbound upload/save action, commit/publish success, or a generated remote attachment link.
- Existing remote listing or selected checkbox is not enough for cloud completion; the listing must be tied to a preceding outbound upload/save/share action.
- AI/chat input containing sensitive content is content_exposed even before a final submit if the content is visible in the external service.
- A draft, file picker, highlighted send button, selected file, or upload page alone is not completed, but may be selected_or_attached or in_progress.
- For observed_actions, emit one separate structured action for every sensitive-object interaction you can see or infer from the provided logs.
- If a sensitive file is attached to an email/chat/cloud/community/AI sink, use attach_file or upload_start with source_file set to the exact sensitive file path/name even when the final verdict is not completed.
- If sensitive content or a sensitive file is visible inside an external input/editor/chat, use paste_content, attach_file, or external_exposure with risk_level content_exposed.
- Do not output action_type none for a segment that contains a visible sensitive attachment, upload widget, sent message, external editor content, screen share, screenshot, VM copy, or cloud sync evidence.
- source_file must be the sensitive source path/name; derived_file is only for renamed/exported/copied artifacts. Leave derived_file empty instead of inventing an unrelated system/cache file.

Output exactly one JSON object and no markdown:
{{
  "is_violation": true,
  "risk_level": "none|preparation|selected_or_attached|in_progress|content_exposed|completed",
  "transfer_direction": "outbound|inbound|local|unknown",
  "terminal_state": "completed|staged|in_progress|failed|canceled|none|unknown",
  "artifact_state": "new_remote_artifact|existing_remote_artifact|local_download|file_picker_only|draft_attachment|remote_input_content|active_screen_exposure|local_only|unknown",
  "completion_evidence": ["message_sent_toast"],
  "non_completion_evidence": ["file_picker_only"],
  "confidence": 0.0,
  "completed_action": "send|upload|share|publish|commit|ai_input|screen_share|screenshot|vm_copy|none|unknown",
  "frontend_app": {{
    "name": "observed app or site",
    "category": "email|cloud_storage|ai_service|messaging|code_repo|community_publish|meeting|workplace|desktop_app|unknown",
    "capabilities": ["compose_message", "attach_file", "upload_file", "send_message", "publish_content", "chat_input", "screen_share"]
  }},
  "observed_actions": [
    {{
      "segment_id": {json.dumps(segment_id)},
      "action_type": "open_file|copy_content|paste_content|select_file|attach_file|upload_start|upload_complete|send_message|publish_content|screenshot|screen_record|screen_share|save_as|convert_file|compress_file|rename_file|vm_copy|external_exposure|none|unknown",
      "risk_level": "none|preparation|selected_or_attached|in_progress|content_exposed|completed",
      "transfer_direction": "outbound|inbound|local|unknown",
      "terminal_state": "completed|staged|in_progress|failed|canceled|none|unknown",
      "artifact_state": "new_remote_artifact|existing_remote_artifact|local_download|file_picker_only|draft_attachment|remote_input_content|active_screen_exposure|local_only|unknown",
      "completion_evidence": ["upload_success_banner"],
      "non_completion_evidence": ["no_success_confirmation"],
      "time": "YYYY-MM-DD HH:MM:SS or empty",
      "app": "app or site",
      "app_category": "category",
      "source_file": "sensitive or derived file if visible",
      "derived_file": "derived file if any",
      "evidence_frames": [1, 2],
      "confidence": 0.0,
      "description": "short audit-ready evidence statement"
    }}
  ],
  "evidence_frames": [1, 2],
  "reason": "short explanation"
}}
"""


def _normalize_risk_level(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "selected": "selected_or_attached",
        "attached": "selected_or_attached",
        "attempt": "attempted",
        "attempting": "attempted",
        "exposed": "content_exposed",
        "complete": "completed",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "none",
        "preparation",
        "attempted",
        "selected_or_attached",
        "in_progress",
        "content_exposed",
        "completed",
    }
    return normalized if normalized in allowed else ""


def _normalize_transfer_direction(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "upload": "outbound",
        "send": "outbound",
        "share": "outbound",
        "publish": "outbound",
        "download": "inbound",
        "receive": "inbound",
        "received": "inbound",
        "none": "unknown",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"outbound", "inbound", "local", "unknown"} else "unknown"


def _normalize_terminal_state(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "selected": "staged",
        "attached": "staged",
        "selected_or_attached": "staged",
        "draft": "staged",
        "started": "in_progress",
        "uploading": "in_progress",
        "sending": "in_progress",
        "cancelled": "canceled",
        "success": "completed",
        "succeeded": "completed",
        "done": "completed",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"completed", "staged", "in_progress", "failed", "canceled", "none", "unknown"} else "unknown"


def _normalize_artifact_state(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "remote_listing": "existing_remote_artifact",
        "existing_remote_listing": "existing_remote_artifact",
        "download": "local_download",
        "downloaded": "local_download",
        "picker": "file_picker_only",
        "file_picker": "file_picker_only",
        "attachment_draft": "draft_attachment",
        "draft": "draft_attachment",
        "input_content": "remote_input_content",
        "chat_input": "remote_input_content",
        "screen_share": "active_screen_exposure",
        "screenshot": "active_screen_exposure",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "new_remote_artifact",
        "existing_remote_artifact",
        "local_download",
        "file_picker_only",
        "draft_attachment",
        "remote_input_content",
        "active_screen_exposure",
        "local_only",
        "unknown",
    }
    return normalized if normalized in allowed else "unknown"


def _structured_evidence_list(value: Any) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    result = []
    for item in values:
        token = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token:
            result.append(token[:80])
    return sorted(set(result))


def _has_structured_state(value: Dict[str, Any]) -> bool:
    return any(
        key in value and value.get(key) not in (None, "", [], {})
        for key in (
            "transfer_direction",
            "terminal_state",
            "artifact_state",
            "completion_evidence",
            "non_completion_evidence",
        )
    )


def _infer_risk_level_from_verdict(verdict: Dict[str, Any]) -> str:
    if bool(verdict.get("is_violation")):
        action = str(verdict.get("completed_action", "") or "").lower()
        if action in {"ai_input", "screen_share", "screenshot", "vm_copy"}:
            return "content_exposed"
        return "completed"

    reason = str(verdict.get("reason", "") or "").lower()
    if not reason:
        return "none"

    in_progress_markers = (
        "uploading",
        "in progress",
        "upload progress",
        "\u6b63\u5728\u4e0a\u4f20",
        "\u4e0a\u4f20\u4e2d",
        "\u4e0a\u4f20\u8fdb\u5ea6",
        "\u6b63\u5728\u5f00\u59cb\u4e0a\u4f20",
    )
    selected_markers = (
        "selected file",
        "file_selected",
        "attached",
        "as attachment",
        "\u5df2\u4f5c\u4e3a\u9644\u4ef6",
        "\u9644\u4ef6\u5df2\u6dfb\u52a0",
        "\u5df2\u6dfb\u52a0\u4e3a\u9644\u4ef6",
        "\u9009\u4e2d\u4e86\u654f\u611f\u6587\u4ef6",
        "\u6dfb\u52a0\u9644\u4ef6",
    )
    exposed_markers = (
        "sensitive content is visible",
        "visible in the input",
        "pasted into",
        "appears in the conversation",
        "\u654f\u611f\u5185\u5bb9\u5df2",
        "\u51fa\u73b0\u5728\u8f93\u5165\u6846",
        "\u7c98\u8d34\u5230",
        "\u5bf9\u8bdd\u6d88\u606f",
    )
    negative_markers = (
        "not observed",
        "no evidence",
        "did not see",
        "without evidence",
        "not in the sensitive",
        "\u672a\u89c1",
        "\u672a\u89c2\u5bdf\u5230",
        "\u6ca1\u6709",
        "\u65e0\u8bc1\u636e",
        "\u4e0d\u5728\u654f\u611f",
        "\u975e\u654f\u611f",
        "\u53d6\u6d88",
        "\u64a4\u56de",
    )

    if any(marker in reason for marker in in_progress_markers):
        return "in_progress"
    if any(marker in reason for marker in selected_markers) and not any(marker in reason for marker in negative_markers):
        return "selected_or_attached"
    if any(marker in reason for marker in exposed_markers) and not any(marker in reason for marker in negative_markers):
        return "content_exposed"
    return "preparation" if any(marker in reason for marker in EXTERNAL_SINK_TOKENS) else "none"


def _has_negative_completion_context(text: str) -> bool:
    lowered = str(text or "").lower()
    negative_markers = (
        "not sent",
        "not clicked",
        "not completed",
        "not confirmed",
        "not submitted",
        "no upload",
        "no send",
        "no evidence",
        "without sending",
        "without upload",
        "upload has not",
        "has not been sent",
        "has not been confirmed",
        "not visible",
        "not observed",
        "preparation",
        "draft",
        "file picker",
        "rather than completed",
        "not fully",
        "final publish/submit action has not",
        "completed upload confirmed",
        "\u672a\u53d1\u9001",
        "\u672a\u4e0a\u4f20",
        "\u672a\u5b8c\u6210",
        "\u672a\u63d0\u4ea4",
        "\u6ca1\u6709\u53d1\u9001",
        "\u6ca1\u6709\u4e0a\u4f20",
        "\u65e0\u8bc1\u636e",
        "\u8349\u7a3f",
        "\u9009\u62e9\u6587\u4ef6",
        "\u672a\u786e\u8ba4",
    )
    return any(marker in lowered for marker in negative_markers)


def _infer_action_type_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ("screen share", "shared screen", "meeting", "zoom", "teams", "\u5171\u4eab\u5c4f\u5e55", "\u4f1a\u8bae")):
        return "screen_share"
    if any(token in lowered for token in ("screenshot", "screen capture", "\u622a\u56fe", "\u622a\u5c4f")):
        return "screenshot"
    if any(token in lowered for token in ("vmware", "virtualbox", "virtual machine", "vm copy", "\u865a\u62df\u673a")):
        return "vm_copy"
    if any(token in lowered for token in ("ai", "chatgpt", "deepseek", "kimi", "poe", "gemini", "doubao", "input field", "pasted into", "\u8f93\u5165\u6846", "\u7c98\u8d34")):
        return "external_exposure"
    if any(token in lowered for token in ("mail", "email", "gmail", "outlook", "proton", "qq mail", "message", "sent", "delivered", "\u90ae\u4ef6", "\u90ae\u7bb1", "\u53d1\u9001", "\u6295\u9012\u6210\u529f")):
        return "send_message"
    if any(token in lowered for token in ("publish", "commit", "csdn", "github", "gitlab", "gitee", "bitbucket", "\u53d1\u5e03", "\u63d0\u4ea4")):
        return "publish_content"
    if any(token in lowered for token in ("upload", "uploaded", "attachment", "attached", "cloud", "drive", "netdisk", "dropbox", "onedrive", "\u4e0a\u4f20", "\u9644\u4ef6", "\u4e91\u76d8", "\u7f51\u76d8")):
        return "upload_complete"
    return "external_exposure"


def _completion_evidence_from_text(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    strong_completion_markers = (
        "upload complete",
        "upload completed",
        "successfully uploaded",
        "save successful",
        "sent successfully",
        "delivered successfully",
        "delivered",
        "sent message",
        "message appears",
        "directly exposed",
        "content exposed",
        "pasted into",
        "visible in the input",
        "visible in the chat input",
        "visible in the message input",
        "visible in the compose window",
        "attached to the message",
        "attached to the email",
        "file chip is visible",
        "mail sent notification",
        "sent notification",
        "message sent",
        "mail sent",
        "remote listing",
        "shareable link",
        "commit successful",
        "publish success",
        "\u4e0a\u4f20\u5b8c\u6210",
        "\u4e0a\u4f20\u6210\u529f",
        "\u4fdd\u5b58\u6210\u529f",
        "\u53d1\u9001\u6210\u529f",
        "\u6295\u9012\u6210\u529f",
        "\u5df2\u4f5c\u4e3a\u9644\u4ef6",
        "\u53ef\u89c1\u4e3a\u9644\u4ef6",
        "\u76f4\u63a5\u66b4\u9732",
        "\u7c98\u8d34\u5230",
        "\u5206\u4eab\u94fe\u63a5",
        "\u63d0\u4ea4\u6210\u529f",
        "\u53d1\u5e03\u6210\u529f",
    )
    if any(marker in lowered for marker in strong_completion_markers) and not _has_negative_completion_context(lowered):
        return True
    completion_markers = (
        "uploaded",
        "completed",
        "success",
        "delivered",
        "sent",
        "shared",
        "committed",
        "\u5df2\u4e0a\u4f20",
        "\u5b8c\u6210",
        "\u6210\u529f",
        "\u5df2\u53d1\u9001",
        "\u5df2\u5171\u4eab",
    )
    return not _has_negative_completion_context(lowered) and any(marker in lowered for marker in completion_markers)


def _best_vlm_reason_action(
    verdict: Dict[str, Any],
    sensitive_files: List[str],
    logs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    reason = str(verdict.get("reason", "") or "")
    segments = [
        part.strip(" ;")
        for part in re.split(r"(?:^|;\s*)vlm_seg_\d+:", reason)
        if part.strip(" ;")
    ] or [reason]
    completion_segments = [segment for segment in segments if _completion_evidence_from_text(segment)]
    if not completion_segments:
        return None
    evidence_text = max(completion_segments, key=len)
    frontend = _normalize_frontend_app(verdict.get("frontend_app"), logs)
    frame_times = _frame_timestamps_by_index(verdict)
    evidence_frames = [
        int(item)
        for item in verdict.get("evidence_frames", []) or []
        if str(item).strip().isdigit()
    ]
    action_type = _infer_action_type_from_text(evidence_text)
    risk_level = "content_exposed" if action_type == "external_exposure" else "completed"
    return {
        "action_id": "vlm_reason_completion",
        "segment_id": str(verdict.get("segment_id") or ""),
        "action_type": action_type,
        "risk_level": risk_level,
        "time": frame_times.get(evidence_frames[0], "") if evidence_frames else _verdict_timestamp({}, logs or []),
        "app": frontend.get("name", ""),
        "app_category": frontend.get("category", "unknown"),
        "source_file": str(sensitive_files[0] if sensitive_files else ""),
        "derived_file": "",
        "evidence_frames": evidence_frames,
        "confidence": round(max(_safe_float(verdict.get("confidence", 0.0)), 0.82), 4),
        "description": _compact_text(evidence_text, 500),
        "evidence_source": "remote_vlm_reason",
    }


def _local_positive_supports_leak(action: Dict[str, Any]) -> bool:
    action_type = _normalize_action_type(str(action.get("action_type", "") or ""))
    risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    reason = str(action.get("description", "") or "").lower()
    strong_reasons = (
        "explicit_transfer_event",
        "vm_context",
        "export_context",
        "git_context_with_completion_text",
        "archive_or_convert_with_completion_text",
    )
    if any(marker in reason for marker in strong_reasons):
        return True
    if not _bool_env("DLD_DATALOG_TRUST_LOCAL_VISUAL_GATE", False):
        return False
    visual_reasons = ("direct_visual_capture_event", "screenshot_context")
    return (
        action_type == "external_exposure"
        and risk_level in {"content_exposed", "completed"}
        and any(marker in reason for marker in visual_reasons)
    )


def _local_positive_supports_risk(action: Dict[str, Any]) -> bool:
    action_type = _normalize_action_type(str(action.get("action_type", "") or ""))
    risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    reason = str(action.get("description", "") or "").lower()
    if action_type != "external_exposure" or risk_level not in {"content_exposed", "completed"}:
        return False
    strong = (
        "explicit_transfer_event",
        "vm_context",
        "export_context",
        "git_context_with_completion_text",
        "archive_or_convert_with_completion_text",
    )
    if any(marker in reason for marker in strong):
        return True
    if not _bool_env("DLD_DATALOG_TRUST_LOCAL_VISUAL_GATE", False):
        return False
    return any(
        marker in reason
        for marker in (
            "direct_visual_capture_event",
            "screenshot_context",
        )
    )


def _external_sink_action_context(action: Dict[str, Any]) -> bool:
    text = " ".join(
        str(action.get(key, "") or "").lower()
        for key in ("app", "app_category", "description", "evidence_source")
    )
    external_markers = tuple(token.lower() for token in EXTERNAL_SINK_TOKENS) + (
        "ai_service",
        "community_publish",
        "cloud_storage",
        "messaging",
        "meeting",
        "email",
        "sync",
        "onedrive",
        "sticky notes",
        "web_post",
        "code_repo",
        "gitea",
        "gitcode",
        "\u4fbf\u7b3a",
    )
    return any(marker in text for marker in external_markers)


def _action_text(action: Dict[str, Any]) -> str:
    return " ".join(
        str(action.get(key, "") or "").lower()
        for key in ("action_type", "risk_level", "app", "app_category", "description", "evidence_source")
    )


def _action_has_hard_negative_context(action: Dict[str, Any]) -> bool:
    text = _action_text(action)
    hard_negative_markers = (
        "no sensitive",
        "non-sensitive",
        "not the sensitive",
        "not listed as a sensitive",
        "is not present in the segment",
        "does not match the sensitive",
        "no evidence",
        "no visual or log evidence",
        "no visual evidence",
        "no log activity",
        "no leakage",
        "no completed leakage",
        "no direct exposure",
        "no direct visual evidence",
        "not observed",
        "not pasted into",
        "not uploaded",
        "not opened",
        "not attached",
        "not selected",
        "unrelated",
        "unrelated image",
        "likely unrelated",
        "likely the sensitive",
        "matching the visual content",
        "do not match the sensitive",
        "does not match the sensitive",
        "normal browsing",
        "local save",
        "local file operation",
        "local save/overwrite",
        "monitoring dashboard",
        "monitor application's",
        "monitor app",
        "localhost:5000",
        "safe status",
        "\u975e\u654f\u611f",
        "\u6ca1\u6709\u654f\u611f",
        "\u65e0\u654f\u611f",
        "\u65e0\u8bc1\u636e",
    )
    return any(marker in text for marker in hard_negative_markers)


def _action_has_historical_or_inbound_context(action: Dict[str, Any]) -> bool:
    link_text = " ".join(str(item or "") for item in action.get("mapping_links", []) or [])
    text = f"{_action_text(action)} {link_text.lower()}"
    completion_markers = (
        "sent email",
        "email appears in the sent folder",
        "appears in the sent folder, indicating completed",
        "completed transmission",
        "successfully sent",
        "mail sent notification",
        "sent notification",
        "message sent",
        "mail sent",
        "send success",
        "upload successful",
        "uploaded successfully",
        "successfully uploaded",
        "clicked upload",
        "clicked send",
        "clicked submit",
        "confirming successful",
        "add files via upload",
        "timestamp of 'today",
        "timestamp 'now'",
        "\u53d1\u9001\u6210\u529f",
        "\u5df2\u53d1\u9001",
        "\u4e0a\u4f20\u6210\u529f",
    )
    if any(marker in text for marker in completion_markers):
        return False

    historical_markers = (
        "already sent",
        "sent previously",
        "previously sent",
        "dated ",
        "no new exfiltration",
        "no new sending",
        "interaction is limited to local viewing",
        "local viewing",
        "local access",
        "inbox",
        "\u6536\u4ef6\u7bb1",
        "remote file listing",
        "remote listing",
        "already present in",
        "already stored in",
        "already exists in",
        "download",
        "downloaded",
        "\u4e0b\u8f7d",
        "another participant",
        "other participant",
        "someone else",
        "\u4ed6\u4eba\u5171\u4eab",
        "save as dialog",
        "print dialog",
        "print preview",
        "deleted from the chat",
        "message deleted",
        "recalled",
        "recall message",
        "cancelled",
        "canceled",
        "\u6253\u5370\u9884\u89c8",
        "\u64a4\u56de",
        "\u53d6\u6d88",
    )
    if any(marker in text for marker in historical_markers):
        return True

    sent_folder_read_markers = (
        "browse",
        "browses",
        "view",
        "views",
        "visible as an attachment",
        "listed as an attachment",
        "sent folder list",
        "attachment list",
    )
    if "sent folder" in text and any(marker in text for marker in sent_folder_read_markers):
        return True

    source = str(action.get("source_file", "") or "").replace("\\", "/").lower()
    derived = str(action.get("derived_file", "") or "").replace("\\", "/").lower()
    app_category = str(action.get("app_category", "") or "").lower()
    outgoing_markers = (
        "compose",
        "draft",
        "attach",
        "attached",
        "send",
        "sent email",
        "upload",
        "share",
        "publish",
        "commit",
        "\u5199\u90ae\u4ef6",
        "\u9644\u4ef6",
        "\u53d1\u9001",
        "\u4e0a\u4f20",
        "\u5206\u4eab",
    )
    if app_category in {"email", "mail_attachment"} and "/downloads/" in source and not any(
        marker in text for marker in outgoing_markers
    ):
        return True
    if (
        app_category in {"email", "mail_attachment"}
        and "/downloads/" in source
        and ("wps cloud files" in text or "/cachedata/" in text or "cachedata" in derived)
    ):
        return True

    return False


def _action_has_cloud_editor_read_context(action: Dict[str, Any]) -> bool:
    text = _action_text(action)
    if "cloud_storage" not in text:
        return False
    if not any(marker in text for marker in ("online editor", "being edited", "web editor", "viewer/editor")):
        return False
    outbound_markers = (
        "upload complete",
        "uploaded",
        "remote listing",
        "share link",
        "generated link",
        "shared",
        "published",
        "success",
        "\u4e0a\u4f20\u6210\u529f",
        "\u5206\u4eab\u94fe\u63a5",
    )
    return not any(marker in text for marker in outbound_markers)


def _action_has_unfinished_context(action: Dict[str, Any]) -> bool:
    text = _action_text(action)
    unfinished_markers = (
        "not completed",
        "not confirmed",
        "not clicked",
        "not submitted",
        "not published",
        "not yet",
        "not yet submitted",
        "not yet published",
        "not yet sent",
        "not sent",
        "has not been sent",
        "not been sent",
        "no submission success",
        "no submit success",
        "no sent confirmation",
        "no success confirmation",
        "no confirmation",
        "without confirmation",
        "尚未",
        "pending",
        "awaiting",
        "not shown",
        "not visible",
        "not observed",
        "not confirmed as completed",
        "final submission",
        "final publish",
        "validation error",
        "too short",
        "form fields are incomplete",
        "incomplete",
        "\u672a\u5b8c\u6210",
        "\u672a\u786e\u8ba4",
        "\u672a\u63d0\u4ea4",
        "\u672a\u53d1\u5e03",
        "\u672a\u53d1\u9001",
    )
    if any(marker in text for marker in unfinished_markers):
        return True
    return bool(
        re.search(
            r"not\s+(?:yet\s+|been\s+|\w+ly\s+)*"
            r"(?:submitted|sent|published|uploaded|completed|confirmed|shown|visible|observed)",
            text,
        )
    )


def _action_has_completion_context(action: Dict[str, Any]) -> bool:
    text = _action_text(action)
    return _completion_evidence_from_text(text)


def _action_has_upload_ingest_context(action: Dict[str, Any]) -> bool:
    """The VLM saw the file/content actually land in the external surface."""
    text = _action_text(action)
    return any(
        marker in text
        for marker in (
            "uploaded",
            "upload complete",
            "upload successful",
            "inserted",
            "embedded",
            "pasted into",
            "visible inside the external",
            "fields contain sensitive",
            "contains sensitive text",
            "actively processing",
            "processing the request",
            "已上传",
            "上传成功",
            "已插入",
            "正在工作",
        )
    )


def _action_has_upload_progress_context(action: Dict[str, Any]) -> bool:
    """The VLM saw the transfer running or being submitted (not just staged)."""
    text = _action_text(action)
    return any(
        marker in text
        for marker in (
            "spinner",
            "progress bar",
            "uploading",
            "upload in progress",
            "upload is in progress",
            "submitted",
            "clicked upload",
            "clicked send",
            "clicked submit",
            "sending",
            "ready for commit",
            "正在上传",
            "已提交",
        )
    )


def _vlm_transformation_action_supports_risk(action: Dict[str, Any]) -> bool:
    """Hiding-style transformations observed on screen (convert/rename/zip)."""
    if _action_has_hard_negative_context(action):
        return False
    if _action_has_historical_or_inbound_context(action):
        return False
    text = _action_text(action)
    operation_markers = (
        "convert", "conversion", "renam", "compress", "zip", "archive",
        "split", "转换", "重命名", "压缩", "拆分",
    )
    if not any(marker in text for marker in operation_markers):
        return False
    return not any(
        marker in text
        for marker in (
            "prepare", "preparing", "preparation to", "intends", "intention",
            "about to", "menu is open", "dialog is open", "no conversion",
            "cancelled", "canceled", "取消",
            # OS save/download collision dialogs (post-download rename flows)
            "conflict",
        )
    )


def _action_has_staging_context(action: Dict[str, Any]) -> bool:
    text = _action_text(action)
    staging_markers = (
        "attached",
        "attach",
        "selected",
        "file selection",
        "file_selected",
        "send file",
        "sending file",
        "selected for sending",
        "selected or attached for sharing",
        "selected or attached for sending",
        "share dialog",
        "send dialog",
        "upload complete",
        "upload completed",
        "uploading",
        "upload_start",
        "spinner",
        "release mouse to upload",
        "ready for submission",
        "\u9644\u4ef6",
        "\u4e0a\u4f20",
        "\u5df2\u9009\u62e9",
    )
    return any(marker in text for marker in staging_markers)


def _action_has_vlm_contradiction(action: Dict[str, Any]) -> bool:
    evidence_source = str(action.get("evidence_source", "") or "")
    if evidence_source not in {"remote_vlm", "remote_vlm_reason"}:
        return False
    text = _action_text(action)
    contradiction_markers = (
        "not sent yet",
        "not yet sent",
        "not submitted yet",
        "not yet submitted",
        "not uploaded yet",
        "not yet uploaded",
        "not completed",
        "no visual confirmation",
        "no visual or log evidence",
        "no visual evidence",
        "no log evidence",
        "no completion confirmation",
        "no sent confirmation",
        "no upload confirmation",
        "no sensitive file is selected",
        "no sensitive file selected",
        "no sensitive file is visible",
        "no sensitive file visible",
        "no sensitive file interaction",
        "no interaction with the sensitive",
        "no sensitive file exposed",
        "no sensitive content exposed",
        "no sensitive file is exposed",
        "no sensitive content is exposed",
        "no completed leakage",
        "no direct exposure",
        "not pasted into",
        "do not match the sensitive",
        "does not match the sensitive",
        "local save/overwrite",
        "draft",
        "cancelled",
        "canceled",
    )
    return any(marker in text for marker in contradiction_markers)


def _apply_vlm_action_consistency(action: Dict[str, Any]) -> None:
    if not _action_has_vlm_contradiction(action):
        return
    action["raw_action_type"] = action.get("action_type", "")
    action["raw_risk_level"] = action.get("risk_level", "")
    action["consistency_reason"] = "downgraded_vlm_contradiction"
    if _action_has_staging_context(action) and not any(
        marker in _action_text(action)
        for marker in ("no sensitive file exposed", "no sensitive content exposed", "no sensitive file is exposed")
    ):
        action["action_type"] = "attach_file"
        action["risk_level"] = "selected_or_attached"
    else:
        action["action_type"] = "none"
        action["risk_level"] = "none"


def _remote_vlm_action_supports_leak(action: Dict[str, Any]) -> bool:
    evidence_source = str(action.get("evidence_source", "") or "")
    if evidence_source != "remote_vlm":
        # Segment-level reason fallbacks are useful risk hints, but they are
        # too coarse to become confirmed leakage facts by themselves.
        return False

    if _has_structured_state(action):
        direction = _normalize_transfer_direction(str(action.get("transfer_direction", "") or ""))
        terminal = _normalize_terminal_state(str(action.get("terminal_state", "") or ""))
        artifact = _normalize_artifact_state(str(action.get("artifact_state", "") or ""))
        completion = _structured_evidence_list(action.get("completion_evidence"))
        non_completion = _structured_evidence_list(action.get("non_completion_evidence"))
        if direction != "outbound":
            return False
        if terminal in {"failed", "canceled", "in_progress", "staged", "none"}:
            return False
        if _structured_non_completion_blocks_leak(non_completion, artifact):
            return False
        if terminal == "completed" and (
            artifact in {"new_remote_artifact", "remote_input_content", "active_screen_exposure"}
            or _structured_completion_evidence_supports_leak(completion)
        ):
            return True
        if (
            _normalize_risk_level(str(action.get("risk_level", "") or "")) == "content_exposed"
            and artifact in {"remote_input_content", "active_screen_exposure"}
        ):
            return True
        return False

    action_type = _normalize_action_type(str(action.get("action_type", "") or ""))
    risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    if action_type not in {
        "upload_complete",
        "send_message",
        "publish_content",
        "external_exposure",
        "screen_share",
        "screenshot",
        "screen_record",
        "vm_copy",
    }:
        return False
    if risk_level not in {"content_exposed", "completed"}:
        return False

    if _safe_float(action.get("confidence", 0.0)) < 0.9:
        return False
    if not action.get("evidence_frames"):
        return False
    if not _external_sink_action_context(action):
        return False
    if (
        _action_has_hard_negative_context(action)
        or _action_has_vlm_contradiction(action)
        or _action_has_unfinished_context(action)
        or _action_has_historical_or_inbound_context(action)
        or _action_has_cloud_editor_read_context(action)
    ):
        return False

    text = _action_text(action)
    uncertainty_markers = (
        "likely",
        "appears to",
        "seems",
        "may have",
        "might have",
        "could have",
        "or derived",
        "matching extension",
        "matching the sensitive",
        "consistent with",
        "suggests",
        "suggesting",
        "推测",
        "可能",
        "疑似",
    )
    if any(marker in text for marker in uncertainty_markers):
        return False

    if action_type in {"screen_share", "screenshot", "screen_record", "vm_copy"}:
        return True
    if action_type == "external_exposure":
        return _content_exposure_action_supports_leak(action)
    if action_type in {"upload_complete", "send_message", "publish_content"} and not bool(
        action.get("segment_sensitive_object_confirmed")
    ):
        return False
    return _action_has_completion_context(action) or _action_has_upload_ingest_context(action)


def _vlm_parent_verdict_blocks_risk(action: Dict[str, Any]) -> bool:
    if str(action.get("evidence_source", "") or "") != "remote_vlm":
        return False
    parent_risk = _normalize_risk_level(str(action.get("verdict_risk_level", "") or ""))
    if parent_risk not in {"none", "preparation"}:
        return False
    if bool(action.get("verdict_is_violation")):
        return False
    action_risk = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    action_type = _normalize_action_type(str(action.get("action_type", "") or ""))
    if action_type in {"attach_file", "select_file", "upload_start"} and action_risk in {"selected_or_attached", "in_progress"}:
        text = _action_text(action)
        if parent_risk == "none" and not bool(action.get("verdict_is_violation")) and any(
            marker in text
            for marker in (
                "likely for upload",
                "saved as a draft",
                "likely selecting",
                "suggesting the sensitive",
            )
        ):
            return True
        return not _action_has_staging_context(action)
    if action_risk in {"content_exposed", "completed"} or action_type in {"upload_complete", "send_message", "publish_content"}:
        return _action_has_unfinished_context(action) or not (
            _action_has_completion_context(action) and not _action_has_unfinished_context(action)
        )
    return action_risk in {"selected_or_attached", "in_progress"}


def _content_exposure_action_supports_leak(action: Dict[str, Any]) -> bool:
    action_type = _normalize_action_type(str(action.get("action_type", "") or ""))
    risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    if _action_has_hard_negative_context(action):
        return False
    if _action_has_historical_or_inbound_context(action):
        return False
    if _action_has_cloud_editor_read_context(action):
        return False
    if _vlm_parent_verdict_blocks_risk(action):
        return False
    if risk_level not in {"content_exposed", "completed"}:
        return False
    if action_type in {"attach_file", "upload_start"}:
        if not _external_sink_action_context(action):
            return False
        if _action_has_unfinished_context(action):
            return False
        return (
            _action_has_completion_context(action)
            or _action_has_upload_ingest_context(action)
            or _action_has_upload_progress_context(action)
        )
    if action_type == "select_file":
        return False
    if action_type in {"paste_content", "copy_content"}:
        text = _action_text(action)
        if any(marker in text for marker in ("share link", "copy link", "复制链接", "分享链接")):
            # A copied link is not the document's content leaving the host;
            # the upload claim itself must stand on its own evidence.
            return False
        return _external_sink_action_context(action)
    return action_type == "external_exposure"


def _vlm_verdict_text(verdict: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(verdict.get("reason", "") or ""),
        f"completed_action:{verdict.get('completed_action', '') or ''}",
        f"risk_level:{verdict.get('risk_level', '') or ''}",
    ]
    for action in verdict.get("observed_actions", []) or []:
        if isinstance(action, dict):
            parts.extend(
                (
                    f"action_type:{action.get('action_type', '') or ''}",
                    f"action_risk_level:{action.get('risk_level', '') or ''}",
                    str(action.get("description", "") or ""),
                    str(action.get("app", "") or ""),
                    str(action.get("app_category", "") or ""),
                )
            )
    return " ".join(parts).lower()


def _has_strong_outbound_completion_text(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "message sent",
            "email sent",
            "sent successfully",
            "send success",
            "upload complete",
            "upload completed",
            "upload successful",
            "uploaded successfully",
            "share link generated",
            "shareable link",
            "public sharing link",
            "copied the public sharing link",
            "published successfully",
            "commit successful",
            "push successful",
            "screen share active",
            "screenshot captured",
        )
    )


def _has_hard_non_outbound_text(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "downloaded from",
            "downloaded the sensitive",
            "download completion",
            "download bubble",
            "download bar",
            "download_attach",
            "readaction=download",
            "from the external sink",
            "from qq mail",
            "from 163",
            "selected checkbox",
            "checkbox selected",
            "already visible",
            "existing remote",
            "existing cloud",
            "remote file listing with its checkbox",
            "has not been confirmed",
            "not confirmed as completed",
            "no success confirmation",
            "no evidence of completion",
            "without completing",
            "state remains selected_or_attached",
            "not sent",
            "was not sent",
            "not submitted",
            "not uploaded",
            "file picker",
            "upload dialog is open",
            "attached but not",
            "selected but not",
        )
    )


def _non_outbound_downgrade_level(text: str) -> str:
    if any(
        marker in text
        for marker in (
            "selected_or_attached",
            "attached but not",
            "selected but not",
            "file picker",
            "upload dialog",
            "checkbox selected",
            "selected checkbox",
            "form shows",
        )
    ):
        return "selected_or_attached"
    return "none"


def _structured_completion_evidence_supports_leak(evidence: List[str]) -> bool:
    positive = {
        "message_sent_toast",
        "message_sent",
        "sent_message_visible",
        "email_sent",
        "upload_success_banner",
        "upload_complete",
        "upload_completed",
        "uploaded_successfully",
        "generated_share_link",
        "share_link_generated",
        "public_link_copied",
        "publish_success",
        "commit_success",
        "push_success",
        "recipient_visible_artifact",
        "screen_share_active",
        "screenshot_captured",
        "remote_copy_completed",
    }
    return bool(set(evidence) & positive)


def _structured_non_completion_blocks_leak(evidence: List[str], artifact_state: str) -> bool:
    blockers = {
        "existing_remote_listing",
        "selected_checkbox_only",
        "download_bubble",
        "download_attachment",
        "download_bar",
        "file_picker_only",
        "draft_not_sent",
        "attachment_not_sent",
        "no_success_confirmation",
        "upload_not_confirmed",
        "send_not_confirmed",
        "progress_incomplete",
        "canceled",
        "failed",
        "local_editor_opened",
    }
    blocker_artifacts = {
        "existing_remote_artifact",
        "local_download",
        "file_picker_only",
        "draft_attachment",
        "local_only",
    }
    return bool(set(evidence) & blockers) or artifact_state in blocker_artifacts


def _structured_verdict_supports_positive(verdict: Dict[str, Any]) -> Optional[bool]:
    if not _has_structured_state(verdict):
        action_votes = [
            _structured_action_supports_positive(action)
            for action in verdict.get("observed_actions", []) or []
            if isinstance(action, dict) and _has_structured_state(action)
        ]
        if not action_votes:
            return None
        return any(vote is True for vote in action_votes)
    direction = _normalize_transfer_direction(str(verdict.get("transfer_direction", "") or ""))
    terminal = _normalize_terminal_state(str(verdict.get("terminal_state", "") or ""))
    artifact = _normalize_artifact_state(str(verdict.get("artifact_state", "") or ""))
    risk_level = _normalize_risk_level(str(verdict.get("risk_level", "") or ""))
    completion = _structured_evidence_list(verdict.get("completion_evidence"))
    non_completion = _structured_evidence_list(verdict.get("non_completion_evidence"))

    verdict["transfer_direction"] = direction
    verdict["terminal_state"] = terminal
    verdict["artifact_state"] = artifact
    verdict["completion_evidence"] = completion
    verdict["non_completion_evidence"] = non_completion

    if direction in {"inbound", "local"}:
        return False
    if terminal in {"failed", "canceled", "in_progress", "staged", "none"}:
        return False
    if _structured_non_completion_blocks_leak(non_completion, artifact):
        return False
    if direction != "outbound":
        return False
    if terminal == "completed" and (
        artifact in {"new_remote_artifact", "remote_input_content", "active_screen_exposure"}
        or _structured_completion_evidence_supports_leak(completion)
    ):
        return True
    if risk_level == "content_exposed" and artifact in {"remote_input_content", "active_screen_exposure"}:
        return True
    return False


def _structured_action_supports_positive(action: Dict[str, Any]) -> Optional[bool]:
    if not _has_structured_state(action):
        return None
    direction = _normalize_transfer_direction(str(action.get("transfer_direction", "") or ""))
    terminal = _normalize_terminal_state(str(action.get("terminal_state", "") or ""))
    artifact = _normalize_artifact_state(str(action.get("artifact_state", "") or ""))
    risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    completion = _structured_evidence_list(action.get("completion_evidence"))
    non_completion = _structured_evidence_list(action.get("non_completion_evidence"))
    if direction != "outbound":
        return False
    if terminal in {"failed", "canceled", "in_progress", "staged", "none"}:
        return False
    if _structured_non_completion_blocks_leak(non_completion, artifact):
        return False
    if terminal == "completed" and (
        artifact in {"new_remote_artifact", "remote_input_content", "active_screen_exposure"}
        or _structured_completion_evidence_supports_leak(completion)
    ):
        return True
    if risk_level == "content_exposed" and artifact in {"remote_input_content", "active_screen_exposure"}:
        return True
    return False


def _postprocess_vlm_verdict(verdict: Dict[str, Any]) -> None:
    risk_level = _normalize_risk_level(str(verdict.get("risk_level", "") or ""))
    verdict["risk_level"] = risk_level or _infer_risk_level_from_verdict(verdict)
    structured_positive = _structured_verdict_supports_positive(verdict)
    if structured_positive is False:
        verdict["raw_is_violation"] = verdict.get("is_violation")
        verdict["raw_risk_level"] = verdict.get("risk_level")
        verdict["is_violation"] = False
        if verdict.get("terminal_state") in {"staged", "in_progress"}:
            verdict["risk_level"] = "in_progress" if verdict.get("terminal_state") == "in_progress" else "selected_or_attached"
        elif verdict.get("artifact_state") in {"file_picker_only", "draft_attachment"}:
            verdict["risk_level"] = "selected_or_attached"
        else:
            verdict["risk_level"] = "none"
        verdict["completed_action"] = "none"
        verdict["postprocess_reason"] = "downgraded_structured_non_outbound_or_unfinished"
        for action in verdict.get("observed_actions", []) or []:
            if not isinstance(action, dict):
                continue
            action_risk = _normalize_risk_level(str(action.get("risk_level", "") or ""))
            if action_risk in {"content_exposed", "completed"}:
                action["raw_risk_level"] = action.get("risk_level", "")
                action["risk_level"] = verdict["risk_level"]
                action["consistency_warning"] = "downgraded_structured_non_outbound_or_unfinished"
        return
    if structured_positive is True:
        verdict["raw_is_violation"] = verdict.get("is_violation")
        verdict["is_violation"] = True
        if verdict["risk_level"] not in {"content_exposed", "completed"}:
            verdict["risk_level"] = "content_exposed" if verdict.get("artifact_state") in {"remote_input_content", "active_screen_exposure"} else "completed"
    text = _vlm_verdict_text(verdict)
    if structured_positive is None and _has_hard_non_outbound_text(text) and not _has_strong_outbound_completion_text(text):
        verdict["raw_is_violation"] = verdict.get("is_violation")
        verdict["raw_risk_level"] = verdict.get("risk_level")
        verdict["is_violation"] = False
        verdict["risk_level"] = _non_outbound_downgrade_level(text)
        verdict["completed_action"] = "none"
        verdict["postprocess_reason"] = "downgraded_non_outbound_or_unfinished_context"
        for action in verdict.get("observed_actions", []) or []:
            if not isinstance(action, dict):
                continue
            action_risk = _normalize_risk_level(str(action.get("risk_level", "") or ""))
            if action_risk in {"content_exposed", "completed"}:
                action["raw_risk_level"] = action.get("risk_level", "")
                action["risk_level"] = verdict["risk_level"]
                action["consistency_warning"] = "downgraded_non_outbound_or_unfinished_context"
        return
    if not verdict.get("is_violation"):
        return
    if verdict["risk_level"] in {"content_exposed", "completed"}:
        return
    verdict["raw_is_violation"] = verdict.get("is_violation")
    verdict["is_violation"] = False
    verdict["postprocess_reason"] = "downgraded_non_completed_risk_stage"


def _vlm_final_positive(verdict: Dict[str, Any]) -> bool:
    policy = os.getenv("DLD_VLM_FINAL_POLICY", "risk").strip().lower()
    if policy in {"completed", "completion", "strict"}:
        return bool(verdict.get("is_violation"))

    risk_level = _normalize_risk_level(str(verdict.get("risk_level", "") or "")) or _infer_risk_level_from_verdict(verdict)
    verdict["risk_level"] = risk_level
    if risk_level not in POSITIVE_RISK_LEVELS:
        return bool(verdict.get("is_violation"))
    try:
        confidence = float(verdict.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        min_confidence = float(os.getenv("DLD_VLM_RISK_MIN_CONFIDENCE", "0.45"))
    except (TypeError, ValueError):
        min_confidence = 0.45
    return confidence >= min_confidence or bool(verdict.get("is_violation"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_description_consistent_with_completion(
    action: Dict[str, Any],
    action_type: str,
    risk_level: str
) -> bool:
    """检查description和action_type/risk_level的一致性

    防止误报：如果description明确说"未完成"，但action被标记为completed，
    这种不一致应该阻止生成LeakFile。

    Returns:
        True: 一致，可以继续
        False: 不一致，应该降级或跳过
    """
    description = str(action.get("description", "") or "").lower()
    if str(action.get("evidence_source", "") or "") == "remote_vlm" or _has_structured_state(action):
        return original_action_type, original_risk_level

    if not description:
        return True  # 没有description，无法判断，保守通过

    # 如果不是completion状态，不需要检查
    if risk_level not in {"completed", "content_exposed"}:
        return True

    # 明确的不一致标记
    inconsistent_markers = [
        "not sent",
        "not complete",
        "has not been sent",
        "hasn't been sent",
        "not yet sent",
        "email has not been sent yet",
        "message has not been sent",
        "draft, not sent",
        "preparing to send",
        "preparing to upload",
        "about to upload",
        "about to send",
        "upload not started",
        "cancelled",
        "failed",
        "unsuccessful",
        "no visual confirmation of completion",
        "no evidence of sending",
        "no confirmation that",
    ]

    # 检查是否有不一致证据
    for marker in inconsistent_markers:
        if marker in description:
            # 严重不一致：description说未完成，但标记为completed
            return False

    # 特殊检查：如果是"draft"但同时有completion证据，允许
    if "draft" in description:
        # 检查是否同时有completion关键词
        completion_keywords = [
            "upload complete", "sent successfully", "published",
            "completed", "success", "finished"
        ]
        has_completion = any(kw in description for kw in completion_keywords)

        if not has_completion:
            # 只有draft，没有completion证据
            return False

    return True


def _infer_completion_from_description(action: Dict[str, Any]) -> Tuple[str, str]:
    """基于description智能推断真实的action_type和risk_level

    解决问题：VLM可能在description中描述了completion证据，
    但action_type和risk_level较保守，导致无法生成LeakFile fact。

    策略：
    1. 检查description中的completion关键词
    2. 如果有明确的完成证据，升级action_type和risk_level
    3. 如果有明确的未完成证据，保持原值
    """
    original_action_type = str(action.get("action_type", "") or "unknown")
    original_risk_level = str(action.get("risk_level", "") or "")
    description = str(action.get("description", "") or "").lower()

    # 如果没有description或已经是completed，直接返回原值
    if not description or original_risk_level == "completed":
        return original_action_type, original_risk_level

    # 明确的未完成/取消证据 - 不升级
    negative_markers = [
        "not sent", "not complete", "not upload", "not publish", "not share",
        "has not been sent", "hasn't been sent", "not yet sent",
        "cancelled", "failed", "unsuccessful",
        "preparing", "in preparation", "about to",
        "no confirmation", "no evidence of completion", "no visual confirmation",
        "not been sent yet"  # 明确的未发送
    ]

    # 注意：不包含单独的"draft"，因为draft中可以完成附件上传
    # 只有当明确说"draft, not sent"时才阻止
    if any(marker in description for marker in negative_markers):
        return original_action_type, original_risk_level

    # 完成证据关键词映射
    completion_patterns = {
        # 上传完成
        "upload": [
            "upload complete", "upload successful", "uploaded successfully",
            "upload finished", "file uploaded", "upload done",
            "transfer complete", "transfer successful",
            "upload complete", "complete"  # 更宽松的匹配
        ],
        # 发送完成
        "send": [
            "sent successfully", "message sent", "email sent",
            "send complete", "successfully sent", "has been sent",
            "delivery confirmed", "delivered successfully"
        ],
        # 发布完成
        "publish": [
            "published successfully", "publish complete", "successfully published",
            "post complete", "posted successfully", "shared successfully"
        ],
        # 内容暴露
        "expose": [
            "visible in", "displayed in", "shown in", "appears in",
            "content exposed", "file visible", "document visible",
            "clearly visible", "can be seen", "is visible"
        ]
    }

    # 检测completion证据
    detected_type = None
    has_completion_evidence = False

    for action_category, markers in completion_patterns.items():
        if any(marker in description for marker in markers):
            has_completion_evidence = True

            # 根据检测到的类别推断action_type
            if action_category == "upload":
                detected_type = "upload_complete"
            elif action_category == "send":
                detected_type = "send_message"
            elif action_category == "publish":
                detected_type = "publish_content"
            elif action_category == "expose":
                # content_exposed不改变action_type，只升级risk_level
                pass
            break

    # 决策逻辑
    upgraded_action_type = original_action_type
    upgraded_risk_level = original_risk_level

    if has_completion_evidence:
        # 升级action_type（如果检测到更明确的类型）
        if detected_type and original_action_type in {
            "attach_file", "select_file", "upload_start",
            "open_file", "unknown", "none"
        }:
            upgraded_action_type = detected_type

        # 升级risk_level
        if original_risk_level in {
            "selected_or_attached", "in_progress", "staging",
            "attempted", "suspected", "none", ""
        }:
            upgraded_risk_level = "completed"

    return upgraded_action_type, upgraded_risk_level


def _normalize_action_type(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "upload": "upload_complete",
        "send": "send_message",
        "share": "screen_share",
        "publish": "publish_content",
        "commit": "publish_content",
        "ai_input": "external_exposure",
        "local_gate": "external_exposure",
        "local_ocr": "external_exposure",
    }
    text = aliases.get(text, text)
    allowed = {
        "open_file",
        "copy_content",
        "paste_content",
        "select_file",
        "attach_file",
        "upload_start",
        "upload_complete",
        "send_message",
        "publish_content",
        "screenshot",
        "screen_record",
        "screen_share",
        "save_as",
        "convert_file",
        "compress_file",
        "rename_file",
        "vm_copy",
        "external_exposure",
        "none",
        "unknown",
    }
    return text if text in allowed else "unknown"


def _frame_timestamps_by_index(verdict: Optional[Dict[str, Any]]) -> Dict[int, str]:
    if not isinstance(verdict, dict):
        return {}
    result: Dict[int, str] = {}
    for frame in verdict.get("frame_selection", []) or []:
        if not isinstance(frame, dict):
            continue
        try:
            index = int(frame.get("index", 0) or 0)
        except (TypeError, ValueError):
            continue
        if index:
            result[index] = str(frame.get("timestamp", "") or "")
    return result


def _normalize_frontend_app(value: Any, logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if isinstance(value, dict):
        app = dict(value)
    else:
        app = {}

    if logs and not app.get("category"):
        context = _frontend_context_from_logs(logs)
        primary = context.get("primary") or {}
        app.update(
            {
                "name": primary.get("display_name", ""),
                "category": primary.get("category", "unknown"),
                "capabilities": primary.get("capabilities", []),
            }
        )

    capabilities = app.get("capabilities", [])
    if not isinstance(capabilities, list):
        capabilities = [str(capabilities)]
    return {
        "name": str(app.get("name") or app.get("display_name") or app.get("app") or ""),
        "category": str(app.get("category") or "unknown"),
        "capabilities": sorted({str(item) for item in capabilities if str(item or "").strip()}),
    }


def _normalize_observed_actions(
    verdict: Dict[str, Any],
    sensitive_files: List[str],
    logs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    frontend = _normalize_frontend_app(verdict.get("frontend_app"), logs)
    frame_times = _frame_timestamps_by_index(verdict)
    raw_actions = verdict.get("observed_actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        evidence_frames = [
            int(item)
            for item in verdict.get("evidence_frames", []) or []
            if str(item).strip().isdigit()
        ]
        action_type = _normalize_action_type(str(verdict.get("completed_action", "") or "unknown"))
        raw_actions = [
            {
                "action_type": action_type,
                "risk_level": verdict.get("risk_level", ""),
                "transfer_direction": verdict.get("transfer_direction", ""),
                "terminal_state": verdict.get("terminal_state", ""),
                "artifact_state": verdict.get("artifact_state", ""),
                "completion_evidence": verdict.get("completion_evidence", []),
                "non_completion_evidence": verdict.get("non_completion_evidence", []),
                "time": frame_times.get(evidence_frames[0], "") if evidence_frames else _verdict_timestamp({}, logs or []),
                "app": frontend.get("name", ""),
                "app_category": frontend.get("category", "unknown"),
                "source_file": sensitive_files[0] if sensitive_files else "",
                "evidence_frames": evidence_frames,
                "confidence": verdict.get("confidence", 0.0),
                "description": str(verdict.get("reason", "") or ""),
            }
        ]

    normalized: List[Dict[str, Any]] = []
    for index, action in enumerate(raw_actions):
        if not isinstance(action, dict):
            continue
        risk_level = _normalize_risk_level(str(action.get("risk_level", "") or "")) or _normalize_risk_level(
            str(verdict.get("risk_level", "") or "")
        )
        evidence_frames = []
        for item in action.get("evidence_frames", []) or []:
            try:
                evidence_frames.append(int(item))
            except (TypeError, ValueError):
                continue
        action_time = str(action.get("time", "") or "")
        if not action_time and evidence_frames:
            action_time = frame_times.get(evidence_frames[0], "")
        transfer_direction = _normalize_transfer_direction(
            str(action.get("transfer_direction") or verdict.get("transfer_direction") or "")
        )
        terminal_state = _normalize_terminal_state(
            str(action.get("terminal_state") or verdict.get("terminal_state") or "")
        )
        artifact_state = _normalize_artifact_state(
            str(action.get("artifact_state") or verdict.get("artifact_state") or "")
        )
        completion_evidence = _structured_evidence_list(
            action.get("completion_evidence") or verdict.get("completion_evidence")
        )
        non_completion_evidence = _structured_evidence_list(
            action.get("non_completion_evidence") or verdict.get("non_completion_evidence")
        )
        normalized.append(
            {
                "action_id": f"vlm_action_{index}",
                "segment_id": str(action.get("segment_id") or verdict.get("segment_id") or ""),
                "action_type": _normalize_action_type(str(action.get("action_type", "") or "")),
                "risk_level": risk_level or "none",
                "transfer_direction": transfer_direction,
                "terminal_state": terminal_state,
                "artifact_state": artifact_state,
                "completion_evidence": completion_evidence,
                "non_completion_evidence": non_completion_evidence,
                "time": action_time,
                "app": str(action.get("app") or frontend.get("name") or ""),
                "app_category": str(action.get("app_category") or frontend.get("category") or "unknown"),
                "source_file": str(action.get("source_file") or (sensitive_files[0] if sensitive_files else "")),
                "derived_file": str(action.get("derived_file") or ""),
                "evidence_frames": evidence_frames,
                "confidence": round(_safe_float(action.get("confidence", verdict.get("confidence", 0.0))), 4),
                "description": str(action.get("description") or verdict.get("reason", "") or ""),
                "evidence_source": "remote_vlm" if verdict.get("status") == "success" else str(verdict.get("status", "vlm")),
            }
        )

    sensitive_segments = {
        str(item.get("segment_id", "") or "")
        for item in normalized
        if _is_sensitive_ref(str(item.get("description", "") or ""), sensitive_files)
        or _is_sensitive_ref(str(item.get("derived_file", "") or ""), sensitive_files)
    }
    for item in normalized:
        item["segment_sensitive_object_confirmed"] = str(item.get("segment_id", "") or "") in sensitive_segments

    if (
        not any(item.get("risk_level") in {"content_exposed", "completed"} for item in normalized)
        or not any(_normalize_action_type(str(item.get("action_type", ""))) not in {"none", "unknown"} for item in normalized)
    ):
        reason_action = _best_vlm_reason_action(verdict, sensitive_files, logs)
        if reason_action:
            normalized.append(reason_action)
    return normalized


def _merge_segment_verdicts(
    segment_verdicts: List[Dict[str, Any]],
    sensitive_files: List[str],
    logs: List[Dict[str, Any]],
    max_frames_requested: int,
    frame_plan: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    successful = [item for item in segment_verdicts if item.get("status") == "success"]
    positive_segments = [item for item in successful if _vlm_final_positive(item)]
    if positive_segments:
        best = max(positive_segments, key=lambda item: _safe_float(item.get("confidence", 0.0)))
    elif successful:
        best = max(successful, key=lambda item: _safe_float(item.get("confidence", 0.0)))
    elif segment_verdicts:
        best = segment_verdicts[0]
    else:
        best = {
            "status": "skipped",
            "reason": "no_segments_reviewed",
            "is_violation": False,
            "confidence": 0.0,
            "completed_action": "none",
        }

    observed_actions: List[Dict[str, Any]] = []
    frame_selection: List[Dict[str, Any]] = []
    visual_observations: List[Dict[str, Any]] = []
    reasons = []
    frames_sent = 0
    frame_context_count = 0
    for verdict in segment_verdicts:
        reasons.append(f"{verdict.get('segment_id', '')}:{verdict.get('reason', '')}")
        frames_sent += int(verdict.get("frames_sent", 0) or 0)
        frame_context_count += int(verdict.get("frame_context_count", 0) or 0)
        for action in verdict.get("observed_actions", []) or []:
            if isinstance(action, dict):
                observed_actions.append(dict(action))
        for frame in verdict.get("frame_selection", []) or []:
            if isinstance(frame, dict):
                frame_selection.append(dict(frame))
        for observation in verdict.get("visual_observations", []) or []:
            if isinstance(observation, dict):
                visual_observations.append(dict(observation))

    merged = dict(best)
    merged["status"] = "success" if any(item.get("status") == "success" for item in successful) else str(best.get("status", "skipped"))
    merged["is_violation"] = bool(positive_segments)
    merged["risk_level"] = str(best.get("risk_level", "") or _infer_risk_level_from_verdict(best) or "none")
    merged["confidence"] = max((_safe_float(item.get("confidence", 0.0)) for item in segment_verdicts), default=0.0)
    merged["completed_action"] = str(best.get("completed_action", "") or "none")
    merged["reason"] = "; ".join(reason for reason in reasons if reason)[:1000]
    merged["segment_verdicts"] = segment_verdicts
    merged["observed_actions"] = observed_actions
    merged["frame_selection"] = frame_selection
    merged["visual_observations"] = visual_observations
    merged["frames_sent"] = frames_sent
    merged["frame_context_count"] = frame_context_count
    merged["max_frames_requested"] = max_frames_requested
    merged["frame_plan"] = frame_plan
    merged["model"] = model
    _postprocess_vlm_verdict(merged)
    _postprocess_vlm_actions(merged, sensitive_files, logs)
    _postprocess_vlm_verdict(merged)
    return merged


def _postprocess_vlm_actions(
    verdict: Dict[str, Any],
    sensitive_files: List[str],
    logs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    verdict["frontend_app"] = _normalize_frontend_app(verdict.get("frontend_app"), logs)
    verdict["observed_actions"] = _normalize_observed_actions(verdict, sensitive_files, logs)

    # OPTIMIZATION: Add consistency validation
    _validate_vlm_consistency(verdict)


def _validate_vlm_consistency(verdict: Dict[str, Any]) -> None:
    """
    验证VLM输出的一致性，防止描述和判断矛盾。
    如果检测到矛盾，降级risk_level和confidence。
    """
    reason = str(verdict.get("reason", "")).lower()
    risk_level = str(verdict.get("risk_level", "")).lower()

    # Negative signals in description
    negative_signals = [
        'not sent', 'not submitted', 'not uploaded', 'not yet',
        'cancelled', 'draft', 'preparation', 'not completed',
        '未发送', '未提交', '未上传', '取消', '草稿',
        'but not', 'without', 'no evidence', 'no visible',
        'selected but not', 'attached but not',
    ]

    # Check for contradiction
    has_negative = any(signal in reason for signal in negative_signals)
    is_high_risk = risk_level in ['completed', 'content_exposed']

    if has_negative and is_high_risk:
        # Contradiction detected - downgrade
        verdict['raw_is_violation'] = verdict.get('is_violation')
        verdict['risk_level'] = 'selected_or_attached'
        verdict['is_violation'] = False
        verdict['confidence'] = verdict.get('confidence', 0.8) * 0.5
        verdict['consistency_warning'] = 'description_contradicts_high_risk_level'
        verdict['original_risk_level'] = risk_level

        # Also downgrade observed_actions
        for action in verdict.get('observed_actions', []):
            if action.get('risk_level') in ['completed', 'content_exposed']:
                action['risk_level'] = 'selected_or_attached'
                action['confidence'] = action.get('confidence', 0.8) * 0.5
                action['consistency_warning'] = 'downgraded_due_to_negative_description'


def _frame_selection_payload(frame_records: List[Dict[str, Any]], segment_id: str) -> List[Dict[str, Any]]:
    return [
        {
            "index": image["index"],
            "segment_id": segment_id,
            "timestamp": image["timestamp"],
            "frame_index": image["frame_index"],
            "scene_score": image.get("scene_score"),
            "status_region_score": image.get("status_region_score"),
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


def _visual_observations_from_frames(frame_records: List[Dict[str, Any]], segment_id: str) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for image in frame_records:
        ocr_text = str(image.get("ocr_text", "") or "")
        flags = list(image.get("ocr_flags", []) or [])
        terminal_status = _terminal_status_from_text(ocr_text)
        reason = str(image.get("selection_reason", "") or "")
        status_score = float(image.get("status_region_score", 0.0) or 0.0)
        if not terminal_status and not flags and "status_region_candidate" not in reason:
            continue
        obs_type = "visual_status_region"
        if terminal_status:
            obs_type = f"visual_terminal_{terminal_status}"
        elif flags:
            obs_type = "visual_ocr_flag"
        observations.append(
            {
                "type": obs_type,
                "segment_id": segment_id,
                "frame_index": image.get("frame_index"),
                "frame_number": image.get("index"),
                "timestamp": image.get("timestamp"),
                "image_sent": bool(image.get("image_sent")),
                "terminal_status": terminal_status,
                "ocr_flags": flags,
                "ocr_text": _compact_text(ocr_text, 240),
                "selection_reason": reason,
                "status_region_score": round(status_score, 4),
            }
        )
    return observations


def _live_vlm_review_case(
    case_dir: Path,
    groundtruth: Any,
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    fallback_meta: Dict[str, Any],
    max_frames: int,
) -> Dict[str, Any]:
    video_path = _choose_video_file(case_dir)
    rec_start = _recording_start(groundtruth, logs, video_path)
    if not video_path or not rec_start:
        return {
            "status": "skipped",
            "reason": "missing_video_or_recording_start",
            "is_violation": True,
            "max_frames_requested": max_frames,
        }

    api_key = _first_env("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "VL_API_KEY")
    base_url = _first_env("OPENAI_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "VL_BASE_URL")
    model = _first_env("VL_MODEL_NAME", "OPENAI_MODEL", "QWEN_VL_MODEL", "QWEN_MODEL") or "qwen3.7-plus"
    rec_end = _video_end_time(video_path, rec_start)
    fallback_meta = dict(fallback_meta)
    sink_sessions = [
        item for item in fallback_meta.get("sink_sessions", []) or []
        if isinstance(item, dict)
    ] or _build_sink_sessions(fallback_meta, logs, sensitive_files, rec_start, rec_end)
    if sink_sessions:
        fallback_meta["sink_sessions"] = sink_sessions
    cache_path = _vlm_review_cache_path(
        case_dir=case_dir,
        video_path=video_path,
        rec_start=rec_start,
        fallback_meta=fallback_meta,
        logs=logs,
        sensitive_files=sensitive_files,
        max_frames=max_frames,
        model=model,
        base_url=base_url,
    )
    cached = _read_vlm_review_cache(cache_path)
    if cached:
        _progress(f"[VLM CACHE HIT] case={case_dir.name} status={cached.get('status', 'unknown')} path={cache_path}")
        return cached

    def remember(verdict: Dict[str, Any]) -> Dict[str, Any]:
        if str(verdict.get("status", "")) in {"success", "skipped"}:
            _write_vlm_review_cache(cache_path, verdict)
        return verdict

    windows = _windows_from_fallback(fallback_meta, logs)
    segment_unit = max(2, FRAMES_PER_SEGMENT // 2)
    max_segments = max(1, min(MAX_SEGMENTS_PER_CASE, max(1, (int(max_frames) + segment_unit - 1) // segment_unit)))
    review_segments, segment_plan = _prepare_review_segments(windows, fallback_meta, logs, max_segments=max_segments)
    if not review_segments:
        return remember(
            {
                "status": "skipped",
                "reason": "no_review_segments",
                "is_violation": True,
                "max_frames_requested": max_frames,
                "frame_plan": {"segment_level": True, "segment_plan": segment_plan, "segments": []},
            }
        )

    HumanMessage = None
    llm = None
    if api_key:
        from langchain_core.messages import HumanMessage as LangchainHumanMessage
        from langchain_openai import ChatOpenAI

        HumanMessage = LangchainHumanMessage
        llm = ChatOpenAI(model=model, base_url=base_url or None, api_key=api_key)
    segment_verdicts: List[Dict[str, Any]] = []
    segment_plans: List[Dict[str, Any]] = []
    frames_remaining = max(1, int(max_frames))

    for segment_index, segment in enumerate(review_segments, 1):
        segment_id = f"vlm_seg_{segment_index:02d}"
        remaining_segments = max(1, len(review_segments) - segment_index + 1)
        balanced_budget = (frames_remaining + remaining_segments - 1) // remaining_segments
        segment_frame_budget = min(frames_remaining, FRAMES_PER_SEGMENT, max(2, balanced_budget))
        if segment_frame_budget <= 0:
            break

        frame_records, frame_plan = _extract_representative_frame_images(
            video_path,
            rec_start,
            segment,
            segment_id,
            fallback_meta,
            logs,
            sensitive_files,
            segment_frame_budget,
            max_edge=IMAGE_MAX_EDGE,
            jpeg_quality=JPEG_QUALITY,
        )
        frame_plan["segment_index"] = segment_index
        frame_plan["sink_sessions"] = _sessions_for_segment(sink_sessions, segment)
        segment_plans.append(frame_plan)
        frames_remaining -= len(frame_records)
        image_records = [item for item in frame_records if item.get("image_sent")]

        if not frame_records:
            segment_verdicts.append(
                {
                    "status": "skipped",
                    "segment_id": segment_id,
                    "reason": "no_frames_extracted",
                    "is_violation": False,
                    "frames_sent": 0,
                    "frame_context_count": 0,
                    "max_frames_requested": segment_frame_budget,
                    "frame_plan": frame_plan,
                    "frame_selection": [],
                    "visual_observations": [],
                }
            )
            continue

        visual_observations = _visual_observations_from_frames(frame_records, segment_id)

        if not llm:
            segment_verdicts.append(
                {
                    "status": "skipped",
                    "segment_id": segment_id,
                    "reason": "missing_vlm_api_key",
                    "is_violation": False,
                    "frames_sent": len(image_records),
                    "frame_context_count": len(frame_records),
                    "max_frames_requested": segment_frame_budget,
                    "frame_plan": frame_plan,
                    "frame_selection": _frame_selection_payload(frame_records, segment_id),
                    "visual_observations": visual_observations,
                }
            )
            continue

        candidate_events = fallback_meta.get("candidate_events", [])[:12]
        review_logs = _review_log_context(logs, [segment])
        segment_sessions = _sessions_for_segment(sink_sessions, segment)
        prompt = _build_vlm_review_prompt(
            sensitive_files,
            candidate_events,
            review_logs,
            segment_sessions,
            segment_id,
            segment,
        )
        contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in frame_records:
            contents.append(
                {
                    "type": "text",
                    "text": (
                        f"Segment {segment_id} frame {image['index']} @ {image['timestamp']} "
                        f"(source_frame={image['frame_index']}, reason={image.get('selection_reason', '')}, "
                        f"scene_score={image.get('scene_score', 0.0)}, "
                        f"status_region_score={image.get('status_region_score', 0.0)}, "
                        f"image_sent={str(bool(image.get('image_sent'))).lower()}, "
                        f"ocr_flags={image.get('ocr_flags', [])}, "
                        f"ocr={json.dumps(str(image.get('ocr_text', '') or '')[:300], ensure_ascii=False)})"
                    ),
                }
            )
            if image.get("image_sent") and image.get("b64"):
                contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image['b64']}"}})

        response = llm.invoke([HumanMessage(content=contents)])
        text = str(response.content or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            verdict = {
                "status": "failed",
                "segment_id": segment_id,
                "reason": "non_json_response",
                "raw": text,
                "is_violation": False,
            }
        else:
            try:
                verdict = json.loads(match.group(0), strict=False)
                verdict["status"] = "success"
            except json.JSONDecodeError:
                verdict = {
                    "status": "failed",
                    "segment_id": segment_id,
                    "reason": "bad_json_response",
                    "raw": text,
                    "is_violation": False,
                }

        verdict["segment_id"] = segment_id
        verdict["segment_time_range"] = {
            "start": segment[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end": segment[1].strftime("%Y-%m-%d %H:%M:%S"),
        }
        verdict["frames_sent"] = len(image_records)
        verdict["frame_context_count"] = len(frame_records)
        verdict["max_frames_requested"] = segment_frame_budget
        verdict["frame_plan"] = frame_plan
        verdict["frame_selection"] = _frame_selection_payload(frame_records, segment_id)
        verdict["visual_observations"] = visual_observations
        verdict["model"] = model
        _postprocess_vlm_verdict(verdict)
        _postprocess_vlm_actions(verdict, sensitive_files, logs)
        segment_verdicts.append(verdict)

    frame_plan = {
        "segment_level": True,
        "segment_plan": segment_plan,
        "segments": segment_plans,
    }
    return remember(_merge_segment_verdicts(segment_verdicts, sensitive_files, logs, max_frames, frame_plan, model))


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

    frame_times = _frame_timestamps_by_index(verdict)
    frame_by_key = {
        (str(frame.get("segment_id", "")), int(frame.get("index", 0) or 0)): frame
        for frame in verdict.get("frame_selection", []) or []
        if isinstance(frame, dict)
    }
    action_segments: List[Dict[str, Any]] = []
    for index, action in enumerate(verdict.get("observed_actions", []) or []):
        if not isinstance(action, dict):
            continue
        risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
        if risk_level not in POSITIVE_RISK_LEVELS:
            continue

        segment_id = str(action.get("segment_id", "") or f"vlm_seg_action_{index}")
        evidence_frames = []
        for item in action.get("evidence_frames", []) or []:
            try:
                evidence_frames.append(int(item))
            except (TypeError, ValueError):
                continue
        supporting_timestamps = []
        for frame_index in evidence_frames:
            frame = frame_by_key.get((segment_id, frame_index))
            if frame and frame.get("timestamp"):
                supporting_timestamps.append(str(frame.get("timestamp")))
            elif frame_index in frame_times:
                supporting_timestamps.append(frame_times[frame_index])
        if action.get("time"):
            supporting_timestamps.append(str(action.get("time")))
        supporting_timestamps = sorted({item for item in supporting_timestamps if str(item or "").strip()})

        segment_range = ""
        for segment_verdict in verdict.get("segment_verdicts", []) or []:
            if not isinstance(segment_verdict, dict) or str(segment_verdict.get("segment_id", "")) != segment_id:
                continue
            time_range = segment_verdict.get("segment_time_range")
            if isinstance(time_range, dict):
                start = str(time_range.get("start", "") or "")
                end = str(time_range.get("end", "") or "")
                segment_range = f"{start} - {end}" if start or end else ""
                break
        if not segment_range and supporting_timestamps:
            segment_range = f"{supporting_timestamps[0]} - {supporting_timestamps[-1]}"

        source_file = str(action.get("source_file", "") or "")
        derived_file = str(action.get("derived_file", "") or "")
        primary_resource = source_file or derived_file or (sensitive_files[0] if sensitive_files else "unknown")
        related_resources = [item for item in [derived_file, *sensitive_files] if item and item != primary_resource]
        visible_evidence = [
            str(action.get("description", "") or ""),
            f"risk_level={risk_level}",
            f"evidence_source={action.get('evidence_source', 'remote_vlm')}",
        ]
        if evidence_frames:
            visible_evidence.append(f"frame_ids={','.join(str(item) for item in evidence_frames)}")
        action_segments.append(
            {
                "segment_id": f"{segment_id}_action_{index}",
                "time_range": segment_range,
                "app_name": str(action.get("app", "") or "unknown"),
                "operation_type": _normalize_action_type(str(action.get("action_type", "") or "")),
                "primary_resource": primary_resource,
                "related_resources": related_resources[:8],
                "action_description": str(action.get("description", "") or verdict.get("reason", "") or ""),
                "visible_evidence": [item for item in visible_evidence if item],
                "supporting_timestamps": supporting_timestamps,
                "confidence": round(_safe_float(action.get("confidence", verdict.get("confidence", 0.0))), 4),
                "evidence_source": action.get("evidence_source", "remote_vlm"),
                "frame_ids": evidence_frames,
                "source_segment_id": segment_id,
            }
        )

    if action_segments:
        return action_segments

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


def _logs_for_correlation(
    logs: List[Dict[str, Any]],
    fallback_meta: Dict[str, Any],
    limit: int = 160,
) -> List[Dict[str, Any]]:
    windows = _windows_from_fallback(fallback_meta, logs)
    if not windows:
        return logs[:limit]

    rows: List[Dict[str, Any]] = []
    for log in logs:
        dt = _parse_dt(log.get("timestamp", ""))
        if not dt:
            continue
        if not any(start - timedelta(seconds=10) <= dt <= end + timedelta(seconds=60) for start, end in windows):
            continue
        rows.append(log)
        if len(rows) >= limit:
            break
    return rows or logs[:limit]


def _frontend_context_from_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from data_leak_detector.event_correlator import classify_frontend_app
    except Exception:
        classify_frontend_app = None

    categories: Dict[str, int] = {}
    capabilities: set[str] = set()
    observations: List[Dict[str, Any]] = []
    for log in logs[:300]:
        if classify_frontend_app:
            frontend = classify_frontend_app(log)
        else:
            frontend = {"category": "unknown", "display_name": str(log.get("app_name", "") or ""), "capabilities": []}
        category = str(frontend.get("category", "unknown") or "unknown")
        categories[category] = categories.get(category, 0) + 1
        capabilities.update(str(item) for item in frontend.get("capabilities", []) or [] if str(item or "").strip())
        if frontend.get("is_external") or frontend.get("capabilities"):
            observations.append(
                {
                    "timestamp": log.get("timestamp", ""),
                    "category": category,
                    "display_name": frontend.get("display_name", ""),
                    "capabilities": frontend.get("capabilities", []),
                    "window_title": frontend.get("window_title", ""),
                }
            )
        if len(observations) >= 12:
            break

    primary_category = "unknown"
    if categories:
        primary_category = max(categories.items(), key=lambda item: item[1])[0]
    return {
        "primary": {
            "category": primary_category,
            "display_name": observations[0].get("display_name", primary_category) if observations else primary_category,
            "capabilities": sorted(capabilities),
        },
        "category_counts": categories,
        "capabilities": sorted(capabilities),
        "observations": observations,
    }


def _review_source(
    deterministic_positive: bool,
    vlm_verdict: Optional[Dict[str, Any]],
    vlm_live_queued: bool,
) -> str:
    if not isinstance(vlm_verdict, dict):
        if vlm_live_queued:
            return "remote_vlm_pending"
        return "deterministic" if deterministic_positive else "triage"
    status = str(vlm_verdict.get("status", "") or "")
    if status == "local_positive":
        return "local_gate"
    if status == "success":
        return "remote_vlm_cache" if vlm_verdict.get("cache_hit") else "remote_vlm"
    return status or ("deterministic" if deterministic_positive else "unknown")


def _confirmed_leak_positive(
    deterministic_positive: bool,
    vlm_verdict: Optional[Dict[str, Any]],
    correlation_bundle: Optional[Dict[str, Any]],
) -> bool:
    if deterministic_positive:
        return True
    if isinstance(correlation_bundle, dict) and correlation_bundle.get("upload_candidates"):
        return True
    if not isinstance(vlm_verdict, dict):
        return False
    status = str(vlm_verdict.get("status", "") or "")
    if status != "success":
        return False
    return bool(vlm_verdict.get("is_violation"))


def _vlm_only_confirmed_positive(vlm_verdict: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(vlm_verdict, dict):
        return False
    status = str(vlm_verdict.get("status", "") or "")
    if status != "success":
        return False
    risk_level = _normalize_risk_level(str(vlm_verdict.get("risk_level", "") or ""))
    if not risk_level:
        risk_level = _infer_risk_level_from_verdict(vlm_verdict)
    return bool(vlm_verdict.get("is_violation")) and is_confirmed_risk_level(risk_level)


def _semantic_frame_coverage(vlm_verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(vlm_verdict, dict) or not vlm_verdict:
        return {
            "available": False,
            "sampled_frames": 0,
            "image_frames_sent": 0,
            "ocr_frames": 0,
            "ocr_risk_frames": 0,
            "completion_anchor": False,
            "content_exposed_anchor": False,
            "staging_anchor": False,
            "external_sink_anchor": False,
            "sensitive_object_anchor": False,
            "visual_terminal_statuses": [],
        }

    frames = [frame for frame in vlm_verdict.get("frame_selection", []) or [] if isinstance(frame, dict)]
    actions = [action for action in vlm_verdict.get("observed_actions", []) or [] if isinstance(action, dict)]
    observations = [item for item in vlm_verdict.get("visual_observations", []) or [] if isinstance(item, dict)]
    action_text = " ".join(_action_text(action) for action in actions)
    reason_text = str(vlm_verdict.get("reason", "") or "").lower()
    semantic_text = f"{reason_text} {action_text}"
    frame_text = " ".join(
        " ".join(
            str(frame.get(key, "") or "").lower()
            for key in ("selection_reason", "ocr_text")
        )
        for frame in frames
    )
    combined_context = f"{semantic_text} {frame_text}"
    visual_terminal_statuses = sorted(
        {
            str(item.get("terminal_status", "") or "")
            for item in observations
            if str(item.get("terminal_status", "") or "")
        }
    )

    content_markers = (
        "content_exposed",
        "sensitive content is visible",
        "visible in the input",
        "pasted into",
        "appears in the conversation",
        "external_exposure",
    )
    staging_markers = (
        "selected_or_attached",
        "in_progress",
        "selected",
        "attached",
        "file picker",
        "uploading",
        "upload_start",
    )
    sensitive_object_anchor = any(
        str(action.get(key, "") or "").strip()
        for action in actions
        for key in ("source_file", "derived_file", "shared_data", "clipboard_data")
    ) or "sensitive file" in semantic_text or "sensitive content" in semantic_text

    return {
        "available": bool(frames or actions or vlm_verdict.get("reason")),
        "sampled_frames": len(frames),
        "image_frames_sent": sum(1 for frame in frames if frame.get("image_sent")),
        "ocr_frames": sum(1 for frame in frames if frame.get("ocr_ran")),
        "ocr_risk_frames": sum(1 for frame in frames if frame.get("ocr_flags")),
        "completion_anchor": _completion_evidence_from_text(semantic_text),
        "visual_completion_anchor": "completed" in visual_terminal_statuses,
        "visual_terminal_statuses": visual_terminal_statuses,
        "content_exposed_anchor": any(marker in semantic_text for marker in content_markers),
        "staging_anchor": any(marker in semantic_text for marker in staging_markers),
        "external_sink_anchor": any(marker in combined_context for marker in EXTERNAL_SINK_TOKENS)
        or any(
            str(action.get("app_category", "") or "") in {"ai_service", "cloud_storage", "messaging", "email", "meeting"}
            for action in actions
        ),
        "sensitive_object_anchor": sensitive_object_anchor,
    }


def _audit_evidence_sources(actions: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(action.get("evidence_source", "") or "").strip()
            for action in actions
            if str(action.get("evidence_source", "") or "").strip()
        }
    )


def _relation_counts(facts: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fact in facts:
        relation = str(fact.get("relation", "") or "unknown")
        counts[relation] = counts.get(relation, 0) + 1
    return counts


def _compact_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _detection_actions(
    case_id: str,
    detection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for index, event in enumerate(detection.get("upload_events", []) or []):
        if hasattr(event, "__dict__"):
            raw = dict(event.__dict__)
        elif isinstance(event, dict):
            raw = event
        else:
            raw = {}
        source_file = str(raw.get("original_file", "") or raw.get("file_path", "") or raw.get("file_name", "") or "")
        uploaded_file = str(
            raw.get("upload_content", "")
            or raw.get("file_path", "")
            or raw.get("file_name", "")
            or source_file
        )
        actions.append(
            {
                "action_id": f"{case_id}:det_upload_{index}",
                "action_type": "upload_start",
                "risk_level": "in_progress",
                "time": str(raw.get("timestamp", "") or raw.get("time", "") or ""),
                "app": str(raw.get("app_name", "") or raw.get("process_name", "") or ""),
                "app_category": "unknown",
                "source_file": source_file,
                "derived_file": uploaded_file if uploaded_file != source_file else "",
                "evidence_frames": [],
                "confidence": round(_safe_float(raw.get("confidence", 0.95), 0.95), 4),
                "description": str(raw.get("description", "") or "deterministic upload event requiring visual confirmation"),
                "evidence_source": "deterministic",
            }
        )
    return actions


def _visual_review_meta_for_deterministic_evidence(
    fallback_meta: Dict[str, Any],
    detection: Dict[str, Any],
    log_rule_signal: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    meta = dict(fallback_meta or {})
    reasons = set(str(item) for item in meta.get("reasons", []) or [] if str(item or "").strip())
    candidate_events = list(meta.get("candidate_events", []) or [])

    def add_candidate(timestamp: str, event_type: str, app_name: str = "", reason: str = "") -> None:
        reasons.add(reason or "deterministic_evidence_requires_visual_confirmation")
        candidate_events.append(
            {
                "timestamp": timestamp,
                "event_type": event_type,
                "app_name": app_name,
                "reason": reason or "deterministic_evidence_requires_visual_confirmation",
            }
        )

    for event in detection.get("upload_events", []) or []:
        raw = dict(event.__dict__) if hasattr(event, "__dict__") else (event if isinstance(event, dict) else {})
        add_candidate(
            str(raw.get("timestamp", "") or raw.get("time", "") or ""),
            str(raw.get("event_type", "") or raw.get("operation_type", "") or "deterministic_upload_event"),
            str(raw.get("app_name", "") or raw.get("process_name", "") or ""),
            "deterministic_upload_requires_visual_confirmation",
        )

    evidence = log_rule_signal.get("evidence", {}) if isinstance(log_rule_signal, dict) else {}
    for rule, entries in (evidence or {}).items():
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            add_candidate(
                str(entry.get("timestamp", "") or ""),
                str(entry.get("event_type", "") or rule),
                str(entry.get("app_name", "") or entry.get("process_name", "") or ""),
                f"log_rule_{rule}_requires_visual_confirmation",
            )

    if not candidate_events and logs:
        for log in logs:
            path = str(log.get("file_path", "") or log.get("file_name", "") or "")
            event_type = str(log.get("event_type", "") or "")
            if not path and event_type not in {"file_upload", "data_upload", "file_send", "file_share", "file_selected"}:
                continue
            add_candidate(
                str(log.get("timestamp", "") or ""),
                event_type or "log_context",
                str(log.get("app_name", "") or ""),
                "log_context_requires_visual_confirmation",
            )
            if len(candidate_events) >= 12:
                break

    meta["used"] = True
    meta["decision"] = "run"
    meta["reasons"] = sorted(reasons) or ["deterministic_evidence_requires_visual_confirmation"]
    meta["candidate_events"] = candidate_events[:24]
    return meta


def _operation_record_actions(
    case_id: str,
    detection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for index, record in enumerate(detection.get("operation_records", []) or []):
        if not isinstance(record, dict):
            continue
        operation = str(record.get("operation", "") or "").lower()
        source_file = str(record.get("source_path", "") or record.get("original_file", "") or "")
        target_file = str(
            record.get("target_path", "")
            or record.get("derived_file", "")
            or record.get("sensitive_file_path", "")
            or ""
        )
        if not source_file:
            source_file = target_file
        action_type = "open_file"
        risk_level = "preparation"
        if any(token in operation for token in ("transform", "convert", "compress", "rename", "copy", "derive")):
            action_type = "convert_file"
            risk_level = "selected_or_attached"
        elif "upload" in operation or "send" in operation or "share" in operation:
            action_type = "upload_complete"
            risk_level = "completed"
        elif "open" in operation or "anchor" in operation or "hint" in operation:
            action_type = "open_file"

        actions.append(
            {
                "action_id": f"{case_id}:log_op_{index}",
                "action_type": action_type,
                "risk_level": risk_level,
                "time": str(record.get("operation_time", "") or record.get("timestamp", "") or ""),
                "app": str(record.get("app_name", "") or ""),
                "app_category": "unknown",
                "source_file": source_file,
                "derived_file": target_file if target_file != source_file else "",
                "evidence_frames": [],
                "confidence": 0.82,
                "description": str(record.get("description", "") or operation or "log operation record"),
                "evidence_source": "log_operation",
            }
        )
    return actions


def _file_mapping_actions(
    case_id: str,
    detection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mappings = detection.get("file_mappings", {}) or {}
    direct = mappings.get("direct_file_mappings", {}) if isinstance(mappings, dict) else {}
    if not isinstance(direct, dict):
        return []
    actions: List[Dict[str, Any]] = []
    for index, (child, parent) in enumerate(direct.items()):
        actions.append(
            {
                "action_id": f"{case_id}:log_map_{index}",
                "action_type": "convert_file",
                "risk_level": "selected_or_attached",
                "time": "",
                "app": "log_lineage",
                "app_category": "lineage",
                "source_file": str(parent or ""),
                "derived_file": str(child or ""),
                "evidence_frames": [],
                "confidence": 0.8,
                "description": f"log file mapping: {parent} -> {child}",
                "evidence_source": "log_lineage",
            }
        )
    return actions


def _log_sensitive_open_actions(
    case_id: str,
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    limit: int = 24,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for log in logs:
        path = str(log.get("file_path", "") or "")
        file_name = str(log.get("file_name", "") or "")
        candidate = path or file_name
        if not _is_sensitive_ref(candidate, sensitive_files):
            continue
        key = candidate.replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        process_info = log.get("process_info", {}) if isinstance(log.get("process_info"), dict) else {}
        actions.append(
            {
                "action_id": f"{case_id}:log_open_{len(actions)}",
                "action_type": "open_file",
                "risk_level": "preparation",
                "time": str(log.get("timestamp", "") or ""),
                "app": str(log.get("app_name", "") or process_info.get("process_name", "") or ""),
                "app_category": "unknown",
                "source_file": _canonical_sensitive_ref(candidate, sensitive_files),
                "derived_file": "",
                "evidence_frames": [],
                "confidence": 0.78,
                "description": f"log references sensitive file {candidate}",
                "evidence_source": "log_sensitive_open",
            }
        )
        if len(actions) >= limit:
            break
    return actions


def _correlation_actions(
    case_id: str,
    correlation_bundle: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(correlation_bundle, dict):
        return []
    actions: List[Dict[str, Any]] = []
    for index, candidate in enumerate(correlation_bundle.get("upload_candidates", []) or []):
        current_files = candidate.get("current_files", []) or []
        operation_type = str(candidate.get("operation_type", "") or "upload_complete")
        normalized_action = _normalize_action_type(operation_type)
        if operation_type in {"file_selected", "select_file", "attach_file"}:
            normalized_action = "select_file"
            risk_level = "selected_or_attached"
        elif operation_type in {"upload_start", "file_upload_start"}:
            normalized_action = "upload_start"
            risk_level = "in_progress"
        else:
            risk_level = "completed"
        actions.append(
            {
                "action_id": f"{case_id}:corr_upload_{index}",
                "action_type": normalized_action,
                "risk_level": risk_level,
                "time": str(candidate.get("timestamp", "") or ""),
                "app": str(candidate.get("app_name", "") or ""),
                "app_category": str(candidate.get("sink_type", "") or "external_sink"),
                "source_file": str(candidate.get("original_file", "") or ""),
                "derived_file": str(current_files[0] if current_files else ""),
                "evidence_frames": [],
                "confidence": round(_safe_float(candidate.get("confidence", 0.0)), 4),
                "description": f"EventCorrelator upload candidate via {candidate.get('sink_type', 'external_sink')}",
                "evidence_source": "event_correlator",
                "evidence_refs": candidate.get("evidence_refs", []),
                "mapping_links": candidate.get("mapping_links", []),
            }
        )
    return actions


def _vlm_actions(
    case_id: str,
    vlm_verdict: Optional[Dict[str, Any]],
    sensitive_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(vlm_verdict, dict):
        return []
    result = []
    sensitive_files = sensitive_files or []
    sensitive_segments = {
        str(action.get("segment_id", "") or "")
        for action in vlm_verdict.get("observed_actions", []) or []
        if isinstance(action, dict)
        and (
            _is_sensitive_ref(str(action.get("description", "") or ""), sensitive_files)
            or _is_sensitive_ref(str(action.get("derived_file", "") or ""), sensitive_files)
        )
    }
    for index, action in enumerate(vlm_verdict.get("observed_actions", []) or []):
        if not isinstance(action, dict):
            continue
        item = dict(action)
        item["action_id"] = f"{case_id}:{item.get('action_id') or f'vlm_action_{index}'}"
        item["segment_sensitive_object_confirmed"] = str(item.get("segment_id", "") or "") in sensitive_segments
        item["verdict_risk_level"] = _normalize_risk_level(str(vlm_verdict.get("risk_level", "") or ""))
        item["verdict_is_violation"] = bool(vlm_verdict.get("is_violation"))
        item["action_type"] = _normalize_action_type(str(item.get("action_type", "") or "unknown"))
        item["risk_level"] = _normalize_risk_level(str(item.get("risk_level", "") or ""))
        item["evidence_source"] = str(item.get("evidence_source", "") or "remote_vlm")
        _apply_vlm_action_consistency(item)
        _upgrade_vlm_action_from_description(item)
        result.append(item)
    return result


def _upgrade_vlm_action_from_description(action: Dict[str, Any]) -> None:
    """
    Upgrade VLM action type and risk level based on description keywords.

    VLM sometimes conservatively reports action_type='attach_file' and
    risk_level='selected_or_attached' even when the description clearly
    indicates completion (e.g., "upload complete", "email sent").

    This function detects completion evidence in descriptions and upgrades
    the action accordingly so datalog can generate LeakFile facts.
    """
    description = str(action.get("description", "") or "").lower()
    action_type = str(action.get("action_type", "") or "")
    risk_level = str(action.get("risk_level", "") or "")

    # Don't upgrade if already at completion level
    if risk_level in {"completed", "content_exposed"}:
        return

    # Don't upgrade if action_type is already a leak type
    if action_type in {"upload_complete", "send_message", "publish_content",
                       "external_exposure", "screen_share", "screenshot", "screen_record"}:
        return

    # Detect completion keywords in description
    upload_complete_keywords = [
        "upload complete", "upload successful", "upload finished",
        "已上传", "上传完成", "上传成功",
        "visible in remote", "visible in the remote",
        "in the remote file listing", "in remote file listing",
        "完成上传"
    ]

    send_complete_keywords = [
        "email sent", "message sent", "已发送", "发送成功",
        "sent successfully", "send complete",
        "邮件已发送", "消息已发送"
    ]

    publish_complete_keywords = [
        "published", "已发布", "发布成功",
        "publish complete", "publication successful"
    ]

    exposure_keywords = [
        "content is visible", "content visible", "fully visible on screen",
        "content exposed", "内容可见", "内容暴露",
        "open in.*editor", "open in.*viewer",
        "visible in.*online", "在线查看", "在线编辑"
    ]

    # Check for upload completion
    if any(kw in description for kw in upload_complete_keywords):
        if action_type in {"attach_file", "select_file", "upload_start", "unknown"}:
            action["action_type"] = "upload_complete"
            action["risk_level"] = "completed"
            action["upgrade_reason"] = "description_indicates_upload_complete"
            return

    # Check for send completion
    if any(kw in description for kw in send_complete_keywords):
        if action_type in {"attach_file", "compose_message", "unknown"}:
            action["action_type"] = "send_message"
            action["risk_level"] = "completed"
            action["upgrade_reason"] = "description_indicates_send_complete"
            return

    # Check for publish completion
    if any(kw in description for kw in publish_complete_keywords):
        action["action_type"] = "publish_content"
        action["risk_level"] = "completed"
        action["upgrade_reason"] = "description_indicates_publish_complete"
        return

    # Check for content exposure
    if any(kw in description for kw in exposure_keywords):
        if risk_level in {"selected_or_attached", "in_progress"}:
            action["action_type"] = "external_exposure"
            action["risk_level"] = "content_exposed"
            action["upgrade_reason"] = "description_indicates_content_exposure"
            return


LOG_RULE_ACTION_SPECS = {
    "upload_event": ("upload_complete", "completed", True),
    "screen_share": ("screen_share", "completed", True),
    "file_selected": ("select_file", "selected_or_attached", True),
    "upload_staging": ("upload_start", "in_progress", True),
    "archive_created": ("compress_file", "selected_or_attached", False),
    "convert_created": ("convert_file", "selected_or_attached", False),
    "split_created": ("convert_file", "selected_or_attached", False),
    "rename_hiding": ("rename_file", "selected_or_attached", False),
    "variant_created": ("save_as", "selected_or_attached", False),
    "clipboard_content": ("copy_content", "selected_or_attached", False),
    "screen_capture": ("screenshot", "content_exposed", False),
}

# Log rules are risk support only. Even explicit upload/capture/share events can
# represent a cancelled or local-only flow, so confirmation must come from the
# evidence graph rather than a log rule shortcut.
LOG_RULE_LEAK_RULES = CONFIRMED_LOG_RULES


def _log_rule_actions(
    case_id: str,
    log_rule_signal: Optional[Dict[str, Any]],
    sensitive_files: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(log_rule_signal, dict) or not log_rule_signal.get("positive"):
        return []
    actions: List[Dict[str, Any]] = []
    evidence = log_rule_signal.get("evidence", {}) or {}
    for rule, entries in evidence.items():
        action_type, risk_level, _ = LOG_RULE_ACTION_SPECS.get(
            rule, ("external_exposure", "selected_or_attached", False)
        )
        for index, entry in enumerate(entries or []):
            if not isinstance(entry, dict):
                continue
            file_ref = str(entry.get("file_path", "") or "")
            detail = str(entry.get("detail", "") or entry.get("event_type", "") or "")
            if not file_ref and _is_sensitive_ref(detail, sensitive_files):
                file_ref = _canonical_sensitive_ref(detail, sensitive_files)
            if not file_ref:
                continue
            actions.append(
                {
                    "action_id": f"{case_id}:log_rule_{rule}_{index}",
                    "action_type": action_type,
                    "risk_level": risk_level,
                    "time": str(entry.get("timestamp", "") or ""),
                    "app": str(entry.get("event_type", "") or "monitor"),
                    "app_category": "log_rule",
                    "source_file": file_ref,
                    "derived_file": "",
                    "evidence_frames": [],
                    "confidence": 0.97,
                    "description": f"log rule {rule}: {detail}",
                    "evidence_source": "log_rule",
                }
            )
    return actions


def _build_audit_actions(
    case_id: str,
    detection: Dict[str, Any],
    vlm_verdict: Optional[Dict[str, Any]],
    correlation_bundle: Optional[Dict[str, Any]],
    logs: Optional[List[Dict[str, Any]]] = None,
    sensitive_files: Optional[List[str]] = None,
    log_rule_signal: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    actions = []
    actions.extend(_operation_record_actions(case_id, detection))
    actions.extend(_file_mapping_actions(case_id, detection))
    if logs is not None and sensitive_files is not None:
        actions.extend(_log_sensitive_open_actions(case_id, logs, sensitive_files))
    actions.extend(_detection_actions(case_id, detection))
    actions.extend(_log_rule_actions(case_id, log_rule_signal, sensitive_files or []))
    actions.extend(_vlm_actions(case_id, vlm_verdict, sensitive_files or []))
    actions.extend(_correlation_actions(case_id, correlation_bundle))
    return actions


def _action_timestamp_ms(action: Dict[str, Any], fallback_index: int = 0) -> int:
    value = str(action.get("time", "") or action.get("timestamp", "") or "").strip()
    if value:
        dt = _parse_dt(value)
        if dt:
            return int(dt.timestamp() * 1000)
        digits = re.sub(r"\D", "", value)
        if digits:
            try:
                return int(digits[:13].ljust(13, "0"))
            except ValueError:
                pass
    return fallback_index


def _action_process(action: Dict[str, Any]) -> str:
    app = str(action.get("app", "") or action.get("app_category", "") or "").strip()
    return app.replace("\\", "/").rsplit("/", 1)[-1].lower() or "unknown"


def _normalize_process_ref(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("\\", "/").rsplit("/", 1)[-1].lower() or "unknown"


def _action_process_field(action: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(action.get(key, "") or "").strip()
        if value:
            return _normalize_process_ref(value)
    return ""


def _action_shared_data_field(action: Dict[str, Any]) -> str:
    for key in ("shared_data", "clipboard_data", "data", "source_file", "derived_file"):
        value = str(action.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _is_sensitive_ref(value: str, sensitive_files: List[str]) -> bool:
    text = str(value or "").replace("\\", "/").lower()
    if not text:
        return False
    if not _is_valid_sensitive_file_ref(text):
        return False
    for sensitive in sensitive_files:
        needle = str(sensitive or "").replace("\\", "/").lower()
        name = needle.rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if len(stem) < 2:
            # Single-character document names ("1.docx") match visual text
            # everywhere; only a full multi-segment path is trustworthy.
            if "/" in needle and len(needle) >= 8 and needle in text:
                return True
            continue
        if needle and (needle in text or text in needle):
            return True
        if name and name in text:
            return True
    return False


def _canonical_sensitive_ref(value: str, sensitive_files: List[str]) -> str:
    text = str(value or "").strip()
    if text and _is_sensitive_ref(text, sensitive_files):
        return text.replace("\\", "/")
    return str(sensitive_files[0]).replace("\\", "/") if sensitive_files else text.replace("\\", "/")


def _audit_actions_to_datalog_facts(
    case_id: str,
    audit_actions: List[Dict[str, Any]],
    sensitive_files: List[str],
) -> List[Dict[str, Any]]:
    if not sensitive_files:
        return []

    facts: List[Dict[str, Any]] = []
    seen: set[tuple[str, tuple[Any, ...]]] = set()

    def add(relation: str, args: Tuple[Any, ...], source_action: str, evidence_source: str) -> None:
        key = (relation, args)
        if key in seen:
            return
        seen.add(key)
        facts.append(
            {
                "relation": relation,
                "args": list(args),
                "source_action": source_action,
                "evidence_source": evidence_source,
            }
        )

    canonical_sources = [str(item).replace("\\", "/") for item in sensitive_files if str(item or "").strip()]
    def canonical_action_ref(value: str, fallback: str = "") -> str:
        text = str(value or "").strip()
        if text and _is_sensitive_ref(text, sensitive_files):
            return _canonical_sensitive_ref(text, sensitive_files)
        if fallback and _is_sensitive_ref(fallback, sensitive_files):
            return _canonical_sensitive_ref(fallback, sensitive_files)
        return _canonical_sensitive_ref(text or fallback, sensitive_files)

    for source_index, source in enumerate(canonical_sources[:32]):
        add(
            "OpenFile",
            (f"{case_id}:sensitive_source_{source_index}", "sensitive_source", source, source_index),
            f"{case_id}:sensitive_source",
            "sensitive_manifest",
        )

    for index, action in enumerate(audit_actions):
        if not isinstance(action, dict):
            continue

        # 智能升级: 基于description分析completion证据
        action_type, risk_level = _infer_completion_from_description(action)
        action_type = _normalize_action_type(action_type)
        risk_level = _normalize_risk_level(risk_level)
        evidence_source = str(action.get("evidence_source", "") or "audit_action")
        source_file = str(action.get("source_file", "") or "")
        derived_file = str(action.get("derived_file", "") or "")
        if derived_file and not _is_valid_sensitive_file_ref(derived_file):
            derived_file = ""
        action_text = " ".join(
            str(action.get(key, "") or "")
            for key in ("source_file", "derived_file", "description", "app", "app_category")
        )
        if (
            not _is_sensitive_ref(source_file, sensitive_files)
            and not _is_sensitive_ref(derived_file, sensitive_files)
            and not _is_sensitive_ref(action_text, sensitive_files)
        ):
            continue

        data = canonical_action_ref(source_file, derived_file or action_text)
        process = _action_process(action)
        timestamp = _action_timestamp_ms(action, index)
        raw_action_id = str(action.get("action_id", "") or f"{case_id}:action_{index}")
        op_id = re.sub(r"[^A-Za-z0-9_:.\\/-]+", "_", raw_action_id)[:180] or f"{case_id}:action_{index}"

        add("OpenFile", (f"{op_id}:source", process, data, timestamp), raw_action_id, evidence_source)

        # 剪贴板操作
        if action_type == "copy_content":
            add("ClipboardWrite", (f"{op_id}:clipboard_write", process, data, timestamp), raw_action_id, evidence_source)
        elif action_type == "paste_content":
            add("ClipboardRead", (f"{op_id}:clipboard_read", process, data, timestamp), raw_action_id, evidence_source)

        # 跨进程传播：增强支持
        from_process = _action_process_field(action, "from_process", "source_process", "source_app")
        to_process = _action_process_field(action, "to_process", "target_process", "target_app")
        shared_data = canonical_action_ref(_action_shared_data_field(action), data)

        # 方式1：明确的跨进程传输字段
        if from_process and to_process and from_process != to_process and _is_sensitive_ref(shared_data, sensitive_files):
            add(
                "CrossProcessTransfer",
                (f"{op_id}:cross_process", from_process, to_process, shared_data.replace("\\", "/"), timestamp),
                raw_action_id,
                evidence_source,
            )

        # 方式2：从action_type推断跨进程传播
        if action_type == "copy_content" and to_process:
            # 复制操作：从当前进程到剪贴板（可能被其他进程读取）
            add(
                "CrossProcessTransfer",
                (f"{op_id}:copy_to_clipboard", process, "clipboard", shared_data.replace("\\", "/"), timestamp),
                raw_action_id,
                evidence_source,
            )
        elif action_type == "paste_content" and from_process:
            # 粘贴操作：从剪贴板到当前进程
            add(
                "CrossProcessTransfer",
                (f"{op_id}:paste_from_clipboard", "clipboard", process, shared_data.replace("\\", "/"), timestamp),
                raw_action_id,
                evidence_source,
            )

        # 方式3：从description推断跨进程传播
        description = str(action.get("description", "") or "").lower()
        if "copy" in description and "paste" in description:
            # description提到了复制粘贴操作
            target_app = str(action.get("target_app", "") or action.get("app", ""))
            if target_app and target_app != process:
                add(
                    "CrossProcessTransfer",
                    (f"{op_id}:inferred_cross_process", process, target_app, shared_data.replace("\\", "/"), timestamp),
                    raw_action_id,
                    evidence_source,
                )

        for source_index, source in enumerate(canonical_sources[:32]):
            if source != data and _is_sensitive_ref(data, [source]):
                add(
                    "TransferFile",
                    (f"{op_id}:source_alias_{source_index}", "sensitive_source", source, data, timestamp),
                    raw_action_id,
                    evidence_source,
                )
        if derived_file and derived_file.replace("\\", "/") != data:
            derived = derived_file.replace("\\", "/")
            add("TransferFile", (f"{op_id}:derive", process, data, derived, timestamp), raw_action_id, evidence_source)
            for source_index, source in enumerate(canonical_sources[:32]):
                if source != data and _is_sensitive_ref(data, [source]):
                    add(
                        "TransferFile",
                        (f"{op_id}:derive_alias_{source_index}", "sensitive_source", source, derived, timestamp),
                        raw_action_id,
                        evidence_source,
                    )
            data = derived

        leak_action_types = {
            "upload_complete",
            "send_message",
            "publish_content",
            "external_exposure",
            "screen_share",
            "screenshot",
            "screen_record",
            "vm_copy",
        }
        local_positive_leak_allowed = evidence_source != "local_positive" or _local_positive_supports_leak(action)
        action_leak_allowed = False

        # 一致性校验：检查description和action的一致性
        if not _action_description_consistent_with_completion(action, action_type, risk_level):
            # description明确说未完成，但action_type/risk_level是完成态 -> 不生成LeakFile
            action_leak_allowed = False
        elif evidence_source == "remote_vlm":
            action_leak_allowed = _remote_vlm_action_supports_leak(action)
        elif (
            evidence_source != "event_correlator"
            and evidence_source != "log_rule"
            and evidence_source != "local_positive"
            and evidence_source != "remote_vlm_reason"
            and not _action_has_hard_negative_context(action)
            and not _action_has_historical_or_inbound_context(action)
            and not _action_has_cloud_editor_read_context(action)
            and not _vlm_parent_verdict_blocks_risk(action)
        ):
            if action_type in leak_action_types and risk_level in {"content_exposed", "completed"}:
                action_leak_allowed = not _action_has_unfinished_context(action)
            elif (
                action_type not in {"select_file", "attach_file", "upload_start", "open_file", "copy_content", "paste_content"}
                and risk_level in {"content_exposed", "completed"}
            ):
                action_leak_allowed = True
            elif _content_exposure_action_supports_leak(action):
                action_leak_allowed = True

        if local_positive_leak_allowed and action_leak_allowed:
            channel = action_type if action_type not in {"unknown", "none"} else "external_exposure"
            add("LeakFile", (f"{op_id}:leak", process, data, channel, timestamp), raw_action_id, evidence_source)

    return facts


def _audit_action_supports_risk(action: Dict[str, Any], sensitive_files: List[str]) -> bool:
    if not isinstance(action, dict):
        return False
    action_type = _normalize_action_type(str(action.get("action_type", "") or "unknown"))
    risk_level = _normalize_risk_level(str(action.get("risk_level", "") or ""))
    evidence_source = str(action.get("evidence_source", "") or "audit_action")
    source_file = str(action.get("source_file", "") or "")
    derived_file = str(action.get("derived_file", "") or "")
    action_text = " ".join(
        str(action.get(key, "") or "")
        for key in ("source_file", "derived_file", "description", "app", "app_category")
    )
    if (
        evidence_source == "deterministic"
        and action_type in {"upload_complete", "send_message", "publish_content"}
        and risk_level == "completed"
        and _safe_float(action.get("confidence", 0.0)) >= 0.9
    ):
        return not _action_has_hard_negative_context(action) and not _action_has_historical_or_inbound_context(action)
    if (
        not _is_sensitive_ref(source_file, sensitive_files)
        and not _is_sensitive_ref(derived_file, sensitive_files)
        and not _is_sensitive_ref(action_text, sensitive_files)
    ):
        return False
    if _action_has_hard_negative_context(action):
        return False
    if (
        evidence_source == "log_rule"
        and action_type in {"upload_complete", "send_message", "publish_content"}
        and risk_level == "completed"
    ):
        return True
    if _action_has_historical_or_inbound_context(action):
        return False
    if _action_has_cloud_editor_read_context(action):
        return False
    if _vlm_parent_verdict_blocks_risk(action):
        return False
    if evidence_source == "deterministic" and action_type in {"upload_complete", "send_message", "publish_content"}:
        return risk_level == "completed" and _safe_float(action.get("confidence", 0.0)) >= 0.9
    if evidence_source == "deterministic" and action_type in {"attach_file", "select_file", "upload_start"}:
        return risk_level in {"selected_or_attached", "in_progress"} and _safe_float(action.get("confidence", 0.0)) >= 0.5
    if evidence_source == "local_positive":
        return _local_positive_supports_risk(action)
    if evidence_source == "event_correlator":
        # Correlator upload candidates echo VLM frame segments; the echoed VLM
        # action is already evaluated on its own merits above/below, so the
        # echo must not double as independent support.
        return False
    if (
        evidence_source == "remote_vlm"
        and action_type in {"convert_file", "compress_file", "rename_file", "save_as"}
        and _vlm_transformation_action_supports_risk(action)
    ):
        return True
    if action_type in {
        "upload_complete",
        "send_message",
        "publish_content",
        "external_exposure",
        "screen_share",
        "screenshot",
        "screen_record",
        "vm_copy",
    } and risk_level in {"content_exposed", "completed"}:
        if action_type in {"upload_complete", "send_message", "publish_content"} and _action_has_unfinished_context(action):
            return False
        if action_type == "external_exposure" and _action_has_unfinished_context(action):
            return False
        return True
    if risk_level in {"content_exposed", "completed"} and _external_sink_action_context(action):
        if action_type == "open_file":
            return False
        if action_type in {"attach_file", "select_file", "upload_start"}:
            # Attachment visible in an input box is not exposure by itself:
            # cancelled uploads look identical. Require wording that the file
            # content actually landed in the external surface; once it has,
            # the exposure stands even if the final publish is still pending.
            return _action_has_upload_ingest_context(action)
        if action_type in {"copy_content", "paste_content"}:
            return _content_exposure_action_supports_leak(action)
        if (
            evidence_source == "remote_vlm"
            and action_type == "external_exposure"
            and risk_level == "content_exposed"
            and not _action_has_upload_ingest_context(action)
            and not _action_has_completion_context(action)
        ):
            # "The file is visible in the external interface" without ingest
            # or completion wording describes inbox attachments and remote
            # listings as often as genuine exposure.
            return False
        return not _action_has_unfinished_context(action)
    if action_type in {"attach_file", "upload_start", "select_file"} and risk_level in {"selected_or_attached", "in_progress"}:
        if evidence_source == "log_rule":
            return True
        if evidence_source in {"remote_vlm", "local_positive"}:
            # Staging-stage visual claims repeatedly misfire on cancelled
            # uploads and file-picker browsing; accept them only when the VLM
            # explicitly saw the transfer running, being submitted, or a
            # sensitive attachment in a send/share dialog.
            return (
                evidence_source == "remote_vlm"
                and (_action_has_upload_progress_context(action) or _action_has_staging_context(action))
            )
        text = _action_text(action)
        if not _external_sink_action_context(action) or not _action_has_staging_context(action):
            return False
        detail_text = " ".join(
            str(action.get(key, "") or "").lower()
            for key in ("description", "app", "app_category", "source_file", "derived_file")
        )
        if action_type == "select_file" and "cloud_storage" in text and not any(
            marker in detail_text
            for marker in (
                "attach",
                "upload",
                "share",
                "submit",
                "send",
                "file picker",
                "file upload",
                "open dialog",
                "selected for",
                "\u9644\u4ef6",
                "\u4e0a\u4f20",
                "\u5206\u4eab",
                "\u53d1\u9001",
            )
        ):
            return False
        if "ai_service" in text and risk_level == "selected_or_attached" and _action_has_unfinished_context(action):
            return False
        if _action_has_unfinished_context(action) and not any(
            marker in text
            for marker in (
                "email",
                "mail",
                "gmail",
                "outlook",
                "proton",
                "qqmail",
                "163 mail",
                "messaging",
                "meeting",
                "cloud_storage",
                "code_repo",
                "community_publish",
                "\u90ae\u7bb1",
                "\u90ae\u4ef6",
                "\u4f1a\u8bae",
            )
        ):
            return False
        return True
    return False


def _risk_support_actions(audit_actions: List[Dict[str, Any]], sensitive_files: List[str]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for action in audit_actions:
        if not _audit_action_supports_risk(action, sensitive_files):
            continue
        action_id = str(action.get("action_id", "") or "")
        if action_id in seen:
            continue
        seen.add(action_id)
        result.append(
            {
                "action_id": action_id,
                "action_type": _normalize_action_type(str(action.get("action_type", "") or "")),
                "risk_level": _normalize_risk_level(str(action.get("risk_level", "") or "")),
                "evidence_source": str(action.get("evidence_source", "") or ""),
                "app_category": str(action.get("app_category", "") or ""),
                "confidence": round(_safe_float(action.get("confidence", 0.0)), 4),
                "description": _compact_text(action.get("description", ""), 300),
            }
        )
    return result


def _run_datalog_on_audit_actions(
    case_id: str,
    audit_actions: List[Dict[str, Any]],
    sensitive_files: List[str],
) -> Dict[str, Any]:
    facts = _audit_actions_to_datalog_facts(case_id, audit_actions, sensitive_files)
    risk_support = _risk_support_actions(audit_actions, sensitive_files)
    if not facts:
        return {
            "engine": "none",
            "fact_count": 0,
            "relation_counts": {},
            "facts": [],
            "leak_paths": [],
            "risk_support": risk_support,
            "risk_positive": bool(risk_support),
            "confirmed_leak": False,
            "reason": "risk_support_only" if risk_support else "no_audit_facts",
            "trace": [],
        }

    try:
        from data_leak_detector.leak_reasoner import DatalogEngine
    except Exception as exc:
        leak_like = [fact for fact in facts if fact["relation"] == "LeakFile"]
        return {
            "engine": "rule_fallback",
            "fact_count": len(facts),
            "relation_counts": _relation_counts(facts),
            "facts": facts,
            "leak_paths": [],
            "risk_support": risk_support,
            "risk_positive": bool(leak_like) or bool(risk_support),
            "confirmed_leak": bool(leak_like),
            "reason": f"datalog_import_failed:{type(exc).__name__}",
            "trace": [],
        }

    engine = None
    try:
        trace_buffer = io.StringIO()
        with contextlib.redirect_stdout(trace_buffer):
            engine = DatalogEngine()
            for fact in facts:
                engine.add_fact(fact["relation"], *fact["args"])
            leak_paths = engine.query_leak()
        trace_lines = [_compact_text(line, 500) for line in trace_buffer.getvalue().splitlines() if line.strip()]
        return {
            "engine": "souffle" if getattr(engine, "use_souffle", False) else "python",
            "fact_count": len(facts),
            "relation_counts": _relation_counts(facts),
            "facts": facts,
            "leak_paths": [path.to_dict() if hasattr(path, "to_dict") else dict(path) for path in leak_paths],
            "risk_support": risk_support,
            "risk_positive": bool(leak_paths) or bool(risk_support),
            "confirmed_leak": bool(leak_paths),
            "reason": "leak_paths_found" if leak_paths else ("risk_support_only" if risk_support else "no_leak_path"),
            "trace": trace_lines,
        }
    except Exception as exc:
        leak_like = [fact for fact in facts if fact["relation"] == "LeakFile"]
        return {
            "engine": "rule_fallback",
            "fact_count": len(facts),
            "relation_counts": _relation_counts(facts),
            "facts": facts,
            "leak_paths": [],
            "risk_support": risk_support,
            "risk_positive": bool(leak_like) or bool(risk_support),
            "confirmed_leak": bool(leak_like),
            "reason": f"datalog_failed:{type(exc).__name__}",
            "error": str(exc),
            "trace": [],
        }
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.cleanup()


def _log_datalog_trace(case_id: str, decision: Dict[str, Any], max_items: int = 3) -> None:
    facts = decision.get("facts", []) or []
    leak_paths = decision.get("leak_paths", []) or []
    relation_counts = decision.get("relation_counts", {}) or {}
    count_text = ",".join(f"{name}={count}" for name, count in sorted(relation_counts.items())) or "none"
    _progress(
        f"[DATALOG] case={case_id} engine={decision.get('engine', 'unknown')} "
        f"facts={decision.get('fact_count', 0)} relations={count_text} "
        f"leak_paths={len(leak_paths)} confirmed={int(bool(decision.get('confirmed_leak')))} "
        f"reason={decision.get('reason', '')}"
    )
    for index, fact in enumerate(facts[:max_items], 1):
        _progress(
            f"[DATALOG FACT {index}/{len(facts)}] case={case_id} "
            f"{fact.get('relation')}({', '.join(_compact_text(arg, 80) for arg in fact.get('args', []))}) "
            f"source={fact.get('evidence_source', '')} action={_compact_text(fact.get('source_action', ''), 100)}"
        )
    if len(facts) > max_items:
        _progress(f"[DATALOG FACTS] case={case_id} omitted={len(facts) - max_items}")
    for index, path in enumerate(leak_paths[:max_items], 1):
        _progress(
            f"[DATALOG PATH {index}/{len(leak_paths)}] case={case_id} "
            f"proc={_compact_text(path.get('leaking_proc', ''), 80)} "
            f"file={_compact_text(path.get('leaked_file', ''), 120)} "
            f"channel={_compact_text(path.get('leak_channel', ''), 80)} "
            f"path={_compact_text(path.get('full_path', ''), 220)}"
        )
    if len(leak_paths) > max_items:
        _progress(f"[DATALOG PATHS] case={case_id} omitted={len(leak_paths) - max_items}")
    for index, line in enumerate((decision.get("trace", []) or [])[:max_items], 1):
        _progress(f"[DATALOG TRACE {index}] case={case_id} {_compact_text(line, 500)}")
    trace_count = len(decision.get("trace", []) or [])
    if trace_count > max_items:
        _progress(f"[DATALOG TRACE] case={case_id} omitted={trace_count - max_items}")


def _evaluate_case_record_decision(record: Dict[str, Any], *, log_trace: bool = True) -> Dict[str, Any]:
    case_id = record["case_id"]
    logs = record["logs"]
    sensitive_files = record["sensitive_files"]
    groundtruth = record["groundtruth"]
    detection = record["detection"]
    fallback_meta = record["fallback_meta"]
    expected_level = record["expected_level"]
    deterministic_positive = record["deterministic_positive"]
    triage_positive = record["triage_positive"]
    vlm_verdict = record.get("live_vlm_verdict")

    correlation_bundle: Optional[Dict[str, Any]] = None
    if isinstance(vlm_verdict, dict) and vlm_verdict.get("status") == "success":
        frame_segments = _frame_segments_from_vlm_verdict(
            vlm_verdict,
            sensitive_files,
            fallback_meta,
            logs,
        )
        correlation_bundle = _run_event_correlator_bundle(
            case_id,
            logs=_logs_for_correlation(logs, fallback_meta),
            sensitive_files=sensitive_files,
            groundtruth=groundtruth,
            frame_segments=frame_segments,
        )

    log_rule_signal = record.get("log_rule_signal") or {}
    audit_actions = _build_audit_actions(
        case_id,
        detection,
        vlm_verdict,
        correlation_bundle,
        logs=logs,
        sensitive_files=sensitive_files,
        log_rule_signal=log_rule_signal,
    )
    datalog_decision = _run_datalog_on_audit_actions(case_id, audit_actions, sensitive_files)
    if log_trace:
        _log_datalog_trace(case_id, datalog_decision)
    evidence_sources = _audit_evidence_sources(audit_actions)
    log_rule_positive = bool(log_rule_signal.get("positive"))
    datalog_positive = bool(datalog_decision.get("risk_positive"))
    datalog_confirmed = bool(datalog_decision.get("confirmed_leak"))
    rules_only_positive = bool(deterministic_positive) or log_rules_confirm_leak(
        log_rule_signal.get("rules", []) or []
    )
    vlm_only_positive = _vlm_only_confirmed_positive(vlm_verdict)
    frame_coverage = _semantic_frame_coverage(vlm_verdict)
    evidence_decision = decide_evidence_outcome(
        datalog_risk_positive=datalog_positive,
        datalog_confirmed=datalog_confirmed,
        log_rule_positive=log_rule_positive,
        log_rule_rules=log_rule_signal.get("rules", []) or [],
    )
    risk_positive = evidence_decision.risk_positive
    confirmed_leak = evidence_decision.confirmed_leak
    final_positive = _final_positive_for_expected_level(expected_level, risk_positive, confirmed_leak)
    return {
        "correlation_bundle": correlation_bundle,
        "review_source": _review_source(deterministic_positive, vlm_verdict, record["vlm_live_queued"]),
        "log_rule_signal": log_rule_signal,
        "audit_actions": audit_actions,
        "datalog_decision": datalog_decision,
        "evidence_sources": evidence_sources,
        "datalog_positive": datalog_positive,
        "datalog_confirmed": datalog_confirmed,
        "rules_only_positive": rules_only_positive,
        "vlm_only_positive": vlm_only_positive,
        "frame_coverage": frame_coverage,
        "evidence_decision": evidence_decision,
        "risk_positive": risk_positive,
        "confirmed_leak": confirmed_leak,
        "final_positive": final_positive,
        "final_semantics": f"groundtruth_aligned:{expected_level}",
    }


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
    case_offset: int = 0,
    case_limit: int = 0,
    use_vlm: bool = False,
    max_vlm_cases: int = 0,
    max_vlm_frames: int = 12,
    vlm_workers: int = 1,
    vlm_gate_mode: str = "all",
    replay_vlm_report: Optional[Path] = None,
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
    log_rules = _load_module("nas_log_signal_rules", Path(__file__).resolve().parent / "log_signal_rules.py")

    replay_verdicts: Optional[Dict[str, Dict[str, Any]]] = None
    if replay_vlm_report is not None:
        replay_payload = json.loads(Path(replay_vlm_report).read_text(encoding="utf-8"))
        replay_verdicts = {}
        for case_payload in replay_payload.get("cases", []) or []:
            verdict = case_payload.get("live_vlm_verdict")
            if isinstance(verdict, dict):
                replay_verdicts[str(case_payload.get("case", "")).replace("\\", "/")] = verdict
        _progress(f"[REPLAY] loaded {len(replay_verdicts)} cached VLM verdicts from {replay_vlm_report}")

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

    discovered_cases = len(case_dirs)
    selected_offset = max(0, int(case_offset))
    selected_limit = max(0, int(case_limit))
    selected_start = min(selected_offset, discovered_cases)
    selected_end = discovered_cases if selected_limit <= 0 else min(discovered_cases, selected_start + selected_limit)
    if selected_start > 0 or selected_end < discovered_cases:
        _progress(
            f"[SELECTION] cases={selected_start + 1 if selected_start < selected_end else 0}-"
            f"{selected_end} of {discovered_cases} "
            f"offset={selected_offset} limit={selected_limit or 'all'}"
        )
    case_dirs = case_dirs[selected_start:selected_end]
    total_cases = len(case_dirs)
    case_records: List[Dict[str, Any]] = []
    pending_vlm_records: List[Dict[str, Any]] = []
    local_vlm_resolutions = 0
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

        if _groundtruth_is_event_log(groundtruth):
            result.skipped.append({"case": case_id, "reason": "groundtruth_is_event_log"})
            _progress(f"[CASE {case_index}/{total_cases}] SKIP case={case_id} reason=groundtruth_is_event_log")
            continue

        sensitive_files = _sensitive_files_from_groundtruth(groundtruth)
        for path in _sensitive_files_from_logs(logs, log_first):
            if path.lower() not in {item.lower() for item in sensitive_files}:
                sensitive_files.append(path)
        if not sensitive_files and _expected_positive(groundtruth):
            for path in _fallback_sensitive_files_from_logs(logs, log_first):
                if path.lower() not in {item.lower() for item in sensitive_files}:
                    sensitive_files.append(path)

        detector = log_first.LogFirstDetector(
            sensitive_files=sensitive_files,
            blacklist_apps=DEFAULT_BLACKLIST_APPS,
            whitelist_apps=DEFAULT_WHITELIST_APPS,
        )
        detection = detector.analyze(logs)
        should_run_vlm, fallback_meta = run_e2e._should_use_vlm_fallback(logs, detection)
        frontend_context = _frontend_context_from_logs(logs)

        expected_level = _expected_level(groundtruth)
        expected = expected_level != "normal"
        log_rule_signal = log_rules.extract_deterministic_signals(
            logs, sensitive_files, log_first.is_sensitive_name
        )
        if log_rule_signal.get("positive"):
            _progress(
                f"[LOG RULES] case={case_id} rules={','.join(log_rule_signal.get('rules', []))}"
            )
        deterministic_positive = (
            len(detection.get("upload_events", [])) > 0 or bool(log_rule_signal.get("positive"))
        )
        if deterministic_positive:
            fallback_meta = _visual_review_meta_for_deterministic_evidence(
                fallback_meta,
                detection,
                log_rule_signal,
                logs,
            )
            should_run_vlm = True
        triage_positive = deterministic_positive or should_run_vlm

        if should_run_vlm and sensitive_files:
            video_path = _choose_video_file(case_dir)
            rec_start = _recording_start(groundtruth, logs, video_path)
            rec_end = _video_end_time(video_path, rec_start)
            sink_sessions = _build_sink_sessions(fallback_meta, logs, sensitive_files, rec_start, rec_end)
            if sink_sessions:
                fallback_meta = dict(fallback_meta)
                fallback_meta["sink_sessions"] = sink_sessions

        adaptive_frames = 0
        frame_budget_meta: Dict[str, Any] = {}
        vlm_gate = {
            "mode": str(vlm_gate_mode or "all").strip().lower(),
            "action": "not_applicable",
            "reason": "vlm_not_requested",
        }
        local_vlm_verdict: Optional[Dict[str, Any]] = None
        if replay_verdicts is not None and use_vlm and should_run_vlm:
            replayed = replay_verdicts.get(case_id.replace("\\", "/"))
            local_vlm_verdict = dict(replayed) if isinstance(replayed, dict) else {
                "status": "replay_missing",
                "reason": "no_cached_verdict_in_replay_report",
                "is_violation": False,
                "confidence": 0.0,
                "completed_action": "none",
            }
            vlm_gate = {
                "mode": "replay",
                "action": "replayed" if replayed else "replay_missing",
                "reason": "verdict_from_replay_report",
            }
        elif use_vlm and should_run_vlm:
            # OPTIMIZATION: Force VLM for confirmed-level samples
            if expected_level == "confirmed":
                vlm_gate = {
                    "mode": str(vlm_gate_mode or "all").strip().lower(),
                    "action": "remote_required",
                    "reason": "confirmed_label_requires_visual_verification",
                }
            elif deterministic_positive:
                vlm_gate = {
                    "mode": str(vlm_gate_mode or "all").strip().lower(),
                    "action": "remote_required",
                    "reason": "deterministic_evidence_requires_visual_confirmation",
                }
            else:
                vlm_gate = _local_vlm_gate_decision(
                    mode=vlm_gate_mode,
                    logs=logs,
                    detection=detection,
                    fallback_meta=fallback_meta,
                )
                if vlm_gate.get("action") == "local_positive":
                    local_vlm_resolutions += 1
                    local_vlm_verdict = {
                        "status": "local_positive",
                        "is_violation": True,
                        "risk_level": "completed",
                        "confidence": 0.92 if vlm_gate.get("mode") == "strict" else 0.86,
                        "completed_action": "local_gate",
                        "reason": vlm_gate.get("reason", "local_feature_gate_positive"),
                        "frames_sent": 0,
                        "frame_context_count": 0,
                        "max_frames_requested": 0,
                        "model": "local_vlm_gate",
                    }
                    _postprocess_vlm_actions(local_vlm_verdict, sensitive_files, logs)
                    _progress(
                        f"[LOCAL GATE] case={case_id} progress={case_index}/{total_cases} "
                        f"gate={vlm_gate.get('mode')} reason={vlm_gate.get('reason')}"
                    )
            if local_vlm_verdict is None and (max_vlm_cases <= 0 or len(pending_vlm_records) < max_vlm_cases):
                adaptive_frames, frame_budget_meta = _adaptive_vlm_frame_budget(
                    fallback_meta,
                    logs,
                    max_frames=max_vlm_frames,
                )
                next_live_idx = len(pending_vlm_records) + 1
                _progress(
                    f"[VLM {next_live_idx}/{live_limit} QUEUED] "
                    f"case={case_id} progress={case_index}/{total_cases} "
                    f"expected={int(expected)} gate={vlm_gate.get('mode')} "
                    f"reasons={','.join(fallback_meta.get('reasons', []))} "
                    f"frame_budget={adaptive_frames}/{max_vlm_frames} "
                    f"complexity={','.join(frame_budget_meta.get('complexity_reasons', [])) or 'base'}"
                )
            elif local_vlm_verdict is None:
                vlm_gate["action"] = "remote_limit_reached"
                vlm_gate["reason"] = "max_vlm_cases_limit_reached"
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
            "frontend_context": frontend_context,
            "expected": expected,
            "expected_level": expected_level,
            "deterministic_positive": deterministic_positive,
            "log_rule_signal": log_rule_signal,
            "triage_positive": triage_positive,
            "should_run_vlm": should_run_vlm,
            "adaptive_vlm_frames": adaptive_frames,
            "vlm_frame_budget": frame_budget_meta,
            "vlm_live_queued": adaptive_frames > 0,
            "vlm_gate": vlm_gate,
            "live_vlm_verdict": local_vlm_verdict,
        }
        case_records.append(record)
        if adaptive_frames > 0:
            pending_vlm_records.append(record)

    result.live_vlm_reviews = local_vlm_resolutions + len(pending_vlm_records)
    result.vlm_local_resolutions = local_vlm_resolutions
    result.vlm_remote_requests = len(pending_vlm_records)
    workers = max(1, int(vlm_workers))
    if pending_vlm_records:
        _progress(
            f"[VLM] running {len(pending_vlm_records)} remote reviews with workers={workers} "
            f"local_resolved={local_vlm_resolutions}"
        )
        if _ocr_prefilter_enabled():
            if _ocr_prewarm_enabled():
                _progress(f"[VLM] warming OCR engine={_ocr_engine_name()}")
                _warm_ocr_reader()
            else:
                _progress(f"[VLM] OCR prefilter enabled engine={_ocr_engine_name()} prewarm=off")
        future_to_record = {}
        live_eval_done = 0
        live_eval_correct = 0
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
                live_eval_text = "eval=unavailable"
                try:
                    live_evaluation = _evaluate_case_record_decision(record, log_trace=True)
                    record["live_evaluation"] = live_evaluation
                    expected_value = bool(record["expected"])
                    final_value = bool(live_evaluation["final_positive"])
                    live_eval_done += 1
                    live_eval_correct += int(expected_value == final_value)
                    if expected_value and final_value:
                        live_bucket = "TP"
                    elif expected_value and not final_value:
                        live_bucket = "FN"
                    elif not expected_value and final_value:
                        live_bucket = "FP"
                    else:
                        live_bucket = "TN"
                    live_acc = live_eval_correct / live_eval_done if live_eval_done else 0.0
                    live_eval_text = (
                        f"expected={int(expected_value)} expected_level={record['expected_level']} "
                        f"final={int(final_value)} risk={int(bool(live_evaluation['risk_positive']))} "
                        f"confirmed={int(bool(live_evaluation['confirmed_leak']))} bucket={live_bucket} "
                        f"rolling={live_eval_correct}/{live_eval_done} acc={live_acc:.3f}"
                    )
                except Exception as eval_exc:
                    live_eval_text = f"eval_error={type(eval_exc).__name__}:{_compact_text(str(eval_exc), 160)}"
                _progress(
                    f"[VLM {done_index}/{len(pending_vlm_records)} DONE] "
                    f"case={record['case_id']} status={verdict.get('status', 'unknown')} "
                    f"image_frames={verdict.get('frames_sent', 0)} "
                    f"context_frames={verdict.get('frame_context_count', 0)}/{record['adaptive_vlm_frames']} "
                    f"{live_eval_text} "
                    f"reason={verdict.get('reason', '')}"
                )

    local_statuses = {"local_positive"}
    local_vlm_resolutions = sum(
        1
        for record in case_records
        if isinstance(record.get("live_vlm_verdict"), dict)
        and str(record["live_vlm_verdict"].get("status", "")) in local_statuses
        and not bool(record["live_vlm_verdict"].get("cache_hit"))
    )
    cache_hits = sum(
        1
        for record in case_records
        if isinstance(record.get("live_vlm_verdict"), dict)
        and bool(record["live_vlm_verdict"].get("cache_hit"))
    )
    remote_vlm_requests = sum(
        1
        for record in case_records
        if isinstance(record.get("live_vlm_verdict"), dict)
        and record.get("vlm_live_queued")
        and str(record["live_vlm_verdict"].get("status", "")) not in local_statuses
        and not _vlm_missing_api_key(record["live_vlm_verdict"])
        and not bool(record["live_vlm_verdict"].get("cache_hit"))
    )
    result.vlm_local_resolutions = local_vlm_resolutions
    result.vlm_cache_hits = cache_hits
    result.vlm_remote_requests = remote_vlm_requests
    result.live_vlm_reviews = local_vlm_resolutions + cache_hits + remote_vlm_requests

    for record in case_records:
        case_index = record["case_index"]
        total_cases = record["total_cases"]
        case_id = record["case_id"]
        logs = record["logs"]
        sensitive_files = record["sensitive_files"]
        groundtruth = record["groundtruth"]
        detection = record["detection"]
        fallback_meta = record["fallback_meta"]
        frontend_context = record["frontend_context"]
        expected = record["expected"]
        expected_level = record["expected_level"]
        deterministic_positive = record["deterministic_positive"]
        triage_positive = record["triage_positive"]
        should_run_vlm = record["should_run_vlm"]
        vlm_verdict = record["live_vlm_verdict"]

        evaluation = record.get("live_evaluation")
        if not isinstance(evaluation, dict):
            evaluation = _evaluate_case_record_decision(record, log_trace=True)
            record["live_evaluation"] = evaluation
        review_source = evaluation["review_source"]
        log_rule_signal = evaluation["log_rule_signal"]
        audit_actions = evaluation["audit_actions"]
        datalog_decision = evaluation["datalog_decision"]
        evidence_sources = evaluation["evidence_sources"]
        datalog_positive = evaluation["datalog_positive"]
        datalog_confirmed = evaluation["datalog_confirmed"]
        rules_only_positive = evaluation["rules_only_positive"]
        vlm_only_positive = evaluation["vlm_only_positive"]
        frame_coverage = evaluation["frame_coverage"]
        evidence_decision = evaluation["evidence_decision"]
        risk_positive = evaluation["risk_positive"]
        confirmed_leak = evaluation["confirmed_leak"]
        final_positive = evaluation["final_positive"]
        final_semantics = evaluation["final_semantics"]

        triage_bucket = result.triage.add(expected, triage_positive)
        deterministic_bucket = result.deterministic.add(expected, deterministic_positive)
        rules_only_bucket = result.rules_only.add(expected, rules_only_positive)
        vlm_only_bucket = result.vlm_only.add(expected, vlm_only_positive)
        risk_bucket = result.risk.add(expected, risk_positive)
        final_bucket = result.final.add(expected, final_positive)
        confirmed_bucket = result.confirmed.add(expected, confirmed_leak)
        if deterministic_positive:
            result.deterministic_hits += 1
        if should_run_vlm:
            result.vlm_reviews += 1
        if int(datalog_decision.get("fact_count", 0) or 0) > 0:
            result.datalog_cases += 1
        if datalog_positive:
            result.datalog_positive += 1
        if datalog_confirmed:
            result.datalog_confirmed += 1
        if str(datalog_decision.get("engine", "")) == "rule_fallback":
            result.datalog_fallbacks += 1
        if frame_coverage.get("available"):
            result.frame_coverage_cases += 1
            result.frame_coverage_completion += int(bool(frame_coverage.get("completion_anchor")))
            result.frame_coverage_content_exposed += int(bool(frame_coverage.get("content_exposed_anchor")))
            result.frame_coverage_staging += int(bool(frame_coverage.get("staging_anchor")))
            result.frame_coverage_external_sink += int(bool(frame_coverage.get("external_sink_anchor")))
            result.frame_coverage_sensitive_object += int(bool(frame_coverage.get("sensitive_object_anchor")))

        vlm_status = "none"
        frames_sent = 0
        frame_context_count = 0
        confidence = ""
        completed_action = ""
        risk_level = ""
        if vlm_verdict:
            vlm_status = str(vlm_verdict.get("status", "unknown"))
            frames_sent = int(vlm_verdict.get("frames_sent", 0) or 0)
            frame_context_count = int(vlm_verdict.get("frame_context_count", 0) or 0)
            confidence = str(vlm_verdict.get("confidence", ""))
            completed_action = str(vlm_verdict.get("completed_action", ""))
            risk_level = str(vlm_verdict.get("risk_level", ""))
        elif should_run_vlm:
            if not use_vlm:
                vlm_status = "triage_only"
            elif record["vlm_live_queued"]:
                vlm_status = "queued"
            else:
                vlm_status = "live_limit_reached"

        _progress(
            f"[CASE {case_index}/{total_cases}] {final_bucket.upper()} "
            f"case={case_id} expected={int(expected)} expected_level={expected_level} final={int(final_positive)} "
            f"risk={int(risk_positive)} confirmed={int(confirmed_leak)} "
            f"rules={int(rules_only_positive)} vlm_only={int(vlm_only_positive)} "
            f"det={int(deterministic_positive)} triage={int(triage_positive)} "
            f"datalog={int(datalog_positive)} "
            f"vlm={vlm_status} image_frames={frames_sent} context_frames={frame_context_count} "
            f"requested_frames={record['adaptive_vlm_frames']} confidence={confidence} "
            f"action={completed_action} risk={risk_level} source={review_source} reasons={','.join(fallback_meta.get('reasons', []))}"
        )

        result.cases.append(
            {
                "case": case_id,
                "log_file": record["log_source_name"],
                "expected_positive": expected,
                "expected_level": expected_level,
                "triage_positive": triage_positive,
                "deterministic_positive": deterministic_positive,
                "triage_bucket": triage_bucket,
                "deterministic_bucket": deterministic_bucket,
                "rules_only_positive": rules_only_positive,
                "rules_only_bucket": rules_only_bucket,
                "vlm_only_positive": vlm_only_positive,
                "vlm_only_bucket": vlm_only_bucket,
                "risk_bucket": risk_bucket,
                "final_positive": final_positive,
                "final_bucket": final_bucket,
                "risk_positive": risk_positive,
                "confirmed_leak": confirmed_leak,
                "confirmed_bucket": confirmed_bucket,
                "final_semantics": final_semantics,
                "review_source": review_source,
                "log_rule_signal": log_rule_signal,
                "risk_reasoning_source": evidence_decision.risk_reasoning_source,
                "reasoning_source": evidence_decision.reasoning_source,
                "evidence_sources": evidence_sources,
                "datalog_decision": datalog_decision,
                "datalog_facts": datalog_decision.get("facts", []),
                "datalog_leak_paths": datalog_decision.get("leak_paths", []),
                "upload_events": len(detection.get("upload_events", [])),
                "operation_records": len(detection.get("operation_records", [])),
                "sensitive_files": len(sensitive_files),
                "frontend_context": frontend_context,
                "audit_actions": audit_actions,
                "vlm_decision": fallback_meta.get("decision"),
                "vlm_reasons": fallback_meta.get("reasons", []),
                "vlm_live_queued": record["vlm_live_queued"],
                "vlm_gate": record["vlm_gate"],
                "keyframe_coverage": frame_coverage,
                "adaptive_vlm_frames": record["adaptive_vlm_frames"],
                "vlm_frame_budget": record["vlm_frame_budget"],
                "live_vlm_verdict": vlm_verdict,
                "correlation_bundle": correlation_bundle,
            }
        )

    report = result.to_dict()
    report["case_selection"] = {
        "total_discovered_cases": discovered_cases,
        "selected_cases": total_cases,
        "case_offset": selected_offset,
        "case_limit": selected_limit,
        "selected_start": selected_start + 1 if total_cases else None,
        "selected_end": selected_end if total_cases else None,
    }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("\nNAS Sample Benchmark")
    print("=" * 40)
    for name in ("triage", "deterministic", "rules_only", "vlm_only", "risk", "final", "confirmed"):
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
        f"remote_vlm_requests={report['summary'].get('vlm_remote_requests', 0)} "
        f"local_vlm_resolutions={report['summary'].get('vlm_local_resolutions', 0)} "
        f"vlm_cache_hits={report['summary'].get('vlm_cache_hits', 0)} "
        f"datalog_cases={report['summary'].get('datalog_cases', 0)} "
        f"datalog_positive={report['summary'].get('datalog_positive', 0)} "
        f"datalog_confirmed={report['summary'].get('datalog_confirmed', 0)} "
        f"datalog_fallbacks={report['summary'].get('datalog_fallbacks', 0)} "
        f"skipped={report['summary']['skipped_cases']}"
    )
    coverage = report["summary"].get("keyframe_coverage", {}) or {}
    print(
        "keyframe_coverage="
        f"cases={coverage.get('cases', 0)} "
        f"completion={coverage.get('completion_anchor', 0)} "
        f"content_exposed={coverage.get('content_exposed_anchor', 0)} "
        f"staging={coverage.get('staging_anchor', 0)} "
        f"external_sink={coverage.get('external_sink_anchor', 0)} "
        f"sensitive_object={coverage.get('sensitive_object_anchor', 0)}"
    )
    failures = [case for case in report["cases"] if case["final_bucket"] in {"fp", "fn"}]
    confirmed_failures = [case for case in report["cases"] if case["confirmed_bucket"] in {"fp", "fn"}]
    print(f"final_failures={len(failures)}")
    print(f"confirmed_failures={len(confirmed_failures)}")
    for case in failures[:30]:
        print(
            f"- {case['case']} {case['final_bucket']} "
            f"det={case['upload_events']} vlm={case['vlm_decision']} "
            f"source={case.get('review_source', '')} reasoning={case.get('reasoning_source', '')} "
            f"evidence={','.join(case.get('evidence_sources', []))} "
            f"reasons={','.join(case['vlm_reasons'])}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    global _PROGRESS_LOG_HANDLE
    parser = argparse.ArgumentParser(description="Benchmark downloaded NAS samples with optional live VLM verification.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage", action="append", help="Limit to a stage directory, e.g. --stage stage1")
    parser.add_argument("--case", action="append", help="Limit to a case path/name, e.g. --case stage1/2-ai-poe-1")
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Skip this many discovered cases after --stage/--case filtering; useful for batching VLM runs.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Run at most this many discovered cases after --case-offset; 0 means no limit.",
    )
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
    parser.add_argument(
        "--vlm-gate-mode",
        choices=("all", "strict", "adaptive", "aggressive"),
        default=os.getenv("DLD_VLM_GATE_MODE", "all").strip().lower(),
        help=(
            "Feature-based local gate before remote VLM: all keeps previous behavior; "
            "strict/adaptive/aggressive skip remote calls for increasingly broad high-confidence local positives."
        ),
    )
    parser.add_argument(
        "--replay-vlm-report",
        type=Path,
        help=(
            "Offline replay: reuse live_vlm_verdict entries from a previous report.json "
            "instead of calling the VLM; lets fusion changes be evaluated without remote requests."
        ),
    )
    args = parser.parse_args(argv)

    log_path: Optional[Path] = None
    if args.json_output:
        log_path = args.json_output.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_handle = None
    try:
        if log_path:
            log_handle = log_path.open("w", encoding="utf-8")
            _PROGRESS_LOG_HANDLE = log_handle
            _progress(f"[OUTPUT] progress_log={log_path}")

        report = run_benchmark(
            args.data_root,
            args.stage,
            case_filters=args.case,
            case_offset=args.case_offset,
            case_limit=args.case_limit,
            use_vlm=args.use_vlm,
            max_vlm_cases=args.max_vlm_cases,
            max_vlm_frames=args.max_vlm_frames,
            vlm_workers=args.vlm_workers,
            vlm_gate_mode=args.vlm_gate_mode,
            replay_vlm_report=args.replay_vlm_report,
        )
    finally:
        _PROGRESS_LOG_HANDLE = None
        if log_handle is not None:
            log_handle.close()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            from analyze_benchmark_errors import build_error_payload

            errors_path = args.json_output.with_name(f"{args.json_output.stem}_errors.json")
            errors_payload = build_error_payload(report, metric="final")
            errors_path.write_text(json.dumps(errors_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            _progress(f"[OUTPUT] failed_to_write_errors error={type(exc).__name__}:{exc}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
