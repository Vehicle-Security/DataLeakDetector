"""Deterministic log-signal rules for NAS leak/hiding detection.

Each rule inspects the merged case logs plus the case's sensitive-file
manifest and fires only on patterns validated to be clean (zero false
hits) on the NAS groundtruth corpus:

- ``file_selected``      sensitive document staged into an upload/attach dialog
- ``upload_event``       explicit upload/send events emitted by the monitor
- ``screen_share``       explicit screen-share start events
- ``archive_created``    sensitive content packed into an archive
- ``convert_created``    sensitive document re-materialized in another format
- ``split_created``      sensitive document split into numbered parts
- ``rename_hiding``      sensitive file renamed outside save/download flows
- ``variant_created``    hiding-style copy of a sensitive document
                         (``副本``/``新版``/``最终版``/numbered variants)
- ``upload_staging``     browser re-reads a pre-existing sensitive document
                         (paired ``created`` + ``browser_file_access`` probes)
- ``clipboard_content``  rich sensitive document content captured on the
                         clipboard (content exfil staging / OCR extraction)

Benign flows that also touch sensitive-looking files are explicitly
vetoed: Office atomic saves (``modified`` on the same path moments after
the event), browser downloads (``created``+``deleted`` pairs and the
post-download rename), and save-as dialogs.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ARCHIVE_EXTENSIONS = (".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz")
DOCUMENT_EXTENSIONS = (
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".txt",
    ".md", ".csv", ".png", ".jpg", ".jpeg",
)
UPLOAD_EVENT_TYPES = {"file_upload", "data_upload", "file_send", "file_share"}
SCREEN_SHARE_EVENT_TYPES = {"screen_share_start", "screen_recording_started"}
SCREEN_CAPTURE_EVENT_TYPES = {"screenshot_capture", "clipboard_image", "record_start"}
SELECTION_EVENT_TYPES = {"file_selected"}
CLIPBOARD_EVENT_TYPES = {"clipboard_text", "clipboard_copy"}
SPLIT_STEM_PATTERN = re.compile(r"(?:^|[_\-\s(])(?:part\s*)?\d{1,3}\)?$|拆分|split", re.IGNORECASE)
VARIANT_SUFFIX_PATTERN = re.compile(
    r"^[\s\-_()（）]*(副本|拷贝|备份|新版|最终版|终版|修订版|copy|backup|final|v?\d{1,4})[\s\-_()（）]*$",
    re.IGNORECASE,
)

# Two distinct sensitive keywords inside a rich clipboard capture mark
# document content (vs. a copied file path, which stays short and path-like).
CLIPBOARD_MIN_COMPACT_CHARS = 40
CLIPBOARD_MIN_KEYWORD_HITS = 2

_NOISE_PATH_MARKERS = (
    "/appdata/", "/programdata/", "/windows/", "/recent/", "/temp/", "/tmp/",
    "/cache", "/$recycle.bin/", "/.git/",
)

_SENSITIVE_KEYWORDS_FALLBACK = (
    "薪资", "工资", "机密", "绝密", "合同", "财务", "客户", "密码", "核心",
    "秘密", "内部", "报表", "预算", "战略", "规划", "会议纪要", "员工",
    "confidential", "salary", "payroll", "contract", "customer", "client",
    "finance", "budget", "strategy", "roadmap",
)


def _normalize(path: str) -> str:
    return str(path or "").replace("\\", "/").strip()


def _basename(path: str) -> str:
    return _normalize(path).rstrip("/").rsplit("/", 1)[-1]


def _stem_ext(name: str) -> Tuple[str, str]:
    base = _basename(name)
    match = re.match(r"^(.*?)(\.[A-Za-z0-9]{1,8})?$", base)
    stem = (match.group(1) if match else base).strip().lower()
    ext = (match.group(2) or "").lower() if match else ""
    return stem, ext


def _parse_ts(value: str) -> Optional[datetime]:
    text = str(value or "").strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _is_user_visible_path(path: str) -> bool:
    lowered = _normalize(path).lower()
    if not lowered:
        return False
    return not any(marker in lowered for marker in _NOISE_PATH_MARKERS)


def _sensitive_stems(sensitive_files: Iterable[str]) -> List[str]:
    stems: List[str] = []
    for item in sensitive_files or []:
        stem, _ = _stem_ext(str(item or ""))
        stem = stem.strip()
        if len(stem) >= 3 and stem not in stems:
            stems.append(stem)
    return stems


def _matches_sensitive(
    text: str,
    stems: List[str],
    is_sensitive_name: Callable[[str], bool],
) -> bool:
    lowered = str(text or "").replace("\\", "/").lower()
    if not lowered:
        return False
    if any(stem in lowered for stem in stems):
        return True
    return bool(is_sensitive_name(lowered))


def _keyword_hits(text: str) -> int:
    lowered = str(text or "").lower()
    return sum(1 for keyword in _SENSITIVE_KEYWORDS_FALLBACK if keyword in lowered)


def extract_deterministic_signals(
    logs: List[Dict[str, Any]],
    sensitive_files: List[str],
    is_sensitive_name: Callable[[str], bool],
) -> Dict[str, Any]:
    """Return fired deterministic rules with per-rule evidence entries."""

    stems = _sensitive_stems(sensitive_files)
    fired: Dict[str, List[Dict[str, Any]]] = {}

    def add(rule: str, log: Dict[str, Any], detail: str = "") -> None:
        entries = fired.setdefault(rule, [])
        if len(entries) >= 8:
            return
        entries.append(
            {
                "timestamp": str(log.get("timestamp", "") or ""),
                "event_type": str(log.get("event_type", "") or ""),
                "file_path": _normalize(str(log.get("file_path", "") or ""))[:200],
                "detail": detail[:200],
            }
        )

    rows: List[Dict[str, Any]] = [log for log in logs if isinstance(log, dict)]
    fs_events: List[Tuple[Optional[datetime], str, str]] = []
    for log in rows:
        fs_events.append(
            (
                _parse_ts(log.get("timestamp", "")),
                str(log.get("event_type", "") or "").lower(),
                str(log.get("file_path", "") or ""),
            )
        )

    def same_path_near(
        path: str,
        when: Optional[datetime],
        types: Iterable[str],
        window_seconds: float,
        direction: str = "both",
    ) -> bool:
        if when is None:
            return False
        wanted = set(types)
        normalized = _normalize(path).lower()
        for ts, event_type, event_path in fs_events:
            if event_type not in wanted or ts is None:
                continue
            if _normalize(event_path).lower() != normalized:
                continue
            delta = (ts - when).total_seconds()
            if direction == "before" and not (-window_seconds <= delta <= 0):
                continue
            if direction == "after" and not (0 <= delta <= window_seconds):
                continue
            if direction == "both" and abs(delta) > window_seconds:
                continue
            return True
        return False

    def event_ref(log: Dict[str, Any]) -> str:
        path = str(log.get("file_path", "") or "")
        return path if path else str(log.get("file_name", "") or "")

    def is_sensitive_ref(ref: str) -> bool:
        return _matches_sensitive(_basename(ref), stems, is_sensitive_name)

    clipboard_path_times: List[Tuple[Optional[datetime], str]] = []
    for log in rows:
        event_type = str(log.get("event_type", "") or "").lower()
        if event_type not in CLIPBOARD_EVENT_TYPES:
            continue
        preview = str(log.get("content_preview", "") or "").strip().strip("\"'")
        if ("/" in preview or "\\" in preview) and is_sensitive_ref(preview):
            clipboard_path_times.append((_parse_ts(log.get("timestamp", "")), preview))

    # Moments where a keyword-sensitive document is visibly in play (open
    # window or user-visible file event); used to ground screen captures.
    monitor_ui_markers = ("win monitor", "localhost:5000", "数据泄露行为监控", "监控系统")
    sensitive_context_times: List[Optional[datetime]] = []
    for log in rows:
        window_info = log.get("window_info") if isinstance(log.get("window_info"), dict) else {}
        window_title = str(window_info.get("window_title", "") or log.get("window_title", "") or "")
        if any(marker in window_title.lower() for marker in monitor_ui_markers):
            continue  # the DLP dashboard itself surfaces sensitive names
        path = str(log.get("file_path", "") or "")
        name = str(log.get("file_name", "") or "")
        context_hit = False
        if window_title and is_sensitive_name(window_title.lower()):
            context_hit = True
        elif (path or name) and _is_user_visible_path(path or name) and is_sensitive_name(
            _basename(path or name).lower()
        ):
            context_hit = True
        if context_hit:
            sensitive_context_times.append(_parse_ts(log.get("timestamp", "")))

    def variant_suffix_of(stem: str) -> str:
        for candidate in stems:
            if candidate != stem and stem.startswith(candidate):
                return stem[len(candidate):]
            if candidate != stem and stem.endswith(candidate):
                return stem[: len(stem) - len(candidate)]
        return ""

    for log in rows:
        event_type = str(log.get("event_type", "") or "").lower()
        ref = event_ref(log)
        path = str(log.get("file_path", "") or "")
        preview = str(log.get("content_preview", "") or "")
        when = _parse_ts(log.get("timestamp", ""))
        stem, ext = _stem_ext(ref)

        if event_type in UPLOAD_EVENT_TYPES:
            add("upload_event", log)
            continue
        if event_type in SCREEN_SHARE_EVENT_TYPES:
            add("screen_share", log)
            continue
        if event_type in SCREEN_CAPTURE_EVENT_TYPES:
            if when is not None and any(
                ts is not None and abs((when - ts).total_seconds()) <= 30
                for ts in sensitive_context_times
            ):
                add("screen_capture", log, "capture while sensitive document in play")
            continue

        if event_type in SELECTION_EVENT_TYPES:
            selected_sensitive = is_sensitive_ref(ref) or _matches_sensitive(
                preview, stems, is_sensitive_name
            )
            document_like = not ext or ext in DOCUMENT_EXTENSIONS + ARCHIVE_EXTENSIONS
            if selected_sensitive and document_like:
                add("file_selected", log)
            elif not ref and when is not None and any(
                ts is not None and 0 <= (when - ts).total_seconds() <= 45
                for ts, _ in clipboard_path_times
            ):
                add("file_selected", log, "path-less selection after sensitive path on clipboard")
            continue

        if event_type in CLIPBOARD_EVENT_TYPES:
            compact = re.sub(r"\s+", "", preview)
            path_like = ("/" in preview or "\\" in preview) and len(compact) <= 160
            if len(compact) >= CLIPBOARD_MIN_COMPACT_CHARS and not path_like:
                stem_hit = any(s in compact.lower() for s in stems)
                if stem_hit or _keyword_hits(compact) >= CLIPBOARD_MIN_KEYWORD_HITS:
                    add("clipboard_content", log, "document content on clipboard")
            continue

        if event_type in ("created", "opened") and is_sensitive_ref(ref) and _is_user_visible_path(path):
            suffix = variant_suffix_of(stem)
            if suffix and ext in DOCUMENT_EXTENSIONS and SPLIT_STEM_PATTERN.search(suffix.strip(" -_()") or suffix):
                add("split_created", log, "numbered fragment of sensitive document")
                continue
            if (
                suffix
                and VARIANT_SUFFIX_PATTERN.match(suffix)
                and ext in DOCUMENT_EXTENSIONS
                and not same_path_near(path, when, ("modified",), 6.0)
            ):
                add("variant_created", log, f"hiding-style variant of sensitive stem (suffix {suffix.strip()})")
                continue

        if event_type == "created":
            if ext in ARCHIVE_EXTENSIONS and is_sensitive_ref(ref):
                add("archive_created", log)
                continue
            if not is_sensitive_ref(ref) or not _is_user_visible_path(path):
                continue
            sensitive_exts = {
                _stem_ext(item)[1] for item in sensitive_files or [] if _stem_ext(item)[0] == stem
            }
            if ext == ".pdf" and any(
                candidate not in ("", ".pdf") for candidate in sensitive_exts
            ):
                add("convert_created", log, "pdf materialized from non-pdf sensitive source")
                continue
            if ext not in DOCUMENT_EXTENSIONS:
                continue
            if same_path_near(path, when, ("deleted",), 3.0):
                continue  # download finalize / transient copy artifact
            if same_path_near(path, when, ("renamed",), 60.0, direction="after"):
                continue  # download then user rename
            if same_path_near(path, when, ("modified",), 6.0):
                continue  # local save flow
            if same_path_near(path, when, ("browser_file_access",), 3.0):
                add("upload_staging", log, "browser re-read of pre-existing sensitive document")
            continue

        if event_type == "renamed":
            if not is_sensitive_ref(ref) or not _is_user_visible_path(path):
                continue
            if same_path_near(path, when, ("created",), 600.0, direction="before"):
                continue  # renamed shortly after appearing (download/save-as flows)
            if same_path_near(path, when, ("modified", "created", "deleted"), 6.0):
                variant = ""
                if when is not None:
                    for other in rows:
                        other_type = str(other.get("event_type", "") or "").lower()
                        if other_type not in ("created", "modified"):
                            continue
                        other_path = str(other.get("file_path", "") or "")
                        if not other_path or _normalize(other_path).lower() == _normalize(path).lower():
                            continue
                        other_stem, _ = _stem_ext(other_path)
                        other_when = _parse_ts(other.get("timestamp", ""))
                        if other_when is None or abs((other_when - when).total_seconds()) > 6.0:
                            continue
                        if stem and stem in other_stem and other_stem != stem:
                            variant = other_path
                            break
                if variant:
                    add("rename_hiding", log, f"renamed to variant {_basename(variant)}")
                continue
            add("rename_hiding", log, "isolated rename outside save/download flows")
            continue

    # Weak staging: log traces of the sensitive document being handed toward
    # an external surface. Not a positive signal by itself, but VLM claims of
    # a *completed* transfer are only trustworthy when such a trace exists —
    # inbound flows (downloads, inbox attachments, remote listings) lack it.
    weak_staging = bool(
        fired.get("file_selected")
        or fired.get("upload_event")
        or fired.get("screen_share")
        or fired.get("upload_staging")
        or clipboard_path_times
    )
    if not weak_staging:
        for log in rows:
            event_type = str(log.get("event_type", "") or "").lower()
            if event_type != "browser_file_access":
                continue
            ref = event_ref(log)
            if ref and _is_user_visible_path(ref) and is_sensitive_ref(ref):
                weak_staging = True
                break

    positive_rules = sorted(fired.keys())
    return {
        "positive": bool(positive_rules),
        "rules": positive_rules,
        "evidence": fired,
        "sensitive_stems": stems[:16],
        "weak_staging": weak_staging,
    }
