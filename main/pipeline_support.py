from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import os
import re
from pathlib import Path
from typing import Any


TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)


@dataclass(frozen=True)
class SampleContext:
    record_id: str
    sample_root: str
    video_path: str
    recording_start_time: str
    search_start_time: str
    search_end_time: str
    sensitive_files: list[str]
    target_keywords: list[str]
    groundtruth: dict[str, Any]
    index_metadata: dict[str, str]
    context_inference: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pipeline_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    config_dir = module_root / "config"

    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))

    env_path = os.getenv("DLD_PIPELINE_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(config_dir / "pipeline.local.json")
    candidates.append(config_dir / "pipeline.default.json")

    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        "No pipeline config found. "
        f"Checked: {[str(candidate) for candidate in candidates]}"
    )


def discover_sample_roots(sample_base: Path, sample_names: list[str] | None = None) -> list[Path]:
    if sample_names:
        return [sample_base / sample_name for sample_name in sample_names]

    sample_roots: list[Path] = []
    if not sample_base.exists():
        return sample_roots

    for child in sorted(sample_base.iterdir(), key=lambda item: item.name):
        if child.is_dir() and is_valid_sample_root(child):
            sample_roots.append(child)
    return sample_roots


def is_valid_sample_root(sample_root: Path) -> bool:
    return (
        sample_root.exists()
        and sample_root.is_dir()
        and (sample_root / "logs" / "keyevents.json").exists()
        and any((sample_root / "video").glob("*.mp4"))
    )


def load_sample_context(
    sample_root: Path,
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
    *,
    mode: str = "full",
) -> SampleContext:
    if mode not in {"full", "demo"}:
        raise ValueError(f"Unsupported context mode: {mode}")

    groundtruth = _load_json_if_exists(sample_root / "groundtruth.json") if mode == "demo" else {}
    index_metadata = _parse_index_metadata(sample_root / "INDEX.md")
    recording_start_time = _infer_recording_start_time(index_metadata, log_events)
    if mode == "demo":
        sensitive_files, sensitive_inference = _infer_sensitive_files_demo(groundtruth, log_events, pipeline_config)
        search_start_time, search_end_time, search_inference = _infer_search_window_demo(
            recording_start_time=recording_start_time,
            groundtruth=groundtruth,
            log_events=log_events,
            pipeline_config=pipeline_config,
        )
        target_keywords, keyword_inference = _infer_target_keywords_demo(
            sensitive_files=sensitive_files,
            groundtruth=groundtruth,
            log_events=log_events,
            search_start_time=search_start_time,
            search_end_time=search_end_time,
            pipeline_config=pipeline_config,
        )
    else:
        sensitive_files, sensitive_inference = _infer_sensitive_files_from_logs(log_events, pipeline_config)
        search_start_time, search_end_time, search_inference = _infer_search_window_from_logs(
            recording_start_time=recording_start_time,
            log_events=log_events,
            pipeline_config=pipeline_config,
            sensitive_files=sensitive_files,
        )
        target_keywords, keyword_inference = _infer_target_keywords_from_logs(
            sensitive_files=sensitive_files,
            log_events=log_events,
            search_start_time=search_start_time,
            search_end_time=search_end_time,
            pipeline_config=pipeline_config,
            search_inference=search_inference,
        )

    context_inference = {
        "mode": mode,
        "groundtruth_used": mode == "demo",
        "sensitive_files": sensitive_inference,
        "search_window": search_inference,
        "target_keywords": keyword_inference,
    }

    return SampleContext(
        record_id=sample_root.name,
        sample_root=str(sample_root),
        video_path=str(resolve_video_path(sample_root)),
        recording_start_time=recording_start_time,
        search_start_time=search_start_time,
        search_end_time=search_end_time,
        sensitive_files=sensitive_files,
        target_keywords=target_keywords,
        groundtruth=groundtruth,
        index_metadata=index_metadata,
        context_inference=context_inference,
    )


def resolve_video_path(sample_root: Path) -> Path:
    videos = sorted((sample_root / "video").glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No video file found under {sample_root / 'video'}")
    return videos[0]


def build_demo_segments(
    sample_root: Path,
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> list[dict[str, Any]]:
    context = load_sample_context(sample_root, log_events, pipeline_config, mode="demo")
    operations = list((context.groundtruth or {}).get("operations", []) or [])
    if not operations:
        return []

    demo_keywords = [
        str(keyword or "").strip()
        for keyword in pipeline_config.get("demo_operation_keywords", []) or []
        if str(keyword or "").strip()
    ]

    segments: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        operation_text = str(operation.get("operation", "") or "").strip()
        if not operation_text:
            continue
        if not any(keyword in operation_text for keyword in demo_keywords):
            continue

        operation_time = normalize_timestamp_text(str(operation.get("operation_time", "") or ""))
        sensitive_file = normalize_file_path(str(operation.get("sensitive_file_path", "") or ""))
        related_resources = infer_related_resources(
            sensitive_file=sensitive_file,
            log_events=log_events,
            search_start_time=context.search_start_time,
            search_end_time=context.search_end_time,
        )
        app_name = infer_demo_app_name(operation_text)
        operation_type = infer_demo_operation_type(operation_text)

        primary_resource = Path(sensitive_file).name if sensitive_file else ""
        segment_related = list(related_resources)
        if operation_type == "邮件附件外发" and related_resources:
            primary_resource = related_resources[0]
            segment_related = related_resources[1:]

        segments.append(
            {
                "segment_id": f"demo_segment_{index}",
                "time_range": f"{operation_time} - {operation_time}",
                "app_name": app_name,
                "operation_type": operation_type,
                "primary_resource": primary_resource,
                "related_resources": segment_related,
                "action_description": operation_text,
                "visible_evidence": [
                    item
                    for item in [primary_resource, *segment_related, app_name]
                    if item
                ],
                "supporting_timestamps": [operation_time] if operation_time else [],
                "confidence": 0.9,
            }
        )

    return segments


def infer_related_resources(
    sensitive_file: str,
    log_events: list[dict[str, Any]],
    search_start_time: str,
    search_end_time: str,
) -> list[str]:
    start_dt = parse_timestamp(search_start_time)
    end_dt = parse_timestamp(search_end_time)
    sensitive_stem = Path(sensitive_file).stem
    resources: list[str] = []
    seen = set()

    if not sensitive_stem or start_dt is None or end_dt is None:
        return resources

    for event in log_events:
        event_time = parse_timestamp(str(event.get("timestamp", "") or ""))
        if event_time is None or event_time < start_dt or event_time > end_dt:
            continue
        file_path = normalize_file_path(str(event.get("file_path", "") or ""))
        basename = Path(file_path).name
        stem = Path(file_path).stem
        if not basename or stem == sensitive_stem or not stem.startswith(sensitive_stem):
            continue
        if basename not in seen:
            seen.add(basename)
            resources.append(basename)
    return resources


def infer_demo_app_name(operation_text: str) -> str:
    if "邮箱" in operation_text or "邮件" in operation_text:
        return "QQ邮箱"
    if "会议" in operation_text:
        return "腾讯会议"
    if "分片" in operation_text or "导出" in operation_text:
        return "WindowsTerminal"
    return "UnknownApp"


def infer_demo_operation_type(operation_text: str) -> str:
    if "屏幕共享" in operation_text or "共享屏幕" in operation_text:
        return "共享屏幕"
    if "邮箱" in operation_text or "邮件" in operation_text:
        return "邮件附件外发"
    if "分片" in operation_text or "导出" in operation_text:
        return "分片导出"
    if "加入会议" in operation_text or "会议客户端" in operation_text:
        return "加入会议"
    return "观察到操作"


def normalize_file_path(file_path: str) -> str:
    normalized = str(file_path or "").strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def normalize_timestamp_text(value: str) -> str:
    text = str(value or "").strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None

    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    normalized = normalize_timestamp_text(text)
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_index_metadata(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}

    metadata: dict[str, str] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\*\*(.+?)\*\*:\s*(.+?)\s*$", line.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        metadata[key] = match.group(2).strip()
    return metadata


def _infer_recording_start_time(
    index_metadata: dict[str, str],
    log_events: list[dict[str, Any]],
) -> str:
    recording_time = normalize_timestamp_text(index_metadata.get("recording_time", ""))
    if recording_time:
        return recording_time
    if log_events:
        first_timestamp = normalize_timestamp_text(str(log_events[0].get("timestamp", "") or ""))
        if first_timestamp:
            return first_timestamp
    raise ValueError("Unable to infer recording start time")


def _infer_sensitive_files_demo(
    groundtruth: dict[str, Any],
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    sensitive_files: list[str] = []
    seen = set()

    for operation in list(groundtruth.get("operations", []) or []):
        file_path = normalize_file_path(str(operation.get("sensitive_file_path", "") or ""))
        if file_path and file_path not in seen:
            seen.add(file_path)
            sensitive_files.append(file_path)

    if sensitive_files:
        return sensitive_files, {
            "strategy": "groundtruth_operations",
            "matched_paths": list(sensitive_files),
        }

    sensitive_keywords = [
        str(keyword or "").strip()
        for keyword in pipeline_config.get("sensitive_filename_keywords", []) or []
        if str(keyword or "").strip()
    ]
    for event in log_events:
        file_path = normalize_file_path(str(event.get("file_path", "") or ""))
        file_name = Path(file_path).name
        if not file_name:
            continue
        if not any(keyword in file_name for keyword in sensitive_keywords):
            continue
        if file_path not in seen:
            seen.add(file_path)
            sensitive_files.append(file_path)

    return sensitive_files, {
        "strategy": "log_keyword_fallback",
        "matched_paths": list(sensitive_files),
    }


def _infer_search_window_demo(
    recording_start_time: str,
    groundtruth: dict[str, Any],
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    suspicious_keywords = [
        str(keyword or "").strip()
        for keyword in pipeline_config.get("suspicious_operation_keywords", []) or []
        if str(keyword or "").strip()
    ]
    operation_rules = list(pipeline_config.get("operation_rules", []) or [])
    best_span: tuple[datetime, datetime] | None = None
    best_priority = -1
    best_start_time: datetime | None = None

    for operation in list(groundtruth.get("operations", []) or []):
        operation_text = str(operation.get("operation", "") or "").strip()
        if not any(keyword in operation_text for keyword in suspicious_keywords):
            continue
        operation_time = parse_timestamp(str(operation.get("operation_time", "") or ""))
        if operation_time is None:
            continue

        rule = _find_matching_rule(operation_text, operation_rules)
        search_window = pipeline_config.get("search_window", {}) or {}
        pre_buffer = int(
            (rule or {}).get("pre_buffer_seconds", search_window.get("fallback_pre_buffer_seconds", 10)) or 10
        )
        post_buffer = int(
            (rule or {}).get("post_buffer_seconds", search_window.get("fallback_post_buffer_seconds", 20)) or 20
        )
        span = (
            operation_time - timedelta(seconds=pre_buffer),
            operation_time + timedelta(seconds=post_buffer),
        )
        priority = int((rule or {}).get("priority", 0) or 0)
        if (
            best_span is None
            or priority > best_priority
            or (priority == best_priority and operation_time > (best_start_time or operation_time))
        ):
            best_span = span
            best_priority = priority
            best_start_time = operation_time

    if best_span is not None:
        return format_timestamp(best_span[0]), format_timestamp(best_span[1]), {
            "strategy": "groundtruth_operation_rule",
            "priority": best_priority,
            "operation_time": format_timestamp(best_start_time) if best_start_time else "",
        }

    interesting_times = [
        parse_timestamp(str(event.get("timestamp", "") or ""))
        for event in log_events
        if str(event.get("event_type", "") or "") in {"file_upload", "file_selected", "upload_detected"}
    ]
    interesting_times = [item for item in interesting_times if item is not None]
    if interesting_times:
        search_window = pipeline_config.get("search_window", {}) or {}
        pre_buffer = int(search_window.get("fallback_pre_buffer_seconds", 10) or 10)
        post_buffer = int(search_window.get("fallback_post_buffer_seconds", 20) or 20)
        return (
            format_timestamp(min(interesting_times) - timedelta(seconds=pre_buffer)),
            format_timestamp(max(interesting_times) + timedelta(seconds=post_buffer)),
            {
                "strategy": "upload_event_fallback",
                "event_count": len(interesting_times),
            },
        )

    recording_start_dt = parse_timestamp(recording_start_time)
    if recording_start_dt is None:
        raise ValueError("Unable to infer search window")
    fallback_duration = int(
        (pipeline_config.get("search_window", {}) or {}).get("fallback_duration_seconds", 90) or 90
    )
    return (
        recording_start_time,
        format_timestamp(recording_start_dt + timedelta(seconds=fallback_duration)),
        {
            "strategy": "recording_start_fallback",
            "duration_seconds": fallback_duration,
        },
    )


def _infer_target_keywords_demo(
    sensitive_files: list[str],
    groundtruth: dict[str, Any],
    log_events: list[dict[str, Any]],
    search_start_time: str,
    search_end_time: str,
    pipeline_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    sensitive_keywords: list[str] = []
    app_keywords: list[str] = []
    scene_keywords: list[str] = []
    seen = set()

    alias_values = {
        str(alias.get("keyword", "") or "").strip()
        for alias in list(pipeline_config.get("app_keyword_aliases", []) or [])
        if str(alias.get("keyword", "") or "").strip()
    }

    def add_keyword(value: str, bucket: list[str]) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        bucket.append(text)

    sensitive_stems: list[str] = []
    for file_path in sensitive_files:
        stem = Path(file_path).stem
        if stem:
            sensitive_stems.append(stem)
            add_keyword(stem, sensitive_keywords)

    suspicious_keywords = [
        str(keyword or "").strip()
        for keyword in pipeline_config.get("suspicious_operation_keywords", []) or []
        if str(keyword or "").strip()
    ]
    operation_rules = list(pipeline_config.get("operation_rules", []) or [])
    for operation in list(groundtruth.get("operations", []) or []):
        operation_text = str(operation.get("operation", "") or "").strip()
        if not any(keyword in operation_text for keyword in suspicious_keywords):
            continue
        rule = _find_matching_rule(operation_text, operation_rules)
        if rule is None:
            continue
        for item in list(rule.get("target_keywords", []) or []):
            if str(item or "").strip() in alias_values:
                add_keyword(item, app_keywords)
            else:
                add_keyword(item, scene_keywords)

    start_dt = parse_timestamp(search_start_time)
    end_dt = parse_timestamp(search_end_time)
    aliases = list(pipeline_config.get("app_keyword_aliases", []) or [])
    if start_dt is not None and end_dt is not None:
        matching_rule = None
        best_rule_priority = -1
        for operation in list(groundtruth.get("operations", []) or []):
            operation_text = str(operation.get("operation", "") or "").strip()
            if not any(keyword in operation_text for keyword in suspicious_keywords):
                continue
            rule = _find_matching_rule(operation_text, operation_rules)
            if rule is None:
                continue
            priority = int(rule.get("priority", 0) or 0)
            if priority > best_rule_priority:
                best_rule_priority = priority
                matching_rule = rule

        allowed_alias_keywords = None
        if matching_rule is not None:
            allowed_alias_keywords = {
                str(item or "").strip()
                for item in matching_rule.get("target_keywords", []) or []
                if str(item or "").strip() in alias_values
            }

        for event in log_events:
            event_time = parse_timestamp(str(event.get("timestamp", "") or ""))
            if event_time is None or event_time < start_dt or event_time > end_dt:
                continue

            haystack = " ".join(
                [
                    str(event.get("app_name", "") or ""),
                    str(((event.get("process_info") or {}).get("process_name", "")) or ""),
                    str(((event.get("window_info") or {}).get("window_title", "")) or ""),
                    str(event.get("content_preview", "") or ""),
                ]
            ).lower()
            for alias in aliases:
                match_any = [str(item or "").strip().lower() for item in alias.get("match_any", []) or []]
                if any(item and item in haystack for item in match_any):
                    keyword = str(alias.get("keyword", "") or "").strip()
                    if allowed_alias_keywords is not None and keyword not in allowed_alias_keywords:
                        continue
                    add_keyword(keyword, app_keywords)

        for event in log_events:
            event_time = parse_timestamp(str(event.get("timestamp", "") or ""))
            if event_time is None or event_time < start_dt or event_time > end_dt:
                continue
            file_path = normalize_file_path(str(event.get("file_path", "") or ""))
            stem = Path(file_path).stem
            if not stem:
                continue
            for sensitive_stem in sensitive_stems:
                if stem == sensitive_stem or not stem.startswith(sensitive_stem):
                    continue
                suffix = stem[len(sensitive_stem):].strip("_-. ")
                if not suffix:
                    continue
                for token in re.split(r"[_\-.]+", suffix):
                    if token:
                        add_keyword(token, scene_keywords)

    keywords = [*sensitive_keywords, *app_keywords, *scene_keywords]
    return keywords, {
        "strategy": "groundtruth_and_log_context",
        "sensitive_keywords": list(sensitive_keywords),
        "app_keywords": list(app_keywords),
        "scene_keywords": list(scene_keywords),
    }


def _infer_sensitive_files_from_logs(
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    sensitive_keywords = [
        str(keyword or "").strip()
        for keyword in pipeline_config.get("sensitive_filename_keywords", []) or []
        if str(keyword or "").strip()
    ]

    sorted_events = sorted(
        list(log_events),
        key=lambda event: parse_timestamp(str(event.get("timestamp", "") or "")) or datetime.max,
    )
    sensitive_files: list[str] = []
    seen = set()
    evidence: list[dict[str, Any]] = []

    def add_candidate(file_path: str, reason: str, event: dict[str, Any]) -> None:
        normalized = normalize_file_path(file_path)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        sensitive_files.append(normalized)
        evidence.append(
            {
                "file_path": normalized,
                "reason": reason,
                "event_type": str(event.get("event_type", "") or ""),
                "timestamp": normalize_timestamp_text(str(event.get("timestamp", "") or "")),
            }
        )

    for event in sorted_events:
        event_type = str(event.get("event_type", "") or "").strip().lower()
        if event_type not in {"file_open", "file_selected"}:
            continue
        file_path = normalize_file_path(str(event.get("file_path", "") or ""))
        basename = Path(file_path).name
        if not basename:
            continue
        if any(keyword in basename for keyword in sensitive_keywords):
            add_candidate(file_path, "sensitive_keyword_file_open", event)

    if sensitive_files:
        return sensitive_files, {
            "strategy": "log_file_open_keywords",
            "matched_files": evidence,
        }

    sink_signal = _infer_primary_log_signal(log_events, pipeline_config)
    sink_dt = parse_timestamp(str((sink_signal or {}).get("timestamp", "") or ""))
    latest_open_before_sink: tuple[datetime, str, dict[str, Any]] | None = None
    for event in sorted_events:
        event_type = str(event.get("event_type", "") or "").strip().lower()
        if event_type != "file_open":
            continue
        event_dt = parse_timestamp(str(event.get("timestamp", "") or ""))
        if event_dt is None or sink_dt is None or event_dt > sink_dt:
            continue
        file_path = normalize_file_path(str(event.get("file_path", "") or ""))
        if not file_path:
            continue
        latest_open_before_sink = (event_dt, file_path, event)

    if latest_open_before_sink is not None:
        _, file_path, event = latest_open_before_sink
        add_candidate(file_path, "latest_file_open_before_sink", event)

    if sensitive_files:
        return sensitive_files, {
            "strategy": "latest_file_open_before_sink",
            "matched_files": evidence,
            "sink_signal": sink_signal or {},
        }

    for event in sorted_events:
        file_path = normalize_file_path(str(event.get("file_path", "") or ""))
        basename = Path(file_path).name
        if not basename:
            continue
        if any(keyword in basename for keyword in sensitive_keywords):
            add_candidate(file_path, "sensitive_keyword_any_event", event)

    return sensitive_files, {
        "strategy": "log_keyword_scan",
        "matched_files": evidence,
        "sink_signal": sink_signal or {},
    }


def _infer_search_window_from_logs(
    recording_start_time: str,
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
    sensitive_files: list[str],
) -> tuple[str, str, dict[str, Any]]:
    search_window = pipeline_config.get("search_window", {}) or {}
    fallback_pre_buffer = int(search_window.get("fallback_pre_buffer_seconds", 10) or 10)
    fallback_post_buffer = int(search_window.get("fallback_post_buffer_seconds", 20) or 20)
    fallback_duration = int(search_window.get("fallback_duration_seconds", 90) or 90)

    sink_signal = _infer_primary_log_signal(log_events, pipeline_config)
    sink_dt = parse_timestamp(str((sink_signal or {}).get("timestamp", "") or ""))
    if sink_signal is not None and sink_dt is not None:
        pre_buffer = int(sink_signal.get("pre_buffer_seconds", fallback_pre_buffer) or fallback_pre_buffer)
        post_buffer = int(sink_signal.get("post_buffer_seconds", fallback_post_buffer) or fallback_post_buffer)
        return (
            format_timestamp(sink_dt - timedelta(seconds=pre_buffer)),
            format_timestamp(sink_dt + timedelta(seconds=post_buffer)),
            {
                "strategy": "log_sink_signal",
                **sink_signal,
            },
        )

    interesting_times = [
        parse_timestamp(str(event.get("timestamp", "") or ""))
        for event in log_events
        if str(event.get("event_type", "") or "").strip().lower()
        in {"file_upload", "file_selected", "upload_detected", "screen_share_start"}
    ]
    interesting_times = [item for item in interesting_times if item is not None]
    if interesting_times:
        return (
            format_timestamp(min(interesting_times) - timedelta(seconds=fallback_pre_buffer)),
            format_timestamp(max(interesting_times) + timedelta(seconds=fallback_post_buffer)),
            {
                "strategy": "interesting_event_fallback",
                "event_count": len(interesting_times),
                "sensitive_files_considered": list(sensitive_files),
            },
        )

    recording_start_dt = parse_timestamp(recording_start_time)
    if recording_start_dt is None:
        raise ValueError("Unable to infer search window")

    return (
        recording_start_time,
        format_timestamp(recording_start_dt + timedelta(seconds=fallback_duration)),
        {
            "strategy": "recording_start_fallback",
            "duration_seconds": fallback_duration,
            "sensitive_files_considered": list(sensitive_files),
        },
    )


def _infer_target_keywords_from_logs(
    sensitive_files: list[str],
    log_events: list[dict[str, Any]],
    search_start_time: str,
    search_end_time: str,
    pipeline_config: dict[str, Any],
    search_inference: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    sensitive_keywords: list[str] = []
    app_keywords: list[str] = []
    scene_keywords: list[str] = []
    seen = set()

    alias_values = {
        str(alias.get("keyword", "") or "").strip()
        for alias in list(pipeline_config.get("app_keyword_aliases", []) or [])
        if str(alias.get("keyword", "") or "").strip()
    }

    def add_keyword(value: str, bucket: list[str]) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        bucket.append(text)

    sensitive_stems: list[str] = []
    for file_path in sensitive_files:
        stem = Path(file_path).stem
        if stem:
            sensitive_stems.append(stem)
            add_keyword(stem, sensitive_keywords)

    selected_rule = str(search_inference.get("rule_name", "") or "").strip()
    operation_rules = list(pipeline_config.get("operation_rules", []) or [])
    alias_map = {
        str(alias.get("keyword", "") or "").strip(): alias
        for alias in list(pipeline_config.get("app_keyword_aliases", []) or [])
        if str(alias.get("keyword", "") or "").strip()
    }
    if selected_rule:
        matching_rule = next(
            (rule for rule in operation_rules if str(rule.get("name", "") or "").strip() == selected_rule),
            None,
        )
        if matching_rule is not None:
            for item in list(matching_rule.get("target_keywords", []) or []):
                keyword = str(item or "").strip()
                if not keyword:
                    continue
                if keyword in alias_values:
                    add_keyword(keyword, app_keywords)
                else:
                    add_keyword(keyword, scene_keywords)

    start_dt = parse_timestamp(search_start_time)
    end_dt = parse_timestamp(search_end_time)
    aliases = list(pipeline_config.get("app_keyword_aliases", []) or [])
    if start_dt is not None and end_dt is not None:
        for event in log_events:
            event_time = parse_timestamp(str(event.get("timestamp", "") or ""))
            if event_time is None or event_time < start_dt or event_time > end_dt:
                continue

            haystack = _build_log_event_haystack(event)
            for alias in aliases:
                match_any = [str(item or "").strip().lower() for item in alias.get("match_any", []) or []]
                if any(item and item in haystack for item in match_any):
                    add_keyword(str(alias.get("keyword", "") or "").strip(), app_keywords)

        for event in log_events:
            event_time = parse_timestamp(str(event.get("timestamp", "") or ""))
            if event_time is None or event_time < start_dt or event_time > end_dt:
                continue
            file_path = normalize_file_path(str(event.get("file_path", "") or ""))
            stem = Path(file_path).stem
            if not stem:
                continue
            for sensitive_stem in sensitive_stems:
                if stem == sensitive_stem or not stem.startswith(sensitive_stem):
                    continue
                suffix = stem[len(sensitive_stem):].strip("_-. ")
                if not suffix:
                    continue
                for token in re.split(r"[_\-.]+", suffix):
                    if token:
                        add_keyword(token, scene_keywords)

    if not app_keywords and selected_rule:
        for keyword, alias in alias_map.items():
            match_any = [str(item or "").strip().lower() for item in alias.get("match_any", []) or []]
            sink_haystack = str(search_inference.get("matched_text", "") or "").strip().lower()
            if any(item and item in sink_haystack for item in match_any):
                add_keyword(keyword, app_keywords)

    keywords = [*sensitive_keywords, *app_keywords, *scene_keywords]
    return keywords, {
        "strategy": "log_context",
        "selected_rule": selected_rule,
        "sensitive_keywords": list(sensitive_keywords),
        "app_keywords": list(app_keywords),
        "scene_keywords": list(scene_keywords),
    }


def _infer_primary_log_signal(
    log_events: list[dict[str, Any]],
    pipeline_config: dict[str, Any],
) -> dict[str, Any] | None:
    operation_rules = list(pipeline_config.get("operation_rules", []) or [])
    search_window = pipeline_config.get("search_window", {}) or {}
    fallback_pre_buffer = int(search_window.get("fallback_pre_buffer_seconds", 10) or 10)
    fallback_post_buffer = int(search_window.get("fallback_post_buffer_seconds", 20) or 20)

    best_signal: dict[str, Any] | None = None
    best_priority = -1
    best_timestamp: datetime | None = None

    for event in log_events:
        event_time = parse_timestamp(str(event.get("timestamp", "") or ""))
        if event_time is None:
            continue

        signal_text = _build_log_signal_text(event)
        rule = _find_matching_rule(signal_text, operation_rules)
        event_type = str(event.get("event_type", "") or "").strip().lower()
        if rule is None and event_type not in {"file_upload", "file_selected", "upload_detected", "screen_share_start"}:
            continue

        priority = int((rule or {}).get("priority", 0) or 0)
        if event_type == "screen_share_start":
            priority = max(priority, 100)
        if event_type in {"file_upload", "file_selected", "upload_detected"}:
            priority = max(priority, 80)

        if (
            best_signal is None
            or priority > best_priority
            or (priority == best_priority and event_time > (best_timestamp or event_time))
        ):
            pre_buffer = int((rule or {}).get("pre_buffer_seconds", fallback_pre_buffer) or fallback_pre_buffer)
            post_buffer = int((rule or {}).get("post_buffer_seconds", fallback_post_buffer) or fallback_post_buffer)
            if event_type in {"file_upload", "file_selected", "upload_detected"}:
                pre_buffer = max(pre_buffer, fallback_pre_buffer)

            best_signal = {
                "rule_name": str((rule or {}).get("name", "") or ""),
                "timestamp": normalize_timestamp_text(str(event.get("timestamp", "") or "")),
                "event_type": str(event.get("event_type", "") or ""),
                "matched_text": signal_text,
                "app_name": str(event.get("app_name", "") or ""),
                "file_path": normalize_file_path(str(event.get("file_path", "") or "")),
                "pre_buffer_seconds": pre_buffer,
                "post_buffer_seconds": post_buffer,
            }
            best_priority = priority
            best_timestamp = event_time

    return best_signal


def _build_log_signal_text(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type", "") or "").strip().lower()
    parts = [
        str(event.get("event_type", "") or ""),
        str(event.get("app_name", "") or ""),
        str(((event.get("process_info") or {}).get("process_name", "")) or ""),
        str(((event.get("window_info") or {}).get("window_title", "")) or ""),
        str(event.get("content_preview", "") or ""),
        str(((event.get("extra") or {}).get("raw_operation", "")) or ""),
        str(((event.get("extra") or {}).get("category", "")) or ""),
    ]

    lowered_haystack = _build_log_event_haystack(event)
    if event_type == "screen_share_start":
        parts.extend(["屏幕共享", "共享屏幕"])
    if event_type == "meeting_join":
        parts.extend(["加入会议", "会议客户端"])
    if event_type in {"file_upload", "file_selected", "upload_detected"}:
        parts.append("上传")
        if any(marker in lowered_haystack for marker in ("qq邮箱", "mail.qq.com", "qqmail", "邮箱", "邮件")):
            parts.extend(["直接外发", "邮件附件", "QQ邮箱"])

    return " ".join(item for item in parts if str(item or "").strip())


def _build_log_event_haystack(event: dict[str, Any]) -> str:
    return " ".join(
        [
            str(event.get("event_type", "") or ""),
            str(event.get("file_path", "") or ""),
            str(event.get("file_name", "") or ""),
            str(event.get("app_name", "") or ""),
            str(((event.get("process_info") or {}).get("process_name", "")) or ""),
            str(((event.get("window_info") or {}).get("window_title", "")) or ""),
            str(event.get("content_preview", "") or ""),
            str(((event.get("extra") or {}).get("raw_operation", "")) or ""),
            str(((event.get("extra") or {}).get("category", "")) or ""),
        ]
    ).lower()


def _find_matching_rule(
    operation_text: str,
    operation_rules: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best_rule: dict[str, Any] | None = None
    best_score = -1
    for rule in operation_rules:
        match_any = [str(item or "").strip() for item in rule.get("match_any", []) or []]
        matched = [item for item in match_any if item and item in operation_text]
        if not matched:
            continue
        score = sum(len(item) for item in matched)
        if score > best_score:
            best_score = score
            best_rule = rule
    return best_rule
