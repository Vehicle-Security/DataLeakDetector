"""
Rule-driven ThreatDetector entrypoint.

The detector builds Datalog facts from module3 results and a reduced log
evidence bundle first. LLM fact generation is optional and only supplements the
deterministic facts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from datalog.datalog_engine import DatalogEngine


@dataclass
class DatalogFact:
    relation: str
    operation_id: str
    process: str
    file: str
    dst_file: Optional[str] = None
    timestamp: Optional[str] = None
    description: Optional[str] = None
    from_process: Optional[str] = None
    to_process: Optional[str] = None
    shared_data: Optional[str] = None
    leak_channel: str = "network"
    source: str = "rule"

    def to_souffle_args(self):
        ts = parse_timestamp_ms(self.timestamp)
        if self.relation == "OpenFile":
            return (self.operation_id, self.process, self.file, ts)
        if self.relation == "TransferFile":
            return (self.operation_id, self.process, self.file, self.dst_file or "", ts)
        if self.relation == "CrossProcessTransfer":
            return (
                self.operation_id,
                self.from_process or self.process,
                self.to_process or "unknown",
                self.shared_data or self.file,
                ts,
            )
        if self.relation == "LeakFile":
            return (self.operation_id, self.process, self.file, self.leak_channel or "network", ts)
        return None

    def identity(self) -> tuple:
        args = self.to_souffle_args()
        return (self.relation, args)


def event_attr(event: Any, name: str, default: Any = "") -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def parse_timestamp_ms(timestamp: Any) -> int:
    if not timestamp:
        return 0
    text = str(timestamp).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def normalize_path(path: Any) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def path_key(path: Any) -> str:
    return normalize_path(path).casefold()


def basename(path: Any) -> str:
    normalized = normalize_path(path)
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def basename_key(path: Any) -> str:
    return basename(path).casefold()


def normalize_process(process: Any) -> str:
    text = normalize_path(process)
    return (text.rsplit("/", 1)[-1] if text else "unknown").casefold()


def is_unknown_file(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.casefold() in {"\u672a\u77e5", "unknown", "none", "null"}


def paths_match(left: Any, right: Any) -> bool:
    left_norm = normalize_path(left)
    right_norm = normalize_path(right)
    if not left_norm or not right_norm:
        return False
    if path_key(left_norm) == path_key(right_norm):
        return True
    return basename_key(left_norm) == basename_key(right_norm)


def infer_channel(app_name: str, window_title: str = "") -> str:
    text = f"{app_name} {window_title}".casefold()
    if any(token in text for token in ["mail", "\u90ae\u7bb1", "gmail", "outlook", "163", "qq\u90ae\u7bb1"]):
        return "email"
    if any(token in text for token in ["drive", "\u7f51\u76d8", "onedrive", "baidu", "google drive"]):
        return "cloud"
    if any(token in text for token in ["wechat", "\u5fae\u4fe1", "qq", "tim", "ding", "\u9489\u9489", "lark", "\u98de\u4e66"]):
        return "chat"
    return "network"


def build_video_events(module3_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen = set()
    for key in ["upload_events", "alert_events", "info_events"]:
        for upload_event in module3_result.get(key, []):
            extra = event_attr(upload_event, "extra_info", {}) or {}
            if not isinstance(extra, dict):
                extra = {}
            item = dict(extra)
            item.update(
                {
                    "timestamp": event_attr(upload_event, "timestamp", item.get("timestamp", "")),
                    "app_name": event_attr(upload_event, "app_name", item.get("app_name", "")),
                    "operation_type": event_attr(upload_event, "operation_type", item.get("operation_type", "")),
                    "behavior_category": event_attr(upload_event, "behavior_category", item.get("behavior_category", "")),
                    "description": event_attr(upload_event, "description", item.get("description", "")),
                    "file_path": event_attr(upload_event, "file_path", ""),
                    "upload_content": event_attr(upload_event, "upload_content", ""),
                }
            )
            identity = (
                item.get("time_range", ""),
                item.get("timestamp", ""),
                item.get("app_name", ""),
                item.get("operation_type", ""),
                item.get("upload_content", ""),
            )
            if identity not in seen:
                seen.add(identity)
                events.append(item)
    return events


class EvidenceReducer:
    def __init__(self, logs: List[Dict[str, Any]], module3_result: Dict[str, Any], max_logs: int, window_seconds: int):
        self.logs = logs
        self.module3_result = module3_result
        self.max_logs = max_logs
        self.window_ms = max(1, window_seconds) * 1000

    def reduce(self) -> Dict[str, Any]:
        video_events = build_video_events(self.module3_result)
        target_times = self._target_times(video_events)
        target_paths = self._target_paths(video_events)
        target_names = {basename_key(path) for path in target_paths if path}

        selected: List[Dict[str, Any]] = []
        seen_ids = set()

        def add(log: Dict[str, Any]) -> None:
            ident = id(log)
            if ident not in seen_ids:
                seen_ids.add(ident)
                selected.append(log)

        for log in self.logs:
            log_path = normalize_path(log.get("file_path", ""))
            log_name = basename_key(log_path or log.get("file_name", ""))
            log_ts = parse_timestamp_ms(log.get("timestamp", ""))
            if log_path and (path_key(log_path) in target_paths or log_name in target_names):
                add(log)
                continue
            if log_ts and any(abs(log_ts - target_ts) <= self.window_ms for target_ts in target_times):
                add(log)

        selected = sorted(selected, key=lambda item: parse_timestamp_ms(item.get("timestamp", "")))
        if self.max_logs > 0 and len(selected) > self.max_logs:
            selected = selected[: self.max_logs]

        return {
            "logs": selected,
            "video_events": video_events,
            "stats": {
                "total_logs": len(self.logs),
                "evidence_logs": len(selected),
                "video_events": len(video_events),
                "max_logs": self.max_logs,
                "window_seconds": self.window_ms // 1000,
            },
        }

    def _target_times(self, video_events: List[Dict[str, Any]]) -> set[int]:
        times = set()
        for event in video_events:
            for value in event.get("involved_timestamps", []) or []:
                ts = parse_timestamp_ms(value)
                if ts:
                    times.add(ts)
            for key in ["timestamp", "operation_time"]:
                ts = parse_timestamp_ms(event.get(key, ""))
                if ts:
                    times.add(ts)

        for key in ["alert_events", "info_events", "upload_events"]:
            for event in self.module3_result.get(key, []):
                ts = parse_timestamp_ms(event_attr(event, "timestamp", ""))
                if ts:
                    times.add(ts)
        return times

    def _target_paths(self, video_events: List[Dict[str, Any]]) -> set[str]:
        paths = set()
        for event in video_events:
            for key in ["file_path", "original_filename", "modified_filename", "upload_content"]:
                value = event.get(key, "")
                if not is_unknown_file(value):
                    paths.add(path_key(value))
        for key in ["alert_events", "info_events", "upload_events"]:
            for event in self.module3_result.get(key, []):
                for attr in ["file_path", "original_file", "upload_content"]:
                    value = event_attr(event, attr, "")
                    if not is_unknown_file(value):
                        paths.add(path_key(value))
        return paths


class FactBuilder:
    def __init__(self, evidence_bundle: Dict[str, Any], module3_result: Dict[str, Any]):
        self.logs = evidence_bundle.get("logs", [])
        self.video_events = evidence_bundle.get("video_events", [])
        self.module3_result = module3_result
        self.facts: List[DatalogFact] = []
        self.seen_fact_keys = set()
        self.op_counter = 1000

    def build(self) -> List[DatalogFact]:
        self._add_file_mapping_facts()
        self._add_module3_event_facts("alert_events")
        self._add_module3_event_facts("info_events")
        self._add_log_upload_facts()
        return self.facts

    def _next_id(self, prefix: str) -> str:
        self.op_counter += 1
        return f"{prefix}_{self.op_counter}"

    def _add_fact(self, fact: DatalogFact) -> bool:
        if not fact.file:
            return False
        key = fact.identity()
        if key in self.seen_fact_keys:
            return False
        self.seen_fact_keys.add(key)
        self.facts.append(fact)
        return True

    def _add_open(self, process: str, file_path: str, timestamp: str, description: str) -> None:
        self._add_fact(
            DatalogFact(
                relation="OpenFile",
                operation_id=self._next_id("open"),
                process=normalize_process(process),
                file=normalize_path(file_path),
                timestamp=timestamp,
                description=description,
            )
        )

    def _add_transfer(self, process: str, src_path: str, dst_path: str, timestamp: str, description: str) -> None:
        self._add_fact(
            DatalogFact(
                relation="TransferFile",
                operation_id=self._next_id("transfer"),
                process=normalize_process(process),
                file=normalize_path(src_path),
                dst_file=normalize_path(dst_path),
                timestamp=timestamp,
                description=description,
            )
        )

    def _add_cross(self, from_process: str, to_process: str, data_path: str, timestamp: str, description: str) -> None:
        self._add_fact(
            DatalogFact(
                relation="CrossProcessTransfer",
                operation_id=self._next_id("cross"),
                process=normalize_process(from_process),
                from_process=normalize_process(from_process),
                to_process=normalize_process(to_process),
                shared_data=normalize_path(data_path),
                file=normalize_path(data_path),
                timestamp=timestamp,
                description=description,
            )
        )

    def _add_leak(self, process: str, file_path: str, channel: str, timestamp: str, description: str) -> None:
        self._add_fact(
            DatalogFact(
                relation="LeakFile",
                operation_id=self._next_id("leak"),
                process=normalize_process(process),
                file=normalize_path(file_path),
                timestamp=timestamp,
                description=description,
                leak_channel=channel,
            )
        )

    def _add_file_mapping_facts(self) -> None:
        mappings = self.module3_result.get("file_mappings", {}).get("direct_file_mappings", {})
        for derived_file, original_file in mappings.items():
            timestamp = self._nearest_timestamp(derived_file) or self._nearest_timestamp(original_file)
            process = self._owner_process(original_file, timestamp) or self._owner_process(derived_file, timestamp)
            self._add_open(process, original_file, timestamp, f"Opened original file for mapped artifact {basename(original_file)}")
            self._add_transfer(process, original_file, derived_file, timestamp, f"Mapped {basename(original_file)} to {basename(derived_file)}")

    def _add_module3_event_facts(self, events_key: str) -> None:
        for event in self.module3_result.get(events_key, []):
            upload_content = event_attr(event, "upload_content", "")
            file_path = event_attr(event, "file_path", "")
            original_file = event_attr(event, "original_file", file_path)
            event_app = event_attr(event, "app_name", "unknown")
            timestamp = event_attr(event, "timestamp", "")
            leak_file = self._resolve_uploaded_file(upload_content, file_path)
            if not leak_file:
                continue

            owner_process = self._owner_process(original_file or file_path, timestamp)
            uploader_process = normalize_process(event_app)
            matched_log = self._nearest_log(leak_file, timestamp=timestamp)

            self._add_open(owner_process, original_file or leak_file, timestamp, f"Opened sensitive file {basename(original_file or leak_file)}")
            if not paths_match(original_file, leak_file):
                self._add_transfer(owner_process, original_file or file_path, leak_file, timestamp, f"Derived upload content {basename(leak_file)}")
            if owner_process and normalize_process(owner_process) != normalize_process(uploader_process):
                self._add_cross(owner_process, uploader_process, leak_file, timestamp, f"{owner_process} transferred {basename(leak_file)} to {uploader_process}")

            channel = infer_channel(event_app, (matched_log or {}).get("window_info", {}).get("window_title", ""))
            self._add_leak(uploader_process, leak_file, channel, timestamp, f"{uploader_process} leaked {basename(leak_file)}")

    def _add_log_upload_facts(self) -> None:
        for log in self.logs:
            if log.get("event_type") not in {"file_upload", "upload_detected"}:
                continue
            file_path = normalize_path(log.get("file_path", ""))
            if not file_path:
                continue
            timestamp = log.get("timestamp", "")
            process = log.get("process_info", {}).get("process_name", "unknown")
            channel = infer_channel(process, log.get("window_info", {}).get("window_title", ""))
            self._add_leak(process, file_path, channel, timestamp, f"Log upload event for {basename(file_path)}")

    def _resolve_uploaded_file(self, upload_content: str, file_path: str) -> str:
        if is_unknown_file(upload_content):
            return normalize_path(file_path)
        if paths_match(upload_content, file_path):
            return normalize_path(file_path)
        for log in self.logs:
            if paths_match(log.get("file_path", ""), upload_content) or paths_match(log.get("file_name", ""), upload_content):
                return normalize_path(log.get("file_path", ""))
        return normalize_path(upload_content)

    def _owner_process(self, file_path: str, timestamp: str = "") -> str:
        matched = self._nearest_log(file_path, timestamp=timestamp, event_types={"opened", "file_open", "browser_file_access"})
        if matched:
            return matched.get("process_info", {}).get("process_name", "unknown")
        return "unknown"

    def _nearest_timestamp(self, file_path: str) -> str:
        matched = self._nearest_log(file_path)
        return matched.get("timestamp", "") if matched else ""

    def _nearest_log(self, file_path: str, timestamp: str = "", event_types: Optional[set[str]] = None) -> Optional[Dict[str, Any]]:
        target_ts = parse_timestamp_ms(timestamp)
        best_log = None
        best_distance = None
        for log in self.logs:
            if event_types and log.get("event_type", "") not in event_types:
                continue
            if not (paths_match(log.get("file_path", ""), file_path) or paths_match(log.get("file_name", ""), file_path)):
                continue
            log_ts = parse_timestamp_ms(log.get("timestamp", ""))
            distance = abs(log_ts - target_ts) if target_ts and log_ts else 0
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_log = log
        return best_log


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def generate_llm_facts(evidence_bundle: Dict[str, Any]) -> tuple[List[DatalogFact], str, str]:
    try:
        from openai import OpenAI
        from threat_prompts import PromptTemplates

        api_key = os.getenv("DLD_THREAT_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return [], "skipped", "missing_api_key"
        model_name = os.getenv("DLD_THREAT_MODEL_NAME") or os.getenv("VL_MODEL_NAME") or os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL_NAME")
        if not model_name:
            return [], "skipped", "missing_model"
        base_url = os.getenv("DLD_THREAT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        messages = PromptTemplates.get_messages(evidence_bundle.get("logs", []), evidence_bundle.get("video_events", []))
        response = client.chat.completions.create(model=model_name, messages=messages, temperature=0.1)
        content = response.choices[0].message.content or ""
        match = re.search(r"\[.*\]", content, re.S)
        if not match:
            return [], "failed", "no_json_array"
        raw_facts = json.loads(match.group(0))
        facts = []
        for item in raw_facts:
            relation = item.get("relation", "")
            if relation not in {"OpenFile", "TransferFile", "CrossProcessTransfer", "LeakFile"}:
                continue
            facts.append(
                DatalogFact(
                    relation=relation,
                    operation_id=item.get("operation_id", f"llm_{len(facts) + 1}"),
                    process=normalize_process(item.get("process", item.get("from_process", "unknown"))),
                    file=normalize_path(item.get("file", item.get("shared_data", ""))),
                    dst_file=normalize_path(item.get("dst_file", "")) or None,
                    timestamp=item.get("timestamp", ""),
                    description=item.get("description", ""),
                    from_process=normalize_process(item.get("from_process", "")) if item.get("from_process") else None,
                    to_process=normalize_process(item.get("to_process", "")) if item.get("to_process") else None,
                    shared_data=normalize_path(item.get("shared_data", "")) or None,
                    leak_channel=item.get("leak_channel", "network"),
                    source="llm",
                )
            )
        return facts, "success", ""
    except Exception as exc:
        return [], "failed", str(exc)


def run_threat_detection(
    logs: List[Dict[str, Any]],
    module3_result: Dict[str, Any],
    *,
    use_llm: bool = False,
    max_logs: int = 80,
    window_seconds: int = 90,
) -> Dict[str, Any]:
    use_llm = env_flag("DLD_THREAT_USE_LLM", use_llm)
    max_logs = int(os.getenv("DLD_THREAT_MAX_LOGS", str(max_logs)))
    window_seconds = int(os.getenv("DLD_THREAT_LOG_WINDOW_SECONDS", str(window_seconds)))

    reducer = EvidenceReducer(logs, module3_result, max_logs=max_logs, window_seconds=window_seconds)
    evidence_bundle = reducer.reduce()
    rule_facts = FactBuilder(evidence_bundle, module3_result).build()

    llm_facts: List[DatalogFact] = []
    llm_status = "disabled"
    llm_error = ""
    if use_llm:
        llm_facts, llm_status, llm_error = generate_llm_facts(evidence_bundle)

    facts: List[DatalogFact] = []
    seen = set()
    for fact in rule_facts + llm_facts:
        key = fact.identity()
        if key not in seen:
            seen.add(key)
            facts.append(fact)

    engine = DatalogEngine()
    for fact in facts:
        args = fact.to_souffle_args()
        if args:
            engine.add_fact(fact.relation, *args)
    leak_paths = engine.query_leak()
    engine.cleanup()

    return {
        "leak_paths": leak_paths,
        "datalog_facts": facts,
        "evidence_bundle": evidence_bundle,
        "stats": {
            **evidence_bundle.get("stats", {}),
            "rule_facts": len(rule_facts),
            "llm_facts": len(llm_facts),
            "facts": len(facts),
            "llm_enabled": use_llm,
            "llm_status": llm_status,
            "llm_error": llm_error,
        },
    }
