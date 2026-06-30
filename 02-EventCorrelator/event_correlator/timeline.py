from __future__ import annotations

from typing import Any

from .schema import (
    CorrelatorContext,
    EventCorrelatorInput,
    NormalizedFrameSegment,
    NormalizedLogEvent,
)
from .frontend import classify_frontend_app
from .utils import (
    choose_non_empty,
    get_path_basename,
    normalize_app_name,
    normalize_file_path,
    normalize_timestamp_text,
    parse_time_range,
    parse_timestamp,
)


class TimelineNormalizer:
    def normalize(self, payload: EventCorrelatorInput, correlation_config: dict[str, Any]) -> CorrelatorContext:
        session_id = str(payload.get("session_id", "") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")

        log_events = payload.get("log_events")
        frame_segments = payload.get("frame_segments")
        sensitive_files = payload.get("sensitive_files")

        if log_events is None:
            raise ValueError("log_events is required")
        if frame_segments is None:
            raise ValueError("frame_segments is required")
        if sensitive_files is None:
            raise ValueError("sensitive_files is required")

        errors: list[dict[str, Any]] = []
        normalized_logs: list[NormalizedLogEvent] = []
        normalized_segments: list[NormalizedFrameSegment] = []

        for index, raw_event in enumerate(log_events):
            timestamp_text = normalize_timestamp_text(str(raw_event.get("timestamp", "") or ""))
            timestamp = parse_timestamp(str(raw_event.get("timestamp", "") or ""))
            if timestamp is None:
                errors.append(
                    {
                        "stage": "timeline_normalizer",
                        "kind": "invalid_log_timestamp",
                        "index": index,
                        "value": raw_event.get("timestamp", ""),
                    }
                )
                continue

            file_path = normalize_file_path(str(raw_event.get("file_path", "") or ""))
            process_info = raw_event.get("process_info", {}) or {}
            window_info = raw_event.get("window_info", {}) or {}
            frontend_app = classify_frontend_app(raw_event)
            display_app = str(frontend_app.get("display_name", "") or "")

            normalized_logs.append(
                NormalizedLogEvent(
                    event_id=str(raw_event.get("event_id", f"log_{index}")),
                    timestamp=timestamp,
                    timestamp_text=timestamp_text,
                    event_type=str(raw_event.get("event_type", "") or "unknown"),
                    file_path=file_path,
                    file_name=get_path_basename(file_path),
                    process_name=normalize_app_name(str(process_info.get("process_name", "") or "")),
                    app_name=normalize_app_name(
                        choose_non_empty(
                            raw_event.get("app_name", ""),
                            display_app if frontend_app.get("is_external") else "",
                            process_info.get("process_name", ""),
                            window_info.get("window_title", ""),
                        )
                    ),
                    frontend_app=frontend_app,
                    raw=raw_event,
                )
            )

        for index, raw_segment in enumerate(frame_segments):
            time_range = str(raw_segment.get("time_range", "") or "").strip()
            start_time, end_time = parse_time_range(time_range)

            normalized_segments.append(
                NormalizedFrameSegment(
                    segment_id=str(raw_segment.get("segment_id", f"segment_{index}")),
                    start_time=start_time,
                    end_time=end_time,
                    time_range=time_range,
                    app_name=normalize_app_name(str(raw_segment.get("app_name", "") or "")),
                    operation_type=str(raw_segment.get("operation_type", "") or "").strip(),
                    primary_resource=str(raw_segment.get("primary_resource", "") or "").strip(),
                    related_resources=[
                        str(item or "").strip()
                        for item in raw_segment.get("related_resources", []) or []
                        if str(item or "").strip()
                    ],
                    action_description=str(raw_segment.get("action_description", "") or "").strip(),
                    visible_evidence=[
                        str(item or "").strip()
                        for item in raw_segment.get("visible_evidence", []) or []
                        if str(item or "").strip()
                    ],
                    supporting_timestamps=[
                        normalize_timestamp_text(str(item or ""))
                        for item in raw_segment.get("supporting_timestamps", []) or []
                        if str(item or "").strip()
                    ],
                    confidence=float(raw_segment.get("confidence", 0.0) or 0.0),
                    raw=dict(raw_segment),
                )
            )

        normalized_logs.sort(key=lambda item: item.timestamp)
        normalized_segments.sort(
            key=lambda item: (
                item.start_time is None,
                item.start_time or item.end_time,
                item.segment_id,
            )
        )

        return CorrelatorContext(
            session_id=session_id,
            record_id=str(payload.get("record_id", "") or ""),
            sensitive_files=[
                normalize_file_path(str(path or ""))
                for path in sensitive_files
                if str(path or "").strip()
            ],
            recording_start_time=normalize_timestamp_text(
                str(payload.get("recording_start_time", "") or "")
            ),
            session_metadata=dict(payload.get("session_metadata", {}) or {}),
            correlation_config=dict(correlation_config or {}),
            normalized_logs=normalized_logs,
            normalized_segments=normalized_segments,
            errors=errors,
        )
