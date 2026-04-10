"""
Module 3 LangGraph node definitions.
"""

import os
import sys
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../2-FileTracker"))

from behavior_analysis_graph import analyze_sensitive_event_behavior
from upload_detection_config import config
from upload_detector_state import UploadDetectorState, UploadEvent
from upload_detector_stats import sync_processed_statistics
from upload_detector_tools import (
    append_operation_record_with_dedup,
    append_upload_event_with_dedup,
    build_sensitive_operation_record,
    extract_hidden_transformed_paths,
    normalize_file_path,
    normalize_timestamp_display,
    read_recording_start_time,
    refresh_upload_statistics,
    resolve_full_path,
    split_output_filenames,
)
from worklist_manager import WorklistManager, load_log_from_json


def initialize_node(state: UploadDetectorState) -> UploadDetectorState:
    print("\n" + "=" * 80)
    print("Initializing RiskHunter")
    print("=" * 80)

    state["current_step"] = "initialize"
    state["messages"].append("Initializing pipeline")

    try:
        log_events = load_log_from_json(state["log_file"])
        manager = WorklistManager(sensitive_files=state["sensitive_files"])
        added_count = manager.scan_and_build_worklist(log_events)

        print(f"Loaded {len(log_events)} log events")
        print(f"Initialized WorklistManager with {len(state['sensitive_files'])} sensitive files")
        print(f"Discovered {added_count} sensitive worklist events")

        state["worklist_size"] = manager.size()
        state["_worklist_manager"] = manager
        state["_log_events"] = log_events
        state["_operation_record_keys"] = set()
        state["_upload_event_index"] = {}
        state["_hidden_transformed_paths"] = []
        state["recording_start_time"] = ""
        state["should_continue"] = not manager.is_empty()

        stats = manager.get_statistics()
        print(f"Worklist size: {stats['worklist_size']}")
        print(f"Event types: {stats['event_types']}")
    except Exception as exc:
        error_msg = f"Initialization failed: {exc}"
        print(error_msg)
        state["errors"].append(error_msg)
        state["should_continue"] = False
        import traceback

        traceback.print_exc()

    return state


def _snapshot_current_event(event: Any, manager: WorklistManager) -> dict:
    refresh_method = getattr(manager, "_refresh_event_original_file", None)
    if callable(refresh_method):
        refresh_method(event)
    return {
        "event_id": event.event_id,
        "file_path": event.current_file,
        "original_file": event.original_file,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
    }


def process_event_node(state: UploadDetectorState) -> UploadDetectorState:
    state["current_step"] = "process_event"

    try:
        manager: WorklistManager = state["_worklist_manager"]
        log_events = state["_log_events"]

        event = manager.get_next_event()
        if not event:
            print("\nWorklist is empty, stopping")
            state["should_continue"] = False
            return state

        state["processed_count"] += 1
        sync_processed_statistics(state)

        print("\n" + "-" * 80)
        print(f"Processing event #{state['processed_count']}")
        print(f"Event ID: {event.event_id}")
        print(f"Current file: {event.current_file}")
        print(f"Original file: {event.original_file}")
        print(f"Event type: {event.event_type}")
        print(f"Timestamp: {event.timestamp}")

        state["current_event"] = _snapshot_current_event(event, manager)

        result = analyze_sensitive_event_behavior(
            event=event,
            index_path=state["index_path"],
            video_path=state["video_path"],
            worklist_manager=manager,
            log_events=log_events,
            search_duration=state["search_duration"],
        )

        state["_hidden_transformed_paths"] = extract_hidden_transformed_paths(result)
        state["module1_result"] = result.get("frame_analysis_result", result)

        module2_recording_time = ""
        if isinstance(state["module1_result"], dict):
            module2_recording_time = normalize_timestamp_display(
                state["module1_result"].get("recording_start_time", "")
            )

        if module2_recording_time:
            state["recording_start_time"] = module2_recording_time
        elif not state.get("recording_start_time"):
            try:
                fallback_time = normalize_timestamp_display(read_recording_start_time(state.get("index_path", "")))
            except Exception:
                fallback_time = ""

            if not fallback_time and log_events:
                fallback_time = normalize_timestamp_display(log_events[0].get("timestamp", ""))

            state["recording_start_time"] = fallback_time

        state["worklist_size"] = manager.size()
        print(f"Analysis complete, current worklist size: {state['worklist_size']}")

        if result.get("has_hidden_behavior") and result.get("new_events"):
            new_events = result.get("new_events", [])
            print(f"Discovered {len(new_events)} derived events, rescanning logs for follow-up work")
            additional_count = manager.scan_and_build_worklist(log_events)
            if additional_count > 0:
                state["worklist_size"] = manager.size()
                print(f"Added {additional_count} more sensitive events")
            else:
                print("No additional sensitive events discovered")

        state["current_event"] = _snapshot_current_event(event, manager)

        state["should_continue"] = not manager.is_empty()
    except Exception as exc:
        error_msg = f"Event processing failed: {exc}"
        print(error_msg)
        state["errors"].append(error_msg)
        import traceback

        traceback.print_exc()

    return state


def _build_alert_reason(app_name: str, app_category: str, should_alert_flag: bool) -> str:
    if should_alert_flag:
        if app_category == "blacklist":
            return f"Detected file exfiltration via blacklist app '{app_name}'"
        return "Detected suspicious file exfiltration behavior"

    if app_category == "whitelist":
        return f"Whitelisted upload through '{app_name}'"
    if app_category == "unknown":
        return f"Recorded upload via non-blacklist app '{app_name}'"
    return ""


def _should_treat_as_upload(behavior_category: str, operation_type: str) -> bool:
    upload_keywords = ["上传", "发送", "分享", "转发", "附件", "粘贴"]
    return "外发" in behavior_category or any(keyword in operation_type for keyword in upload_keywords)


def _build_upload_content_mapping_link(
    manager: WorklistManager | None,
    upload_content: str,
    current_event: dict[str, Any],
    event_data: dict[str, Any],
    log_events: list[dict[str, Any]],
) -> str:
    if not manager or not upload_content:
        return ""

    mapping_links: list[str] = []
    seen = set()
    base_dir = os.path.dirname(current_event.get("file_path", ""))
    time_range = event_data.get("time_range", "")

    for content_item in split_output_filenames(upload_content) or [upload_content]:
        stripped_item = normalize_file_path(str(content_item or "").strip())
        if not stripped_item:
            continue

        candidate_paths = [stripped_item]
        if base_dir and not os.path.isabs(stripped_item):
            candidate_paths.append(normalize_file_path(os.path.join(base_dir, stripped_item)))

        mapping_chain = ""
        for candidate_path in candidate_paths:
            mapping_chain = manager.get_mapping_chain(candidate_path)
            if mapping_chain:
                break

        if not mapping_chain:
            full_path = resolve_full_path(
                filename=stripped_item,
                base_dir=base_dir,
                log_events=log_events,
                time_range=time_range,
                print_prefix="      ",
            )
            if full_path:
                mapping_chain = manager.get_mapping_chain(full_path)

        if not mapping_chain or mapping_chain in seen:
            continue

        seen.add(mapping_chain)
        mapping_links.append(mapping_chain)

    return " | ".join(mapping_links)


def analyze_upload_node(state: UploadDetectorState) -> UploadDetectorState:
    state["current_step"] = "analyze_upload"

    try:
        module1_result = state["module1_result"]
        current_event = state["current_event"]

        if not module1_result or not current_event:
            refresh_upload_statistics(state)
            sync_processed_statistics(state)
            return state

        events = module1_result.get("events", [])
        if not events:
            print("No relevant behaviors detected")
            refresh_upload_statistics(state)
            sync_processed_statistics(state)
            return state

        hidden_transformed_paths = state.get("_hidden_transformed_paths", [])
        hidden_path_cursor = 0

        for event_data in events:
            app_name = event_data.get("app_name", "unknown")
            behavior_category = event_data.get("behavior_category", "")
            operation_type = event_data.get("operation_type", "")

            transformed_file_path = ""
            if behavior_category == "潜在隐藏行为":
                original_filename = event_data.get("original_filename", "")
                modified_filename = event_data.get("modified_filename", "")
                is_hidden_transform = (
                    original_filename
                    and modified_filename
                    and original_filename != modified_filename
                )
                if is_hidden_transform and hidden_path_cursor < len(hidden_transformed_paths):
                    transformed_file_path = hidden_transformed_paths[hidden_path_cursor]
                    hidden_path_cursor += 1

            operation_record = build_sensitive_operation_record(
                recording_start_time=state.get("recording_start_time", ""),
                sensitive_file_path=current_event.get("file_path", ""),
                event_data=event_data,
                fallback_timestamp=current_event.get("timestamp", ""),
                transformed_file_path=transformed_file_path,
            )
            if append_operation_record_with_dedup(state, operation_record):
                print(
                    "Recorded sensitive operation: "
                    f"{operation_record['operation_time']} | "
                    f"{operation_record['sensitive_file_path']} | "
                    f"{operation_record['operation']}"
                )
            else:
                print("Sensitive operation duplicated, skipped")

            print("\nAnalyzing event result")
            print(f"App: {app_name}")
            print(f"Behavior category: {behavior_category}")
            print(f"Operation type: {operation_type}")

            if not _should_treat_as_upload(behavior_category, operation_type):
                print("Non-upload behavior, skipping")
                continue

            app_category = config.get_app_category(app_name)
            should_alert_flag, alert_level = config.should_alert(app_category, behavior_category)
            alert_reason = _build_alert_reason(app_name, app_category, should_alert_flag)

            upload_content = event_data.get("original_filename", "")
            if not upload_content or upload_content == "未知":
                upload_content = current_event["file_path"]

            upload_content_mapping_link = ""
            try:
                manager = state.get("_worklist_manager")
                upload_content_mapping_link = _build_upload_content_mapping_link(
                    manager=manager,
                    upload_content=upload_content,
                    current_event=current_event,
                    event_data=event_data,
                    log_events=state.get("_log_events", []),
                )
            except Exception as exc:
                print(f"Failed to build mapping chain: {exc}")

            upload_event = UploadEvent(
                event_id=current_event["event_id"],
                timestamp=current_event["timestamp"],
                file_path=current_event["file_path"],
                file_name=os.path.basename(current_event["file_path"]),
                original_file=current_event["original_file"],
                upload_content=upload_content,
                upload_content_mapping_link=upload_content_mapping_link,
                app_name=app_name,
                app_category=app_category,
                behavior_category=behavior_category,
                operation_type=operation_type,
                time_range=event_data.get("time_range", ""),
                involved_timestamps=event_data.get("involved_timestamps", []),
                description=event_data.get("description", ""),
                should_alert=should_alert_flag,
                alert_level=alert_level,
                alert_reason=alert_reason,
                extra_info=event_data,
            )

            was_added = append_upload_event_with_dedup(state, upload_event)
            refresh_upload_statistics(state)

            if was_added:
                if should_alert_flag:
                    print("Added alert event")
                else:
                    print("Added informational upload event")
            else:
                print("Duplicate upload event merged into existing result")
    except Exception as exc:
        error_msg = f"Upload analysis failed: {exc}"
        print(error_msg)
        state["errors"].append(error_msg)
        import traceback

        traceback.print_exc()

    refresh_upload_statistics(state)
    sync_processed_statistics(state)
    return state


def finalize_node(state: UploadDetectorState) -> UploadDetectorState:
    state["current_step"] = "finalize"
    refresh_upload_statistics(state)
    sync_processed_statistics(state)

    manager = state.get("_worklist_manager")
    if manager and hasattr(manager, "export_file_mappings"):
        state["file_mappings"] = manager.export_file_mappings()

    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)

    stats = state["statistics"]
    print(f"Processed events: {stats['total_events_processed']}")
    print(f"Upload events detected: {stats['upload_events_detected']}")
    print(f"Sensitive operation records: {len(state['operation_records'])}")
    print(f"Blacklist alerts: {stats['blacklist_alerts']}")
    print(f"Whitelist uploads: {stats['whitelist_uploads']}")
    print(f"Other uploads: {stats['unknown_uploads']}")

    if state["alert_events"]:
        print(f"Alert events: {len(state['alert_events'])}")
        for idx, event in enumerate(state["alert_events"], start=1):
            print(f"[{idx}] {event.alert_level.upper()} {event.file_name} -> {event.app_name}")

    if state["info_events"]:
        print(f"Informational events: {len(state['info_events'])}")
        for idx, event in enumerate(state["info_events"], start=1):
            print(f"[{idx}] {event.file_name} -> {event.app_name}")

    if state["errors"]:
        print(f"Errors: {len(state['errors'])}")
        for error in state["errors"]:
            print(f"- {error}")

    print("=" * 80)
    state["should_continue"] = False
    return state


def should_continue_processing(state: UploadDetectorState) -> str:
    if state["should_continue"] and state["worklist_size"] > 0:
        return "continue"
    return "end"
