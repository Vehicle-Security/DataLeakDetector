"""
模块3状态定义。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
import json


@dataclass
class UploadEvent:
    """上传事件数据结构。"""

    event_id: str
    timestamp: str

    file_path: str
    file_name: str
    original_file: str

    app_name: str
    app_category: str

    behavior_category: str
    operation_type: str

    time_range: str
    involved_timestamps: List[str]
    description: str

    should_alert: bool
    alert_level: str

    upload_content: str = ""
    upload_content_mapping_link: str = ""
    alert_reason: str = ""
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "original_file": self.original_file,
            "upload_content": self.upload_content,
            "upload_content_mapping_link": self.upload_content_mapping_link,
            "app_name": self.app_name,
            "app_category": self.app_category,
            "behavior_category": self.behavior_category,
            "operation_type": self.operation_type,
            "time_range": self.time_range,
            "involved_timestamps": self.involved_timestamps,
            "description": self.description,
            "should_alert": self.should_alert,
            "alert_level": self.alert_level,
            "alert_reason": self.alert_reason,
            "extra_info": self.extra_info,
        }


class UploadDetectorState(TypedDict):
    record_id: int
    base_path: str
    log_file: str
    video_path: str
    index_path: str

    sensitive_files: List[str]
    blacklist_apps: List[str]
    whitelist_apps: List[str]
    search_duration: int

    worklist_size: int
    processed_count: int
    current_event: Optional[Dict[str, Any]]
    module1_result: Optional[Dict[str, Any]]

    upload_events: List[UploadEvent]
    recording_start_time: str
    operation_records: List[Dict[str, Any]]
    alert_events: List[UploadEvent]
    info_events: List[UploadEvent]
    file_mappings: Dict[str, Any]
    statistics: Dict[str, Any]

    errors: List[str]
    should_continue: bool
    current_step: str
    messages: List[str]

    _worklist_manager: Any
    _log_events: List[Dict[str, Any]]
    _operation_record_keys: Any
    _upload_event_index: Any
    _hidden_transformed_paths: List[str]


def create_initial_state(
    record_id: int,
    base_path: str,
    log_file: str,
    video_path: str,
    index_path: str,
    sensitive_files: List[str],
    blacklist_apps: List[str],
    whitelist_apps: List[str],
    search_duration: int = 30,
) -> UploadDetectorState:
    return UploadDetectorState(
        record_id=record_id,
        base_path=base_path,
        log_file=log_file,
        video_path=video_path,
        index_path=index_path,
        sensitive_files=sensitive_files,
        blacklist_apps=blacklist_apps,
        whitelist_apps=whitelist_apps,
        search_duration=search_duration,
        worklist_size=0,
        processed_count=0,
        current_event=None,
        module1_result=None,
        upload_events=[],
        recording_start_time="",
        operation_records=[],
        alert_events=[],
        info_events=[],
        file_mappings={},
        statistics={
            "total_events_processed": 0,
            "upload_events_detected": 0,
            "blacklist_alerts": 0,
            "whitelist_uploads": 0,
            "unknown_uploads": 0,
        },
        errors=[],
        should_continue=True,
        current_step="initialize",
        messages=[],
        _worklist_manager=None,
        _log_events=[],
        _operation_record_keys=set(),
        _upload_event_index={},
        _hidden_transformed_paths=[],
    )


def save_state_to_json(state: UploadDetectorState, output_path: str) -> None:
    state_dict = dict(state)
    state_dict["upload_events"] = [event.to_dict() for event in state["upload_events"]]
    state_dict["alert_events"] = [event.to_dict() for event in state["alert_events"]]
    state_dict["info_events"] = [event.to_dict() for event in state["info_events"]]

    state_dict.pop("_worklist_manager", None)
    state_dict.pop("_log_events", None)
    state_dict.pop("_operation_record_keys", None)
    state_dict.pop("_upload_event_index", None)
    state_dict.pop("_hidden_transformed_paths", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, ensure_ascii=False, indent=2)
