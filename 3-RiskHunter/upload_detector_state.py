# upload_detector_state.py
"""
模块3 State定义
定义LangGraph Agent的状态结构
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class UploadEvent:
    """上传事件数据类"""
    
    # 基本信息
    event_id: str
    timestamp: str
    
    # 文件信息
    file_path: str
    file_name: str
    original_file: str  # 原始敏感文件
    
    # 应用信息
    app_name: str
    app_category: str  # blacklist/whitelist/unknown
    
    # 行为分析
    behavior_category: str  # 直接外发/正常操作/潜在隐藏行为
    operation_type: str  # 邮件附件外发/聊天转发/云盘上传等
    
    # 视频分析结果
    time_range: str
    involved_timestamps: List[str]
    description: str
    
    # 报警信息
    should_alert: bool
    alert_level: str  # critical/warning/info
    alert_reason: str = ""
    
    # 额外信息
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "original_file": self.original_file,
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
    """
    LangGraph Agent状态
    
    记录整个分析流程的状态信息
    """
    
    # 输入配置
    record_id: int
    base_path: str
    log_file: str
    video_path: str
    index_path: str
    
    # 配置信息
    sensitive_files: List[str]
    blacklist_apps: List[str]
    whitelist_apps: List[str]
    
    # WorklistManager状态
    worklist_size: int
    processed_count: int
    
    # 当前处理的事件
    current_event: Optional[Dict[str, Any]]
    
    # 模块1的分析结果（从模块2获取）
    module1_result: Optional[Dict[str, Any]]
    
    # 检测到的上传事件列表
    upload_events: List[UploadEvent]
    
    # 报警事件列表（黑名单应用上传）
    alert_events: List[UploadEvent]
    
    # 信息事件列表（非黑名单应用上传）
    info_events: List[UploadEvent]
    
    # 统计信息
    statistics: Dict[str, Any]
    
    # 错误信息
    errors: List[str]
    
    # 流程控制
    should_continue: bool
    current_step: str
    
    # 日志消息
    messages: List[str]
    
    # 内部状态（不序列化到JSON）
    _worklist_manager: Any  # WorklistManager实例
    _log_events: List[Dict[str, Any]]  # 日志事件列表


def create_initial_state(
    record_id: int,
    base_path: str,
    log_file: str,
    video_path: str,
    index_path: str,
    sensitive_files: List[str],
    blacklist_apps: List[str],
    whitelist_apps: List[str]
) -> UploadDetectorState:
    """
    创建初始状态
    
    Args:
        record_id: 记录ID
        base_path: 基础路径
        log_file: 日志文件路径
        video_path: 视频文件路径
        index_path: INDEX.md路径
        sensitive_files: 敏感文件列表
        blacklist_apps: 黑名单应用列表
        whitelist_apps: 白名单应用列表
        
    Returns:
        初始化的状态对象
    """
    return UploadDetectorState(
        # 输入配置
        record_id=record_id,
        base_path=base_path,
        log_file=log_file,
        video_path=video_path,
        index_path=index_path,
        
        # 配置信息
        sensitive_files=sensitive_files,
        blacklist_apps=blacklist_apps,
        whitelist_apps=whitelist_apps,
        
        # WorklistManager状态
        worklist_size=0,
        processed_count=0,
        
        # 当前处理的事件
        current_event=None,
        
        # 模块1的分析结果
        module1_result=None,
        
        # 检测到的上传事件列表
        upload_events=[],
        
        # 报警事件列表
        alert_events=[],
        
        # 信息事件列表
        info_events=[],
        
        # 统计信息
        statistics={
            "total_events_processed": 0,
            "upload_events_detected": 0,
            "blacklist_alerts": 0,
            "whitelist_uploads": 0,
            "unknown_uploads": 0,
        },
        
        # 错误信息
        errors=[],
        
        # 流程控制
        should_continue=True,
        current_step="initialize",
        
        # 日志消息
        messages=[],
        
        # 内部状态（稍后在initialize_node中设置）
        _worklist_manager=None,
        _log_events=[],
    )


def save_state_to_json(state: UploadDetectorState, output_path: str):
    """
    保存状态到JSON文件
    
    Args:
        state: 状态对象
        output_path: 输出文件路径
    """
    # 转换UploadEvent对象为字典
    state_dict = dict(state)
    state_dict["upload_events"] = [event.to_dict() for event in state["upload_events"]]
    state_dict["alert_events"] = [event.to_dict() for event in state["alert_events"]]
    state_dict["info_events"] = [event.to_dict() for event in state["info_events"]]
    
    # 排除不可序列化的内部状态字段
    state_dict.pop("_worklist_manager", None)
    state_dict.pop("_log_events", None)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, ensure_ascii=False, indent=2)
