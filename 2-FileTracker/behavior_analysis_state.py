# behavior_analysis_state.py
"""
隐藏行为分析的状态定义
"""

from typing import TypedDict, List, Dict, Any, Optional
from worklist_manager import SensitiveFileEvent


class BehaviorAnalysisState(TypedDict):
    """
    隐藏行为分析的状态
    """
    # 输入
    current_event: SensitiveFileEvent  # 当前待分析的敏感事件
    index_path: str  # INDEX.md 文件路径
    video_path: str  # 视频文件路径
    log_events: List[Dict[str, Any]]  # 日志事件列表（用于查找跨目录文件）
    
    # 模块1分析结果
    frame_analysis_result: Optional[Dict[str, Any]]  # 帧分析结果
    
    # 隐藏操作提取结果
    hidden_operations: List[Dict[str, Any]]  # 识别出的隐藏操作列表
    file_mappings: List[Dict[str, str]]  # 文件映射关系列表
    
    # 新创建的事件
    new_events: List[SensitiveFileEvent]  # 新发现的敏感事件列表
    
    # 状态标记
    has_hidden_behavior: bool  # 是否存在隐藏行为
    analysis_complete: bool  # 分析是否完成
    error_message: Optional[str]  # 错误信息（如果有）
    
    # LLM 消息历史
    messages: List[Any]  # LangChain 消息列表
