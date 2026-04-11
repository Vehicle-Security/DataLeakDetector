# behavior_analysis_nodes.py
"""
模块2 LangGraph节点定义
定义隐藏行为分析流程中的各个节点
"""

import os
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import SystemMessage

from behavior_analysis_state import BehaviorAnalysisState
from behavior_analysis_tools import (
    analyze_frame_behavior,
    extract_hidden_operations,
    normalize_file_path,
    resolve_full_path
)
from behavior_analysis_prompts import BEHAVIOR_ANALYSIS_SYSTEM_PROMPT
from worklist_manager import SensitiveFileEvent


def initialize_node(state: BehaviorAnalysisState, worklist_manager) -> Dict[str, Any]:
    """
    初始化节点
    """
    print("\n🚀 [Initialize] 初始化隐藏行为分析...")
    
    event = state["current_event"]
    print(f"   - 事件ID: {event.event_id}")
    print(f"   - 文件: {event.current_file}")
    print(f"   - 时间: {event.timestamp}")
    print(f"   - 事件类型: {event.event_type}")
    
    messages = [
        SystemMessage(content=BEHAVIOR_ANALYSIS_SYSTEM_PROMPT)
    ]
    
    return {
        "messages": messages
    }


def analyze_frames_node(state: BehaviorAnalysisState, worklist_manager) -> Dict[str, Any]:
    """
    分析视频帧节点
    """
    print("\n🎬 [AnalyzeFrames] 调用模块1分析视频帧...")
    
    event = state["current_event"]
    
    try:
        result = analyze_frame_behavior.invoke({
            "event_timestamp": event.timestamp,
            "current_file": event.current_file,
            "index_path": state["index_path"],
            "video_path": state["video_path"],
            "search_duration": state["search_duration"]
        })
        
        print(f"   ✅ 帧分析完成")
        
        return {
            "frame_analysis_result": result
        }
        
    except Exception as e:
        error_msg = f"帧分析失败: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {
            "frame_analysis_result": None,
            "error_message": error_msg,
            "analysis_complete": True
        }


def extract_operations_node(state: BehaviorAnalysisState, worklist_manager) -> Dict[str, Any]:
    """
    提取隐藏操作节点
    """
    print("\n🔎 [ExtractOperations] 提取隐藏操作...")
    
    frame_result = state.get("frame_analysis_result")
    
    if not frame_result or frame_result.get("total_events", 0) == 0:
        print("   ⚠️ 没有发现相关事件")
        return {
            "has_hidden_behavior": False,
            "hidden_operations": [],
            "file_mappings": []
        }
    
    try:
        extraction_result = extract_hidden_operations.invoke({
            "frame_analysis_result": frame_result
        })
        
        has_hidden = extraction_result.get("has_hidden_behavior", False)
        operations = extraction_result.get("hidden_operations", [])
        mappings = extraction_result.get("file_mappings", [])
        
        print(f"   ✅ 提取完成: 发现 {len(operations)} 个隐藏操作")
        
        return {
            "has_hidden_behavior": has_hidden,
            "hidden_operations": operations,
            "file_mappings": mappings
        }
        
    except Exception as e:
        error_msg = f"提取隐藏操作失败: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {
            "has_hidden_behavior": False,
            "hidden_operations": [],
            "file_mappings": [],
            "error_message": error_msg
        }


def create_new_events_node(state: BehaviorAnalysisState, worklist_manager) -> Dict[str, Any]:
    """
    创建新的敏感事件节点
    """
    print("\n📝 [CreateNewEvents] 创建新的敏感事件...")
    
    operations = state.get("hidden_operations", [])
    current_event = state["current_event"]
    log_events = state.get("log_events", [])
    new_events = []
    
    for op in operations:
        # 从模块1返回的是文件名，需要推断完整路径
        new_filename = op["new_file"]
        operation_type = op["operation_type"]
        time_range = op.get("time_range", "")
        
        current_dir = os.path.dirname(current_event.current_file)
        
        # 使用统一的路径解析函数
        new_file_path = resolve_full_path(
            filename=new_filename,
            base_dir=current_dir,
            log_events=log_events,
            time_range=time_range,
            print_prefix="   "
        )
        
        # 使用隐藏行为的实际发生时间（从involved_timestamps取中间时间戳作为事件发生时间）
        involved_timestamps = op.get("involved_timestamps", [])
        if involved_timestamps:
            mid_index = (len(involved_timestamps)-1) // 2
            timestamp_str = involved_timestamps[mid_index]
            # 转换为ISO格式（如果需要）
            try:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except:
                timestamp = timestamp_str if 'T' in timestamp_str else timestamp_str.replace(' ', 'T') + '.000000'
        else:
            timestamp = current_event.timestamp
        
        event_id = f"derived_{operation_type}_{hash(new_file_path)}_{timestamp}"
        
        new_event = SensitiveFileEvent(
            event_id=event_id,
            original_file=current_event.original_file,  # 追溯到原始文件
            current_file=new_file_path,  # 使用完整路径
            event_type=f"derived_from_{operation_type}",
            process_info={"app_name": op["app_name"]},
            timestamp=timestamp,
            is_hidden=True,
            raw_event={
                "operation_type": operation_type,
                "description": op["description"],
                "time_range": op["time_range"]
            }
        )
        
        new_events.append(new_event)
        print(f"   ✅ 创建事件: {new_file_path} (来自 {operation_type})")
    
    return {
        "new_events": new_events
    }


def update_worklist_node(state: BehaviorAnalysisState, worklist_manager) -> Dict[str, Any]:
    """
    更新 worklist 节点
    """
    print("\n📋 [UpdateWorklist] 更新 worklist...")
    
    new_events = state.get("new_events", [])
    
    new_sensitive_files = []
    for event in new_events:
        normalized_current = normalize_file_path(event.current_file)
        normalized_original = normalize_file_path(event.original_file)
        existing_original = normalize_file_path(worklist_manager.get_original_file(event.current_file) or "")
        is_known_sensitive = normalized_current in worklist_manager.sensitive_files
        mapping_exists = existing_original == normalized_original and normalized_current != ""

        if event.event_type.startswith("derived_from_") and is_known_sensitive:
            print(f"   ♻️ 已知敏感文件，跳过重复入队: {event.current_file}")
        else:
            worklist_manager.add_event(event)
            print(f"   ✅ 添加到 worklist: {event.current_file}")
        
        if not is_known_sensitive:
            worklist_manager.add_sensitive_file(event.current_file)
            new_sensitive_files.append(event.current_file)
            print(f"   ✅ 添加到敏感文件列表: {event.current_file}")
        
        # 获取relationship信息
        relationship = event.raw_event.get("operation_type", "未知关系") if event.raw_event else "未知关系"

        if normalized_current and normalized_current != normalized_original and not mapping_exists:
            worklist_manager.update_file_mapping(
                original_file=event.original_file,
                new_file=event.current_file
            )
            print(f"   ✅ 添加映射 ({relationship}): {event.original_file} -> {event.current_file}")
        elif mapping_exists:
            print(f"   ♻️ 映射已存在，跳过: {event.original_file} -> {event.current_file}")

    worklist_manager.refresh_pending_event_origins()
    
    # 如果有新的敏感文件，重新扫描日志
    if new_sensitive_files:
        print(f"\n   🔄 重新扫描日志以查找新敏感文件的相关操作...")
        # TODO:这里先输出提示，具体实现需要调整工作流
        print(f"   ⚠️ 提示：发现 {len(new_sensitive_files)} 个新敏感文件，建议重新扫描完整日志")
    
    return {}


def finalize_node(state: BehaviorAnalysisState, worklist_manager) -> Dict[str, Any]:
    """
    完成节点
    """
    print("\n✅ [Finalize] 分析完成")
    
    has_hidden = state.get("has_hidden_behavior", False)
    operations_count = len(state.get("hidden_operations", []))
    new_events_count = len(state.get("new_events", []))
    
    print(f"   - 发现隐藏行为: {'是' if has_hidden else '否'}")
    print(f"   - 隐藏操作数: {operations_count}")
    print(f"   - 新增事件数: {new_events_count}")
    
    return {
        "analysis_complete": True
    }


def should_create_events(state: BehaviorAnalysisState) -> str:
    """
    条件边：判断是否需要创建新事件
    """
    if state.get("has_hidden_behavior", False):
        return "create_events"
    else:
        return "skip"
