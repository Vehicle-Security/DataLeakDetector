# behavior_analysis_tools.py
"""
隐藏行为分析工具
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool

sys.path.append(os.path.join(os.path.dirname(__file__), "../1-FrameAnalyzer"))
from relavance_frame import analyze_video_behavior


@tool
def analyze_frame_behavior(
    event_timestamp: str,
    current_file: str,
    index_path: str,
    video_path: str,
    search_duration: int = 30
) -> Dict[str, Any]:
    """
    调用模块1分析视频帧，获取敏感文件的操作行为
    
    Args:
        event_timestamp: 事件时间戳 (格式: "2026-01-05 17:48:33")
        current_file: 当前文件路径
        index_path: INDEX.md 文件路径
        video_path: 视频文件路径
        search_duration: 搜索时长（秒），默认30秒
        
    Returns:
        模块1的分析结果
    """
    print(f"   [Tool] 调用模块1分析帧行为...")
    print(f"   - 事件时间: {event_timestamp}")
    print(f"   - 文件: {current_file}")
    
    try:
        rec_start_time = _read_recording_start_time(index_path)
        print(f"   - 录屏开始时间: {rec_start_time}")
        
        # 支持两种时间格式：带毫秒和不带毫秒
        try:
            event_dt = datetime.strptime(event_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            try:
                event_dt = datetime.strptime(event_timestamp, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                try:
                    # 如果都失败，尝试去掉T的格式
                    event_dt = datetime.strptime(event_timestamp.replace('T', ' '), "%Y-%m-%d %H:%M:%S")
                except ValueError as e:
                    raise ValueError(f"无法解析事件时间戳: {event_timestamp}") from e
        
        search_start_time = event_dt.strftime("%Y-%m-%d %H:%M:%S")
        search_end_dt = event_dt + timedelta(seconds=search_duration)
        search_end_time = search_end_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        filename = os.path.splitext(os.path.basename(current_file))[0]
        target_keywords = [filename]
        
        print(f"   - 搜索范围: {search_start_time} ~ {search_end_time}")
        print(f"   - 目标关键词: {target_keywords}")
        
        # 调用模块1
        result = analyze_video_behavior(
            rec_start_time_str=rec_start_time,
            search_start_time_str=search_start_time,
            search_end_time_str=search_end_time,
            target_keywords=target_keywords,
            video_path=video_path
        )
        
        if result:
            print(f"   ✅ 模块1分析完成，发现 {result.get('total_events', 0)} 个事件")
            print(f"   - 结果预览: {json.dumps(result.get('events', [])[:3], ensure_ascii=False)}")
            return result
        else:
            print(f"   ⚠️ 模块1未返回结果")
            return {
                "search_range": {
                    "start": search_start_time,
                    "end": search_end_time
                },
                "total_events": 0,
                "events": []
            }
            
    except Exception as e:
        print(f"   ❌ 调用模块1失败: {e}")
        return {
            "error": str(e),
            "search_range": {"start": event_timestamp, "end": event_timestamp},
            "total_events": 0,
            "events": []
        }


def _read_recording_start_time(index_path: str) -> str:
    """
    从 INDEX.md 文件读取录屏开始时间
    
    Args:
        index_path: INDEX.md 文件路径
        
    Returns:
        录屏开始时间字符串 (格式: "2026-01-05 17:48:33")
    """
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找 "**Recording Time**: 2026-01-05 17:48:33" 格式
        import re
        match = re.search(r'\*\*Recording Time\*\*:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', content)
        if match:
            return match.group(1)
        else:
            raise ValueError("无法从 INDEX.md 中提取录屏开始时间")
            
    except Exception as e:
        print(f"   ❌ 读取 INDEX.md 失败: {e}")
        raise


@tool
def extract_hidden_operations(frame_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    从帧分析结果中提取隐藏操作信息
    
    Args:
        frame_analysis_result: 模块1的分析结果
        
    Returns:
        包含隐藏操作和文件映射的字典
    """
    print(f"   [Tool] 提取隐藏操作...")
    
    try:
        events = frame_analysis_result.get("events", [])
        hidden_operations = []
        file_mappings = []
        
        for event in events:
            behavior_category = event.get("behavior_category", "")
            
            if behavior_category == "潜在隐藏行为":
                original_filename = event.get("original_filename", "")
                modified_filename = event.get("modified_filename", "")
                
                # 文件名必须不同才算隐藏操作
                if original_filename and modified_filename and original_filename != modified_filename:
                    operation = {
                        "operation_type": event.get("operation_type", "未知操作"),
                        "original_file": original_filename,
                        "new_file": modified_filename,
                        "app_name": event.get("app_name", "未知应用"),
                        "time_range": event.get("time_range", ""),
                        "description": event.get("description", ""),
                        "involved_timestamps": event.get("involved_timestamps", [])  # 保存实际发生时间
                    }
                    hidden_operations.append(operation)
                    
                    file_mappings.append({
                        "original": original_filename,
                        "derived": modified_filename,
                        "relationship": event.get("operation_type", "未知")
                    })
        
        has_hidden_behavior = len(hidden_operations) > 0
        
        result = {
            "has_hidden_behavior": has_hidden_behavior,
            "hidden_operations": hidden_operations,
            "file_mappings": file_mappings
        }
        
        print(f"   ✅ 发现 {len(hidden_operations)} 个隐藏操作")
        return result
        
    except Exception as e:
        print(f"   ❌ 提取隐藏操作失败: {e}")
        return {
            "has_hidden_behavior": False,
            "hidden_operations": [],
            "file_mappings": []
        }


# 工具列表
tools_list = [
    analyze_frame_behavior,
    extract_hidden_operations,
]
