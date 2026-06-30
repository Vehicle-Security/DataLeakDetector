# behavior_analysis_tools.py
"""
隐藏行为分析工具
"""

import os
import re
import sys
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langchain_core.tools import tool

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from relavance_frame import analyze_video_behavior


def normalize_file_path(file_path: str) -> str:
    """
    统一文件路径分隔符为 /，并清理重复分隔符

    Args:
        file_path: 原始路径

    Returns:
        规范化路径
    """
    if not file_path:
        return ""

    normalized = str(file_path).strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def get_path_basename(file_path: str) -> str:
    """
    获取文件名（同时兼容 Windows 和 Unix 路径分隔符）
    """
    normalized = normalize_file_path(file_path)
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def is_absolute_file_path(file_path: str) -> bool:
    """
    判断是否为绝对路径（兼容 Windows 盘符/UNC 与 Unix）
    """
    if not file_path:
        return False

    path = str(file_path)
    if os.path.isabs(path):
        return True

    if path.startswith("/") or path.startswith("\\\\"):
        return True

    # Windows 盘符路径，例如 D:\xxx 或 D:/xxx
    return len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/")


def normalize_timestamp_display(timestamp: str) -> str:
    """
    统一时间格式为 YYYY-MM-DD HH:MM:SS

    Args:
        timestamp: 原始时间字符串

    Returns:
        规范化后的时间字符串
    """
    if not timestamp:
        return ""

    text = str(timestamp).strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    if "." in text:
        text = text.split(".", 1)[0]

    return text


def select_operation_time(event_data: Dict[str, Any], fallback_timestamp: str) -> str:
    """
    选择敏感操作时间：优先取 involved_timestamps[0]，否则回退到传入时间

    Args:
        event_data: 模块1单条事件数据
        fallback_timestamp: 回退时间（通常为当前worklist事件时间）

    Returns:
        格式化后的操作时间
    """
    involved_timestamps = event_data.get("involved_timestamps", [])
    if isinstance(involved_timestamps, list) and involved_timestamps:
        selected = involved_timestamps[0]
    else:
        selected = fallback_timestamp

    return normalize_timestamp_display(selected)


def build_operation_text(
    behavior_category: str,
    operation_type: str,
    transformed_file_path: str = "",
) -> str:
    """
    构建操作描述文本

    Args:
        behavior_category: 行为类别
        operation_type: 操作类型
        transformed_file_path: 变换后文件路径（可选）

    Returns:
        操作描述文本
    """
    if behavior_category and operation_type:
        if behavior_category == "潜在隐藏行为" and transformed_file_path:
            return f"{behavior_category}-{operation_type}-{transformed_file_path}"
        return f"{behavior_category}-{operation_type}"

    return behavior_category or operation_type or "未知操作"


def build_sensitive_operation_record(
    recording_start_time: str,
    sensitive_file_path: str,
    event_data: Dict[str, Any],
    fallback_timestamp: str,
    transformed_file_path: str = "",
) -> Dict[str, Any]:
    """
    构建敏感操作记录

    Args:
        recording_start_time: 录屏开始时间
        sensitive_file_path: 敏感文件路径
        event_data: 模块1单条事件数据
        fallback_timestamp: 回退时间（通常为当前worklist事件时间）
        transformed_file_path: 变换后文件路径（可选）

    Returns:
        敏感操作记录字典
    """
    behavior_category = event_data.get("behavior_category", "")
    operation_type = event_data.get("operation_type", "")
    app_name = event_data.get("app_name", "")
    description = event_data.get("description", "")

    return {
        # "recording_start_time": recording_start_time,
        "operation_time": select_operation_time(event_data, fallback_timestamp),
        "sensitive_file_path": sensitive_file_path,
        "app_name": app_name,
        "description": description,
        "operation": build_operation_text(
            behavior_category=behavior_category,
            operation_type=operation_type,
            transformed_file_path=transformed_file_path,
        ),
    }


def build_sensitive_operation_dedup_key(operation_record: Dict[str, Any]) -> str:
    """
    生成敏感操作去重键

    规则：同时间 + 同文件 + 同操作
    """
    return "|".join(
        [
            operation_record.get("operation_time", ""),
            operation_record.get("sensitive_file_path", ""),
            operation_record.get("operation", ""),
        ]
    )


def split_output_filenames(filename_text: Any) -> List[str]:
    """
    拆分模型返回的输出文件名列表。

    VLM 可能同时返回中英文逗号、分号连接的多个文件名，这里统一拆分并去重。
    """
    raw_text = str(filename_text or "").strip()
    if not raw_text:
        return []

    seen = set()
    filenames: List[str] = []

    for part in re.split(r"[，,；;]+", raw_text):
        name = part.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        filenames.append(name)

    return filenames


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

    rec_start_time = ""
    
    try:
        rec_start_time = read_recording_start_time(index_path)
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
        
        pre_seconds = max(0, int(os.getenv("DLD_ANALYSIS_PRE_SECONDS", "8")))
        post_seconds = max(
            int(search_duration),
            int(os.getenv("DLD_ANALYSIS_POST_SECONDS", str(search_duration))),
        )
        search_start_dt = event_dt - timedelta(seconds=pre_seconds)
        search_start_time = search_start_dt.strftime("%Y-%m-%d %H:%M:%S")
        search_end_dt = event_dt + timedelta(seconds=post_seconds)
        search_end_time = search_end_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        filename = os.path.splitext(get_path_basename(current_file))[0]
        target_keywords = [filename]
        
        print(f"   - 搜索范围: {search_start_time} ~ {search_end_time}")
        print(f"   - 目标关键词: {target_keywords}")
        
        # 调用模块1
        cache_enabled = os.getenv("DLD_FRAME_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
        cache_path = ""
        if cache_enabled:
            cache_dir = os.path.join(os.getcwd(), "output", "cache", "frame_analysis")
            os.makedirs(cache_dir, exist_ok=True)
            try:
                video_mtime = os.path.getmtime(video_path)
            except OSError:
                video_mtime = 0
            cache_payload = {
                "video_path": os.path.abspath(video_path),
                "video_mtime": video_mtime,
                "search_start": search_start_time,
                "search_end": search_end_time,
                "target_keywords": target_keywords,
                "search_duration": search_duration,
                "pre_seconds": pre_seconds,
                "post_seconds": post_seconds,
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_result = json.load(f)
                    print(f"   [cache] reused frame analysis: {cache_path}")
                    return cached_result
                except Exception as cache_error:
                    print(f"   [cache] ignored unreadable cache: {cache_error}")

        result = analyze_video_behavior(
            rec_start_time_str=rec_start_time,
            search_start_time_str=search_start_time,
            search_end_time_str=search_end_time,
            target_keywords=target_keywords,
            video_path=video_path
        )
        
        if result:
            result = dict(result)
            result["recording_start_time"] = rec_start_time
            result["review_window"] = {
                "anchor_timestamp": event_timestamp,
                "start": search_start_time,
                "end": search_end_time,
                "pre_seconds": pre_seconds,
                "post_seconds": post_seconds,
            }
            if cache_enabled and cache_path:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"   [cache] saved frame analysis: {cache_path}")
                except Exception as cache_error:
                    print(f"   [cache] save failed: {cache_error}")
            print(f"   ✅ 模块1分析完成，发现 {result.get('total_events', 0)} 个事件")
            print(f"   - 结果预览: {json.dumps(result.get('events', [])[:3], ensure_ascii=False)}")
            return result
        else:
            print(f"   ⚠️ 模块1未返回结果")
            return {
                "recording_start_time": rec_start_time,
                "search_range": {
                    "start": search_start_time,
                    "end": search_end_time
                },
                "review_window": {
                    "anchor_timestamp": event_timestamp,
                    "start": search_start_time,
                    "end": search_end_time,
                    "pre_seconds": pre_seconds,
                    "post_seconds": post_seconds,
                },
                "total_events": 0,
                "events": []
            }
            
    except Exception as e:
        print(f"   ❌ 调用模块1失败: {e}")
        return {
            "error": str(e),
            "recording_start_time": rec_start_time,
            "search_range": {"start": event_timestamp, "end": event_timestamp},
            "total_events": 0,
            "events": []
        }


def read_recording_start_time(index_path: str) -> str:
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
        seen_hidden_operations = set()
        
        for event in events:
            behavior_category = event.get("behavior_category", "")
            
            if behavior_category == "潜在隐藏行为":
                original_filename = event.get("original_filename", "")
                modified_filename = event.get("modified_filename", "")
                # 支持一个操作生成多个文件：统一拆分中英文逗号/分号
                modified_filenames = split_output_filenames(modified_filename)
                
                # 文件名必须不同才算隐藏操作
                if original_filename and modified_filenames:
                    for new_filename in modified_filenames:
                        if original_filename == new_filename:
                            continue

                        operation_key = (
                            event.get("operation_type", "未知操作"),
                            original_filename,
                            new_filename,
                            event.get("time_range", ""),
                            tuple(event.get("involved_timestamps", [])),
                        )
                        if operation_key in seen_hidden_operations:
                            continue

                        seen_hidden_operations.add(operation_key)

                        operation = {
                            "operation_type": event.get("operation_type", "未知操作"),
                            "original_file": original_filename,
                            "new_file": new_filename,
                            "app_name": event.get("app_name", "未知应用"),
                            "time_range": event.get("time_range", ""),
                            "description": event.get("description", ""),
                            "involved_timestamps": event.get("involved_timestamps", [])  # 保存实际发生时间
                        }
                        hidden_operations.append(operation)

                        file_mappings.append({
                            "original": original_filename,
                            "derived": new_filename,
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

# ==================== 共同辅助函数 ====================

def infer_full_path(base_dir: str, filename: str) -> str:
    """
    推断并标准化文件的完整路径
    
    Args:
        base_dir: 基础目录路径
        filename: 文件名（可能是完整路径或仅文件名）
        
    Returns:
        标准化后的完整路径
    """
    # 如果已经是绝对路径，直接使用
    if is_absolute_file_path(filename):
        full_path = filename
    else:
        full_path = os.path.join(base_dir, filename) if base_dir else filename

    return normalize_file_path(full_path)


def find_file_path_in_logs(filename: str, time_range: str, log_events: list) -> str:
    """
    从日志中查找文件的实际完整路径
    
    单次遍历策略（高效）：
    1. 遍历日志时记录：完全匹配（带扩展名）和不带扩展名匹配
    2. 找到完全匹配立即返回
    3. 遍历完没有完全匹配，返回不带扩展名匹配（如果有）
    
    Args:
        filename: 文件名（不含路径）
        time_range: 时间范围字符串（如 "2026-01-05 10:00:00 - 2026-01-05 10:10:00"）
        log_events: 日志事件列表
        
    Returns:
        找到的完整文件路径，如果未找到则返回空字符串
    """
    if not log_events:
        return ""
    
    start_time, end_time = datetime.min, datetime.max
    try:
        time_parts = time_range.split(" - ")
        if len(time_parts) == 2:
            start_time = datetime.strptime(time_parts[0].strip(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(time_parts[1].strip(), "%Y-%m-%d %H:%M:%S")
    except:
        pass
    
    normalized_filename = normalize_file_path(filename)
    target_basename = get_path_basename(normalized_filename)
    target_name_no_ext = os.path.splitext(target_basename)[0]
    
    no_ext_match = None
    no_ext_in_range = False
    
    for event in log_events:
        try:
            file_path = normalize_file_path(event.get("file_path", ""))
            if not file_path:
                continue

            event_filename = get_path_basename(file_path)

            if event_filename == target_basename:
                in_time_range = False
                event_time_str = event.get("timestamp", "")
                if event_time_str:
                    try:
                        event_time = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S.%f")
                        in_time_range = start_time <= event_time <= end_time
                    except:
                        try:
                            event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                            in_time_range = start_time <= event_time <= end_time
                        except:
                            pass
                
                range_info = "在时间范围内" if in_time_range else "不在时间范围内"
                print(f"      ✅ 带扩展名匹配文件路径（{range_info}）")
                return file_path
            
            if not no_ext_match:
                event_name_no_ext = os.path.splitext(event_filename)[0]
                if event_name_no_ext == target_name_no_ext:
                    no_ext_match = file_path
                    
                    event_time_str = event.get("timestamp", "")
                    if event_time_str:
                        try:
                            event_time = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S.%f")
                            no_ext_in_range = start_time <= event_time <= end_time
                        except:
                            try:
                                event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                                no_ext_in_range = start_time <= event_time <= end_time
                            except:
                                pass
                    
        except Exception as e:
            continue
    
    if no_ext_match:
        range_info = "在时间范围内" if no_ext_in_range else "不在时间范围内"
        print(f"      ✅ 不带扩展名匹配文件路径（{range_info}）")
        return no_ext_match
    
    return ""


def resolve_full_path(
    filename: str,
    base_dir: str,
    log_events: list = None,
    time_range: str = "",
    print_prefix: str = ""
) -> str:
    """
    解析文件的完整路径（智能推断）
    
    策略：
    1. 如果已经是完整路径，直接返回
    2. 优先从日志中查找（处理跨目录情况）
    3. 日志未找到则使用同目录推断
    
    Args:
        filename: 文件名或路径
        base_dir: 基础目录路径（用于同目录推断）
        log_events: 日志事件列表（可选）
        time_range: 时间范围字符串（可选）
        print_prefix: 打印信息的前缀（用于缩进）
        
    Returns:
        解析后的完整路径
    """
    normalized_filename = normalize_file_path(filename)

    # 检查是否已经是完整路径
    if is_absolute_file_path(normalized_filename):
        return normalized_filename
    
    # 优先从日志中查找
    if log_events:
        found_path = find_file_path_in_logs(normalized_filename, time_range, log_events)
        if found_path:
            print(f"{print_prefix}📍 从日志找到完整路径: {found_path}")
            return normalize_file_path(found_path)
    else:
        print(f"{print_prefix}📍 日志中未找到文件路径")
    # 日志未找到，使用同目录推断
    full_path = infer_full_path(base_dir, normalized_filename)
    print(f"{print_prefix}📍 使用同目录推断路径: {full_path}")
    return normalize_file_path(full_path)


# 工具列表
tools_list = [
    analyze_frame_behavior,
    extract_hidden_operations,
]
