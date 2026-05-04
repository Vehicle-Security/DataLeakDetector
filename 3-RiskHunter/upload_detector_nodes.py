# upload_detector_nodes.py
"""
模块3 LangGraph节点定义
定义分析流程中的各个节点
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../2-FileTracker"))

from typing import Dict, Any
from datetime import datetime
from upload_detector_state import UploadDetectorState, UploadEvent
from upload_detection_config import config
from worklist_manager import WorklistManager, load_log_from_json, SensitiveFileEvent
from behavior_analysis_graph import analyze_sensitive_event_behavior
from upload_detector_tools import (
    resolve_full_path,
    read_recording_start_time,
    normalize_timestamp_display,
    build_sensitive_operation_record,
    extract_hidden_transformed_paths,
    append_operation_record_with_dedup,
    sync_processed_statistics,
)


UPLOAD_OPERATION_KEYWORDS = [
    "上传", "发送", "分享", "转发", "附件", "粘贴", "同步", "外发",
    "upload", "send", "share", "attach", "attachment",
]

NON_UPLOAD_OPERATION_KEYWORDS = [
    "签名", "设置姓名", "查看监控", "监控系统", "浏览文件", "正常操作",
    "加载", "预览", "存草稿", "取消", "返回收件箱",
]

UNKNOWN_FILE_VALUES = {"", "未知", "unknown", "none", "null"}


def _has_real_filename(value: str) -> bool:
    text = str(value or "").strip()
    if text.lower() in UNKNOWN_FILE_VALUES or text in UNKNOWN_FILE_VALUES:
        return False
    return "." in text or "/" in text or "\\" in text


def _is_upload_event(event_data: Dict[str, Any]) -> bool:
    behavior_category = str(event_data.get("behavior_category", ""))
    operation_type = str(event_data.get("operation_type", ""))
    description = str(event_data.get("description", ""))
    combined = f"{behavior_category} {operation_type} {description}".lower()

    if any(keyword.lower() in combined for keyword in NON_UPLOAD_OPERATION_KEYWORDS):
        return False

    has_upload_operation = any(keyword.lower() in combined for keyword in UPLOAD_OPERATION_KEYWORDS)
    has_named_file = _has_real_filename(event_data.get("original_filename", "")) or _has_real_filename(
        event_data.get("modified_filename", "")
    )

    return has_upload_operation and (has_named_file or "外发" in behavior_category)


def _parse_event_timestamp(timestamp: str):
    """解析事件时间戳，兼容常见格式。"""
    if not timestamp:
        return None

    text = str(timestamp).strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _build_segment_timeline(scene_segments: list) -> list:
    """
    构建分段时间轴：每段包含 [start_time, end_time] 与资源路径。
    """
    timeline = []

    for seg in scene_segments:
        seg_log_events = load_log_from_json(seg["log_file"])
        seg_events = []
        for event in seg_log_events:
            event_copy = dict(event)
            event_copy["__segment_name"] = seg.get("segment_name", "")
            event_copy["__segment_log_file"] = seg.get("log_file", "")
            seg_events.append(event_copy)

        parsed_times = [
            _parse_event_timestamp(item.get("timestamp", ""))
            for item in seg_events
            if item.get("timestamp")
        ]
        parsed_times = [item for item in parsed_times if item is not None]

        start_time = min(parsed_times) if parsed_times else None
        end_time = max(parsed_times) if parsed_times else None

        timeline.append(
            {
                "segment_name": seg.get("segment_name", ""),
                "log_file": seg.get("log_file", ""),
                "index_path": seg.get("index_path", ""),
                "video_path": seg.get("video_path", ""),
                "start_time": start_time,
                "end_time": end_time,
                "log_events": seg_events,
            }
        )

    # 保底按目录名排序，保持顺序稳定
    timeline.sort(key=lambda item: item.get("segment_name", ""))
    return timeline


def _select_segment_for_event(event_timestamp: str, timeline: list) -> Dict[str, Any]:
    """
    根据事件时间戳选择对应分段。
    优先命中时间窗口；若未命中，回退到“开始时间最接近且不晚于事件时间”的分段。
    """
    if not timeline:
        return {}

    event_dt = _parse_event_timestamp(event_timestamp)
    if event_dt is None:
        return timeline[0]

    for seg in timeline:
        start_time = seg.get("start_time")
        end_time = seg.get("end_time")
        if start_time and end_time and start_time <= event_dt <= end_time:
            return seg

    fallback = timeline[0]
    for seg in timeline:
        start_time = seg.get("start_time")
        if start_time and start_time <= event_dt:
            fallback = seg

    return fallback


def initialize_node(state: UploadDetectorState) -> UploadDetectorState:
    """
    初始化节点
    
    功能：
    1. 初始化WorklistManager
    2. 扫描日志构建worklist
    3. 更新状态
    """
    print("\n" + "=" * 80)
    print("📋 初始化上传检测系统")
    print("=" * 80)
    
    state["current_step"] = "initialize"
    state["messages"].append("开始初始化...")
    
    try:
        scene_segments = state.get("_scene_segments", [])

        if scene_segments:
            timeline = _build_segment_timeline(scene_segments)
            merged_log_events = []
            for seg in timeline:
                merged_log_events.extend(seg.get("log_events", []))

            merged_log_events.sort(
                key=lambda item: _parse_event_timestamp(item.get("timestamp", "")) or datetime.min
            )

            log_events = merged_log_events
            state["_segment_timeline"] = timeline
            print(f"✅ 加载多段日志: {len(scene_segments)} 段，共 {len(log_events)} 条事件")
            state["messages"].append(f"加载多段日志: {len(scene_segments)} 段，共 {len(log_events)} 条事件")

            for idx, seg in enumerate(timeline, 1):
                start_text = seg["start_time"].strftime("%Y-%m-%d %H:%M:%S") if seg.get("start_time") else "未知"
                end_text = seg["end_time"].strftime("%Y-%m-%d %H:%M:%S") if seg.get("end_time") else "未知"
                print(
                    f"   [{idx}] {seg.get('segment_name', '')} | "
                    f"时间范围: {start_text} ~ {end_text} | "
                    f"事件数: {len(seg.get('log_events', []))}"
                )
        else:
            log_events = load_log_from_json(state["log_file"])
            print(f"✅ 加载日志: {len(log_events)} 条事件")
            state["messages"].append(f"加载日志: {len(log_events)} 条事件")
        
        manager = WorklistManager(sensitive_files=state["sensitive_files"])
        print(f"✅ 初始化WorklistManager: {len(state['sensitive_files'])} 个敏感文件")
        state["messages"].append(f"初始化WorklistManager: {len(state['sensitive_files'])} 个敏感文件")
        
        added_count = manager.scan_and_build_worklist(log_events)
        print(f"✅ 构建worklist: 发现 {added_count} 个敏感事件")
        state["messages"].append(f"构建worklist: {added_count} 个敏感事件")
        
        state["worklist_size"] = manager.size()
        state["_worklist_manager"] = manager  # 保存manager实例（不会被序列化到JSON）
        state["_log_events"] = log_events  # 保存日志事件（不会被序列化到JSON）
        state["_operation_record_keys"] = set()
        state["_hidden_transformed_paths"] = []
        state["recording_start_time"] = ""
        
        stats = manager.get_statistics()
        print(f"\n📊 Worklist统计:")
        print(f"   - 大小: {stats['worklist_size']}")
        print(f"   - 事件类型: {stats['event_types']}")
        
        state["should_continue"] = not manager.is_empty()
        
    except Exception as e:
        error_msg = f"初始化失败: {e}"
        print(f"❌ {error_msg}")
        state["errors"].append(error_msg)
        state["should_continue"] = False
        import traceback
        traceback.print_exc()
    
    return state


def process_event_node(state: UploadDetectorState) -> UploadDetectorState:
    """
    处理事件节点
    
    功能：
    1. 从worklist获取下一个事件
    2. 调用模块2分析事件（模块2会调用模块1）
    3. 从模块2的结果中提取模块1的分析结果
    4. 更新状态
    """
    state["current_step"] = "process_event"
    
    try:
        manager: WorklistManager = state["_worklist_manager"]
        log_events = state["_log_events"]
        timeline = state.get("_segment_timeline", [])
        
        event = manager.get_next_event()
        if not event:
            print("\n✅ Worklist已空，处理完成")
            state["should_continue"] = False
            return state
        
        state["processed_count"] += 1
        sync_processed_statistics(state)
        
        print("\n" + "-" * 80)
        print(f"🔹 处理事件 {state['processed_count']}")
        print(f"   - 事件ID: {event.event_id}")
        print(f"   - 文件: {event.current_file}")
        print(f"   - 原始文件: {event.original_file}")
        print(f"   - 类型: {event.event_type}")
        print(f"   - 时间戳: {event.timestamp}")
        
        # 保存当前事件到状态
        state["current_event"] = {
            "event_id": event.event_id,
            "file_path": event.current_file,
            "original_file": event.original_file,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
        }
        
        # 调用模块2分析事件（模块2会调用模块1）
        print(f"   🔍 调用模块2分析事件...")

        selected_segment = _select_segment_for_event(event.timestamp, timeline)
        selected_index_path = selected_segment.get("index_path", state["index_path"])
        selected_video_path = selected_segment.get("video_path", state["video_path"])

        if selected_segment:
            print(
                f"   🧭 时间路由到分段: {selected_segment.get('segment_name', 'unknown')}"
            )
            print(f"      - INDEX: {selected_index_path}")
            print(f"      - VIDEO: {selected_video_path}")
        
        result = analyze_sensitive_event_behavior(
            event=event,
            index_path=selected_index_path,
            video_path=selected_video_path,
            worklist_manager=manager,
            log_events=log_events,
            search_duration=state["search_duration"]
        )

        # 直接复用模块2已解析出的变换后路径列表（不重新推断）
        state["_hidden_transformed_paths"] = extract_hidden_transformed_paths(result)
        
        state["module1_result"] = result.get("frame_analysis_result", result)

        # 优先复用模块2（behavior_analysis_tools）已经读取出的录屏开始时间
        module2_recording_time = ""
        if isinstance(state["module1_result"], dict):
            module2_recording_time = normalize_timestamp_display(
                state["module1_result"].get("recording_start_time", "")
            )

        if module2_recording_time:
            state["recording_start_time"] = module2_recording_time
        elif not state.get("recording_start_time"):
            # 仅在模块2结果缺失时兜底读取，避免重复解析
            try:
                fallback_time = normalize_timestamp_display(
                    read_recording_start_time(selected_index_path)
                )
            except Exception:
                fallback_time = ""

            if not fallback_time and log_events:
                fallback_time = normalize_timestamp_display(log_events[0].get("timestamp", ""))

            state["recording_start_time"] = fallback_time
        
        state["worklist_size"] = manager.size()
        
        print(f"   ✅ 分析完成")
        print(f"   📊 当前worklist大小: {state['worklist_size']}")
        
        # 如果发现了新的派生文件，重新扫描日志
        if result.get("has_hidden_behavior") and result.get("new_events"):
            new_events = result.get("new_events", [])
            print(f"\n   🔄 发现 {len(new_events)} 个新的派生事件，重新扫描日志，动态更新worklist...")
            additional_count = manager.scan_and_build_worklist(log_events)
            if additional_count > 0:
                print(f"   ✅ 新增 {additional_count} 个敏感事件到worklist")
                state["worklist_size"] = manager.size()
                print(f"   📊 更新后worklist大小: {state['worklist_size']}")
            else:
                print(f"   ℹ️ 未发现额外的敏感事件")
        
        state["should_continue"] = not manager.is_empty()
        
    except Exception as e:
        error_msg = f"处理事件失败: {e}"
        print(f"   ❌ {error_msg}")
        state["errors"].append(error_msg)
        import traceback
        traceback.print_exc()
    
    return state


def analyze_upload_node(state: UploadDetectorState) -> UploadDetectorState:
    """
    分析上传行为节点
    
    功能：
    1. 从模块1的结果中判断是否为上传行为
    2. 判断应用类别（黑名单/白名单/未知）
    3. 决定是否报警
    4. 创建UploadEvent并添加到相应列表
    """
    state["current_step"] = "analyze_upload"
    
    try:
        module1_result = state["module1_result"]
        current_event = state["current_event"]
        
        if not module1_result or not current_event:
            sync_processed_statistics(state)
            return state
        
        # 检查是否有外发行为
        events = module1_result.get("events", [])
        if not events:
            print(f"   ℹ️ 未检测到相关行为")
            sync_processed_statistics(state)
            return state

        hidden_transformed_paths = state.get("_hidden_transformed_paths", [])
        hidden_path_cursor = 0
        
        # 分析每个检测到的事件
        for event_data in events:
            app_name = event_data.get("app_name", "未知应用")
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

            # 为每个worklist事件增量记录敏感操作（先记录，再做上传过滤）
            operation_record = build_sensitive_operation_record(
                recording_start_time=state.get("recording_start_time", ""),
                sensitive_file_path=current_event.get("file_path", ""),
                event_data=event_data,
                fallback_timestamp=current_event.get("timestamp", ""),
                transformed_file_path=transformed_file_path,
            )
            if append_operation_record_with_dedup(state, operation_record):
                print(
                    "      📝 记录敏感操作: "
                    f"{operation_record['operation_time']} | "
                    f"{operation_record['sensitive_file_path']} | "
                    f"{operation_record['operation']}"
                )
            else:
                print("      ♻️ 敏感操作重复，已去重")
            
            print(f"\n   📊 分析检测结果:")
            print(f"      - 应用: {app_name}")
            print(f"      - 行为类别: {behavior_category}")
            print(f"      - 操作类型: {operation_type}")
            
            # 判断是否为上传/外发行为。VLM 偶尔会把“签名设置”等上下文事件标成
            # “直接外发”，这里要求同时具备明确上传动作，避免黑名单应用上下文误报。
            is_upload = _is_upload_event(event_data)
            
            if not is_upload:
                print(f"      ℹ️ 非上传行为，跳过")
                continue
            
            app_category = config.get_app_category(app_name)
            print(f"      - 应用类别: {app_category}")
            
            should_alert_flag, alert_level = config.should_alert(app_category, behavior_category)
            
            alert_reason = ""
            if should_alert_flag:
                if app_category == "blacklist":
                    alert_reason = f"检测到黑名单应用 '{app_name}' 的文件外发行为"
                else:
                    alert_reason = f"检测到可疑的文件外发行为"
            else:
                if app_category == "whitelist":
                    alert_reason = f"白名单应用 '{app_name}' 的正常文件上传操作"
                elif app_category == "unknown":
                    alert_reason = f"非黑名单应用 '{app_name}' 的文件外发（仅记录）"
            
            print(f"      - 报警: {'是' if should_alert_flag else '否'}")
            print(f"      - 级别: {alert_level}")
            print(f"      - 原因: {alert_reason}")
            
            # 提取真正外发的文件/内容
            # 对于直接外发行为，upload_content是真正外发的内容
            upload_content = event_data.get("original_filename", "")
            if not upload_content or upload_content == "未知":
                upload_content = current_event["file_path"]  # 如果没有，默认使用当前文件
            
            # 使用统一的路径解析函数
            upload_content_full_path = resolve_full_path(
                filename=upload_content,
                base_dir=os.path.dirname(current_event["file_path"]),
                log_events=state.get("_log_events", []),
                time_range=event_data.get("time_range", ""),
                print_prefix="      "
            )
            
            # 构建映射链：从worklist_manager获取文件映射
            upload_content_mapping_link = "无"
            try:
                manager = state.get("_worklist_manager")
                if manager and upload_content_full_path:
                    # 使用 get_mapping_chain 方法获取完整映射链
                    mapping_chain = manager.get_mapping_chain(upload_content_full_path)
                    if mapping_chain:
                        upload_content_mapping_link = mapping_chain
            except Exception as e:
                print(f"      ⚠️ 构建映射链失败: {e}")
            
            print(f"      - 外发内容: {upload_content}")
            print(f"      - 映射链: {upload_content_mapping_link}")
            
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
                extra_info=event_data
            )
            
            state["upload_events"].append(upload_event)
            
            if should_alert_flag:
                state["alert_events"].append(upload_event)
                print(f"      ⚠️ 添加到报警列表")
            else:
                state["info_events"].append(upload_event)
                print(f"      ℹ️ 添加到信息列表")
            
            # 更新统计
            state["statistics"]["upload_events_detected"] += 1
            if app_category == "blacklist" and should_alert_flag:
                state["statistics"]["blacklist_alerts"] += 1
            elif app_category == "whitelist":
                state["statistics"]["whitelist_uploads"] += 1
            elif app_category == "unknown":
                state["statistics"]["unknown_uploads"] += 1
        
    except Exception as e:
        error_msg = f"分析上传行为失败: {e}"
        print(f"   ❌ {error_msg}")
        state["errors"].append(error_msg)
        import traceback
        traceback.print_exc()
    
    sync_processed_statistics(state)
    
    return state


def finalize_node(state: UploadDetectorState) -> UploadDetectorState:
    """
    完成节点
    
    功能：
    1. 生成最终报告
    2. 保存结果到文件
    3. 显示统计信息
    """
    state["current_step"] = "finalize"
    sync_processed_statistics(state)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
    
    # 显示统计
    stats = state["statistics"]
    print(f"\n📊 最终统计:")
    print(f"   - 已处理事件: {stats['total_events_processed']}")
    print(f"   - 检测到的上传事件: {stats['upload_events_detected']}")
    print(f"   - 敏感操作记录数（去重后）: {len(state['operation_records'])}")
    print(f"   - 黑名单应用报警: {stats['blacklist_alerts']}")
    print(f"   - 白名单应用上传: {stats['whitelist_uploads']}")
    print(f"   - 其他应用上传: {stats['unknown_uploads']}")
    
    # 显示报警事件
    if state["alert_events"]:
        print(f"\n⚠️ 报警事件 ({len(state['alert_events'])} 个):")
        for i, event in enumerate(state["alert_events"], 1):
            print(f"\n   [{i}] {event.alert_level.upper()}")
            print(f"      时间: {event.timestamp}")
            print(f"      文件: {event.file_name}")
            print(f"      应用: {event.app_name} ({event.app_category})")
            print(f"      操作: {event.operation_type}")
            print(f"      外发内容: {event.upload_content}")
            print(f"      映射链: {event.upload_content_mapping_link}")
            print(f"      原因: {event.alert_reason}")
    else:
        print(f"\n✅ 无报警事件")
    
    # 显示信息事件
    if state["info_events"]:
        print(f"\nℹ️ 信息事件 ({len(state['info_events'])} 个):")
        for i, event in enumerate(state["info_events"], 1):
            print(f"\n   [{i}]")
            print(f"      时间: {event.timestamp}")
            print(f"      文件: {event.file_name}")
            print(f"      应用: {event.app_name} ({event.app_category})")
            print(f"      操作: {event.operation_type}")
            print(f"      外发内容: {event.upload_content}")
            print(f"      映射链: {event.upload_content_mapping_link}")
    
    if state["errors"]:
        print(f"\n❌ 错误 ({len(state['errors'])} 个):")
        for error in state["errors"]:
            print(f"   - {error}")
    
    print("\n" + "=" * 80)
    
    state["should_continue"] = False
    
    return state


def should_continue_processing(state: UploadDetectorState) -> str:
    """
    条件边：判断是否继续处理
    
    Returns:
        "continue": 继续处理下一个事件
        "end": 结束处理
    """
    if state["should_continue"] and state["worklist_size"] > 0:
        return "continue"
    else:
        return "end"
