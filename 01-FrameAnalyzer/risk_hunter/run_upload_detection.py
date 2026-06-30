# run_upload_detection.py
"""
模块3使用示例
演示如何使用上传检测Agent分析文件上传行为
"""

import json
import glob
import os
from datetime import datetime
from typing import Dict, Any, List
from upload_detector_graph import create_upload_detector_graph
from upload_detector_state import create_initial_state, save_state_to_json
from upload_detection_config import config


def _find_segment_assets(segment_dir: str) -> Dict[str, Any]:
    """
    在单个分段目录中查找日志/INDEX/视频资源。
    返回空字典表示该目录不是有效分段。
    """
    log_file = os.path.join(segment_dir, "logs", "keyevents.json")
    index_path = os.path.join(segment_dir, "INDEX.md")
    video_candidates = sorted(
        glob.glob(os.path.join(segment_dir, "video", "*.mp4"))
        + glob.glob(os.path.join(segment_dir, "video", "*.mov"))
    )

    if not (os.path.exists(log_file) and os.path.exists(index_path) and video_candidates):
        return {}

    return {
        "segment_dir": segment_dir,
        "segment_name": os.path.basename(segment_dir),
        "log_file": log_file,
        "index_path": index_path,
        "video_path": video_candidates[0],
    }


def discover_scene_segments(base_path: str) -> List[Dict[str, Any]]:
    """
    发现一个场景下的分段数据。

    支持两种结构：
    1) 旧结构：base_path 本身就是单段目录
    2) 新结构：base_path 下有多个按时间命名的子目录，每个子目录是一段
    """
    segments: List[Dict[str, Any]] = []

    # 兼容旧结构：base_path 自身就是一段
    current = _find_segment_assets(base_path)
    if current:
        segments.append(current)
        return segments

    # 新结构：按目录名排序后作为时间顺序
    child_dirs = sorted(
        [
            os.path.join(base_path, name)
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ]
    ) if os.path.isdir(base_path) else []

    for child_dir in child_dirs:
        segment = _find_segment_assets(child_dir)
        if segment:
            segments.append(segment)

    return segments


def main():
    """
    主函数：演示上传检测的完整流程
    
    使用步骤：
    1. 修改配置参数（record_id、敏感文件、黑白名单）
    2. 运行程序
    3. 查看输出报告
    """
    
    print("=" * 80)
    print("文件上传检测系统 - 模块3")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    # 记录ID（可修改为不同的会话）
    # 可用ID：41 42 43 44 49 50 52 53 54 55 56 57 59 61 62 64 65 70
    #        
    # 无用ID：45\58(模块1无法识别截图操作及生成的照片文件)
    #        46\47(两块屏幕像素低)  48 51 60
    # 待重测ID：65
    # 现有逻辑无法解决ID：63
    record_id = "10-1" #"5-1" (90s) #"2-filestruct-pdfconvert-1" #"5-screen-record-1"
    
    # 模块1视频搜索时间范围（秒）
    # 从敏感文件事件发生时刻开始往后搜索的时长
    search_duration = 30
    
    # base_path = f"records/{record_id}"
    base_path = f"recordings/stage4/{record_id}"
    
    """
    分段前的旧代码
    # log_files = glob.glob(f"{base_path}/key_events/key_events_*.json")
    log_files = glob.glob(f"{base_path}/logs/keyevents.json")
    if not log_files:
        # print(f"❌ 错误: 在 {base_path}/key_events/ 中找不到 key_events_*.json 文件")
        print(f"❌ 错误: 在 {base_path}/logs/ 中找不到 keyevents.json 文件")
        return
    """
    # 分段新代码-begin
    segments = discover_scene_segments(base_path)
    if not segments:
        print(f"❌ 错误: 在 {base_path} 下找不到有效分段（需包含 logs/keyevents.json、INDEX.md、video/*.mp4|*.mov）")
        return
    # 分段新代码-end
    
    """
    log_file = log_files[0]
    print(f"📄 日志文件: {os.path.basename(log_file)}")
    
    index_path = f"{base_path}/INDEX.md"
    if not os.path.exists(index_path):
        print(f"❌ 错误: INDEX.md 不存在: {index_path}")
        return
    print(f"📄 INDEX文件: {index_path}")
    
    video_files = glob.glob(f"{base_path}/video/*.mp4") + glob.glob(f"{base_path}/video/*.mov")
    if not video_files:
        print(f"❌ 错误: 在 {base_path}/video/ 中找不到 .mp4/.mov 文件")
        return
    video_path = video_files[0]
    print(f"🎥 视频文件: {os.path.basename(video_path)}")
    """
    #分段新代码-begin
    print(f"📦 场景分段数: {len(segments)}")
    for idx, seg in enumerate(segments, 1):
        print(f"   [{idx}] {seg['segment_name']}")
        print(f"       📄 日志: {seg['log_file']}")
        print(f"       📄 INDEX: {seg['index_path']}")
        print(f"       🎥 视频: {seg['video_path']}")

    # 保留单段字段以兼容现有流程，默认取第一段；真正按时间段路由在节点中处理
    log_file = segments[0]["log_file"]
    index_path = segments[0]["index_path"]
    video_path = segments[0]["video_path"]
    #分段新代码-end

    sensitive_files = config.sensitive_files
    blacklist_apps = config.blacklist_apps
    whitelist_apps = config.whitelist_apps
    
    print(f"\n📋 配置信息:")
    print(f"   - 敏感文件: {len(sensitive_files)} 个")
    print(f"   - 黑名单应用: {len(blacklist_apps)} 个")
    print(f"   - 白名单应用: {len(whitelist_apps)} 个")
    print(f"   - 视频搜索间隔时长: {search_duration} 秒")
    
    # ========== 创建初始状态 ==========
    print(f"\n🔧 创建初始状态...")
    initial_state = create_initial_state(
        record_id=record_id,
        base_path=base_path,
        log_file=log_file,
        video_path=video_path,
        index_path=index_path,
        sensitive_files=sensitive_files,
        blacklist_apps=blacklist_apps,
        whitelist_apps=whitelist_apps,
        search_duration=search_duration
    )
    initial_state["_scene_segments"] = segments
    
    # ========== 创建并运行图 ==========
    print(f"🚀 启动上传检测Agent...")
    
    try:
        app = create_upload_detector_graph()
        # 默认 recursion_limit=25，worklist事件稍多时会触发上限
        graph_config = {"recursion_limit": 35}
        
        final_state = None
        for state in app.stream(initial_state, config=graph_config):
            final_state = state
        
        if final_state:
            # 提取实际的state（从字典中获取最后一个节点的输出）
            if isinstance(final_state, dict):
                final_state = list(final_state.values())[-1]
        
        # ========== 保存结果 ==========
        print(f"\n💾 保存结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = f"{base_path}/results"
        output_dir = f"{output_base_dir}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        '''
        # 保存完整状态
        state_output_path = f"{output_dir}/full_state.json"
        save_state_to_json(final_state, state_output_path)
        print(f"   ✅ 完整状态: {state_output_path}")
        '''
        # 保存报警事件
        if final_state["alert_events"]:
            alerts_output_path = f"{output_dir}/alert_events.json"
            alerts_data = {
                "record_id": record_id,
                "timestamp": timestamp,
                "total_alerts": len(final_state["alert_events"]),
                "alerts": [event.to_dict() for event in final_state["alert_events"]]
            }
            with open(alerts_output_path, 'w', encoding='utf-8') as f:
                json.dump(alerts_data, f, ensure_ascii=False, indent=2)
            print(f"   ⚠️ 报警事件: {alerts_output_path}")
        
        # 保存信息事件
        if final_state["info_events"]:
            info_output_path = f"{output_dir}/info_events.json"
            info_data = {
                "record_id": record_id,
                "timestamp": timestamp,
                "total_events": len(final_state["info_events"]),
                "events": [event.to_dict() for event in final_state["info_events"]]
            }
            with open(info_output_path, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, ensure_ascii=False, indent=2)
            print(f"   ℹ️ 信息事件: {info_output_path}")

        # 保存敏感操作明细（用于与groundtruth做正确率评估）
        operation_records = final_state.get("operation_records", [])
        operation_records = sorted(
            operation_records,
            key=lambda item: item.get("operation_time", "")
        )
        operations_output_path = f"{output_dir}/sensitive_operations.json"
        operations_data = {
            "record_id": record_id,
            "timestamp": timestamp,
            "recording_start_time": final_state.get("recording_start_time", ""),
            "total_operations": len(operation_records),
            "operations": operation_records
        }
        with open(operations_output_path, 'w', encoding='utf-8') as f:
            json.dump(operations_data, f, ensure_ascii=False, indent=2)
        print(f"   🧪 敏感操作明细: {operations_output_path}")

        # 保存文件映射关系（直接映射关系 + 完整映射链）
        mappings_output_path = f"{output_dir}/file_mappings.json"
        mappings_data = {
            "record_id": record_id,
            "timestamp": timestamp,
            "direct_file_mappings": {},
            "full_file_mapping_chains": {}
        }

        manager = final_state.get("_worklist_manager") if isinstance(final_state, dict) else None
        if manager and hasattr(manager, "export_file_mappings"):
            mappings_data.update(manager.export_file_mappings())
        else:
            print("   ⚠️ 未获取到WorklistManager，文件映射关系将输出为空")

        with open(mappings_output_path, 'w', encoding='utf-8') as f:
            json.dump(mappings_data, f, ensure_ascii=False, indent=2)
        print(f"   🔗 文件映射关系: {mappings_output_path}")
        
        # 保存简要报告
        report_output_path = f"{output_dir}/report.txt"
        with open(report_output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("文件上传检测报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"记录ID: {record_id}\n")
            f.write(f"生成时间: {timestamp}\n\n")
            
            f.write("统计信息:\n")
            stats = final_state["statistics"]
            f.write(f"  - 已处理事件: {stats['total_events_processed']}\n")
            f.write(f"  - 检测到的上传事件: {stats['upload_events_detected']}\n")
            f.write(f"  - 敏感操作记录数（去重后）: {len(final_state.get('operation_records', []))}\n")
            f.write(f"  - 黑名单应用报警: {stats['blacklist_alerts']}\n")
            f.write(f"  - 白名单应用上传: {stats['whitelist_uploads']}\n")
            f.write(f"  - 其他应用上传: {stats['unknown_uploads']}\n\n")
            
            if final_state["alert_events"]:
                f.write("=" * 80 + "\n")
                f.write(f"报警事件 ({len(final_state['alert_events'])} 个)\n")
                f.write("=" * 80 + "\n\n")
                
                for i, event in enumerate(final_state["alert_events"], 1):
                    f.write(f"[{i}] {event.alert_level.upper()}\n")
                    f.write(f"  时间: {event.timestamp}\n")
                    f.write(f"  文件: {event.file_name}\n")
                    f.write(f"  原始文件: {event.original_file}\n")
                    f.write(f"  应用: {event.app_name} ({event.app_category})\n")
                    f.write(f"  操作: {event.operation_type}\n")
                    f.write(f"  原因: {event.alert_reason}\n")
                    f.write(f"  描述: {event.description}\n\n")
            
            if final_state["info_events"]:
                f.write("=" * 80 + "\n")
                f.write(f"信息事件 ({len(final_state['info_events'])} 个)\n")
                f.write("=" * 80 + "\n\n")
                
                for i, event in enumerate(final_state["info_events"], 1):
                    f.write(f"[{i}]\n")
                    f.write(f"  时间: {event.timestamp}\n")
                    f.write(f"  文件: {event.file_name}\n")
                    f.write(f"  原始文件: {event.original_file}\n")
                    f.write(f"  应用: {event.app_name} ({event.app_category})\n")
                    f.write(f"  操作: {event.operation_type}\n")
                    f.write(f"  描述: {event.description}\n\n")
        
        print(f"   📄 简要报告: {report_output_path}")
        
        print("\n✅ 处理完成！")
        print(f"📁 结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
