# run_behavior_analysis.py
"""
隐藏行为分析示例
演示如何使用 behavior_analysis_graph 分析敏感事件
"""

import json
import glob
import os
from worklist_manager import WorklistManager, load_log_from_json
from behavior_analysis_graph import analyze_sensitive_event_behavior


def main():
    """
    主函数：演示完整的工作流
    
    流程：
    1. 初始化 WorklistManager 并添加敏感文件
    2. 从日志构建 worklist
    3. 循环处理 worklist 中的事件
    4. 对每个事件调用隐藏行为分析
    5. 分析会自动更新 worklist（发现新的派生文件）
    """
    
    print("=" * 80)
    print("隐藏行为分析示例")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    # 记录ID（可修改为不同的会话）
    record_id = 42
    
    # 基础路径
    base_path = f"../risk_hunter/records/{record_id}"
    
    # 敏感文件列表（由用户提供）
    sensitive_files = [
        "/Users/tujiali/Documents/项目3prd设计.docx", # 60 重命名
        "/Users/tujiali/Documents/项目1详细规划.docx", # 43 压缩zip
        "/Users/tujiali/Documents/项目2需求分析.docx", # 42 格式转换：导出为pdf
        "D:\\dingxinyao\\desktop\\智能图像增强系统技术设计文档.docx" # 64 重命名(windows)
    ]
    
    log_files = glob.glob(f"{base_path}/key_events/key_events_*.json")
    if not log_files:
        print(f"❌ 错误: 在 {base_path}/key_events/ 中找不到 key_events_*.json 文件")
        return
    log_file = log_files[0]  # 取第一个匹配的文件
    print(f"📄 使用日志文件: {os.path.basename(log_file)}")
    
    index_path = f"{base_path}/INDEX.md"
    if not os.path.exists(index_path):
        print(f"❌ 错误: INDEX.md 不存在: {index_path}")
        return
    
    video_files = glob.glob(f"{base_path}/video/*.mp4")
    if not video_files:
        print(f"❌ 错误: 在 {base_path}/video/ 中找不到 .mp4 文件")
        return
    video_path = video_files[0]  # 取第一个匹配的文件
    print(f"🎥 使用视频文件: {os.path.basename(video_path)}")
    
    # ========== 步骤1: 初始化 WorklistManager ==========
    print("\n📋 步骤1: 初始化 WorklistManager")
    manager = WorklistManager(sensitive_files=sensitive_files)
    print(f"   - 敏感文件数: {len(sensitive_files)}")
    
    # ========== 步骤2: 扫描日志并构建 worklist ==========
    print("\n🔍 步骤2: 扫描日志构建 worklist")
    
    try:
        log_events = load_log_from_json(log_file)
        added_count = manager.scan_and_build_worklist(log_events)
        print(f"   ✅ 成功扫描 {len(log_events)} 条日志")
        print(f"   ✅ 发现 {added_count} 个敏感事件")
    except FileNotFoundError:
        print(f"   ⚠️ 日志文件不存在: {log_file}")
    except json.JSONDecodeError:
        print(f"   ❌ 日志文件格式错误，无法解析 JSON")
    except Exception as e:
        print(f"   ❌ 扫描日志失败: {e}")
    
    stats = manager.get_statistics()
    print(f"\n📊 初始 Worklist 统计:")
    print(f"   - Worklist 大小: {stats['worklist_size']}")
    print(f"   - 事件类型分布: {stats['event_types']}")
    
    # ========== 步骤3: 循环处理 worklist ==========
    print("\n🔄 步骤3: 循环处理 worklist")
    print("-" * 80)
    
    processed_count = 0
    max_iterations = 10  # 防止无限循环
    iteration = 0
    
    while not manager.is_empty() and iteration < max_iterations:
        iteration += 1
        
        event = manager.get_next_event()
        if not event:
            break
        
        processed_count += 1
        print(f"\n🔹 处理事件 {processed_count} (迭代 {iteration})")
        print(f"   - 事件ID: {event.event_id}")
        print(f"   - 文件: {event.current_file}")
        print(f"   - 原始文件: {event.original_file}")
        print(f"   - 类型: {event.event_type}")
        print(f"   - 是否隐藏: {event.is_hidden}")
        
        # ========== 步骤4: 调用隐藏行为分析 ==========
        print(f"\n   🔍 调用隐藏行为分析...")
        
        try:
            # 调用模块1分析视频帧
            # 并自动更新 worklist（如果发现新的派生文件）
            result = analyze_sensitive_event_behavior(
                event=event,
                index_path=index_path,
                video_path=video_path,
                worklist_manager=manager,
                log_events=log_events  # 传递日志事件以支持跨目录文件查找
            )
            
            if result.get("has_hidden_behavior"):
                print(f"   ⚠️ 发现隐藏行为!")
                operations = result.get("hidden_operations", [])
                for op in operations:
                    print(f"      - {op['operation_type']}: {op['original_file']} → {op['new_file']}")
                
                new_events = result.get("new_events", [])
                print(f"   ✅ 新增 {len(new_events)} 个派生事件到 worklist")
                
                # 重要：发现新的敏感文件后，重新扫描日志
                if new_events:
                    print(f"\n   🔄 重新扫描日志以查找新敏感文件的相关操作...")
                    try:
                        additional_count = manager.scan_and_build_worklist(log_events)
                        if additional_count > 0:
                            print(f"   ✅ 发现 {additional_count} 个额外的敏感事件")
                        else:
                            print(f"   ℹ️ 没有发现额外的敏感事件")
                    except Exception as rescan_error:
                        print(f"   ⚠️ 重新扫描日志失败: {rescan_error}")
            else:
                print(f"   ✅ 未发现隐藏行为")
                
        except Exception as e:
            print(f"   ❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()
        
        current_size = manager.size()
        print(f"\n   📊 当前 Worklist 大小: {current_size}")
    
    # ========== 最终统计 ==========
    print("\n" + "=" * 80)
    print("✅ 处理完成")
    print("=" * 80)
    
    final_stats = manager.get_statistics()
    print(f"\n📊 最终统计:")
    print(f"   - 敏感文件数: {final_stats['sensitive_files_count']}")
    print(f"   - 已加入待处理事件数: {final_stats['processed_events_count']}")  # 原始统计
    print(f"   - 已处理事件: {processed_count}")  # 使用循环中的实际处理数
    print(f"   - 文件映射关系: {final_stats['file_mappings_count']}")
    print(f"   - Worklist 剩余: {final_stats['worklist_size']}")
    
    if manager.file_mapping:
        print(f"\n📁 文件映射关系:")
        for derived, original in manager.file_mapping.items():
            print(f"   {derived}")
            print(f"   └─ 派生自: {original}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
