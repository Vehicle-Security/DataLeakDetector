# run_worklist.py
"""
Worklist 使用示例
演示如何在模块3中调用模块2的worklist功能
"""

import json
from worklist_manager import WorklistManager, load_log_from_json


def example_basic_usage():
    """
    基本使用示例
    """
    print("=" * 60)
    print("示例1: 基本使用流程")
    print("=" * 60)
    
    # 1. 创建 WorklistManager，传入敏感文件列表
    sensitive_files = [
        "/Users/admin/Documents/secret.pdf",
        "/Users/admin/Documents/confidential.docx",
        "/home/user/sensitive_data.txt"
    ]
    
    manager = WorklistManager(sensitive_files)
    print(f"\n✓ 初始化完成，监控 {len(sensitive_files)} 个敏感文件")
    
    # 2. 模拟日志事件
    log_events = [
        {
            "timestamp": "2026-01-05T15:45:10.941",
            "event_type": "opened",
            "file_path": "/Users/admin/Documents/secret.pdf",
            "file_name": "secret.pdf",
            "process_info": {
                "pid": "12345",
                "process_name": "Preview",
                "process_path": "/Applications/Preview.app"
            }
        },
        {
            "timestamp": "2026-01-05T15:46:20.123",
            "event_type": "opened",
            "file_path": "/Users/admin/Documents/normal.txt",
            "file_name": "normal.txt",
            "process_info": {
                "pid": "12346",
                "process_name": "TextEdit"
            }
        },
        {
            "timestamp": "2026-01-05T15:47:30.456",
            "event_type": "opened",
            "file_path": "/Users/admin/Documents/confidential.docx",
            "file_name": "confidential.docx",
            "process_info": {
                "pid": "12347",
                "process_name": "Microsoft Word"
            }
        }
    ]
    
    # 3. 扫描日志并构建工作列表
    added_count = manager.scan_and_build_worklist(log_events)
    print(f"\n✓ 扫描完成，发现 {added_count} 个敏感事件")
    
    # 4. 查看工作列表状态
    stats = manager.get_statistics()
    print(f"\n工作列表统计:")
    print(f"  - 敏感文件数: {stats['sensitive_files_count']}")
    print(f"  - 待处理事件: {stats['worklist_size']}")
    print(f"  - 事件类型分布: {stats['event_types']}")
    
    # 5. 模块3处理工作列表
    print(f"\n开始处理工作列表...")
    event_count = 0
    while not manager.is_empty():
        event = manager.get_next_event()
        event_count += 1
        print(f"\n  事件 {event_count}:")
        print(f"    - 文件: {event.current_file}")
        print(f"    - 类型: {event.event_type}")
        print(f"    - 进程: {event.process_info.get('process_name', 'N/A')}")
        print(f"    - 时间: {event.timestamp}")
    
    print(f"\n✓ 所有事件处理完成")


def example_file_tracking():
    """
    文件追踪示例（重命名、复制等）
    """
    print("\n\n" + "=" * 60)
    print("示例2: 文件追踪（隐藏行为检测）")
    print("=" * 60)
    
    # 1. 初始化
    manager = WorklistManager(["/Users/admin/secret.pdf"])
    print("\n✓ 监控文件: /Users/admin/secret.pdf")
    
    # 2. 模拟重命名操作
    print("\n场景: 用户将 secret.pdf 重命名为 backup.pdf")
    manager.update_file_mapping("/Users/admin/secret.pdf", "/Users/admin/backup.pdf")
    
    # 3. 检查 backup.pdf 是否是敏感文件
    is_sensitive = manager.is_sensitive_file("/Users/admin/backup.pdf")
    print(f"  backup.pdf 是敏感文件吗? {is_sensitive}")
    
    if is_sensitive:
        original = manager.get_original_file("/Users/admin/backup.pdf")
        print(f"  原始文件: {original}")
    
    # 4. 继续追踪后续操作
    print("\n场景: 用户又将 backup.pdf 压缩为 backup.zip")
    manager.update_file_mapping("/Users/admin/backup.pdf", "/Users/admin/backup.zip")
    
    is_sensitive = manager.is_sensitive_file("/Users/admin/backup.zip")
    print(f"  backup.zip 是敏感文件吗? {is_sensitive}")
    
    if is_sensitive:
        original = manager.get_original_file("/Users/admin/backup.zip")
        print(f"  原始文件: {original}")


def example_dynamic_addition():
    """
    动态添加敏感文件示例
    """
    print("\n\n" + "=" * 60)
    print("示例3: 动态添加敏感文件")
    print("=" * 60)
    
    # 1. 初始化（空列表）
    manager = WorklistManager()
    print(f"\n初始敏感文件数: {len(manager.sensitive_files)}")
    
    # 2. 动态添加敏感文件
    new_sensitive_files = [
        "/home/user/project/credentials.json",
        "/home/user/project/api_keys.txt"
    ]
    
    manager.add_sensitive_files(new_sensitive_files)
    print(f"\n添加后敏感文件数: {len(manager.sensitive_files)}")
    
    # 3. 查询文件
    test_files = [
        "/home/user/project/credentials.json",
        "/home/user/project/readme.md"
    ]
    
    print("\n文件敏感性检查:")
    for file_path in test_files:
        is_sensitive = manager.is_sensitive_file(file_path)
        status = "✓ 敏感" if is_sensitive else "✗ 普通"
        print(f"  {file_path}: {status}")


def example_real_log():
    """
    使用真实日志文件示例
    """
    print("\n\n" + "=" * 60)
    print("示例4: 处理真实日志文件")
    print("=" * 60)
    
    # 指定日志文件路径
    log_file = "/home/dxy/Projects/DataLeakDetector/3-RiskHunter/records/60/key_events/key_events_20260105_154508.json"
    
    # 定义敏感文件（示例）
    sensitive_files = [
        "/Users/tujiali/Documents/项目3prd设计.docx", 
        "/Users/tujiali/Documents/计划.docx"
    ]
    
    # 创建管理器
    manager = WorklistManager(sensitive_files)
    
    try:
        # 加载日志
        print(f"\n加载日志文件: {log_file}")
        events = load_log_from_json(log_file)
        print(f"✓ 共加载 {len(events)} 个事件")
        
        # 扫描并构建工作列表
        added_count = manager.scan_and_build_worklist(events)
        print(f"✓ 发现 {added_count} 个敏感事件")
        
        # 显示统计信息
        stats = manager.get_statistics()
        print(f"\n统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 处理前几个事件作为示例
        print(f"\n处理前5个事件:")
        for i in range(min(5, manager.size())):
            event = manager.get_next_event()
            if event:
                print(f"\n  事件 {i+1}:")
                print(f"    文件: {event.current_file}")
                print(f"    类型: {event.event_type}")
                print(f"    进程: {event.process_info.get('process_name', 'N/A')} (PID: {event.process_info.get('pid', 'N/A')})")
                print(f"    隐藏: {event.is_hidden}")
        
        remaining = manager.size()
        if remaining > 0:
            print(f"\n还有 {remaining} 个事件待处理...")
            
    except FileNotFoundError:
        print(f"✗ 文件未找到: {log_file}")
        print("  请确保路径正确")
    except Exception as e:
        print(f"✗ 错误: {e}")


if __name__ == "__main__":
    # 运行所有示例
    # example_basic_usage()
    # example_file_tracking()
    # example_dynamic_addition()
    example_real_log()
    
    print("\n\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
