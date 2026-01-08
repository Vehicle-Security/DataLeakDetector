#  -*- coding: utf-8 -*-
"""
关键日志提取器 - 从监控日志中提取关键事件
"""
import json
import os
from datetime import datetime


class KeyLogExtractor:
    """从日志文件中提取关键事件"""
    
    def __init__(self, config=None):
        """
        初始化提取器
        
        Args:
            config: 配置对象
        """
        self.config = config or {}
        
        # 从配置获取关键扩展名和应用
        self.key_extensions = self.config.get("log_extraction", {}).get(
            "key_extensions", 
            [".docx", ".doc", ".pdf", ".xlsx", ".txt"]
        )
        self.key_apps = self.config.get("log_extraction", {}).get(
            "key_apps",
            ["QQ", "微信", "WeChat", "浏览器"]
        )
        self.include_events = self.config.get("log_extraction", {}).get(
            "include_events",
            ["opened", "created", "modified", "deleted", "renamed", "file_selected", "upload_detected"]
        )
    
    def extract_key_events(self, log_file_path):
        """
        从日志文件中提取关键事件
        
        Args:
            log_file_path: 日志文件路径
            
        Returns:
            list: 关键事件列表
        """
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"日志文件不存在: {log_file_path}")
        
        key_events = []
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    
                    # 检查是否是关键事件
                    if self._is_key_event(event):
                        key_events.append(event)
                        
                except json.JSONDecodeError as e:
                    print(f"[EXTRACTOR] 警告: 第{line_num}行JSON解析失败: {e}")
                    continue
        
        print(f"[EXTRACTOR] 从 {log_file_path} 提取了 {len(key_events)} 个关键事件")
        return key_events
    
    def _is_key_event(self, event):
        """
        判断事件是否是关键事件
        
        Args:
            event: 事件字典
            
        Returns:
            bool: True表示是关键事件
        """
        event_type = event.get("event_type", "")
        detection_method = event.get("detection_method", "")
        file_ext = event.get("file_extension", "").lower()
        file_path = event.get("file_path", "")
        
        # ====================================================================
        # PRIORITY 0: 完全过滤 Temp 目录下的所有 .tmp 临时文件
        # 这些是浏览器上传时的临时缓存,不是用户选择的原始文件
        # ====================================================================
        temp_extensions = [".tmp", ".temp", ".crdownload", ".part"]
        if file_ext in temp_extensions:
            # 检查是否在任何临时目录中
            temp_patterns = [
                "\\AppData\\Local\\Temp\\",
                "\\Windows\\Temp\\",
                "\\Temp\\",
            ]
            for pattern in temp_patterns:
                if pattern in file_path:
                    return False  # 完全过滤临时目录的临时文件
        
        # PRIORITY 1: file_dialog events - ALWAYS keep (user explicitly selected)
        if detection_method == "file_dialog":
            return True
        
        # Direct upload detection
        if event_type == "upload_detected":
            return True
        
        # Filter out system images and assets from WindowsApps
        system_image_patterns = [
            "\\WindowsApps\\",
            "\\Microsoft\\Edge\\",
            "\\Program Files (x86)\\Microsoft\\",
            "\\Program Files\\Microsoft\\",
        ]
        if any(pattern in file_path for pattern in system_image_patterns):
            # System images - skip
            if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg']:
                return False
        
        # Priority 2: Has app_name (upload target identified)
        # Keep these UNLESS they are browser-specific cache files
        if event.get("app_name"):
            # Only filter out browser cache (not general Temp folder)
            browser_cache_patterns = [
                "\\GPUCache\\",
                "\\Code Cache\\",
                "\\Service Worker\\",
                "\\DawnCache\\",
                "scoped_dir"
            ]
            if any(pattern in file_path for pattern in browser_cache_patterns):
                return False
            # Keep events with app_name (already filtered .tmp from Temp above)
            return True
        
        # Priority 3: Events from browser_file_monitor
        # These are detected file accesses, very valuable
        if detection_method == "browser_file_monitor":
            # Filter only obvious browser cache
            browser_cache_patterns = [
                "\\GPUCache\\",
                "\\Code Cache\\",
                "\\Service Worker\\",
                "scoped_dir"
            ]
            if any(pattern in file_path for pattern in browser_cache_patterns):
                return False
            # .tmp files from Temp already filtered above
            return True
        
        # Priority 4: Filter remaining temp files without context
        if file_ext in temp_extensions:
            return False
        
        # Priority 5: Standard event type check
        if event_type not in self.include_events:
            return False
        
        # For DLP: if no extension filter defined, accept all
        if not self.key_extensions:
            return True
        
        # Check file extension
        if file_ext and file_ext not in self.key_extensions:
            return False
        
        return True
    
    def filter_by_app(self, events, app_name):
        """
        按应用名称过滤事件
        
        Args:
            events: 事件列表
            app_name: 应用名称（支持部分匹配）
            
        Returns:
            list: 过滤后的事件列表
        """
        filtered = []
        app_name_lower = app_name.lower()
        
        for event in events:
            # 检查上传事件的app_name字段
            if event.get("event_type") == "upload_detected":
                event_app = event.get("app_name", "").lower()
                if app_name_lower in event_app or event_app in app_name_lower:
                    filtered.append(event)
                    continue
            
            # 检查窗口标题
            window_title = event.get("window_info", {}).get("window_title", "").lower()
            if app_name_lower in window_title:
                filtered.append(event)
                continue
            
            # 检查进程名
            process_name = event.get("process_info", {}).get("process_name", "").lower()
            if app_name_lower in process_name:
                filtered.append(event)
        
        return filtered
    
    def filter_by_file_extension(self, events, extensions):
        """
        按文件扩展名过滤事件
        
        Args:
            events: 事件列表
            extensions: 扩展名列表（如[".docx", ".pdf"]）
            
        Returns:
            list: 过滤后的事件列表
        """
        extensions_lower = [ext.lower() for ext in extensions]
        
        filtered = []
        for event in events:
            file_ext = event.get("file_extension", "").lower()
            if file_ext in extensions_lower:
                filtered.append(event)
            
            # 对于上传事件，也检查uploaded_file的扩展名
            if event.get("event_type") == "upload_detected":
                uploaded_file = event.get("uploaded_file", "")
                if uploaded_file:
                    _, ext = os.path.splitext(uploaded_file)
                    if ext.lower() in extensions_lower:
                        filtered.append(event)
        
        return filtered
    
    def get_upload_events(self, events):
        """
        获取所有上传事件
        
        Args:
            events: 事件列表
            
        Returns:
            list: 上传事件列表
        """
        return [
            event for event in events
            if event.get("event_type") == "upload_detected"
        ]
    
    def group_by_time_window(self, events, window_seconds=5.0):
        """
        按时间窗口分组事件
        
        Args:
            events: 事件列表
            window_seconds: 时间窗口大小（秒）
            
        Returns:
            list: 分组后的事件列表，每组是一个列表
        """
        if not events:
            return []
        
        # 按时间戳排序
        sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
        
        groups = []
        current_group = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            # 获取时间戳
            current_time = self._parse_timestamp(event.get("timestamp", ""))
            group_start_time = self._parse_timestamp(current_group[0].get("timestamp", ""))
            
            if current_time and group_start_time:
                time_diff = (current_time - group_start_time).total_seconds()
                
                if time_diff <= window_seconds:
                    current_group.append(event)
                else:
                    groups.append(current_group)
                    current_group = [event]
            else:
                current_group.append(event)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _parse_timestamp(self, timestamp_str):
        """
        解析时间戳字符串
        
        Args:
            timestamp_str: 时间戳字符串
            
        Returns:
            datetime: 时间对象，失败返回None
        """
        if not timestamp_str:
            return None
        
        try:
            # 支持ISO格式: 2025-12-10T14:10:30.123
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def generate_summary(self, events):
        """
        生成事件摘要统计
        
        Args:
            events: 事件列表
            
        Returns:
            dict: 摘要统计
        """
        summary = {
            "total_events": len(events),
            "event_types": {},
            "file_extensions": {},
            "apps": {},
            "upload_count": 0,
            "time_range": {
                "start": None,
                "end": None
            }
        }
        
        if not events:
            return summary
        
        # 统计事件类型
        for event in events:
            event_type = event.get("event_type", "unknown")
            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1
            
            # 统计文件扩展名
            file_ext = event.get("file_extension", "")
            if file_ext:
                summary["file_extensions"][file_ext] = summary["file_extensions"].get(file_ext, 0) + 1
            
            # 统计应用（对于上传事件）
            if event_type == "upload_detected":
                summary["upload_count"] += 1
                app_name = event.get("app_name", "Unknown")
                summary["apps"][app_name] = summary["apps"].get(app_name, 0) + 1
        
        # 计算时间范围
        timestamps = [self._parse_timestamp(e.get("timestamp", "")) for e in events]
        valid_timestamps = [t for t in timestamps if t]
        
        if valid_timestamps:
            summary["time_range"]["start"] = min(valid_timestamps).isoformat()
            summary["time_range"]["end"] = max(valid_timestamps).isoformat()
        
        return summary


if __name__ == "__main__":
    # 简单测试
    print("=== 关键日志提取器测试 ===")
    
    extractor = KeyLogExtractor()
    
    # 测试日志文件
    test_log = "d:/code/win_monitor/logs/monitor_20251207_19.json"
    
    if os.path.exists(test_log):
        try:
            events = extractor.extract_key_events(test_log)
            print(f"\n提取了 {len(events)} 个关键事件")
            
            # 生成摘要
            summary = extractor.generate_summary(events)
            print("\n事件摘要:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            
            # 获取上传事件
            upload_events = extractor.get_upload_events(events)
            print(f"\n上传事件数量: {len(upload_events)}")
            
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"测试日志文件不存在: {test_log}")
