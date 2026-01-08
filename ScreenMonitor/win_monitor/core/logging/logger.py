# -*- coding: utf-8 -*-
"""
logger.py - 结构化日志记录器
职责：输出符合 Mac Protocol 的 JSON 日志
确保 Windows 和 Mac 的日志格式完全一致

对应架构角色：Logger（日志器）
"""

import json
import os
import socket
from datetime import datetime
from typing import Optional, IO, Dict, Any

try:
    import win32api
except ImportError:
    win32api = None


class Logger:
    """
    结构化日志记录器
    
    输出格式严格遵循 Mac Protocol，确保后端无法区分日志来源
    所有事件统一使用嵌套结构: process_info, window_info, user_info, disk_info
    """
    
    def __init__(self):
        self.log_file: Optional[IO] = None
        self.start_time: Optional[float] = None
        self.first_entry = True
        self._hostname = socket.gethostname()
        self._username = self._get_username()
        self._event_count = 0
    
    def _get_username(self) -> str:
        """获取当前用户名"""
        try:
            if win32api:
                return win32api.GetUserName()
        except Exception:
            pass
        return os.environ.get("USERNAME", "Unknown")
    
    def open(self, output_path: str, start_time: float) -> bool:
        """
        打开日志文件
        
        Args:
            output_path: 日志文件路径
            start_time: 录制开始时间戳（用于计算相对时间）
            
        Returns:
            True 如果成功打开
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.log_file = open(output_path, 'w', encoding='utf-8')
            self.log_file.write("[\n")
            self.start_time = start_time
            self.first_entry = True
            self._event_count = 0
            return True
        except Exception as e:
            print(f"[ERROR] 无法打开日志文件: {e}")
            return False
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.write("\n]")
            self.log_file.close()
            self.log_file = None
            print(f"📊 日志已保存，共 {self._event_count} 条记录")
    
    def log(self, window_data, match_result, current_time: float):
        """
        记录窗口切换日志（app_switch/website_visit）
        使用与文件事件相同的嵌套格式，确保前端一致解析
        
        Args:
            window_data: 传感器数据
            match_result: 匹配结果
            current_time: 当前时间戳
        """
        # 计算相对时间戳
        relative_ts = None
        if self.start_time:
            relative_ts = round(current_time - self.start_time, 3)
        
        # 使用与文件事件相同的嵌套结构
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "event_type": "website_visit" if match_result.match_type == "website" else "app_switch",
            "process_info": {
                "pid": str(window_data.process_id),
                "process_name": window_data.process_name,
                "process_path": window_data.process_path,
                "cmdline": ""
            },
            "window_info": {
                "window_handle": str(window_data.window_handle),
                "window_title": window_data.window_title,
                "window_class": window_data.window_class
            },
            "user_info": {
                "username": self._username,
                "hostname": self._hostname
            },
            "app_name": match_result.app_name,
            "category": match_result.category,
            "risk_level": "高" if match_result.is_match else "",
            "match_type": match_result.match_type
        }
        
        if relative_ts is not None:
            entry["relative_timestamp"] = relative_ts
        
        self._write_entry(entry)
        
        # 控制台输出
        if match_result.is_match:
            print(f"🚨 [高] {match_result.app_name} - {window_data.window_title[:50]}... ({match_result.category})")
    
    def log_file_event(self, event: dict):
        """
        记录文件系统事件（created/modified/deleted/renamed/opened）
        
        Args:
            event: 文件系统事件字典，来自 FileSystemMonitor
                   包含: timestamp, event_type, file_path, file_name, file_size,
                         file_extension, process_info, window_info, user_info, disk_info
        """
        if not self.log_file:
            return
        
        # 提取进程名称用于 app_name
        proc_info = event.get("process_info", {})
        process_name = proc_info.get("process_name", "")
        app_name = self._normalize_app_name(process_name)
        
        # 构建统一格式的事件
        entry = {
            "timestamp": event.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]),
            "event_type": event.get("event_type", ""),
            "file_path": event.get("file_path", ""),
            "file_name": event.get("file_name", ""),
            "file_size": event.get("file_size", 0),
            "file_extension": event.get("file_extension", ""),
            "process_info": event.get("process_info", {}),
            "window_info": event.get("window_info", {}),
            "user_info": event.get("user_info", {
                "username": self._username,
                "hostname": self._hostname
            }),
            "disk_info": event.get("disk_info", {}),
            "app_name": app_name
        }
        
        self._write_entry(entry)
        
        # 控制台输出（简化）
        event_emoji = {
            "created": "✨",
            "modified": "✏️",
            "deleted": "❌",
            "renamed": "📦",
            "opened": "📂"
        }
        emoji = event_emoji.get(event.get("event_type", ""), "📄")
        print(f"{emoji} [{event.get('event_type', '')}] {event.get('file_name', '')} <- {app_name}")
    
    def _normalize_app_name(self, process_name: str) -> str:
        """规范化应用名称"""
        if not process_name:
            return ""
        
        # 移除 .exe 后缀
        if process_name.lower().endswith('.exe'):
            process_name = process_name[:-4]
        
        # 常见应用名称映射
        app_name_map = {
            "chrome": "Chrome",
            "msedge": "Edge",
            "firefox": "Firefox",
            "explorer": "Explorer",
            "notepad": "记事本",
            "code": "VS Code",
            "wechat": "微信",
            "qq": "QQ",
        }
        
        return app_name_map.get(process_name.lower(), process_name)
    
    def _write_entry(self, entry: dict):
        """写入日志条目到文件"""
        if not self.log_file:
            return
        
        try:
            if not self.first_entry:
                self.log_file.write(",\n")
            self.log_file.write(json.dumps(entry, ensure_ascii=False))
            self.log_file.flush()
            self.first_entry = False
            self._event_count += 1
        except Exception as e:
            print(f"[ERROR] 写入日志失败: {e}")
    
    def get_event_count(self) -> int:
        """获取已记录的事件数量"""
        return self._event_count
