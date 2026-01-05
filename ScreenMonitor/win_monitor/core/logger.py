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
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, IO

try:
    import win32api
except ImportError:
    win32api = None

from .sensor import WindowData
from .rule_matcher import MatchResult


@dataclass
class LogEntry:
    """
    日志条目 - 与 Mac_monitor/server/session_manager.go 的 LogEntry 完全一致
    """
    timestamp: str  # ISO8601 格式: "2006-01-02T15:04:05.000"
    event_type: str  # "app_switch" | "website_visit"
    
    # 窗口信息
    window_handle: str
    window_title: str
    window_class: str
    
    # 进程信息
    pid: str
    process_name: str
    process_path: str
    
    # 用户信息
    username: str
    hostname: str
    
    # 匹配结果
    app_name: str
    category: str
    risk_level: str  # "高" | ""
    match_type: str  # "app" | "website" | "none"
    
    # 相对时间戳（可选，用于录制同步）
    relative_timestamp: Optional[float] = None
    
    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Logger:
    """
    结构化日志记录器
    
    输出格式严格遵循 Mac Protocol，确保后端无法区分日志来源
    """
    
    def __init__(self):
        self.log_file: Optional[IO] = None
        self.start_time: Optional[float] = None
        self.first_entry = True
        self._hostname = socket.gethostname()
        self._username = self._get_username()
    
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
    
    def log(self, window_data: WindowData, match_result: MatchResult, current_time: float):
        """
        记录一条日志
        
        Args:
            window_data: 传感器数据
            match_result: 匹配结果
            current_time: 当前时间戳
        """
        # 计算相对时间戳
        relative_ts = None
        if self.start_time:
            relative_ts = round(current_time - self.start_time, 3)
        
        entry = LogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            event_type="website_visit" if match_result.match_type == "website" else "app_switch",
            window_handle=str(window_data.window_handle),
            window_title=window_data.window_title,
            window_class=window_data.window_class,
            pid=str(window_data.process_id),
            process_name=window_data.process_name,
            process_path=window_data.process_path,
            username=self._username,
            hostname=self._hostname,
            app_name=match_result.app_name,
            category=match_result.category,
            risk_level="高" if match_result.is_match else "",
            match_type=match_result.match_type,
            relative_timestamp=relative_ts
        )
        
        self._write_entry(entry)
        
        # 控制台输出
        if match_result.is_match:
            print(f"🚨 [高] {match_result.app_name} - {window_data.window_title[:50]}... ({match_result.category})")
    
    def _write_entry(self, entry: LogEntry):
        """写入日志条目到文件"""
        if not self.log_file:
            return
        
        try:
            if not self.first_entry:
                self.log_file.write(",\n")
            self.log_file.write(entry.to_json())
            self.log_file.flush()
            self.first_entry = False
        except Exception as e:
            print(f"[ERROR] 写入日志失败: {e}")
