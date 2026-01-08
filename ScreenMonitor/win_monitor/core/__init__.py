# -*- coding: utf-8 -*-
"""
core - win_monitor 核心模块
统一导出所有子模块，保持向后兼容
"""

# monitors - 监控模块
from .monitors.engine import Engine
from .monitors.sensor import Sensor, WindowData, ProcessTracker, WindowSpy
from .monitors.file_system_monitor import FileSystemMonitor, get_file_system_monitor, EventType
from .monitors.browser_file_monitor import BrowserFileMonitor
from .monitors.screen_recorder import ScreenRecorder

# detection - 检测模块
from .detection.rule_matcher import RuleMatcher, MatchResult
from .detection.file_dialog_detector import FileDialogDetector
from .detection.upload_detector import UploadDetector
from .detection.recent_file_tracker import RecentFileTracker, get_recent_file_tracker

# logging - 日志模块
from .logging.logger import Logger, LogEntry
from .logging.key_log_extractor import KeyLogExtractor
from .logging.processor import EventBatchProcessor

# utils - 工具模块
from .utils.stats import StatisticsCollector

__all__ = [
    # monitors
    'Engine', 'Sensor', 'WindowData', 'ProcessTracker', 'WindowSpy',
    'FileSystemMonitor', 'get_file_system_monitor', 'EventType',
    'BrowserFileMonitor', 'ScreenRecorder',
    # detection
    'RuleMatcher', 'MatchResult', 'FileDialogDetector',
    'UploadDetector', 'RecentFileTracker', 'get_recent_file_tracker',
    # logging
    'Logger', 'LogEntry', 'KeyLogExtractor', 'EventBatchProcessor',
    # utils
    'StatisticsCollector',
]
