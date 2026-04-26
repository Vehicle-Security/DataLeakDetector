# -*- coding: utf-8 -*-
"""
core - win_monitor 核心模块
统一导出所有子模块
"""

# monitors - 监控模块
from .monitors.engine import Engine
from .monitors.sensor import Sensor, WindowData
from .monitors.file_system_monitor import FileSystemMonitor
from .monitors.clipboard_monitor import ClipboardMonitor

# ETW 监控器（可选，需要 pywintrace）
try:
    from .monitors.etw_file_monitor import ETWFileMonitor
except ImportError:
    ETWFileMonitor = None

# detection - 检测模块
from .detection.rule_matcher import RuleMatcher, MatchResult

# logging - 日志模块
from .logging.logger import Logger
from .logging.key_log_extractor import KeyLogExtractor

__all__ = [
    # monitors
    'Engine', 'Sensor', 'WindowData', 'FileSystemMonitor',
    'ClipboardMonitor', 'ETWFileMonitor',
    # detection
    'RuleMatcher', 'MatchResult',
    # logging
    'Logger', 'KeyLogExtractor',
]
