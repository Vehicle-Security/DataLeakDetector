# -*- coding: utf-8 -*-
"""
core - win_monitor 核心模块
统一导出所有子模块
"""

# monitors - 监控模块
from .monitors.engine import Engine
from .monitors.sensor import Sensor, WindowData
from .monitors.screen_recorder import ScreenRecorder
from .monitors.file_system_monitor import FileSystemMonitor

# detection - 检测模块
from .detection.rule_matcher import RuleMatcher, MatchResult

# logging - 日志模块
from .logging.logger import Logger
from .logging.key_log_extractor import KeyLogExtractor
from .logging.processor import EventBatchProcessor

__all__ = [
    # monitors
    'Engine', 'Sensor', 'WindowData', 'ScreenRecorder', 'FileSystemMonitor',
    # detection
    'RuleMatcher', 'MatchResult',
    # logging
    'Logger', 'KeyLogExtractor', 'EventBatchProcessor',
]
