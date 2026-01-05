# -*- coding: utf-8 -*-
"""
win_monitor.core - 核心监控模块

架构设计（Sensor 模式）：
- sensor.py: 传感器，获取原始数据
- rule_matcher.py: 规则匹配器，纯布尔逻辑
- logger.py: 结构化日志，Mac Protocol 兼容
- engine.py: 引擎，状态机控制器
"""

from .sensor import Sensor, WindowData
from .rule_matcher import RuleMatcher, MatchResult
from .logger import Logger, LogEntry
from .engine import Engine, State

__all__ = [
    # 传感器
    'Sensor', 'WindowData',
    # 规则匹配
    'RuleMatcher', 'MatchResult',
    # 日志
    'Logger', 'LogEntry',
    # 引擎
    'Engine', 'State',
]
