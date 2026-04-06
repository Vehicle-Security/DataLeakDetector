# -*- coding: utf-8 -*-
"""
app_logger.py - 统一日志模块

提供标准化的日志记录功能，替代分散的 print() 调用。
支持同时输出到控制台和日志文件。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class AppLogger:
    """统一日志记录器"""
    
    _instance: Optional['AppLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if AppLogger._initialized:
            return
        
        AppLogger._initialized = True
        
        # 创建主日志记录器
        self.logger = logging.getLogger("win_monitor")
        self.logger.setLevel(logging.DEBUG)
        
        # 防止日志重复
        if self.logger.handlers:
            return
        
        # 控制台处理器 - 带颜色和 emoji
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（可选，稍后初始化）
        self._file_handler: Optional[logging.FileHandler] = None
    
    def setup_file_logging(self, log_dir: str):
        """设置文件日志"""
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
        
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"monitor_{datetime.now().strftime('%Y%m%d')}.log")
        
        self._file_handler = logging.FileHandler(log_file, encoding='utf-8')
        self._file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self._file_handler.setFormatter(file_formatter)
        self.logger.addHandler(self._file_handler)
        
        self.logger.info(f"文件日志已启用: {log_file}")
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(msg, *args, **kwargs)


class ColoredFormatter(logging.Formatter):
    """带颜色的控制台格式化器"""
    
    # ANSI 颜色代码（Windows 10+ 支持）
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'
    }
    
    # Emoji 映射
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥'
    }
    
    def format(self, record):
        # 保存原始 levelname
        levelname = record.levelname
        
        # 添加颜色和 emoji（仅在终端中）
        if sys.stdout.isatty():
            color = self.COLORS.get(levelname, '')
            reset = self.COLORS['RESET']
            emoji = self.EMOJIS.get(levelname, '')
            record.msg = f"{emoji} {color}{record.msg}{reset}"
        
        return super().format(record)


# 全局日志实例
_logger: Optional[AppLogger] = None


def get_logger() -> AppLogger:
    """获取全局日志实例"""
    global _logger
    if _logger is None:
        _logger = AppLogger()
    return _logger


# 便捷函数
def debug(msg: str, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)

def exception(msg: str, *args, **kwargs):
    get_logger().exception(msg, *args, **kwargs)
