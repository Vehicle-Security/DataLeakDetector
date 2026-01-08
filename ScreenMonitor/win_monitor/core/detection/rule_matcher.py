# -*- coding: utf-8 -*-
"""
rule_matcher.py - 规则匹配器
职责：纯布尔逻辑判断（输入数据 + 配置 → True/False）
不做任何数据采集，只负责匹配

对应架构角色：Matcher（匹配器）
"""

from dataclasses import dataclass
from typing import Optional

from ..monitors.sensor import WindowData


@dataclass
class MatchResult:
    """匹配结果"""
    is_match: bool  # 是否命中规则
    app_name: str  # 规范化的应用名称
    category: str  # 分类（即时通讯、AI、网盘等）
    match_type: str  # 匹配类型：app / website / none


# 浏览器进程名列表
BROWSER_PROCESSES = frozenset([
    "chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe",
    "iexplore.exe", "360se.exe", "qqbrowser.exe", "sogouexplorer.exe"
])


class RuleMatcher:
    """
    规则匹配器 - 纯布尔逻辑
    
    输入：WindowData + ConfigLoader
    输出：MatchResult（是否命中、应用名、分类）
    """
    
    def __init__(self, config_loader):
        """
        Args:
            config_loader: ConfigLoader 实例
        """
        self.config = config_loader
    
    def match(self, window_data: WindowData) -> MatchResult:
        """
        检查窗口数据是否命中黑名单规则
        
        匹配优先级：
        1. 进程名是否在黑名单应用中
        2. 如果是浏览器，窗口标题是否包含黑名单网站
        
        Args:
            window_data: 传感器采集的窗口数据
            
        Returns:
            MatchResult: 匹配结果
        """
        process_name = window_data.process_name
        window_title = window_data.window_title
        
        # 1. 检查进程是否是黑名单应用
        app = self.config.find_blacklist_app(process_name)
        if app:
            return MatchResult(
                is_match=True,
                app_name=app.name,
                category=app.category,
                match_type="app"
            )
        
        # 2. 如果是浏览器，检查窗口标题中是否包含黑名单网站
        if self._is_browser(process_name):
            website = self.config.find_blacklist_website_in_title(window_title)
            if website:
                return MatchResult(
                    is_match=True,
                    app_name=website.name,
                    category=website.category,
                    match_type="website"
                )
        
        # 3. 无匹配
        return MatchResult(
            is_match=False,
            app_name=self._normalize_app_name(process_name),
            category="",
            match_type="none"
        )
    
    def match_blacklist_app(self, process_name: str) -> bool:
        """
        简单布尔判断：进程是否在黑名单
        
        Args:
            process_name: 进程名
            
        Returns:
            True 如果命中黑名单
        """
        return self.config.find_blacklist_app(process_name) is not None
    
    def match_blacklist_website(self, window_title: str) -> bool:
        """
        简单布尔判断：窗口标题是否包含黑名单网站
        
        Args:
            window_title: 窗口标题
            
        Returns:
            True 如果命中黑名单
        """
        return self.config.find_blacklist_website_in_title(window_title) is not None
    
    def _is_browser(self, process_name: str) -> bool:
        """检查是否是浏览器进程"""
        return process_name.lower() in BROWSER_PROCESSES
    
    def _normalize_app_name(self, process_name: str) -> str:
        """规范化应用名称（移除 .exe 后缀）"""
        if process_name.lower().endswith('.exe'):
            return process_name[:-4]
        return process_name
