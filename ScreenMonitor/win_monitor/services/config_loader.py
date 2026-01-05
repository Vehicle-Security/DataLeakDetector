# -*- coding: utf-8 -*-
"""
config_loader.py - YAML 配置加载器
职责：加载 config.yaml 并提供快速查找接口

对应架构角色：Service（服务层）
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import yaml


@dataclass
class AppConfig:
    """应用配置"""
    name: str
    aliases: List[str] = field(default_factory=list)
    category: str = ""


@dataclass
class WebsiteConfig:
    """网站配置"""
    domain: str
    name: str
    category: str = ""


@dataclass
class SystemWhitelistConfig:
    """系统白名单配置"""
    ignore_processes: List[str] = field(default_factory=list)
    ignore_path_prefixes: List[str] = field(default_factory=list)
    correlation_processes: List[str] = field(default_factory=list)


@dataclass
class MonitorSettings:
    """监控设置"""
    poll_interval_ms: int = 500
    buffer_time_seconds: int = 5
    enable_screen_recording: bool = True
    enable_window_monitoring: bool = True


@dataclass
class MonitorConfig:
    """监控配置"""
    blacklist_apps: List[AppConfig] = field(default_factory=list)
    blacklist_websites: List[WebsiteConfig] = field(default_factory=list)
    sensitive_keywords: List[str] = field(default_factory=list)
    sensitive_extensions: List[str] = field(default_factory=list)
    system_whitelist: SystemWhitelistConfig = field(default_factory=SystemWhitelistConfig)
    monitor_settings: MonitorSettings = field(default_factory=MonitorSettings)


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Optional[MonitorConfig] = None
        
        # 快速查找 maps
        self.blacklist_app_map: Dict[str, AppConfig] = {}
        self.blacklist_website_map: Dict[str, WebsiteConfig] = {}
        self.ignore_process_set: Set[str] = set()
        
        self.load_config()
    
    def load_config(self) -> MonitorConfig:
        """从 YAML 文件加载配置"""
        if not os.path.exists(self.config_path):
            print(f"[WARNING] 配置文件不存在: {self.config_path}")
            self.config = MonitorConfig()
            return self.config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] 加载配置文件失败: {e}")
            self.config = MonitorConfig()
            return self.config
        
        # 解析配置
        blacklist_apps = [
            AppConfig(
                name=app.get('name', ''),
                aliases=app.get('aliases', []),
                category=app.get('category', '')
            )
            for app in data.get('blacklist_apps', [])
        ]
        
        blacklist_websites = [
            WebsiteConfig(
                domain=site.get('domain', ''),
                name=site.get('name', ''),
                category=site.get('category', '')
            )
            for site in data.get('blacklist_websites', [])
        ]
        
        whitelist_data = data.get('system_whitelist', {})
        system_whitelist = SystemWhitelistConfig(
            ignore_processes=whitelist_data.get('ignore_processes', []),
            ignore_path_prefixes=whitelist_data.get('ignore_path_prefixes', []),
            correlation_processes=whitelist_data.get('correlation_processes', [])
        )
        
        settings_data = data.get('monitor_settings', {})
        monitor_settings = MonitorSettings(
            poll_interval_ms=settings_data.get('poll_interval_ms', 500),
            buffer_time_seconds=settings_data.get('buffer_time_seconds', 5),
            enable_screen_recording=settings_data.get('enable_screen_recording', True),
            enable_window_monitoring=settings_data.get('enable_window_monitoring', True)
        )
        
        self.config = MonitorConfig(
            blacklist_apps=blacklist_apps,
            blacklist_websites=blacklist_websites,
            sensitive_keywords=data.get('sensitive_keywords', []),
            sensitive_extensions=data.get('sensitive_extensions', []),
            system_whitelist=system_whitelist,
            monitor_settings=monitor_settings
        )
        
        self._build_lookup_maps()
        
        print(f"[INFO] 配置加载成功: {len(blacklist_apps)} 个黑名单应用, "
              f"{len(blacklist_websites)} 个黑名单网站")
        
        return self.config
    
    def _build_lookup_maps(self):
        """构建快速查找 maps"""
        self.blacklist_app_map.clear()
        self.blacklist_website_map.clear()
        self.ignore_process_set.clear()
        
        if not self.config:
            return
        
        # 应用查找 map（包含别名）
        for app in self.config.blacklist_apps:
            self.blacklist_app_map[app.name.lower()] = app
            for alias in app.aliases:
                self.blacklist_app_map[alias.lower()] = app
        
        # 网站查找 map
        for site in self.config.blacklist_websites:
            self.blacklist_website_map[site.domain.lower()] = site
            self.blacklist_website_map[site.name.lower()] = site
        
        # 忽略进程 set
        for proc in self.config.system_whitelist.ignore_processes:
            self.ignore_process_set.add(proc.lower())
    
    def find_blacklist_app(self, process_name: str) -> Optional[AppConfig]:
        """查找黑名单应用"""
        name_lower = process_name.lower()
        
        # 直接匹配
        if name_lower in self.blacklist_app_map:
            return self.blacklist_app_map[name_lower]
        
        # 移除 .exe 后缀再匹配
        if name_lower.endswith('.exe'):
            name_without_ext = name_lower[:-4]
            if name_without_ext in self.blacklist_app_map:
                return self.blacklist_app_map[name_without_ext]
        
        # 模糊匹配
        for key, app in self.blacklist_app_map.items():
            if key in name_lower or name_lower in key:
                return app
        
        return None
    
    def find_blacklist_website_in_title(self, window_title: str) -> Optional[WebsiteConfig]:
        """在窗口标题中查找黑名单网站"""
        title_lower = window_title.lower()
        
        for key, site in self.blacklist_website_map.items():
            if key in title_lower:
                return site
        
        return None
    
    def should_ignore_process(self, process_name: str) -> bool:
        """检查进程是否应该被忽略"""
        name_lower = process_name.lower()
        if name_lower.endswith('.exe'):
            name_lower = name_lower[:-4]
        return name_lower in self.ignore_process_set
    
    def get_poll_interval_seconds(self) -> float:
        """获取轮询间隔（秒）"""
        if self.config:
            return self.config.monitor_settings.poll_interval_ms / 1000.0
        return 0.5
    
    def get_buffer_time_seconds(self) -> int:
        """获取缓冲时间（秒）"""
        if self.config:
            return self.config.monitor_settings.buffer_time_seconds
        return 5


# 便捷函数
_default_loader: Optional[ConfigLoader] = None


def load_config(config_path: str = "config.yaml") -> MonitorConfig:
    """加载配置文件"""
    global _default_loader
    _default_loader = ConfigLoader(config_path)
    return _default_loader.config


def get_config_loader() -> Optional[ConfigLoader]:
    """获取默认配置加载器"""
    return _default_loader
