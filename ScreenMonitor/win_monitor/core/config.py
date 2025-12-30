# -*- coding: utf-8 -*-
import json
import os
import time
import threading

# 默认配置
DEFAULT_CONFIG = {
    "monitor_settings": {
        "scan_interval": 1.0,
        "max_log_size": 10485760,
        "log_rotation": "hourly",
        "debounce_seconds": 3.0,
        "same_file_cooldown": 10.0,
        "batch_processing_enabled": True,
        "batch_size": 50,
        "batch_interval_ms": 1000
    },
    "filters": {
        "excluded_extensions": [
            ".tmp", ".temp", ".log", ".etl", ".ini", ".dat", ".cache",
            ".pyc", ".pyo", ".swp", "~", ".bak~", ".autosave", ".journal",
            ".lock", ".pid", ".idx", ".pack", ".db-shm", ".db-wal"
        ],
        "excluded_paths": [
            "C:\\Windows", "C:\\Program Files", "C:\\ProgramData",
            "\\AppData\\Local\\Temp", "\\AppData\\Roaming\\Temp",
            "\\AppData\\Local\\Microsoft\\Edge\\",
            "\\AppData\\Local\\Google\\Chrome\\",
            "\\AppData\\Local\\Microsoft\\Windows\\",
            "\\AppData\\Roaming\\Microsoft\\Windows\\Recent\\",
            "\\$RECYCLE.BIN\\", "\\.git\\", "\\.idea\\", "\\__pycache__\\",
            "\\node_modules\\", "\\.vscode\\", "\\.cache\\", "\\.lingma\\",
            "\\User Data\\Default\\", "\\JetBrains\\consentOptions\\",
            "\\EBWebView\\", "\\Cache\\", "\\Code Cache\\", "\\GPUCache\\",
            "\\Service Worker\\", "\\Session Storage\\", "\\Local Storage\\",
            "\\IndexedDB\\"
        ],
        "excluded_filenames": [
            "desktop.ini", "thumbs.db", ".DS_Store", "~$",
            "Preferences", "Local State", "Cookies", "Cookies-journal",
            "QuotaManager", "QuotaManager-journal", "TransportSecurity",
            "CURRENT", "LOCK", "LOG", "MANIFEST", "bak.txt"
        ],
        "included_extensions": [
            ".doc", ".docx", ".xlsx", ".xls", ".pptx", ".ppt",
            ".pdf", ".txt", ".py", ".java", ".cpp", ".c", ".h",
            ".js", ".html", ".css", ".json", ".xml", ".md",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
            ".mp4", ".avi", ".mov", ".zip", ".rar", ".7z"
        ]
    },
    "resource_management": {
        "max_thread_pool_size": 4,
        "memory_limit_mb": 512,
        "handle_cache_timeout": 300,
        "recent_paths_size": 1000
    },
    "error_handling": {
        "enable_error_tracking": True,
        "error_log_file": "errors.log",
        "max_error_log_size": 1048576,
        "auto_recovery_enabled": True,
        "recovery_check_interval": 30
    },
    "log_enrichment": {
        "include_user_info": True,
        "include_process_info": True,
        "include_system_info": True,
        "include_metadata": True,
        "enable_console_colors": True,
        "show_full_paths": True
    },
    "advanced": {
        "enable_debug_logging": False,
        "enable_statistics": True,
        "statistics_report_interval": 300,
        "enable_handle_tracking": False
    },
    "upload_detection": {
        "enabled": True,
        "enable_dialog_detection": True,
        "dialog_check_interval": 0.5,
        "monitored_apps": {
            "qq_weixin": {
                "enabled": True,
                "display_name": "微信/QQ",
                "process_names": ["Weixin.exe", "QQ.exe", "QQScLauncher.exe"],
                "temp_directories": [
                    "%USERPROFILE%\\Documents\\xwechat_files",
                    "%USERPROFILE%\\Documents\\Tencent Files",
                    "%APPDATA%\\Tencent\\Weixin",
                    "%APPDATA%\\Tencent\\QQ"
                ],
                "upload_type": "instant_messaging"
            },
            "quark": {
                "enabled": True,
                "display_name": "夸克网盘",
                "process_names": ["quark.exe", "Quark.exe"],
                "temp_directories": [
                    "%LOCALAPPDATA%\\Quark",
                    "%APPDATA%\\Quark",
                    "%TEMP%\\Quark"
                ],
                "upload_type": "cloud_storage"
            }
        }
    }
}


class ConfigManager:
    """配置管理器 - 支持热加载"""

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self._last_mtime = 0
        self._check_reload()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    return self._merge_config(DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"警告: 加载配置文件失败,使用默认配置: {e}")
        return DEFAULT_CONFIG.copy()

    def _merge_config(self, default, custom):
        """深度合并配置"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _check_reload(self):
        """检查配置文件是否需要重新加载"""

        def reload_worker():
            while True:
                time.sleep(10)
                try:
                    if os.path.exists(self.config_path):
                        mtime = os.path.getmtime(self.config_path)
                        if mtime > self._last_mtime:
                            self._last_mtime = mtime
                            self.config = self.load_config()
                            print("[INFO] 配置文件已重新加载")
                except Exception as e:
                    pass

        thread = threading.Thread(target=reload_worker, daemon=True)
        thread.start()

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default
