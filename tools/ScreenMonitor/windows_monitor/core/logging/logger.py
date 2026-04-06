# -*- coding: utf-8 -*-
"""
logger.py - 结构化日志记录器
职责：输出符合 Mac Protocol 的 JSON 日志
确保 Windows 和 Mac 的日志格式完全一致

日志文件结构:
- logs.json: 所有事件的完整日志 (JSON Array)
- keyevents.json: 关键事件摘要 (JSON Array) - 用于 LLM 分析

对应架构角色：Logger（日志器）
"""

import json
import os
import socket
from datetime import datetime
from typing import Optional, IO, Dict, Any, List

from ..utils import app_logger
from .log_contract import normalize_app_name, normalize_event_entry

try:
    import win32api
except ImportError:
    win32api = None


class Logger:
    """
    结构化日志记录器
    
    输出格式严格遵循 Mac Protocol，确保后端无法区分日志来源
    所有事件统一使用嵌套结构: process_info, window_info, user_info, disk_info
    
    日志格式:
    - 使用 JSON Array 格式（整个文件是一个数组）
    - 同时维护 keyevents.json 用于 LLM 分析
    """
    
    def __init__(self):
        self.log_file: Optional[IO] = None
        self.keyevents_file: Optional[IO] = None
        self.start_time: Optional[float] = None
        self._hostname = socket.gethostname()
        self._username = self._get_username()
        self._event_count = 0
        self._keyevent_count = 0
        
        # 内存中存储所有日志条目（用于 JSON Array 格式）
        self._log_entries: List[dict] = []
        self._keyevent_entries: List[dict] = []
        
        # 文件路径
        self._log_path: Optional[str] = None
        self._keyevents_path: Optional[str] = None
    
    def _get_username(self) -> str:
        """获取当前用户名"""
        try:
            if win32api:
                return win32api.GetUserName()
        except Exception:
            pass
        return os.environ.get("USERNAME", "Unknown")
    
    def _load_sensitive_keywords(self) -> list:
        """从 config.yaml 加载敏感关键词"""
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    keywords = config.get('sensitive_keywords', [])
                    if keywords:
                        return keywords
        except Exception:
            pass
        # 默认关键词
        return ["机密", "绝密", "合同", "secret", "confidential", "password", "salary"]
    
    def open(self, output_path: str, start_time: float) -> bool:
        """
        打开日志文件
        
        Args:
            output_path: 日志文件路径 (logs.json)
            start_time: 录制开始时间戳（用于计算相对时间）
            
        Returns:
            True 如果成功打开
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self._log_path = output_path
            # keyevents.json 放在同一目录下
            log_dir = os.path.dirname(output_path)
            self._keyevents_path = os.path.join(log_dir, "keyevents.json")
            
            self.start_time = start_time
            self._event_count = 0
            self._keyevent_count = 0
            self._log_entries = []
            self._keyevent_entries = []
            
            app_logger.info(f"📝 日志文件: {self._log_path}")
            app_logger.info(f"📝 关键事件文件: {self._keyevents_path}")
            
            return True
        except Exception as e:
            app_logger.error(f"[ERROR] 无法初始化日志: {e}")
            return False
    
    def close(self):
        """关闭日志文件 - 将内存中的数据写入 JSON 文件"""
        try:
            # 写入 logs.json
            if self._log_path is not None:
                with open(self._log_path, 'w', encoding='utf-8') as f:
                    json.dump(self._log_entries, f, ensure_ascii=False, indent=2)
                app_logger.info(f"📊 日志已保存: {self._log_path} ({self._event_count} 条)")
            
            # 写入 keyevents.json
            if self._keyevents_path is not None:
                with open(self._keyevents_path, 'w', encoding='utf-8') as f:
                    json.dump(self._keyevent_entries, f, ensure_ascii=False, indent=2)
                app_logger.info(f"🔑 关键事件已保存: {self._keyevents_path} ({self._keyevent_count} 条)")
            
            # 清空内存
            self._log_entries = []
            self._keyevent_entries = []
            self._log_path = None
            self._keyevents_path = None
            
        except Exception as e:
            app_logger.error(f"[ERROR] 保存日志失败: {e}")
    
    def is_open(self) -> bool:
        """检查日志是否已打开"""
        return self._log_path is not None
    
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
        
        # 使用与 Mac 一致的嵌套结构
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "event_type": "website_visit" if match_result.match_type == "website" else "app_switch",
            "file_path": "",
            "file_name": "",
            "file_size": 0,
            "file_extension": "",
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
            "disk_info": {
                "drive_letter": "",
                "disk_type": ""
            },
            "app_name": match_result.app_name,
            "extra": {
                "raw_operation": match_result.match_type,
                "category": match_result.category,
                "source": "window_monitor",
                "risk_level": "高" if match_result.is_match else "",
                "relative_timestamp": relative_ts
            }
        }
        entry = normalize_event_entry(entry)
        
        self._write_entry(entry)
        
        # 如果是高风险事件，同时写入 keyevents
        if match_result.is_match:
            self._write_keyevent(entry, "黑名单应用访问")
            app_logger.warning(f"🚨 [高] {match_result.app_name} - {window_data.window_title[:50]}... ({match_result.category})")
    
    def log_file_event(self, event: dict):
        """
        记录文件系统事件（created/modified/deleted/renamed/opened）
        
        Args:
            event: 文件系统事件字典，来自 FileSystemMonitor
        """
        # 提取进程名称用于 app_name
        proc_info = event.get("process_info", {})
        process_name = proc_info.get("process_name", "")
        app_name = self._normalize_app_name(process_name)

        # 构建与 Mac 完全一致的格式
        entry = {
            "timestamp": event.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]),
            "event_type": event.get("event_type", ""),
            "file_path": event.get("file_path", ""),
            "file_name": event.get("file_name", ""),
            "file_size": event.get("file_size", 0),
            "file_extension": event.get("file_extension", ""),
            "process_info": event.get("process_info", {
                "pid": "", "process_name": "", "process_path": "", "cmdline": ""
            }),
            "window_info": event.get("window_info", {
                "window_handle": "", "window_title": "", "window_class": ""
            }),
            "user_info": event.get("user_info", {
                "username": self._username,
                "hostname": self._hostname
            }),
            "disk_info": event.get("disk_info", {
                "drive_letter": "", "disk_type": ""
            }),
            "app_name": app_name,
        }

        # 添加 extra 对象（与 Mac 一致）
        entry["extra"] = {
            "raw_operation": event.get("event_type", ""),
            "category": "",
            "source": event.get("detection_method", "watchdog_fs_monitor")
        }
        if "destination_path" in event:
            entry["destination_path"] = event.get("destination_path", "")
            entry["destination_name"] = event.get("destination_name", "")
            entry["destination_extension"] = event.get("destination_extension", "")

        entry = normalize_event_entry(entry)

        upload_detection = event.get("upload_detection")
        if not isinstance(upload_detection, dict):
            upload_detection = self._check_sensitive_file(
                entry.get("file_name", ""),
                entry.get("file_path", ""),
                app_name,
            )

        if upload_detection:
            entry["upload_detection"] = upload_detection
            entry = normalize_event_entry(entry)
        
        self._write_entry(entry)
        
        # 检查是否为关键事件 (不仅是敏感文件，还包括其他类型)
        key_reason = self._is_key_event(entry, event)
        if key_reason:
            self._write_keyevent(entry, key_reason)
        elif upload_detection:
            # 备用：如果 _is_key_event 没检测到但有敏感文件
            self._write_keyevent(entry, "敏感文件访问")
        
        # 控制台输出（简化）
        event_emoji = {
            "created": "✨",
            "modified": "✏️",
            "deleted": "❌",
            "renamed": "📦",
            "opened": "📂"
        }
        emoji = event_emoji.get(event.get("event_type", ""), "📄")
        app_logger.info(f"{emoji} [{event.get('event_type', '')}] {event.get('file_name', '')} <- {app_name}")

    def log_raw_event(self, event: dict):
        """
        记录原始事件（来自 FileSystemMonitor 或 ClipboardMonitor）
        直接写入，或进行最小化格式适配以符合 Mac Protocol
        """
        # 提取应用名称
        proc_info = event.get("process_info", {})
        process_name = proc_info.get("process_name", "")
        app_name = event.get("app_name") or self._normalize_app_name(process_name)

        # 构建标准格式
        entry = {
            "timestamp": event.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]),
            "event_type": event.get("event_type", "unknown"),
            "file_path": event.get("file_path", ""),
            "file_name": event.get("file_name", ""),
            "file_size": event.get("file_size", 0),
            "file_extension": event.get("file_extension", ""),
            "process_info": proc_info,
            "window_info": event.get("window_info", {
                "window_handle": "", "window_title": "", "window_class": ""
            }),
            "user_info": event.get("user_info", {
                "username": self._username,
                "hostname": self._hostname
            }),
            "disk_info": event.get("disk_info", {
                "drive_letter": "", "disk_type": ""
            }),
            "app_name": app_name,
        }
        if "destination_path" in event:
            entry["destination_path"] = event.get("destination_path", "")
            entry["destination_name"] = event.get("destination_name", "")
            entry["destination_extension"] = event.get("destination_extension", "")

        entry = normalize_event_entry(entry)

        # 敏感信息检测 (如果 Monitor 没做)
        if "upload_detection" in event:
             entry["upload_detection"] = event["upload_detection"]
        elif event.get("event_type") in ["opened", "created", "modified"]:
             det = self._check_sensitive_file(entry["file_name"], entry["file_path"], app_name)
             if det:
                 entry["upload_detection"] = det

        # 剪贴板内容
        if "content_preview" in event:
            entry["content_preview"] = event["content_preview"]
        if "content_hash" in event:
            entry["content_hash"] = event["content_hash"]
            
        # 图片大小
        if "image_size" in event:
            entry["image_size"] = event["image_size"]

        # 确保 extra 存在
        if "extra" in event:
            entry["extra"] = event["extra"]
        else:
            entry["extra"] = {
                "raw_operation": event.get("event_type", ""),
                "category": "",
                "source": event.get("detection_method", "unknown_monitor")
            }
        entry = normalize_event_entry(entry)

        self._write_entry(entry)
        
        # 判断是否为关键事件 (基于场景分析扩展)
        key_reason = self._is_key_event(entry, event)
        
        if key_reason:
            self._write_keyevent(entry, key_reason)
    
    def _normalize_app_name(self, process_name: str) -> str:
        """规范化应用名称"""
        return normalize_app_name(process_name)
    
    def _check_sensitive_file(self, file_name: str, file_path: str, app_name: str) -> Optional[Dict[str, Any]]:
        """检查是否为敏感文件，返回 upload_detection 对象"""
        if not file_name:
            return None
        
        # 敏感关键字 - 从 config 加载
        sensitive_keywords = self._load_sensitive_keywords()
        
        # 敏感扩展名
        sensitive_extensions = [".pem", ".key", ".cert", ".p12", ".pfx"]
        
        file_name_lower = file_name.lower()
        _, ext = os.path.splitext(file_name_lower)
        
        # 检查敏感扩展名
        if ext in sensitive_extensions:
            return {
                "is_upload": True,
                "app_name": app_name,
                "upload_type": "Sensitive File Type",
                "original_file": file_path,
                "temp_directory": ""
            }
        
        # 检查敏感关键字
        for keyword in sensitive_keywords:
            if keyword.lower() in file_name_lower:
                return {
                    "is_upload": True,
                    "app_name": app_name,
                    "upload_type": "File Access",
                    "original_file": file_path,
                    "temp_directory": ""
                }
        
        return None
    
    def _is_key_event(self, entry: dict, original_event: dict) -> Optional[str]:
        """
        判断是否为关键事件，返回原因或 None
        基于场景设计分析，扩展关键事件类型
        """
        event_type = entry.get("event_type", "")
        app_name = entry.get("app_name", "").lower()
        file_name = entry.get("file_name", "").lower()
        window_title = entry.get("window_info", {}).get("window_title", "").lower()
        
        # ========== 1. 敏感文件访问 ==========
        if entry.get("upload_detection"):
            return "敏感文件访问"
        
        # ========== 2. 黑名单应用访问 ==========
        blacklist_apps = [
            # 即时通讯
            "qq", "wechat", "微信", "钉钉", "dingtalk", "飞书", "feishu", "lark",
            "telegram", "slack", "teams",
            # AI 应用
            "kimi", "doubao", "豆包", "chatgpt", "deepseek", "gemini",
            "通义", "tongyi", "文心", "yiyan", "元宝", "yuanbao",
            "chatbox", "cherry studio", "claude",
            # 网盘
            "百度网盘", "baiduyun", "阿里云盘", "aliyundrive",
            "夸克网盘", "quark", "坚果云", "jianguoyun",
            "wps云盘", "wpsclouddrive",
            # 会议
            "zoom", "腾讯会议", "tencent meeting",
            # 办公 (WPS 打开敏感文件时需要记录)
            "wps", "wpsoffice", "et", "wpp",
        ]
        for bl_app in blacklist_apps:
            if bl_app in app_name or bl_app in window_title:
                if event_type in ["app_switch", "website_visit"]:
                    return f"黑名单应用: {entry.get('app_name', '')}"
        
        # ========== 3. 剪贴板操作 ==========
        # clipboard_monitor.py 发送: clipboard_text, clipboard_image
        if event_type in ["clipboard_text", "clipboard_image", "clipboard_copy", "clipboard_paste"]:
            return "剪贴板操作"
        
        # ========== 4. 文件上传推断 ==========
        if event_type == "inferred_upload":
            return "推断上传操作"
        
        # ========== 4.5 文件对话框选择（上传场景）==========
        if event_type == "file_selected":
            return "文件选择/上传"
        

        # ========== 5. 文件重命名 ==========
        if event_type == "renamed":
            return "文件重命名"
        
        # ========== 6. 压缩文件操作 ==========
        if file_name.endswith((".zip", ".rar", ".7z", ".tar", ".gz")):
            if event_type in ["created", "modified"]:
                return "压缩文件创建"
        
        # ========== 7. 屏幕录制/截图 ==========
        screen_keywords = ["screenshot", "screen", "录屏", "截图", "capture", "snip"]
        for kw in screen_keywords:
            if kw in file_name:
                return "屏幕录制/截图"
        
        # ========== 8. U盘/移动存储 ==========
        drive_letter = entry.get("disk_info", {}).get("drive_letter", "")
        if drive_letter and drive_letter not in ["C:", "D:", ""]:
            if event_type in ["created", "modified", "moved"]:
                return f"移动存储操作 ({drive_letter})"
        
        # ========== 9. 邮件附件 (扩展关键词) ==========
        email_keywords = [
            "mail", "邮箱", "邮件", "outlook", 
            "163", "126", "qq邮箱", "qqmail", "foxmail",
            "gmail", "yahoo", "hotmail"
        ]
        for kw in email_keywords:
            if kw in window_title or kw in app_name:
                if event_type in ["opened", "created", "app_switch", "website_visit"]:
                    return "邮箱操作"
        
        # ========== 10. AI应用文件上传 ==========
        ai_apps = [
            "kimi", "doubao", "豆包", "chatgpt", "deepseek", "gemini",
            "通义", "tongyi", "文心", "yiyan", "claude", "元宝"
        ]
        for ai_app in ai_apps:
            if ai_app in window_title or ai_app in app_name:
                if event_type in ["opened", "created", "app_switch", "website_visit"]:
                    return f"AI应用操作: {ai_app}"
        
        # ========== 11. WPS 打开文件 ==========
        wps_apps = ["wps", "wpsoffice", "et", "wpp"]
        process_name = entry.get("process_info", {}).get("process_name", "").lower()
        for wps_app in wps_apps:
            if wps_app in process_name or wps_app in app_name:
                if event_type in ["opened", "created", "modified"]:
                    return "WPS文件操作"
        
        return None
    
    def _write_entry(self, entry: dict):
        """写入日志条目到内存缓存（JSON Array 格式）"""
        if not self._log_path:
            return
        
        try:
            normalized_entry = normalize_event_entry(entry)
            self._log_entries.append(normalized_entry)
            self._event_count += 1
            
            # 每 100 条自动保存一次，防止意外丢失
            if self._event_count % 100 == 0:
                self._flush_logs()
                
        except Exception as e:
            app_logger.error(f"[ERROR] 写入日志失败: {e}")
    
    def _write_keyevent(self, entry: dict, reason: str):
        """写入关键事件 - 直接使用原始日志格式，keyevents 是 logs 的子集"""
        if not self._keyevents_path:
            return
        
        try:
            normalized_entry = normalize_event_entry(entry, drop_invalid_file_event=True)
            if normalized_entry is None:
                app_logger.warning(
                    "⚠️ 跳过 file_path 无法修复的关键事件: "
                    f"type={entry.get('event_type', '')}, "
                    f"path={entry.get('file_path', '')}, "
                    f"process_path={entry.get('process_info', {}).get('process_path', '')}"
                )
                return

            # 直接使用原始日志条目，保持与 logs.json 相同的格式
            # keyevents.json 是 logs.json 的子集
            self._keyevent_entries.append(normalized_entry)
            self._keyevent_count += 1
            
            # 控制台输出
            app_name = normalized_entry.get("app_name", "")
            file_name = normalized_entry.get("file_name", "")
            window_title = normalized_entry.get("window_info", {}).get("window_title", "")[:30] if normalized_entry.get("window_info") else ""
            app_logger.info(f"🔑 关键事件: [{reason}] {app_name} - {file_name or window_title}")
            
        except Exception as e:
            app_logger.error(f"[ERROR] 写入关键事件失败: {e}")
    
    def _flush_logs(self):
        """将内存中的日志刷新到文件（防止丢失）"""
        try:
            if self._log_path and self._log_entries:
                with open(self._log_path, 'w', encoding='utf-8') as f:
                    json.dump(self._log_entries, f, ensure_ascii=False, indent=2)
            
            if self._keyevents_path and self._keyevent_entries:
                with open(self._keyevents_path, 'w', encoding='utf-8') as f:
                    json.dump(self._keyevent_entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            app_logger.warning(f"[WARN] 刷新日志失败: {e}")
    
    def get_event_count(self) -> int:
        """获取已记录的事件数量"""
        return self._event_count
    
    def get_keyevent_count(self) -> int:
        """获取关键事件数量"""
        return self._keyevent_count
