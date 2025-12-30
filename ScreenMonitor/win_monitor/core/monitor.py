# -*- coding: utf-8 -*-
import time
import os
import socket
import threading
from datetime import datetime
import win32api
from watchdog.events import FileSystemEventHandler

class MonitorHandler(FileSystemEventHandler):
    """核心事件处理器 - 优化版"""

    def __init__(self, config, batch_processor, window_spy, process_tracker, error_logger, stats_collector, upload_detector=None):
        self.config = config
        self.batch_processor = batch_processor
        self.window_spy = window_spy
        self.process_tracker = process_tracker
        self.error_logger = error_logger
        self.stats = stats_collector
        self.upload_detector = upload_detector

        self.excluded_exts = set(config.get("filters.excluded_extensions", []))
        self.included_exts = set(config.get("filters.included_extensions", []))
        self.excluded_paths = config.get("filters.excluded_paths", [])
        self.excluded_filenames = config.get("filters.excluded_filenames", [])

        self.debounce_cache = {}
        self.debounce_time = config.get("monitor_settings.debounce_seconds", 3.0)

        self.operation_cooldown = {}
        self.cooldown_time = config.get("monitor_settings.same_file_cooldown", 10.0)

        self._start_cache_cleanup()

    def _start_cache_cleanup(self):
        """定期清理过期缓存"""
        def cleanup():
            while True:
                time.sleep(60)
                now = time.time()

                try:
                    expired_debounce = [k for k, v in self.debounce_cache.items()
                                        if now - v > self.debounce_time * 2]
                    for k in expired_debounce:
                        del self.debounce_cache[k]

                    expired_cooldown = [k for k, v in self.operation_cooldown.items()
                                        if now - v > self.cooldown_time * 2]
                    for k in expired_cooldown:
                        del self.operation_cooldown[k]
                except Exception as e:
                    self.error_logger.log_error("cache_cleanup", "缓存清理失败", e)

        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

    def _is_ignored(self, file_path):
        """智能过滤逻辑"""
        try:
            filename = os.path.basename(file_path)

            # 0. 检查是否在上传检测目录中 - 如果是,则不过滤
            if self.upload_detector:
                for app_name, app_config in self.upload_detector.app_configs.items():
                    if not app_config.get("enabled", True):
                        continue
                    temp_dirs = app_config.get("temp_directories", [])
                    for temp_dir in temp_dirs:
                        expanded_dir = os.path.expandvars(temp_dir)
                        if expanded_dir.lower() in file_path.lower():
                            # 在上传监控目录中,不过滤
                            return False

            # 1. 文件名过滤
            for excluded_name in self.excluded_filenames:
                if filename == excluded_name or filename.startswith(excluded_name):
                    return True

            # 2. 路径过滤
            for path in self.excluded_paths:
                if path.lower() in file_path.lower():
                    return True

            # 3. 扩展名过滤
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext in self.excluded_exts:
                return True

            # 4. 白名单逻辑
            if self.included_exts:
                if ext == "":
                    common_no_ext = ["README", "LICENSE", "Makefile", "Dockerfile"]
                    if filename not in common_no_ext:
                        return True
                elif ext not in self.included_exts:
                    return True

            # 5. 特殊模式过滤
            if filename.startswith(("~$", ".tmp")):
                return True

            # 6. 数字文件名过滤
            if filename.replace('-', '').replace('_', '').replace('.', '').isdigit():
                return True

            return False
        except Exception as e:
            self.error_logger.log_error("filter_check", f"过滤检查失败: {file_path}", e)
            return True

    def _should_log_event(self, event_type, src_path):
        """判断是否应该记录此事件"""
        try:
            now = time.time()

            # 防抖动检查
            if event_type == "modified":
                last_time = self.debounce_cache.get(src_path, 0)
                if now - last_time < self.debounce_time:
                    return False
                self.debounce_cache[src_path] = now

            # 操作冷却检查
            operation_key = (src_path, event_type)
            last_operation_time = self.operation_cooldown.get(operation_key, 0)
            if now - last_operation_time < self.cooldown_time:
                return False

            self.operation_cooldown[operation_key] = now
            return True
        except Exception as e:
            self.error_logger.log_error("event_check", "事件检查失败", e)
            return False

    def _enrich_event(self, event_type, src_path, dest_path=None):
        """构建富日志数据"""
        try:
            if self._is_ignored(src_path):
                return

            if not self._should_log_event(event_type, src_path):
                return

            # 记录统计
            if self.stats:
                self.stats.record_event(event_type)

            # 获取窗口和进程信息
            win_info = self.window_spy.get_active_window_info()

            # 获取用户信息
            try:
                username = win32api.GetUserName()
            except Exception:
                username = win_info.get('username', 'Unknown') if win_info else 'Unknown'

            # 获取文件信息
            file_size = 0
            if event_type != "deleted" and os.path.exists(src_path):
                try:
                    file_size = os.path.getsize(src_path)
                except Exception:
                    pass

            # 规范化路径 - 支持长路径
            try:
                abs_path = os.path.abspath(src_path)
                # Windows长路径支持
                if len(abs_path) > 260 and not abs_path.startswith('\\\\?\\'):
                    abs_path = '\\\\?\\' + abs_path
            except Exception:
                abs_path = src_path

            # 实时识别应用名称（新增）
            app_name = None
            if self.upload_detector and win_info:
                window_title = win_info.get("window_title")
                if window_title:
                    app_name = self.upload_detector.identify_upload_target(window_title)

            # 检测上传操作
            upload_info = None
            if self.upload_detector and win_info:
                process_name = win_info.get("process_name")
                upload_info = self.upload_detector.is_temp_file_for_upload(abs_path, process_name)
                
                if upload_info:
                    # 尝试关联原始文件
                    file_name = os.path.basename(src_path)
                    original_path = self.upload_detector.try_associate_original_file(abs_path, file_name)
                    
                    if original_path:
                        upload_info["original_file_path"] = original_path
                    
                    # 记录上传统计
                    if self.stats:
                        self.stats.record_event("file_upload_detected")

            # 构建日志条目
            log_entry = {
                "timestamp": datetime.now().isoformat(timespec='milliseconds'),
                "event_type": event_type,
                "file_path": abs_path,
                "file_name": os.path.basename(src_path),
                "file_size": file_size,
                "file_extension": os.path.splitext(src_path)[1],
                "process_info": {
                    "pid": win_info.get("pid") if win_info else None,
                    "process_name": win_info.get("process_name") if win_info else None,
                    "process_path": win_info.get("process_path") if win_info else None,
                    "cmdline": win_info.get("cmdline") if win_info else None
                },
                "window_info": {
                    "window_handle": win_info.get("window_handle") if win_info else None,
                    "window_title": win_info.get("window_title") if win_info else None,
                    "window_class": win_info.get("window_class") if win_info else None
                },
                "user_info": {
                    "username": username,
                    "hostname": socket.gethostname()
                },
                "disk_info": {
                    "drive_letter": os.path.splitdrive(src_path)[0],
                    "disk_type": "Fixed"
                }
            }
            
            # 添加应用名称字段（新增）
            if app_name:
                log_entry["app_name"] = app_name
            
            # 添加上传信息
            if upload_info:
                log_entry["upload_detection"] = {
                    "is_upload": True,
                    "app_name": upload_info.get("app_display_name"),
                    "upload_type": upload_info.get("upload_type"),
                    "original_file": upload_info.get("original_file_path"),
                    "temp_directory": upload_info.get("temp_directory")
                }

            if dest_path:
                try:
                    abs_dest = os.path.abspath(dest_path)
                    if len(abs_dest) > 260 and not abs_dest.startswith('\\\\?\\'):
                        abs_dest = '\\\\?\\' + abs_dest
                    log_entry["destination_path"] = abs_dest
                except Exception:
                    log_entry["destination_path"] = dest_path

            # 添加到批处理队列
            self.batch_processor.add_event(log_entry)

        except Exception as e:
            self.error_logger.log_error("event_enrichment", f"事件处理失败: {src_path}", e)

    def on_created(self, event):
        if not event.is_directory:
            self._enrich_event("created", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._enrich_event("deleted", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._enrich_event("modified", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._enrich_event("moved", event.src_path, event.dest_path)
