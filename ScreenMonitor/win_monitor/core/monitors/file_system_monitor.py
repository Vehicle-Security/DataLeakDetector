# -*- coding: utf-8 -*-
"""
file_system_monitor.py - 基于 watchdog 的文件系统监控器
提供完整的文件操作事件：opened, created, modified, deleted, renamed

与 browser_file_monitor.py 配合使用，browser_file_monitor 负责进程关联，
本模块负责捕获所有文件系统事件。

优化:
1. 添加进程关联 - 通过 Sensor 获取当前活动窗口进程
2. 延长事件去重 TTL 到 5 秒
3. 对连续 modified 事件只保留最后一个
"""
import os
import time
import socket
import threading
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class EventType:
    """统一的事件类型定义"""
    OPENED = "opened"
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    FILE_SELECTED = "file_selected"
    UPLOAD_DETECTED = "upload_detected"
    APP_SWITCH = "app_switch"
    WEBSITE_VISIT = "website_visit"


class FileSystemMonitorHandler(FileSystemEventHandler):
    """文件系统事件处理器"""
    
    def __init__(self, config, event_callback, stats=None):
        super().__init__()
        self.config = config or {}
        self.event_callback = event_callback
        self.stats = stats
        self.username = os.environ.get("USERNAME", "Unknown")
        self.hostname = socket.gethostname()
        
        # 进程关联 - 延迟导入避免循环依赖
        self._sensor = None
        
        # 过滤配置
        self.ignore_patterns = self._get_ignore_patterns()
        self.ignore_extensions = self._get_ignore_extensions()
        
        # 去重缓存 - 延长 TTL 到 5 秒
        self._event_cache = {}
        self._cache_ttl = 5.0  # 5秒内相同事件去重
        
        # modified 事件延迟发送（合并连续 modified）
        self._pending_modified = {}
        self._modified_delay = 1.0  # 1秒延迟
        self._flush_thread = None
        self._flush_lock = threading.Lock()
        self._start_flush_thread()
    
    def _get_sensor(self):
        """延迟获取 Sensor 实例"""
        if self._sensor is None:
            try:
                from .sensor import Sensor
                self._sensor = Sensor()
            except ImportError:
                pass
        return self._sensor
    
    def _start_flush_thread(self):
        """启动延迟刷新线程"""
        def flush_worker():
            while True:
                time.sleep(0.5)
                self._flush_pending_modified()
        
        self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self._flush_thread.start()
    
    def _get_ignore_patterns(self):
        """获取忽略的路径模式"""
        return [
            "\\AppData\\Local\\Temp\\",
            "\\Windows\\Temp\\",
            "\\$Recycle.Bin\\",
            "\\System Volume Information\\",
            "\\.git\\",
            "\\node_modules\\",
            "\\__pycache__\\",
            "\\AppData\\Local\\Microsoft\\Edge\\",
            "\\AppData\\Local\\Google\\Chrome\\",
            "\\AppData\\Roaming\\Microsoft\\",
        ]
    
    def _get_ignore_extensions(self):
        """获取忽略的文件扩展名"""
        return [
            '.tmp', '.temp', '.log', '.lock', '.ldb', '.db-wal', '.db-shm',
            '.crdownload', '.part', '.etag', '.cache'
        ]
    
    def _should_ignore(self, path):
        """判断是否应该忽略该路径"""
        # 忽略目录
        if os.path.isdir(path):
            return True
        
        # 忽略路径模式
        for pattern in self.ignore_patterns:
            if pattern in path:
                return True
        
        # 忽略扩展名
        _, ext = os.path.splitext(path)
        if ext.lower() in self.ignore_extensions:
            return True
        
        # 忽略隐藏文件
        basename = os.path.basename(path)
        if basename.startswith('.') or basename.startswith('~$'):
            return True
        
        return False
    
    def _is_duplicate(self, event_type, path):
        """检查是否是重复事件"""
        key = f"{event_type}:{path}"
        now = time.time()
        
        if key in self._event_cache:
            if now - self._event_cache[key] < self._cache_ttl:
                return True
        
        self._event_cache[key] = now
        
        # 清理过期缓存
        expired = [k for k, v in self._event_cache.items() if now - v > self._cache_ttl * 2]
        for k in expired:
            del self._event_cache[k]
        
        return False
    
    def _get_process_info(self):
        """获取当前活动窗口的进程信息"""
        sensor = self._get_sensor()
        if not sensor:
            return {}, {}
        
        try:
            window_data = sensor.get_active_window()
            if window_data:
                process_info = {
                    "pid": str(window_data.process_id),
                    "process_name": window_data.process_name,
                    "process_path": window_data.process_path,
                    "cmdline": ""
                }
                window_info = {
                    "window_handle": str(window_data.window_handle),
                    "window_title": window_data.window_title,
                    "window_class": window_data.window_class
                }
                return process_info, window_info
        except Exception:
            pass
        
        return {
            "pid": "",
            "process_name": "",
            "process_path": "",
            "cmdline": ""
        }, {
            "window_handle": "",
            "window_title": "",
            "window_class": ""
        }
    
    def _build_event(self, event_type, src_path, dest_path=None):
        """构建统一格式的事件（带进程关联）"""
        try:
            file_size = os.path.getsize(src_path) if os.path.exists(src_path) else 0
        except:
            file_size = 0
        
        basename = os.path.basename(src_path)
        _, ext = os.path.splitext(src_path)
        drive = os.path.splitdrive(src_path)[0]
        
        # 获取进程和窗口信息
        process_info, window_info = self._get_process_info()
        
        event = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "event_type": event_type,
            "file_path": src_path,
            "file_name": basename,
            "file_size": file_size,
            "file_extension": ext,
            "process_info": process_info,
            "window_info": window_info,
            "user_info": {
                "username": self.username,
                "hostname": self.hostname
            },
            "disk_info": {
                "drive_letter": drive,
                "disk_type": "Fixed"
            },
            "detection_method": "watchdog_fs_monitor"
        }
        
        if dest_path:
            event["destination_path"] = dest_path
        
        return event
    
    def _emit_event(self, event_type, src_path, dest_path=None):
        """发送事件"""
        if self._should_ignore(src_path):
            return
        
        if self._is_duplicate(event_type, src_path):
            return
        
        event = self._build_event(event_type, src_path, dest_path)
        
        if self.stats:
            self.stats.record_event(event_type)
        
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                print(f"[FS_MONITOR] Event callback error: {e}")
    
    def _flush_pending_modified(self):
        """刷新延迟的 modified 事件"""
        now = time.time()
        to_emit = []
        
        with self._flush_lock:
            expired = []
            for path, (event, timestamp) in self._pending_modified.items():
                if now - timestamp >= self._modified_delay:
                    to_emit.append(event)
                    expired.append(path)
            
            for path in expired:
                del self._pending_modified[path]
        
        # 发送事件（在锁外执行）
        for event in to_emit:
            if self.stats:
                self.stats.record_event(event['event_type'])
            if self.event_callback:
                try:
                    self.event_callback(event)
                except Exception as e:
                    print(f"[FS_MONITOR] Event callback error: {e}")
    
    def on_created(self, event: FileSystemEvent):
        """文件创建事件"""
        if not event.is_directory:
            self._emit_event(EventType.CREATED, event.src_path)
    
    def on_deleted(self, event: FileSystemEvent):
        """文件删除事件"""
        if not event.is_directory:
            self._emit_event(EventType.DELETED, event.src_path)
    
    def on_modified(self, event: FileSystemEvent):
        """文件修改事件 - 使用延迟合并"""
        if event.is_directory:
            return
        
        src_path = event.src_path
        if self._should_ignore(src_path):
            return
        
        # 延迟发送，合并连续的 modified 事件
        with self._flush_lock:
            ev = self._build_event(EventType.MODIFIED, src_path)
            self._pending_modified[src_path] = (ev, time.time())
    
    def on_moved(self, event: FileSystemEvent):
        """文件移动/重命名事件"""
        if not event.is_directory:
            self._emit_event(EventType.RENAMED, event.src_path, event.dest_path)


class FileSystemMonitor:
    """基于 watchdog 的文件系统监控器"""
    
    def __init__(self, config=None, event_callback=None, stats=None):
        self.config = config or {}
        self.event_callback = event_callback
        self.stats = stats
        self.observer = None
        self.is_running = False
        
        # 默认监控路径
        self.watch_paths = self._get_watch_paths()
    
    def _get_watch_paths(self):
        """获取要监控的路径"""
        user_profile = os.environ.get("USERPROFILE", "")
        
        paths = []
        if user_profile:
            paths.extend([
                os.path.join(user_profile, "Desktop"),
                os.path.join(user_profile, "Documents"),
                os.path.join(user_profile, "Downloads"),
            ])
        
        # 从配置中获取额外路径
        extra_paths = self.config.get("monitor_paths", [])
        paths.extend(extra_paths)
        
        # 过滤存在的路径
        return [p for p in paths if os.path.exists(p)]
    
    def start(self):
        """启动监控"""
        if self.is_running:
            print("[FS_MONITOR] Already running")
            return
        
        if not self.watch_paths:
            print("[FS_MONITOR] No valid paths to watch")
            return
        
        handler = FileSystemMonitorHandler(
            self.config, 
            self.event_callback,
            self.stats
        )
        
        self.observer = Observer()
        
        for path in self.watch_paths:
            try:
                self.observer.schedule(handler, path, recursive=True)
                print(f"[FS_MONITOR] Watching: {path}")
            except Exception as e:
                print(f"[FS_MONITOR] Failed to watch {path}: {e}")
        
        self.observer.start()
        self.is_running = True
        print(f"[FS_MONITOR] Started with {len(self.watch_paths)} paths")
    
    def stop(self):
        """停止监控"""
        if not self.is_running:
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=3)
            self.observer = None
        
        self.is_running = False
        print("[FS_MONITOR] Stopped")


# 全局实例
_global_fs_monitor = None


def get_file_system_monitor(config=None, event_callback=None, stats=None):
    """获取全局文件系统监控器实例"""
    global _global_fs_monitor
    if _global_fs_monitor is None:
        _global_fs_monitor = FileSystemMonitor(config, event_callback, stats)
    return _global_fs_monitor


if __name__ == "__main__":
    # 测试
    def print_event(event):
        print(f"[{event['event_type']}] {event['file_path']}")
    
    monitor = FileSystemMonitor(event_callback=print_event)
    monitor.start()
    
    try:
        print("Monitoring... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()
