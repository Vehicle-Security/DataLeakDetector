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
        """获取忽略的路径模式 - 放宽规则以捕获更多文件事件 (修复 Case 48/49/50)"""
        return [
            # 系统临时目录
            "\\Windows\\Temp\\",
            "\\$Recycle.Bin\\",
            "\\System Volume Information\\",
            # 开发相关目录
            "\\.git\\",
            "\\node_modules\\",
            "\\__pycache__\\",
            # 浏览器缓存（仍然过滤）
            "\\AppData\\Local\\Microsoft\\Edge\\User Data\\",
            "\\AppData\\Local\\Google\\Chrome\\User Data\\",
            # 注意：移除了 \\AppData\\Local\\Temp\\ 和 \\AppData\\Roaming\\Microsoft\\
            # 因为某些应用（如WPS、QQ）可能在这些路径下保存重要文件
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
            # 对于重命名事件，获取目标文件的大小
            check_path = dest_path if dest_path and os.path.exists(dest_path) else src_path
            file_size = os.path.getsize(check_path) if os.path.exists(check_path) else 0
        except:
            file_size = 0
        
        basename = os.path.basename(src_path)
        _, ext = os.path.splitext(src_path)
        drive = os.path.splitdrive(src_path)[0]
        
        # 获取进程和窗口信息
        process_info, window_info = self._get_process_info()
        
        # 规范化应用名称
        app_name = self._normalize_app_name(process_info.get("process_name", ""))
        
        # 检查敏感文件
        upload_detection = self._check_sensitive_file(basename, src_path, app_name)
        
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
            "app_name": app_name,
        }
        
        # 添加 upload_detection（如果是敏感文件）
        if upload_detection:
            event["upload_detection"] = upload_detection
        
        # 添加 extra 对象（与 Mac 格式一致）
        event["extra"] = {
            "raw_operation": event_type,
            "category": "",
            "source": "watchdog_fs_monitor"
        }
        
        # 重命名事件：添加目标路径和文件名
        if dest_path:
            dest_basename = os.path.basename(dest_path)
            _, dest_ext = os.path.splitext(dest_path)
            event["destination_path"] = dest_path
            event["destination_name"] = dest_basename
            event["destination_extension"] = dest_ext
        
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
    
    def on_opened(self, event: FileSystemEvent):
        """
        文件打开事件 - 需要 watchdog 3.0+ 或 Windows ReadDirectoryChangesW
        注意: 标准 watchdog 可能不会触发此事件
        """
        if not event.is_directory:
            self._emit_event(EventType.OPENED, event.src_path)
    
    def _normalize_app_name(self, process_name: str) -> str:
        """规范化应用名称"""
        if not process_name:
            return ""
        
        # 移除 .exe 后缀
        if process_name.lower().endswith('.exe'):
            process_name = process_name[:-4]
        
        # 常见应用名称映射
        app_name_map = {
            "chrome": "Chrome",
            "msedge": "Edge",
            "firefox": "Firefox",
            "explorer": "Explorer",
            "notepad": "记事本",
            "code": "VS Code",
            "wechat": "微信",
            "qq": "QQ",
        }
        
        return app_name_map.get(process_name.lower(), process_name)
    
    def _check_sensitive_file(self, file_name: str, file_path: str, app_name: str):
        """检查是否为敏感文件，返回 upload_detection 对象"""
        if not file_name:
            return None
        
        # 敏感关键字（与 Mac 保持一致）
        sensitive_keywords = [
            "合同", "机密", "密码", "password", "secret", "private",
            "财务", "工资", "薪资", "银行", "账号", "证件",
            "身份证", "护照", "驾照", "简历", "resume"
        ]
        
        file_name_lower = file_name.lower()
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


class RecentFilesMonitor:
    """监控 Windows 最近文件夹来检测文件打开事件"""
    
    def __init__(self, event_callback=None):
        self.event_callback = event_callback
        self.is_running = False
        self._monitor_thread = None
        self._known_files = set()
        self._recent_path = os.path.join(os.environ.get("APPDATA", ""), 
                                          "Microsoft", "Windows", "Recent")
        self.username = os.environ.get("USERNAME", "Unknown")
        self.hostname = socket.gethostname()
    
    def start(self):
        """启动监控"""
        if self.is_running:
            return
        
        if not os.path.exists(self._recent_path):
            print(f"[RECENT_MONITOR] Recent folder not found: {self._recent_path}")
            return
        
        # 初始化已知文件列表
        self._known_files = set(os.listdir(self._recent_path))
        
        self.is_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print(f"[RECENT_MONITOR] Started watching: {self._recent_path}")
    
    def stop(self):
        """停止监控"""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        print("[RECENT_MONITOR] Stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                current_files = set(os.listdir(self._recent_path))
                new_files = current_files - self._known_files
                
                for lnk_file in new_files:
                    if lnk_file.endswith('.lnk'):
                        self._process_lnk_file(lnk_file)
                
                self._known_files = current_files
            except Exception as e:
                print(f"[RECENT_MONITOR] Error: {e}")
            
            time.sleep(1)
    
    def _process_lnk_file(self, lnk_filename):
        """处理 .lnk 快捷方式文件，提取原始文件路径"""
        try:
            import pythoncom
            from win32com.shell import shell
            
            lnk_path = os.path.join(self._recent_path, lnk_filename)
            
            pythoncom.CoInitialize()
            try:
                shortcut = pythoncom.CoCreateInstance(
                    shell.CLSID_ShellLink, None,
                    pythoncom.CLSCTX_INPROC_SERVER,
                    shell.IID_IShellLink
                )
                shortcut.QueryInterface(pythoncom.IID_IPersistFile).Load(lnk_path)
                target_path = shortcut.GetPath(0)[0]
                
                if target_path and os.path.isfile(target_path):
                    self._emit_opened_event(target_path)
            finally:
                pythoncom.CoUninitialize()
                
        except ImportError:
            # pywin32 not available, extract from filename
            original_name = lnk_filename[:-4]  # Remove .lnk
            print(f"[RECENT_MONITOR] Opened (name only): {original_name}")
        except Exception as e:
            print(f"[RECENT_MONITOR] Failed to resolve lnk: {e}")
    
    def _emit_opened_event(self, file_path):
        """发送文件打开事件"""
        if not self.event_callback:
            return
        
        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        except:
            file_size = 0
        
        basename = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        drive = os.path.splitdrive(file_path)[0]
        
        event = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "event_type": EventType.OPENED,
            "file_path": file_path,
            "file_name": basename,
            "file_size": file_size,
            "file_extension": ext,
            "process_info": {
                "pid": "",
                "process_name": "",
                "process_path": "",
                "cmdline": ""
            },
            "window_info": {
                "window_handle": "",
                "window_title": "",
                "window_class": ""
            },
            "user_info": {
                "username": self.username,
                "hostname": self.hostname
            },
            "disk_info": {
                "drive_letter": drive,
                "disk_type": "Fixed"
            },
            "app_name": "",
            "extra": {
                "raw_operation": "opened",
                "category": "",
                "source": "recent_folder_monitor"
            }
        }
        
        try:
            self.event_callback(event)
            print(f"📂 [opened] {basename}")
        except Exception as e:
            print(f"[RECENT_MONITOR] Callback error: {e}")


class FileSystemMonitor:
    """基于 watchdog 的文件系统监控器"""
    
    def __init__(self, config=None, event_callback=None, stats=None):
        self.config = config or {}
        self.event_callback = event_callback
        self.stats = stats
        self.observer = None
        self.is_running = False
        
        # 添加最近文件监控器用于检测文件打开
        self.recent_monitor = RecentFilesMonitor(event_callback=event_callback)
        
        # 默认监控路径
        self.watch_paths = self._get_watch_paths()
    
    def _get_watch_paths(self):
        """获取要监控的路径 - 扩展监控范围 (修复 Case 48/49/50)"""
        user_profile = os.environ.get("USERPROFILE", "")
        
        paths = []
        if user_profile:
            paths.extend([
                os.path.join(user_profile, "Desktop"),
                os.path.join(user_profile, "Documents"),
                os.path.join(user_profile, "Downloads"),
                os.path.join(user_profile, "Videos"),  # 屏幕录制常保存在此
                os.path.join(user_profile, "Pictures"),  # 截图可能保存在此
            ])
            # 添加常用应用数据目录
            appdata_local = os.environ.get("LOCALAPPDATA", "")
            if appdata_local:
                # WPS、QQ 等应用的临时/缓存目录
                paths.append(os.path.join(appdata_local, "Kingsoft"))  # WPS
                paths.append(os.path.join(appdata_local, "Tencent"))   # QQ/微信
        
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
        
        # 启动文件打开检测
        self.recent_monitor.start()
        
        print(f"[FS_MONITOR] Started with {len(self.watch_paths)} paths")
    
    def stop(self):
        """停止监控"""
        if not self.is_running:
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=3)
            self.observer = None
        
        # 停止文件打开检测
        self.recent_monitor.stop()
        
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
