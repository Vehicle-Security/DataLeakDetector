# -*- coding: utf-8 -*-
"""
etw_file_monitor.py - 基于 ETW 的文件 I/O 监控器
类似于 Mac 的 fs_usage，精确捕获文件打开事件及进程信息

功能:
- 使用 Windows Event Tracing (ETW) 捕获文件 I/O 事件
- 捕获 FileIo_Create (文件打开) 事件
- 提供完整的进程信息 (PID, 进程名)

要求:
- 需要管理员权限运行
- 需要安装: pip install pywintrace

对应架构角色: ETW 文件监控器 (与 watchdog 互补)
"""

import os
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Dict, Any

# ETW 相关常量
KERNEL_FILE_PROVIDER_GUID = "{EDD08927-9CC4-4E65-B970-C2560FB5C289}"

# FileIo 操作类型
FILE_IO_CREATE = 64    # 文件创建/打开
FILE_IO_READ = 67      # 文件读取
FILE_IO_WRITE = 68     # 文件写入
FILE_IO_CLOSE = 66     # 文件关闭


class ETWFileMonitor:
    """
    基于 ETW 的文件 I/O 监控器
    
    使用 NT Kernel Logger 的 FileIo 事件来精确捕获:
    - 文件打开事件 (FileIo_Create)
    - 完整的进程信息 (PID, 进程名)
    
    与 watchdog 的区别:
    - watchdog: 监控文件系统变化 (创建、修改、删除、重命名)
    - ETW: 监控文件 I/O 操作 (打开、读取、写入)
    
    两者互补使用可实现完整的文件监控。
    """
    
    def __init__(self, event_callback: Callable[[Dict[str, Any]], None] = None):
        """
        Args:
            event_callback: 事件回调函数，接收标准化的事件字典
        """
        self.event_callback = event_callback
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._session = None
        
        # 进程名缓存 (避免频繁查询)
        self._process_cache: Dict[int, str] = {}
        self._cache_ttl = 60  # 缓存 60 秒
        self._cache_times: Dict[int, float] = {}
        
        # 事件去重
        self._event_cache: Dict[str, float] = {}
        self._dedup_ttl = 1.0  # 1秒内相同事件去重
        self._cache_lock = threading.Lock()  # 线程安全锁
        
        # 过滤规则
        self.ignore_extensions = [
            '.tmp', '.temp', '.log', '.lock', '.ldb', 
            '.db-wal', '.db-shm', '.etag', '.cache'
        ]
        self.ignore_patterns = [
            '\\Windows\\',
            '\\$Recycle.Bin\\',
            '\\System Volume Information\\',
            '\\.git\\',
        ]
        
        # 敏感文件关键字
        self.sensitive_keywords = [
            "合同", "机密", "密码", "password", "secret", "private",
            "财务", "工资", "薪资", "银行", "账号", "证件",
            "身份证", "护照", "驾照", "简历", "resume"
        ]
    
    def start(self):
        """启动 ETW 文件监控"""
        if self.is_running:
            print("[ETW_MONITOR] Already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_etw_trace, daemon=True)
        self._thread.start()
        print("[ETW_MONITOR] Started with Kernel File I/O tracing")
    
    def stop(self):
        """停止 ETW 文件监控"""
        self.is_running = False
        if self._session:
            try:
                self._session.stop()
            except:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        print("[ETW_MONITOR] Stopped")
    
    def _run_etw_trace(self):
        """运行 ETW 跟踪 (在独立线程中)"""
        try:
            # 尝试使用 pywintrace (正确的 API 用法)
            import etw
            
            # 定义 provider (Microsoft-Windows-Kernel-File)
            providers = [
                etw.ProviderInfo(
                    'Microsoft-Windows-Kernel-File',
                    etw.GUID(KERNEL_FILE_PROVIDER_GUID)
                )
            ]
            
            # 创建 ETW 实例，传入 event_callback
            self._session = etw.ETW(
                providers=providers,
                event_callback=self._process_etw_event
            )
            
            print("[ETW_MONITOR] ETW session started")
            self._session.start()
            
            # 保持运行直到停止
            while self.is_running:
                time.sleep(0.1)
            
            # 停止会话
            self._session.stop()
            
        except ImportError:
            print("[ETW_MONITOR] pywintrace not installed, falling back to Recent Files monitor")
            self._run_wmi_fallback()
        except Exception as e:
            print(f"[ETW_MONITOR] Error: {e}")
            # 尝试备用方案
            self._run_wmi_fallback()
    
    def _run_wmi_fallback(self):
        """文件打开监控备用方案 - 监控 Recent Files 文件夹"""
        # WMI 不能直接监控文件打开，改用监控 Recent 文件夹
        print("[ETW_MONITOR] Fallback: monitoring Recent Files folder")
        
        recent_path = os.path.join(
            os.environ.get("APPDATA", ""),
            "Microsoft", "Windows", "Recent"
        )
        
        if not os.path.exists(recent_path):
            print(f"[ETW_MONITOR] Recent folder not found: {recent_path}")
            return
        
        known_files = set(os.listdir(recent_path))
        
        while self.is_running:
            try:
                current_files = set(os.listdir(recent_path))
                new_files = current_files - known_files
                
                for lnk_file in new_files:
                    if lnk_file.endswith('.lnk'):
                        # 从 .lnk 文件名提取原始文件名
                        original_name = lnk_file[:-4]
                        if self.event_callback:
                            event = self._build_event(
                                'opened', 
                                f"(Recent) {original_name}",
                                0, 
                                ""
                            )
                            self.event_callback(event)
                
                known_files = current_files
                time.sleep(1)
            except Exception as e:
                print(f"[ETW_MONITOR] Fallback error: {e}")
                time.sleep(1)
    
    def _process_etw_event(self, event):
        """处理 ETW 事件"""
        try:
            # 过滤非文件 I/O 事件
            task_name = getattr(event, 'task_name', '')
            if task_name not in ['Create', 'Open', 'Read']:
                return
            
            # 获取文件路径
            file_path = getattr(event, 'FileName', '') or getattr(event, 'file_path', '')
            if not file_path:
                return
            
            # 过滤
            if self._should_ignore(file_path):
                return
            
            # 去重
            if self._is_duplicate('opened', file_path):
                return
            
            # 获取进程信息
            pid = getattr(event, 'ProcessId', 0) or getattr(event, 'process_id', 0)
            process_name = self._get_process_name(pid)
            
            # 构建标准化事件
            file_event = self._build_event('opened', file_path, pid, process_name)
            
            # 回调
            if self.event_callback:
                self.event_callback(file_event)
                
        except Exception as e:
            print(f"[ETW_MONITOR] Event processing error: {e}")
    
    def _should_ignore(self, path: str) -> bool:
        """判断是否应该忽略该路径"""
        # 忽略路径模式
        for pattern in self.ignore_patterns:
            if pattern.lower() in path.lower():
                return True
        
        # 忽略扩展名
        _, ext = os.path.splitext(path)
        if ext.lower() in self.ignore_extensions:
            return True
        
        return False
    
    def _is_duplicate(self, event_type: str, path: str) -> bool:
        """检查是否是重复事件 (线程安全)"""
        key = f"{event_type}:{path}"
        now = time.time()
        
        with self._cache_lock:
            if key in self._event_cache:
                if now - self._event_cache[key] < self._dedup_ttl:
                    return True
            
            self._event_cache[key] = now
            
            # 清理过期缓存
            expired = [k for k, v in self._event_cache.items() if now - v > self._dedup_ttl * 2]
            for k in expired:
                del self._event_cache[k]
        
        return False
    
    def _get_process_name(self, pid: int) -> str:
        """获取进程名称 (带缓存)"""
        if pid == 0:
            return ""
        
        now = time.time()
        
        # 检查缓存
        if pid in self._process_cache:
            if now - self._cache_times.get(pid, 0) < self._cache_ttl:
                return self._process_cache[pid]
        
        # 查询进程名
        try:
            import psutil
            proc = psutil.Process(pid)
            name = proc.name()
            self._process_cache[pid] = name
            self._cache_times[pid] = now
            return name
        except:
            return ""
    
    def _build_event(self, event_type: str, file_path: str, 
                     pid: int, process_name: str) -> Dict[str, Any]:
        """构建标准化事件 (与 watchdog 事件格式一致)"""
        import socket
        
        basename = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        drive = os.path.splitdrive(file_path)[0]
        
        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        except:
            file_size = 0
        
        # 规范化应用名称
        app_name = self._normalize_app_name(process_name)
        
        event = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "event_type": event_type,
            "file_path": file_path,
            "file_name": basename,
            "file_size": file_size,
            "file_extension": ext,
            "process_info": {
                "pid": str(pid),
                "process_name": process_name,
                "process_path": "",
                "cmdline": ""
            },
            "window_info": {
                "window_handle": "",
                "window_title": "",
                "window_class": ""
            },
            "user_info": {
                "username": os.environ.get("USERNAME", ""),
                "hostname": socket.gethostname()
            },
            "disk_info": {
                "drive_letter": drive,
                "disk_type": "Fixed"
            },
            "app_name": app_name,
        }
        
        # 添加 upload_detection（如果是敏感文件）
        upload_detection = self._check_sensitive_file(basename, file_path, app_name)
        if upload_detection:
            event["upload_detection"] = upload_detection
        
        event["extra"] = {
            "raw_operation": "opened",
            "category": "",
            "source": "etw_file_monitor"
        }
        
        return event
    
    def _check_sensitive_file(self, file_name: str, file_path: str, app_name: str):
        """检查是否为敏感文件"""
        if not file_name:
            return None
        
        file_name_lower = file_name.lower()
        for keyword in self.sensitive_keywords:
            if keyword.lower() in file_name_lower:
                return {
                    "is_upload": True,
                    "app_name": app_name,
                    "upload_type": "File Access",
                    "original_file": file_path,
                    "temp_directory": ""
                }
        return None
    
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
            "wps": "WPS",
            "excel": "Excel",
            "winword": "Word",
            "powerpnt": "PowerPoint",
        }
        
        return app_name_map.get(process_name.lower(), process_name)


# 测试
if __name__ == "__main__":
    def print_event(event):
        print(f"📂 [{event['event_type']}] {event['file_name']} <- {event['app_name']}")
    
    monitor = ETWFileMonitor(event_callback=print_event)
    monitor.start()
    
    try:
        print("ETW Monitoring... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nStopping...")
        monitor.stop()
