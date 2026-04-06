# -*- coding: utf-8 -*-
"""
etw_file_monitor.py - 基于 ETW 的文件 I/O 监控器
类似于 Mac 的 fs_usage，精确捕获文件打开事件及进程信息

功能:
- 使用 Windows Event Tracing (ETW) 捕获文件 I/O 事件
- 捕获 FileIo_Create (文件打开) 事件
- 提供完整的进程信息 (PID, 进程名)
- NT 路径自动转换为 DOS 路径

要求:
- 需要管理员权限运行
- 需要安装: pip install pywintrace pywin32

对应架构角色: ETW 文件监控器 (与 watchdog 互补)
"""

import os
import threading
import time
import string
from datetime import datetime
try:
    import psutil
except ImportError:
    psutil = None
from typing import Callable, Optional, Dict, Any, List, Tuple

from ..utils import app_logger

# ETW 相关常量
KERNEL_FILE_PROVIDER_GUID = "{EDD08927-9CC4-4E65-B970-C2560FB5C289}"

# FileIo Keywords (Based on Microsoft-Windows-Kernel-File provider)
# Ref: https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win10/Microsoft-Windows-Kernel-File.xml
# 0x10  = KERNEL_FILE_KEYWORD_FILENAME
# 0x20  = KERNEL_FILE_KEYWORD_FILEIO
# 0x80  = KERNEL_FILE_KEYWORD_CREATE
# 0x200 = KERNEL_FILE_KEYWORD_WRITE
ALL_FILE_IO_KEYWORDS = 0xFED0  # Combined common keywords

# FileIo 操作类型
FILE_IO_CREATE = 12   # 文件创建/打开 (常见 opcode)
FILE_IO_CREATE_2 = 64 # 文件创建/打开 (备用 opcode)
FILE_IO_READ = 67     # 文件读取
FILE_IO_WRITE = 68    # 文件写入
FILE_IO_CLOSE = 65    # 文件关闭

# NT 路径前缀
NT_DEVICE_PREFIX = "\\Device\\"

# 浏览器和应用进程列表（用于上传检测）
BROWSER_PROCESSES = [
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
    "qq.exe", "wechat.exe", "weixin.exe", "dingtalk.exe", "feishu.exe",
    "tencentmeeting.exe", "doubao.exe", "quark.exe"
]

# 临时目录模式（用于检测上传触发）
TEMP_PATTERNS = [
    "\\AppData\\Local\\Temp\\",
    "\\Windows\\Temp\\",
    "\\Temp\\",
    "\\Cache\\",
    "\\User Data\\",  # 浏览器数据目录（包含缓存和临时文件）
    "\\Temporary Internet Files\\",
]

# 用户文件目录（非系统目录，表示用户主动操作的文件）
USER_FILE_PATTERNS = [
    "\\Documents\\",
    "\\Desktop\\",
    "\\Downloads\\",
    "\\Pictures\\",
    "\\Videos\\",
    "\\Music\\",
    "\\OneDrive\\",
]


class FileReadContext:
    """
    滑动窗口：记录每个进程最近的文件读取
    
    用于将 "Temp 文件创建" 与 "原始文件读取" 关联起来，
    从而捕获浏览器上传的原始文件路径。
    """
    
    def __init__(self, window_ms: int = 2000):
        """
        Args:
            window_ms: 时间窗口大小（毫秒），默认 1000ms
        """
        self.window_ms = window_ms
        # 结构: {pid: [(timestamp_ms, file_path, file_name), ...]}
        self._recent_reads: Dict[int, List[Tuple[float, str, str]]] = {}
        self._lock = threading.Lock()
        self._max_entries_per_pid = 20  # 每个进程最多保留20条记录
    
    def add_read(self, pid: int, file_path: str, file_name: str):
        """
        记录一次文件读取（仅记录用户文件，不记录系统/缓存文件）
        
        Args:
            pid: 进程ID
            file_path: 完整文件路径
            file_name: 文件名
        """
        # 过滤：只记录"用户文件"（非临时目录、非系统目录）
        if self._is_temp_or_system_path(file_path):
            return
        
        now_ms = time.time() * 1000
        
        with self._lock:
            if pid not in self._recent_reads:
                self._recent_reads[pid] = []
            
            # 添加记录
            self._recent_reads[pid].append((now_ms, file_path, file_name))
            
            # 清理过期记录
            cutoff = now_ms - self.window_ms * 2  # 保留2倍窗口时间的数据
            self._recent_reads[pid] = [
                (ts, fp, fn) for ts, fp, fn in self._recent_reads[pid]
                if ts > cutoff
            ][-self._max_entries_per_pid:]  # 限制数量
    
    def correlate_upload(self, pid: int) -> Optional[Tuple[str, str]]:
        """
        当检测到可能的上传行为（如 Temp 文件创建）时，
        回溯查找该进程最近读取的用户文件
        
        Args:
            pid: 进程ID
            
        Returns:
            (file_path, file_name) 如果找到，否则 None
        """
        now_ms = time.time() * 1000
        cutoff = now_ms - self.window_ms
        
        with self._lock:
            if pid not in self._recent_reads:
                return None
            
            # 从最近的记录开始查找
            for ts, file_path, file_name in reversed(self._recent_reads[pid]):
                if ts > cutoff:
                    # 找到时间窗口内的用户文件读取
                    return (file_path, file_name)
            
            return None
    
    def _is_temp_or_system_path(self, path: str) -> bool:
        """判断是否为临时/系统路径"""
        if not path:
            return True
        
        path_lower = path.lower()
        
        # 首先检查是否是用户文件目录 - 这些永远不应该被过滤
        user_file_patterns = [
            "\\documents\\",
            "\\desktop\\",
            "\\downloads\\",
            "\\pictures\\",
            "\\videos\\",
            "\\music\\",
            "\\onedrive\\",
        ]
        for pattern in user_file_patterns:
            if pattern in path_lower:
                return False  # 是用户文件，不过滤
        
        # 临时目录
        for pattern in TEMP_PATTERNS:
            if pattern.lower() in path_lower:
                return True
        
        # 系统目录
        system_dirs = [
            "\\windows\\",
            "\\program files\\",
            "\\program files (x86)\\",
            "\\programdata\\",
            "\\appdata\\local\\microsoft\\",
            "\\appdata\\local\\google\\chrome\\",
            "\\appdata\\roaming\\qq\\",
            "\\appdata\\roaming\\tencent\\",
        ]
        for pattern in system_dirs:
            if pattern in path_lower:
                return True
        
        return False

    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（调试用）"""
        with self._lock:
            total_entries = sum(len(v) for v in self._recent_reads.values())
            return {
                "tracked_pids": len(self._recent_reads),
                "total_entries": total_entries
            }


class ETWFileMonitor:
    """
    基于 ETW 的文件 I/O 监控器
    
    使用 Microsoft-Windows-Kernel-File Provider 来精确捕获:
    - 文件打开事件 (Create, 在内核层 Open 也是 Create)
    - 完整的进程信息 (PID, 进程名)
    
    与 watchdog 的区别:
    - watchdog: 监控文件系统变化 (创建、修改、删除、重命名)
    - ETW: 监控文件 I/O 操作 (打开、读取、写入)
    
    两者互补使用可实现完整的文件监控。
    """
    
    def __init__(self, event_callback: Callable[[Dict[str, Any]], None] = None, debug: bool = False, 
                 browser_processes: List[str] = None, sensitive_keywords: List[str] = None):
        """
        Args:
            event_callback: 事件回调函数
            debug: 是否启用调试输出
            browser_processes: 浏览器进程列表 (带.exe)
            sensitive_keywords: 敏感文件关键词列表
        """
        self.event_callback = event_callback
        self.debug = debug
        self.browser_processes = [name.lower() for name in (browser_processes or [])] or BROWSER_PROCESSES
        
        # 预处理敏感关键词（转为小写以优化性能）
        raw_keywords = sensitive_keywords if sensitive_keywords else ["机密", "绝密", "合同", "secret", "confidential", "password"]
        self.sensitive_keywords_lower = [k.lower() for k in raw_keywords]
        # 保留原始列表用于调试（可选）
        self.sensitive_keywords = raw_keywords

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._session = None
        
        # NT 路径到 DOS 路径的映射缓存
        self._nt_to_dos_map: Dict[str, str] = {}
        self._init_drive_mapping()
        
        # 进程名缓存 (避免频繁查询)
        self._process_cache: Dict[int, str] = {}
        self._cache_ttl = 60  # 缓存 60 秒
        self._cache_times: Dict[int, float] = {}
        
        # 滑动窗口上下文 - 用于关联上传检测
        self._file_read_context = FileReadContext(window_ms=1000)
        self._upload_count = 0  # 上传检测计数器
        
        # 事件去重
        self._event_cache: Dict[str, float] = {}
        self._dedup_ttl = 0.5  # 0.5秒内相同事件去重
        self._cache_lock = threading.Lock()  # 线程安全锁
        
        # 事件计数器 (调试用)
        self._event_count = 0
        self._create_event_count = 0
        self._seen_task_names = set() # 诊断用
        
        # 进程名缓存锁 (线程安全)
        self._process_cache_lock = threading.Lock()
        
        # FileObject -> FileName 映射 (用于解析 READ 事件)
        self._file_object_map = {}
        
        # 过滤规则 - 监控整台机器但忽略系统噪音
        self.ignore_extensions = [
            '.etl', '.pf', '.db-journal', '.db-wal', '.db-shm',
            '.tmp', '.TMP', '.log', '.evtx'
        ]
        self.ignore_patterns = [
            '\\Windows\\Prefetch\\',
            '\\Windows\\System32\\winevt\\',
            '\\$Extend\\',
            '\\Device\\NamedPipe\\',
            '\\Device\\Afd\\',
        ]
        
    
    def _init_drive_mapping(self):
        """初始化 NT 设备路径到 DOS 驱动器的映射"""
        try:
            import win32file
            import win32api
            
            # 获取所有逻辑驱动器
            drives = win32api.GetLogicalDriveStrings().split('\x00')
            drives = [d for d in drives if d]
            
            for drive in drives:
                drive_letter = drive.rstrip('\\')
                try:
                    # 查询设备路径
                    device_path = win32file.QueryDosDevice(drive_letter)
                    if device_path:
                        # QueryDosDevice 返回多个路径，取第一个
                        device_name = device_path.split('\x00')[0]
                        self._nt_to_dos_map[device_name.lower()] = drive_letter
                        if self.debug:
                            app_logger.debug(f"[ETW_MONITOR] Drive mapping: {device_name} -> {drive_letter}")
                except Exception as e:
                    if self.debug:
                        app_logger.warning(f"[ETW_MONITOR] Failed to query device for {drive}: {e}")
            
            app_logger.info(f"[ETW_MONITOR] Initialized {len(self._nt_to_dos_map)} drive mappings")
            
        except ImportError:
            app_logger.warning("[ETW_MONITOR] Warning: pywin32 not installed, NT path conversion disabled")
            # 备用硬编码映射
            self._nt_to_dos_map = {
                "\\device\\harddiskvolume1": "C:",
                "\\device\\harddiskvolume2": "D:",
                "\\device\\harddiskvolume3": "E:",
                "\\device\\harddiskvolume4": "D:",  # 常见配置
            }
        except Exception as e:
            app_logger.error(f"[ETW_MONITOR] Drive mapping init error: {e}")
    
    def _convert_nt_path_to_dos(self, nt_path: str) -> str:
        """将 NT 设备路径转换为 DOS 路径"""
        if not nt_path:
            return nt_path
        
        # 已经是 DOS 路径
        if len(nt_path) >= 2 and nt_path[1] == ':':
            return nt_path
        
        # 检查是否是 NT 设备路径
        nt_path_lower = nt_path.lower()
        
        for device_path, drive_letter in self._nt_to_dos_map.items():
            if nt_path_lower.startswith(device_path):
                # 替换设备路径为驱动器盘符
                return drive_letter + nt_path[len(device_path):]
        
        # 无法转换，返回原路径
        return nt_path
    
    def start(self):
        """启动 ETW 监控"""
        if self.is_running:
            return
            
        self.is_running = True
        self._thread = threading.Thread(target=self._run_etw_trace, daemon=True, name="ETWTraceLoop")
        self._thread.start()
        app_logger.info(f"[ETW] ✅ 文件监控已启动 (debug={self.debug})")

    def stop(self):
        """停止 ETW 监控"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self._session:
            try:
                self._session.stop()
            except Exception as e:
                app_logger.error(f"[ETW_MONITOR] Error stopping session: {e}")
                
        if self._thread:
            self._thread.join(timeout=2.0)
            
        # 打印统计
        app_logger.info(f"[ETW] ✅ 监控已停止 - 共处理 {self._event_count} 事件, 检测到 {self._upload_count} 次上传")

    
    def _get_process_name(self, pid: int) -> str:
        """获取进程名称 (带缓存)"""
        if not pid:
            return ""
            
        with self._process_cache_lock:
            # 检查缓存
            if pid in self._process_cache:
                name, timestamp = self._process_cache[pid]
                if time.time() - timestamp < 60:  # 1分钟缓存有效期
                    return name
        
        # 获取进程名
        name = ""
        try:
            if psutil:
                proc = psutil.Process(pid)
                name = proc.name()
        except:
            pass
            
        # 更新缓存
        with self._process_cache_lock:
            self._process_cache[pid] = (name, time.time())
            
        return name
    
    # _normalize_app_name 定义在文件末尾 (L944+)，包含完整映射表

    def _run_etw_trace(self):
        """ETW 跟踪主循环"""
        try:
            import etw
            import logging as _logging
            
            # 抑制 pywintrace 内部的 WARNING 日志
            # pywintrace 在解析 Kernel-File ETW 事件时，某些字段无法解析会打印:
            # "Failed to get data field data for FileName, incrementing by reported size"
            # 这是已知限制，不影响功能（我们在回调中已处理缺失字段）
            _logging.getLogger('etw.etw').setLevel(_logging.ERROR)
            
            # 定义 Provider
            # Microsoft-Windows-Kernel-File: {EDD08927-9CC4-4E65-B970-C2560FB5C289}
            # 注意: 不使用 any_keywords 过滤，接收所有文件事件，在回调中过滤
            providers = [
                etw.ProviderInfo(
                    'Microsoft-Windows-Kernel-File',
                    etw.GUID(KERNEL_FILE_PROVIDER_GUID)
                )
            ]
            
            # 创建 ETW 实例
            self._session = etw.ETW(
                providers=providers,
                event_callback=self._process_etw_event
            )
            
            # 启动 ETW 会话
            self._session.start()
            
            # 阻塞等待事件 - 这是关键！
            # pywintrace 的 start() 只是启动会话，需要持续运行才能接收事件
            # 使用 while 循环保持线程活跃
            while self.is_running:
                time.sleep(0.1)  # 短暂休眠，避免 CPU 空转
            
        except ImportError as e:
            app_logger.warning(f"[ETW_MONITOR] pywintrace not installed: {e}")
            self.is_running = False
        except Exception as e:
            app_logger.error(f"[ETW_MONITOR] Trace loop error: {e}")
            import traceback
            app_logger.error(traceback.format_exc())
            self.is_running = False

    def _log_debug(self, message):
        """发送调试日志到事件流"""
        if self.debug and self.event_callback:
            self.event_callback({
                'event_type': 'log',
                'level': 'debug',
                'source': 'etw_file_monitor',
                'timestamp': time.time(),
                'file_name': message # Hack to show in UI log view
            })

    def _process_etw_event(self, event):
        """处理 ETW 事件"""
        try:
            if not self.is_running:
                return

            self._event_count += 1
            
            # ETW 事件通常是一个元组: (event_id, event_data_dict)
            if isinstance(event, tuple) and len(event) >= 2:
                event_data = event[1]
                if not isinstance(event_data, dict):
                    return
            else:
                event_data = event
                
            # 1. 获取事件类型/任务名
            task_name = ''
            if isinstance(event_data, dict):
                task_name = event_data.get('Task Name', '')
            else:
                task_name = getattr(event_data, 'task_name', '')

            # DIAGNOSTIC: Dump keys for processed events (limit 50)
            if self.debug:
                 try:
                     current_dump_count = getattr(self, '_dump_count', 0)
                     is_interesting = 'CREATE' in task_name.upper() or 'READ' in task_name.upper()
                     
                     if current_dump_count < 50 or (is_interesting and current_dump_count < 100):
                         keys_list = list(event_data.keys()) if isinstance(event_data, dict) else ["Not Dict"]
                         
                         log_msg = f"[ETW_KEYS] Dump #{current_dump_count} ({task_name}): {keys_list}"
                         self._log_debug(log_msg)
                         app_logger.debug(log_msg)
                         
                         self._dump_count = current_dump_count + 1
                 except Exception as e:
                     app_logger.debug(f"[ETW_DUMP_ERROR] {e}")
                     self._log_debug(f"[ETW_DUMP_ERROR] {e}")

            # 状态追踪：从任何包含 FileName 和 FileObject 的事件中学习映射
            file_object = None
            file_name_in_event = ''
            
            if isinstance(event_data, dict):
                file_object = event_data.get('FileObject')
                file_name_in_event = event_data.get('FileName') or event_data.get('OpenPath') or event_data.get('Path')
            else:
                file_object = getattr(event_data, 'FileObject', None)
                file_name_in_event = getattr(event_data, 'FileName', '') or getattr(event_data, 'OpenPath', '')
                
            # 更新映射
            if file_object and file_name_in_event:
                with self._process_cache_lock: # 复用锁
                    self._file_object_map[str(file_object)] = str(file_name_in_event)
                    # Use a sampling or specific check to avoid flooding logs
                    if 'AAA' in file_name_in_event or 'upload' in file_name_in_event or self._event_count < 200:
                         self._log_debug(f"[ETW_MAP] Mapped {file_object} -> {file_name_in_event}")
            
            # 清理映射 (Close/Cleanup)
            task_name_upper = task_name.upper() # Define task_name_upper here for use in cleanup
            if 'CLOSE' in task_name_upper or 'CLEANUP' in task_name_upper:
                if file_object:
                    with self._process_cache_lock:
                        if str(file_object) in self._file_object_map:
                             self._file_object_map.pop(str(file_object), None)

            # 2. 获取 Opcode
            opcode = 0
            if isinstance(event_data, dict):
                try:
                    opcode = event_data.get('EventHeader', {}).get('EventDescriptor', {}).get('Opcode', 0)
                except:
                    pass
            else:
                opcode = getattr(event_data, 'opcode', 0)

            # 3. 识别事件类型（大小写不敏感）
            # Create/Open 事件: Task Name 包含 CREATE
            # Read 事件: Task Name 包含 READ
            
            is_create_event = (
                'CREATE' in task_name_upper or
                opcode in [12, 64, 32]
            )
            
            is_read_event = (
                'READ' in task_name_upper or
                opcode == 67 or 
                (opcode == 0 and 'READ' in task_name_upper) # 兼容 Opcode=0 的 READ
            )
            
            # 只处理 Create 和 Read 事件
            if not is_create_event and not is_read_event:
                return
            
            # 确定事件类型
            if is_create_event:
                event_type = 'opened'
            else:
                event_type = 'read'

            # 4. 获取文件路径
            file_path = file_name_in_event
            
            # 如果没有路径但有 FileObject (针对 READ 事件)，尝试查表
            mapped_from_obj = False
            if not file_path and file_object and is_read_event:
                with self._process_cache_lock:
                    file_path = self._file_object_map.get(str(file_object))
                    if file_path:
                        mapped_from_obj = True
            
            if not file_path:
                if is_read_event and file_object and self._event_count < 500:
                    self._log_debug(f"[ETW_FAIL] Read event miss: Obj={file_object}, Task={task_name}")
                return

            # 5. 转换 NT 路径为 DOS 路径
            dos_path = self._convert_nt_path_to_dos(file_path)
            
            # 6. 过滤系统噪音
            if self._should_ignore(dos_path):
                return
            
            # 7. 去重 (分别对 opened 和 read 去重)
            if self._is_duplicate(event_type, dos_path):
                return
            
            self._create_event_count += 1
            
            # 8. 获取进程信息
            pid = 0
            if isinstance(event_data, dict):
                try:
                    pid = event_data.get('EventHeader', {}).get('ProcessId', 0)
                except:
                    pass
            else:
                pid = getattr(event_data, 'ProcessId', 0) or getattr(event_data, 'process_id', 0)
            
            process_name = self._get_process_name(pid)
            
            # === 新增: 上传检测逻辑 ===
            # 获取 ThreadID (用于更精确的关联)
            tid = 0
            if isinstance(event_data, dict):
                try:
                    tid = event_data.get('EventHeader', {}).get('ThreadId', 0)
                except:
                    pass
             # 8. 记录读取事件 (用于关联上传)
            # 只有浏览器进程的读取才可能是上传
            is_browser = False
            if process_name:
                is_browser = process_name.lower() in self.browser_processes or process_name.lower() in ['chrome', 'msedge', 'firefox', 'opera', 'brave']
                # Try partial match if not found
                if not is_browser:
                     p_lower = process_name.lower()
                     if 'chrome' in p_lower or 'edge' in p_lower:
                         is_browser = True
            
            if self.debug and file_path and not is_browser:
                 # Debug: 看看我们忽略了哪些进程的读取
                 if pid not in getattr(self, '_ignored_pids', []):
                     app_logger.debug(f"[ETW_DEBUG] Ignored read from non-browser: {file_path} (PID={pid}, Name={process_name})")
                     if not hasattr(self, '_ignored_pids'): self._ignored_pids = []
                     self._ignored_pids.append(pid)

            # 检查是否是临时目录
            is_temp_path = self._is_temp_path(dos_path)
            
            # 记录文件名
            file_name = os.path.basename(dos_path)
            
            # 检查是否是用户文件目录
            is_user_file = self._is_user_file_path(dos_path)
            
            # 上传检测核心逻辑
            upload_detection = None
            
            if is_browser:
                if is_temp_path and is_create_event:
                    # 触发信号: 浏览器在 Temp 目录创建文件
                    # 回溯查找原始文件
                    original = self._file_read_context.correlate_upload(pid)
                    if original:
                        original_path, original_name = original
                        self._upload_count += 1
                        upload_detection = {
                            "is_upload": True,
                            "original_file": original_path,
                            "original_name": original_name,
                            "temp_file": dos_path,
                            "app_name": self._normalize_app_name(process_name),
                            "upload_type": "Browser Upload",
                            "detection_method": "sliding_window_correlation"
                        }
                        app_logger.info(f"[ETW] 📤 检测到上传! 原始文件: {original_path}")
                        app_logger.info(f"[ETW]    → 临时文件: {dos_path}")
                        app_logger.info(f"[ETW]    → 进程: {process_name} (PID: {pid})")
                elif is_user_file:
                    # 🆕 直接标记为浏览器访问用户文件（可能是上传）
                    self._file_read_context.add_read(pid, dos_path, file_name)
                    
                    # 检查文件扩展名是否是常见上传文件类型
                    file_ext = os.path.splitext(dos_path)[1].lower()
                    upload_file_exts = [
                        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',  # 图片
                        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # 文档
                        '.zip', '.rar', '.7z',  # 压缩包
                        '.txt', '.csv', '.json',  # 文本
                        '.mp3', '.mp4', '.avi', '.mov',  # 多媒体
                    ]
                    
                    if file_ext in upload_file_exts:
                        self._upload_count += 1
                        upload_detection = {
                            "is_upload": True,
                            "original_file": dos_path,
                            "original_name": file_name,
                            "app_name": self._normalize_app_name(process_name),
                            "upload_type": "Browser File Access",
                            "detection_method": "browser_user_file_access"
                        }
                        app_logger.info(f"[ETW] 📤 浏览器文件访问: {dos_path}")
                        app_logger.info(f"[ETW]    → 进程: {process_name} (PID: {pid})")
                else:
                    # 其他文件读取: 也记录到滑动窗口
                    self._file_read_context.add_read(pid, dos_path, file_name)
            
            # 打印所有浏览器访问用户文件的事件
            if is_browser and is_user_file:
                app_logger.debug(f"[ETW] 📂 {event_type}: {dos_path} <- {process_name}")

            
            # 构建标准化事件
            file_event = self._build_event(event_type, dos_path, pid, process_name)
            
            # 附加上传检测信息
            if upload_detection:
                file_event["upload_detection"] = upload_detection
                # 🆕 将 detection_method 也添加到 extra 字段，便于 KeyLogExtractor 过滤
                if "detection_method" in upload_detection:
                    file_event["extra"]["detection_method"] = upload_detection["detection_method"]
            
            # 回调
            if self.event_callback:
                self.event_callback(file_event)
            


                
        except Exception as e:
            if self.debug:
                app_logger.error(f"[ETW_MONITOR] Event processing error: {e}")
                import traceback
                app_logger.error(traceback.format_exc())
    
    def _should_ignore(self, path: str) -> bool:
        """判断是否应该忽略该路径"""
        if not path:
            return True
            
        path_lower = path.lower()
        
        # 忽略路径模式
        for pattern in self.ignore_patterns:
            if pattern.lower() in path_lower:
                return True
        
        # 忽略扩展名
        _, ext = os.path.splitext(path)
        if ext.lower() in self.ignore_extensions:
            return True
        
        return False
    
    def _is_temp_path(self, path: str) -> bool:
        """判断是否为临时目录路径（用于上传检测触发）"""
        if not path:
            return False
        
        path_lower = path.lower()
        for pattern in TEMP_PATTERNS:
            if pattern.lower() in path_lower:
                return True
        return False
    
    def _is_user_file_path(self, path: str) -> bool:
        """判断是否为用户文件目录（Documents, Desktop, Pictures等）"""
        if not path:
            return False
        
        path_lower = path.lower()
        
        # 首先排除临时目录和缓存目录
        temp_exclude_patterns = [
            "\\temp\\", "\\cache\\", "\\user data\\", "\\appdata\\",
            "\\gpucache\\", "\\code cache\\", "\\service worker\\",
            "\\microsoft\\edge\\", "\\google\\chrome\\", "\\mozilla\\firefox\\",
            "scoped_dir", ".tmp", ".crdownload"
        ]
        for pattern in temp_exclude_patterns:
            if pattern in path_lower:
                return False
        
        # 用户文件目录模式
        user_patterns = [
            "\\documents\\",
            "\\desktop\\",
            "\\downloads\\",
            "\\pictures\\",
            "\\videos\\",
            "\\music\\",
            "\\onedrive\\",
            "\\我的文档\\",
            "\\桌面\\",
            "\\下载\\",
        ]
        
        for pattern in user_patterns:
            if pattern in path_lower:
                return True
        
        # 也检查根目录下的用户文件（如 D:\photos\kids.jpg）
        # 如果路径是盘符开头且不在系统目录，也可能是用户文件
        if len(path) > 3 and path[1] == ':':
            # 排除系统目录
            system_dirs = ["\\windows\\", "\\program files", "\\programdata\\"]
            is_system = any(p in path_lower for p in system_dirs)
            if not is_system:
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
    
    # _get_process_name 已在上方 L383 定义 (使用 _process_cache_lock 的线程安全版本)
    
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
            "raw_operation": event_type,
            "category": "",
            "source": "etw_file_monitor"
        }
        
        return event
    
    def _check_sensitive_file(self, file_name: str, file_path: str, app_name: str):
        """检查是否为敏感文件"""
        if not file_name:
            return None
        
        file_name_lower = file_name.lower()
        for keyword in self.sensitive_keywords_lower:

            if keyword in file_name_lower:
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
        
        # 常见应用名称映射（与 logger.py 保持一致）
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
            "wpsoffice": "WPS",
            "et": "WPS Excel",
            "wpp": "WPS PPT",
            "wpsclouddrive": "WPS云盘",
            "excel": "Excel",
            "winword": "Word",
            "powerpnt": "PowerPoint",
            "dingtalk": "钉钉",
            "feishu": "飞书",
            "lark": "飞书",
        }
        
        return app_name_map.get(process_name.lower(), process_name)


# 测试
if __name__ == "__main__":
    def print_event(event):
        print(f"📂 [{event['event_type']}] {event['file_path']} <- {event['app_name']} ({event['process_info']['process_name']})")
    
    # 启用调试模式
    monitor = ETWFileMonitor(event_callback=print_event, debug=True)
    monitor.start()
    
    try:
        print("ETW Monitoring... Press Ctrl+C to stop")
        print("Try opening some files (documents, images, etc.) to see events")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()
