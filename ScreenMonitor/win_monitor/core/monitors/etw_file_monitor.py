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
from typing import Callable, Optional, Dict, Any

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

# NT 路径前缀
NT_DEVICE_PREFIX = "\\Device\\"


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
    
    def __init__(self, event_callback: Callable[[Dict[str, Any]], None] = None, debug: bool = False):
        """
        Args:
            event_callback: 事件回调函数，接收标准化的事件字典
            debug: 是否启用调试输出
        """
        self.event_callback = event_callback
        self.debug = debug
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
        
        # 事件去重
        self._event_cache: Dict[str, float] = {}
        self._dedup_ttl = 0.5  # 0.5秒内相同事件去重
        self._cache_lock = threading.Lock()  # 线程安全锁
        
        # 事件计数器 (调试用)
        self._event_count = 0
        self._create_event_count = 0
        self._seen_task_names = set() # 诊断用
        
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
        
        # 敏感文件关键字（扩展版，基于69个场景分析，与 logger.py 保持一致）
        self.sensitive_keywords = [
            # 公司核心文件
            "合同", "机密", "密码", "secret", "private", "confidential", "绝密",
            "财务", "工资", "薪资", "银行", "账号", "证件", "内部",
            "身份证", "护照", "驾照", "简历", "resume",
            # 项目/业务文件
            "规划", "战略", "预算", "报表", "员工守则", "客户",
            "设计", "技术", "算法", "核心", "会议纪要", "项目",
            # 敏感文件名模式
            "accesskey", "credential", "api_key", "token",
            # 场景设计提取的敏感词
            "员工绩效", "薪资表", "薪资明细", "财务报表", "发票",
            "公司合同", "合作合同", "劳务合同", "客户合同", 
            "秘密会议", "内部资料", "公司机密",
            "发展战略", "核心技术", "技术文档", "技术图纸",
            "客户信息", "重点客户", "客户身份",
            "产品设计", "市场分析", "市场调研", "竞品分析",
            "需求分析", "prd设计", "需求设计", "详细规划",
            "培养方案", "管理制度", "组织架构",
            "身份信息", "部署账号", "并购项目", "定价策略",
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
                            print(f"[ETW_MONITOR] Drive mapping: {device_name} -> {drive_letter}")
                except Exception as e:
                    if self.debug:
                        print(f"[ETW_MONITOR] Failed to query device for {drive}: {e}")
            
            print(f"[ETW_MONITOR] Initialized {len(self._nt_to_dos_map)} drive mappings")
            
        except ImportError:
            print("[ETW_MONITOR] Warning: pywin32 not installed, NT path conversion disabled")
            # 备用硬编码映射
            self._nt_to_dos_map = {
                "\\device\\harddiskvolume1": "C:",
                "\\device\\harddiskvolume2": "D:",
                "\\device\\harddiskvolume3": "E:",
                "\\device\\harddiskvolume4": "D:",  # 常见配置
            }
        except Exception as e:
            print(f"[ETW_MONITOR] Drive mapping init error: {e}")
    
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
        self._thread = threading.Thread(target=self._run_etw_trace, daemon=True)
        self._thread.start()
        print(f"[ETW_MONITOR] Started with Kernel File I/O tracing (Keywords: 0x{ALL_FILE_IO_KEYWORDS:X})")

    def stop(self):
        """停止 ETW 监控"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self._session:
            try:
                self._session.stop()
            except Exception as e:
                print(f"[ETW_MONITOR] Error stopping session: {e}")
                
        if self._thread:
            self._thread.join(timeout=2.0)
            
        # 打印统计
        print(f"[ETW_MONITOR] Stopped. Total events: {self._event_count}, Create events: {self._create_event_count}")
        print(f"[ETW_MONITOR] Unique Task Names seen: {list(self._seen_task_names)}")

    def _run_etw_trace(self):
        """ETW 跟踪主循环"""
        try:
            import etw
            
            # 定义 Provider
            # Microsoft-Windows-Kernel-File: {EDD08927-9CC4-4E65-B970-C2560FB5C289}
            providers = [
                etw.ProviderInfo(
                    'Microsoft-Windows-Kernel-File',
                    etw.GUID(KERNEL_FILE_PROVIDER_GUID),
                    any_keywords=ALL_FILE_IO_KEYWORDS
                )
            ]
            
            # 创建 ETW 实例
            self._session = etw.ETW(
                providers=providers,
                event_callback=self._process_etw_event
            )
            
            self._session.start()

        except Exception as e:
            print(f"[ETW_MONITOR] Trace loop error: {e}")
            self.is_running = False

    def _process_etw_event(self, event):
        """处理 ETW 事件"""
        try:
            if not self.is_running:
                return

            self._event_count += 1
            
            # 每100个事件输出一次统计
            if self._event_count % 100 == 0:
                print(f"[ETW_MONITOR] Received {self._event_count} total events, {self._create_event_count} create events")

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
            
            # 记录见过的任务名 (诊断用)
            if task_name and task_name not in self._seen_task_names:
                self._seen_task_names.add(task_name)
                # if self.debug:
                #     print(f"[ETW_DIAG] New Task Name: {task_name}")

            # 2. 获取 Opcode
            opcode = 0
            if isinstance(event_data, dict):
                try:
                    opcode = event_data.get('EventHeader', {}).get('EventDescriptor', {}).get('Opcode', 0)
                except:
                    pass
            else:
                opcode = getattr(event_data, 'opcode', 0)

            # 3. 识别 Create/Open 事件
            # 包括 'Create', 'FileCreate', 'NameCreate' 等
            # Opcode 12 (DelayCreate) 或 64 (Create) 或 32 (FileCreate)
            # 我们放宽条件，不仅看 task_name，也看 opcode
            is_create_event = (
                'Create' in task_name or 
                task_name in ['FileCreate', 'NameCreate'] or
                opcode in [12, 64, 32]
            )

            if not is_create_event:
                return

            # 4. 获取文件路径
            file_path = ''
            if isinstance(event_data, dict):
                for key in ['FileName', 'OpenPath', 'Path', 'FileObject', 'Name']:
                    val = event_data.get(key)
                    if val and isinstance(val, str) and (':' in val or '\\' in val):
                        file_path = val
                        break
            else:
                file_path = (
                    getattr(event_data, 'FileName', None) or
                    getattr(event_data, 'OpenPath', None) or
                    getattr(event_data, 'file_path', None) or
                    getattr(event_data, 'Path', None) or
                    ''
                )

            if not file_path:
                return

            # 5. 转换 NT 路径为 DOS 路径
            dos_path = self._convert_nt_path_to_dos(file_path)
            
            # 6. 过滤系统噪音
            if self._should_ignore(dos_path):
                return
            
            # 7. 去重
            if self._is_duplicate('opened', dos_path):
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
            
            # 打印已捕获的打开事件
            print(f"[ETW] 📂 opened: {dos_path} <- {process_name} (Task={task_name}, Opcode={opcode})")
            
            # 构建标准化事件
            file_event = self._build_event('opened', dos_path, pid, process_name)
            
            # 回调
            if self.event_callback:
                self.event_callback(file_event)
            

                
        except Exception as e:
            if self.debug:
                print(f"[ETW_MONITOR] Event processing error: {e}")
                import traceback
                traceback.print_exc()
    
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
