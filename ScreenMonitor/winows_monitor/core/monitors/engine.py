# -*- coding: utf-8 -*-
"""
engine.py - 监控引擎（主循环/状态机）
职责：协调各组件，实现主循环逻辑
- 从 Sensor 获取数据
- 用 RuleMatcher 判断是否命中
- 用 Logger 记录日志（包括文件事件）
- 控制 RecorderService 启停
- 集成 FileSystemMonitor 监控文件操作

对应架构角色：Controller（控制器）
"""

import os
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from typing import List, Optional

from ..utils import app_logger

from .sensor import Sensor, WindowData
from .file_system_monitor import FileSystemMonitor
from .clipboard_monitor import ClipboardMonitor
from .etw_launcher import get_etw_launcher, EtwLauncher
from ..detection.rule_matcher import RuleMatcher, MatchResult
from ..logging.logger import Logger


class State(Enum):
    """引擎状态"""
    IDLE = "idle"  # 空闲，未录制
    RECORDING = "recording"  # 录制中


class Engine:
    """
    监控引擎 - 状态机实现
    
    核心循环：
    1. Sensor.get_active_window() -> WindowData
    2. RuleMatcher.match(WindowData) -> MatchResult
    3. 状态机决策：是否开始/停止录制
    4. Logger.log() 记录事件
    5. FileSystemMonitor 捕获文件操作
    6. ClipboardMonitor 捕获剪贴板操作
    
    改进：
    - 启动时即开启所有监控器 (File, Clipboard)
    - 使用 event_buffer 缓存录制前的事件
    - 触发录制时，将缓存的“案发前”事件写入日志
    """
    
    # 日志缓冲区最大容量 (用于 Web UI)
    MAX_LOG_BUFFER = 500
    
    # 事件回溯缓冲区容量 (保存录制前的事件)
    MAX_EVENT_BUFFER = 2000
    
    # 最大录制时长（秒）
    MAX_RECORDING_DURATION = 3600  # 1小时
    
    def __init__(self, config_loader, recorder_service=None, output_dir: str = "./recordings"):
        """
        Args:
            config_loader: ConfigLoader 实例
            recorder_service: RecorderService 实例（可选）
            output_dir: 输出目录
        """
        # 核心组件
        self.sensor = Sensor()
        self.rule_matcher = RuleMatcher(config_loader)
        self.logger = Logger()
        self.config = config_loader
        
        # 监控通过 _handle_monitor_event 统一回调
        self.file_monitor: Optional[FileSystemMonitor] = None
        self.etw_monitor = None
        self.clipboard_monitor: Optional[ClipboardMonitor] = None
        
        # EtwMonitor.exe 启动器 (用于捕获浏览器文件访问)
        self.etw_launcher: Optional[EtwLauncher] = None
        
        # 事件回溯缓冲区
        self.event_buffer = deque(maxlen=self.MAX_EVENT_BUFFER)
        self.buffer_lock = threading.Lock()
        
        # 录制服务（可选）
        self.recorder = recorder_service
        
        # 配置
        self.output_dir = output_dir
        self.poll_interval = config_loader.get_poll_interval_seconds()
        
        # 状态机
        self.state = State.IDLE
        self.state_lock = threading.Lock()
        
        # 运行状态
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 窗口变化检测
        self.last_window: Optional[WindowData] = None
        
        # 录制信息
        self.current_session_id: Optional[str] = None
        self.current_session_dir: Optional[str] = None
        self.recording_start_time: Optional[float] = None
        
        # 🆕 文件对话框检测状态
        self.last_dialog_info: Optional[dict] = None  # 记录最近一次文件对话框信息
        self.dialog_trigger_window: Optional[WindowData] = None  # 触发对话框的窗口
        
        # 日志缓冲区（用于 Web UI 显示）
        self._log_buffer: deque = deque(maxlen=self.MAX_LOG_BUFFER)
        self._log_lock = threading.Lock()
    
    def _add_log(self, level: str, message: str, data: dict = None):
        """添加日志到缓冲区（线程安全）"""
        with self._log_lock:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": level,
                "message": message,
                "data": data or {}
            }
            self._log_buffer.append(log_entry)
    
    def get_recent_logs(self, count: int = 50) -> List[dict]:
        """获取最近的日志（用于 Web UI）"""
        with self._log_lock:
            logs = list(self._log_buffer)
            return logs[-count:] if len(logs) > count else logs
    
    def start(self):
        """启动引擎"""
        if self.running:
            self._add_log("warning", "引擎已在运行")
            return False
        
        self.running = True
        self.state = State.IDLE
        
        self._add_log("info", f"监控引擎已启动 (轮询间隔: {self.poll_interval}s)")
        app_logger.info(f"🚀 监控引擎已启动")
        app_logger.info(f"   轮询间隔: {self.poll_interval}s")
        app_logger.info(f"   最大录制时长: {self.MAX_RECORDING_DURATION // 60} 分钟")
        
        # 1. 启动所有监控器 (持续运行)
        self._start_monitors()
        
        # 2. 启动主循环线程
        self.monitor_thread = threading.Thread(
            target=self._main_loop, 
            daemon=True,
            name="EngineMainLoop"
        )
        self.monitor_thread.start()
        return True
    
    # Web UI 别名
    start_monitoring = start
    
    def stop(self):
        """停止引擎"""
        if not self.running:
            return False
        
        self.running = False
        
        # 等待线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # 如果正在录制，停止录制
        if self.state == State.RECORDING:
            self._stop_recording()
            
        # 停止所有监控器
        self._stop_monitors()
        
        self._add_log("info", "监控引擎已停止")
        app_logger.info("🛑 监控引擎已停止")
        return True
    
    # Web UI 别名
    stop_monitoring = stop

    def _start_monitors(self):
        """启动所有底层监控器"""
        # 1. 文件系统监控 (Watchdog)
        try:
            self.file_monitor = FileSystemMonitor(
                event_callback=self._handle_monitor_event
            )
            self.file_monitor.start()
            app_logger.info("📂 文件系统监控已启动 (watchdog)")
        except Exception as e:

            app_logger.error(f"启动 watchdog 监控失败: {e}")
            # Watchdog 失败不应阻止引擎启动，但需记录详细日志
            app_logger.exception("Watchdog init details:")

        # 2. ETW 文件打开监控
        try:
            from .etw_file_monitor import ETWFileMonitor
            browser_list = self.config.get_browser_processes()
            sensitive_list = self.config.get_sensitive_keywords()
            self.etw_monitor = ETWFileMonitor(
                event_callback=self._handle_monitor_event,
                browser_processes=browser_list,
                sensitive_keywords=sensitive_list
            )
            self.etw_monitor.start()
            app_logger.info("🔍 Python ETW 文件监控已启动 (pywintrace)")
            app_logger.info("   💡 C++ ETW (EtwMonitor.exe) 将在录制开始时自动启动，两者可同时工作")
        except ImportError:
            app_logger.warning("Python ETW 监控不可用 (pywintrace 未安装), 仅使用 C++ ETW")
        except Exception as e:
            app_logger.warning(f"Python ETW 监控启动失败 (将依赖 C++ ETW): {e}")
            
        # 3. 剪贴板监控
        try:
            self.clipboard_monitor = ClipboardMonitor(event_callback=self._handle_monitor_event)
            self.clipboard_monitor.start()
            app_logger.info("📋 剪贴板监控已启动")
        except Exception as e:
            app_logger.error(f"启动剪贴板监控失败: {e}")

    def _stop_monitors(self):
        """停止所有底层监控器"""
        if self.file_monitor:
            try:
                self.file_monitor.stop()
            except Exception as e:
                app_logger.error(f"停止 watchdog 失败: {e}")
            self.file_monitor = None
            
        if hasattr(self, 'etw_monitor') and self.etw_monitor:
            try:
                self.etw_monitor.stop()
            except Exception as e:
                app_logger.error(f"停止 ETW 失败: {e}")
            self.etw_monitor = None
            
        if self.clipboard_monitor:
            try:
                self.clipboard_monitor.stop()
            except Exception as e:
                app_logger.error(f"停止剪贴板监控失败: {e}")
            self.clipboard_monitor = None
            
    def _handle_monitor_event(self, event: dict):
        """
        统一处理来自各个监控器(File, Clipboard)的事件
        """
        # 1. 添加到 Web UI 日志缓冲区 (实时显示)
        event_type = event.get("event_type", "unknown")
        desc = event.get("file_name") or event.get("content_preview") or "unknown"
        
        level = "file"
        if "clipboard" in event_type:
            level = "clipboard"
            
        self._add_log(level, f"[{event_type}] {desc}", event)

        # 2. 根据状态处理事件
        if self.state == State.RECORDING:
            # 录制中：直接写入日志文件
            self.logger.log_raw_event(event)  # 需要确保 Logger 有 log_raw_event 方法
        else:
            # 空闲中：存入回溯缓冲区
            with self.buffer_lock:
                self.event_buffer.append(event)

    def _main_loop(self):
        """主循环"""
        while self.running:
            try:
                # 1. 获取当前窗口数据
                window_data = self.sensor.get_active_window()
                
                if window_data:
                    # 检测窗口变化
                    if self._window_changed(window_data):
                        # 2. 规则匹配
                        match_result = self.rule_matcher.match(window_data)
                        
                        # 添加到 Web UI 日志缓冲区
                        if match_result.is_match:
                            self._add_log("alert", f"检测到风险: {match_result.app_name}", {
                                "app": match_result.app_name,
                                "category": match_result.category,
                                "window_title": window_data.window_title[:50]
                            })
                        
                        # 3. 状态机决策
                        self._process_state(match_result, window_data)
                        
                        # 🆕 4. 文件对话框检测和上传推理
                        if self.state == State.RECORDING:
                            self._detect_file_operations(window_data)
                        
                        # 5. 日志记录（持续记录窗口切换）
                        if self.state == State.RECORDING:
                            self.logger.log(window_data, match_result, time.time())
                        
                        # 更新上次窗口
                        self.last_window = window_data
                
                # 检查录制时长
                self._check_recording_duration()
                
            except Exception as e:
                app_logger.error(f"主循环异常: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(self.poll_interval)
    
    def _window_changed(self, current: WindowData) -> bool:
        """检查窗口是否发生变化"""
        if not self.last_window:
            return True
        
        return (current.process_name != self.last_window.process_name or
                current.window_title != self.last_window.window_title)
    
    def _process_state(self, match_result: MatchResult, window_data: WindowData):
        """状态机处理"""
        with self.state_lock:
            if match_result.is_match:
                # 命中规则
                if self.state == State.IDLE:
                    # IDLE -> RECORDING
                    self._start_recording(match_result)
            # 注意：不再有冷却逻辑，录制会持续进行
    
    def _start_recording(self, match_result: MatchResult, session_id: str = None):
        """开始录制"""
        self.state = State.RECORDING
        if session_id:
            self.current_session_id = session_id
        else:
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_start_time = time.time()
        
        # 创建会话目录
        # 如果是绝对路径，直接使用；否则拼接 output_dir
        # 这里假设 session_id 只是 ID
        self.current_session_dir = os.path.join(self.output_dir, f"session_{self.current_session_id}")
        os.makedirs(os.path.join(self.current_session_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.current_session_dir, "video"), exist_ok=True)
        
        # 启动日志记录 (使用 logs.json 以匹配 API 期望)
        log_path = os.path.join(self.current_session_dir, "logs", "logs.json")
        self.logger.open(log_path, time.time())
        
        self._add_log("info", f"开始录制 - 触发: {match_result.app_name} ({match_result.category})")
        print(f"🎬 开始录制 - 触发: {match_result.app_name} ({match_result.category})")
        print(f"   会话目录: {self.current_session_dir}")
        
        # 🌟 关键：将缓冲区中的“案发前”事件写入日志
        self._flush_event_buffer()
        
        # 启动屏幕录制
        if self.recorder:
            video_path = os.path.join(self.current_session_dir, "video", f"recording_{self.current_session_id}.mp4")
            try:
                self.recorder.start(video_path)
            except Exception as e:
                app_logger.error(f"启动屏幕录制失败: {e}")
        
        # 🆕 启动 EtwMonitor.exe (用于捕获浏览器文件访问)
        try:
            self.etw_launcher = get_etw_launcher()
            if self.etw_launcher.is_available:
                logs_dir = os.path.join(self.current_session_dir, "logs")
                self.etw_launcher.start(logs_dir)
        except Exception as e:
            app_logger.warning(f"启动 EtwMonitor 失败: {e}")

    def _flush_event_buffer(self):
        """将缓存的事件写入日志文件"""
        count = 0
        with self.buffer_lock:
            app_logger.debug(f"正在写入缓存的 {len(self.event_buffer)} 个历史事件...")
            while self.event_buffer:
                event = self.event_buffer.popleft()
                # 写入日志
                self.logger.log_raw_event(event)
                count += 1
        return count
    
    # 注意：不再需要 _start_file_monitor 和 _on_file_event，
    # 因为已经由 _start_monitors 和 _handle_monitor_event 统一接管
    
    def _check_recording_duration(self):
        """检查录制时长是否超过最大值"""
        if self.state != State.RECORDING or not self.recording_start_time:
            return
        
        elapsed = time.time() - self.recording_start_time
        if elapsed >= self.MAX_RECORDING_DURATION:
            print(f"⏰ 录制已达到最大时长 ({self.MAX_RECORDING_DURATION // 60} 分钟)，自动停止")
            self._add_log("warning", f"录制已达到最大时长 ({self.MAX_RECORDING_DURATION // 60} 分钟)")
            self._stop_recording()
    
    def _stop_recording(self):
        """停止录制（但不停止监控器，监控器持续运行直到引擎停止）"""
        self.state = State.IDLE
        
        # 注意：不要在这里停止监控器！
        # 监控器应该持续运行，以便捕获下一次触发事件
        # 监控器只在 _stop_monitors() 中停止
        
        # 关闭日志
        event_count = self.logger.get_event_count()
        keyevent_count = self.logger.get_keyevent_count()
        self.logger.close()
        
        # 计算录制时长
        duration = 0
        if self.recording_start_time:
            duration = time.time() - self.recording_start_time
        
        self._add_log("info", f"录制已停止 (时长: {duration:.1f}s, 事件: {event_count}, 关键事件: {keyevent_count})")
        print(f"🛑 录制已停止")
        print(f"   时长: {duration:.1f} 秒")
        print(f"   事件数: {event_count} (关键事件: {keyevent_count})")
        
        # 停止屏幕录制
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                app_logger.error(f"停止屏幕录制失败: {e}")
        
        # 🆕 停止 EtwMonitor.exe 并合并日志
        # 注意：必须在 logger.close() 之后合并，因为 close() 会将内存中的日志写入文件
        etw_merged_count = 0
        if self.etw_launcher and self.etw_launcher.is_running:
            try:
                self.etw_launcher.stop()
            except Exception as e:
                app_logger.error(f"停止 EtwMonitor 失败: {e}")
        
        # 合并 EtwMonitor 日志到 session 的 logs.json
        if self.etw_launcher and self.current_session_dir:
            try:
                logs_path = os.path.join(self.current_session_dir, "logs", "logs.json")
                etw_merged_count = self.etw_launcher.convert_and_merge(logs_path)
            except Exception as e:
                app_logger.error(f"合并 EtwMonitor 日志失败: {e}")
        
        # 🆕 清理 keyevents.json（过滤临时文件、去重等）
        if self.current_session_dir:
            self._cleanup_keyevents()
        
        # 生成INDEX.md (包含 EtwMonitor 事件数)
        self._generate_index(duration, event_count + etw_merged_count)
        
        self.current_session_id = None
        self.current_session_dir = None
        self.recording_start_time = None
    
    def _cleanup_keyevents(self):
        """合并 ETW 事件到 keyevents.json，然后使用严格的白名单过滤"""
        import re
        import json
        import glob
        
        logs_dir = os.path.join(self.current_session_dir, "logs")
        keyevents_path = os.path.join(logs_dir, "keyevents.json")
        
        # ============ 1. 合并 ETW 事件到 keyevents.json ============
        etw_files = glob.glob(os.path.join(logs_dir, "etw_session_*.json"))
        etw_events = []
        
        for etw_file in etw_files:
            try:
                with open(etw_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                # 跳过 monitor_started/stopped 元事件
                                if 'event' in event:
                                    continue
                                # 转换为 keyevent 格式
                                converted = self._convert_etw_event(event)
                                if converted:
                                    etw_events.append(converted)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                app_logger.warning(f"读取 ETW 日志失败: {e}")
        
        app_logger.info(f"📥 读取 {len(etw_events)} 条 ETW 事件")
        
        # ============ 2. 读取现有 keyevents ============
        existing_events = []
        if os.path.exists(keyevents_path):
            try:
                with open(keyevents_path, 'r', encoding='utf-8') as f:
                    existing_events = json.load(f)
            except Exception:
                existing_events = []
        
        # ============ 3. 合并所有事件 ============
        all_events = existing_events + etw_events
        
        # ============ 4. 严格过滤：使用黑名单路径和扩展名 ============
        
        # 系统路径黑名单（不区分大小写）
        system_path_patterns = [
            r'c:\\windows\\',
            r'c:\\program files\\',
            r'c:\\program files \(x86\)\\',
            r'c:\\programdata\\',
            r'\\appdata\\local\\temp\\',
            r'\\appdata\\local\\google\\chrome\\user data\\',
            r'\\appdata\\local\\microsoft\\',
            r'\\appdata\\roaming\\microsoft\\windows\\recent\\',
            r'\\google\\chrome\\application\\',
            r'\\mozilla firefox\\',
            r'\\microsoft\\edge\\',
        ]
        
        # 系统文件扩展名黑名单
        system_extensions = {
            '.dll', '.exe', '.sys', '.drv', '.ocx',  # 可执行/库文件
            '.pf', '.sdb', '.nls', '.mui', '.cat',    # Windows 系统文件
            '.etl', '.log', '.bak', '.tmp', '.temp',  # 日志/临时文件
            '.lnk', '.url',                           # 快捷方式（一般不需要）
            '.dat', '.db', '.sqlite', '.db-journal',  # 数据库/缓存
            '.manifest', '.config',                   # 配置文件
            '.crdownload', '.partial',                # 下载中的文件
        }
        
        # 有意义的文档扩展名白名单
        meaningful_extensions = {
            # 文档
            '.doc', '.docx', '.pdf', '.txt', '.rtf', '.odt', '.wps',
            '.xls', '.xlsx', '.csv', '.ods',
            '.ppt', '.pptx', '.odp',
            # 图片
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg',
            # 压缩包
            '.zip', '.rar', '.7z', '.tar', '.gz',
            # 代码
            '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs',
            '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md',
            # 其他
            '.mp3', '.mp4', '.avi', '.mov', '.mkv',
        }
        
        def should_keep(event):
            """判断事件是否应该保留"""
            event_type = event.get('event_type', '')
            file_path = event.get('file_path', '')
            
            # 1. 非文件事件（如 app_switch, clipboard）：始终保留
            if not file_path or event_type in ('app_switch', 'clipboard_text', 'clipboard_image', 'clipboard_files'):
                return True
            
            file_path_lower = file_path.lower()
            
            # 2. 检查系统路径黑名单
            for pattern in system_path_patterns:
                if re.search(pattern, file_path_lower):
                    return False
            
            # 3. 获取文件扩展名
            file_ext = event.get('file_extension', '') or os.path.splitext(file_path)[1]
            file_ext_lower = file_ext.lower()
            
            # 4. 系统扩展名黑名单
            if file_ext_lower in system_extensions:
                return False
            
            # 5. 没有扩展名的文件（通常是缓存）
            if not file_ext or file_ext == '.':
                basename = os.path.basename(file_path)
                if basename and '.' not in basename:
                    return False
            
            # 6. 目录访问（路径以 \ 结尾）
            if file_path.endswith('\\') or file_path.endswith('/'):
                return False
            
            # 7. 如果有有意义的扩展名，保留
            if file_ext_lower in meaningful_extensions:
                return True
            
            # 8. 用户目录下的文件，即使扩展名未知也保留
            user_paths = [
                r'\\users\\[^\\]+\\documents\\',
                r'\\users\\[^\\]+\\desktop\\',
                r'\\users\\[^\\]+\\downloads\\',
                r'd:\\',  # D盘通常是用户数据
            ]
            for pattern in user_paths:
                if re.search(pattern, file_path_lower):
                    return True
            
            # 9. 默认过滤掉
            return False
        
        # 过滤
        filtered = [e for e in all_events if should_keep(e)]
        
        # ============ 5. 去重 ============
        seen = set()
        unique = []
        for event in filtered:
            key = (event.get('timestamp', ''), event.get('file_path', ''), event.get('event_type', ''))
            if key not in seen:
                seen.add(key)
                unique.append(event)
        
        # 按时间排序
        unique.sort(key=lambda x: x.get('timestamp', ''))
        
        # 写回
        try:
            with open(keyevents_path, 'w', encoding='utf-8') as f:
                json.dump(unique, f, ensure_ascii=False, indent=2)
            app_logger.info(f"📋 Keyevents: {len(existing_events)} 原始 + {len(etw_events)} ETW → 过滤后 {len(unique)} 条")
        except Exception as e:
            app_logger.error(f"保存 keyevents.json 失败: {e}")
    
    def _convert_etw_event(self, etw_event: dict) -> dict:
        """将 ETW 事件转换为 keyevent 格式"""
        path = etw_event.get('path', '')
        if not path:
            return None
        
        raw_timestamp = etw_event.get('timestamp', '')
        process_name = etw_event.get('process', '')
        pid = etw_event.get('pid', 0)
        
        # 转换时间戳格式
        try:
            from datetime import datetime
            dt = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S")
            timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.000")
        except Exception:
            timestamp = raw_timestamp.replace(" ", "T") + ".000" if raw_timestamp else ""
        
        file_name = os.path.basename(path) if path else ''
        _, file_ext = os.path.splitext(path)
        
        drive_letter = path[:2] if len(path) >= 2 and path[1] == ':' else ''
        
        # 应用名称映射
        app_name_map = {'chrome.exe': 'Chrome', 'msedge.exe': 'Edge', 'firefox.exe': 'Firefox'}
        app_name = app_name_map.get(process_name.lower(), process_name.replace('.exe', ''))
        
        return {
            "timestamp": timestamp,
            "event_type": "created",
            "file_path": path,
            "file_name": file_name,
            "file_size": 0,
            "file_extension": file_ext,
            "process_info": {"pid": str(pid), "process_name": process_name, "process_path": "", "cmdline": ""},
            "window_info": {"window_handle": "", "window_title": "", "window_class": ""},
            "user_info": {"username": os.environ.get("USERNAME", ""), "hostname": os.environ.get("COMPUTERNAME", "")},
            "disk_info": {"drive_letter": drive_letter, "disk_type": "Fixed"},
            "app_name": app_name,
            "extra": {"raw_operation": "browser_file_access", "category": "浏览器文件访问", "source": "etw_monitor"}
        }
    
    def _generate_index(self, duration: float, event_count: int):
        """生成会话索引文件"""
        if not self.current_session_dir:
            return
        
        try:
            # 使用录制开始时间而非当前时间（修复 Case 46/47）
            if self.recording_start_time:
                start_time_str = datetime.fromtimestamp(self.recording_start_time).strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            content = f"""# Recording Session Index

**Session ID**: {self.current_session_id}  
**Recording Time**: {start_time_str}  
**Duration**: {duration:.2f} seconds  
**Event Count**: {event_count}

## File List

### Video Files
- `video/recording_{self.current_session_id}.mp4` - Recorded screen video

### Original Logs
- `logs/events_{self.current_session_id}.json` - Complete monitoring log

---
*Auto-generated by win_monitor*
"""
            index_path = os.path.join(self.current_session_dir, "INDEX.md")
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 生成索引文件: INDEX.md")
        except Exception as e:
            print(f"[ERROR] 生成INDEX.md失败: {e}")
    
    def get_status(self) -> dict:
        """获取当前状态"""
        with self.state_lock:
            status = {
                "state": self.state.value,
                "running": self.running,
                "session_id": self.current_session_id,
                "poll_interval": self.poll_interval,
                "max_recording_duration": self.MAX_RECORDING_DURATION
            }
            
            if self.state == State.RECORDING and self.recording_start_time:
                elapsed = time.time() - self.recording_start_time
                status["recording_duration"] = round(elapsed, 1)
                status["remaining_time"] = max(0, self.MAX_RECORDING_DURATION - elapsed)
                status["event_count"] = self.logger.get_event_count()
            
            if self.last_window:
                status["current_app"] = self.last_window.process_name
                status["current_title"] = self.last_window.window_title[:50]
            
            return status
    
    # ======================== 🆕 文件对话框检测和上传推理 ========================
    
    def _detect_file_operations(self, window_data: WindowData):
        """检测文件对话框和上传操作"""
        # 1. 检测文件对话框打开
        if self._is_file_dialog(window_data):
            self._on_file_dialog_opened(window_data)
        
        # 2. 检测从对话框返回浏览器（可能完成上传）
        elif self.last_dialog_info and self._is_browser_window(window_data):
            self._infer_upload_action(window_data)
    
    def _is_file_dialog(self, window_data: WindowData) -> bool:
        """判断是否是文件选择对话框"""
        # Windows 标准文件对话框的窗口类
        dialog_classes = ['#32770', 'Chrome_WidgetWin_1']  # Win32对话框 和 Chrome内置对话框
        
        # 检查窗口类
        if window_data.window_class not in dialog_classes:
            return False
        
        # 检查窗口标题
        dialog_keywords = ['打开', 'Open', '选择文件', 'Choose File', '上传', 'Upload']
        for keyword in dialog_keywords:
            if keyword in window_data.window_title:
                return True
        
        return False
    
    def _on_file_dialog_opened(self, window_data: WindowData):
        """处理文件对话框打开事件 - 使用 UI Automation 读取选中的文件"""
        # 记录对话框信息
        self.last_dialog_info = {
            'opened_at': datetime.now(),
            'dialog_window': window_data,
            'selected_file': None  # 将在用户选择后尝试捕获
        }
        
        # 记录触发对话框的应用（上一个窗口）
        if self.last_window:
            self.dialog_trigger_window = self.last_window
            
            print(f"📂 检测到文件对话框:")
            print(f"   标题: {window_data.window_title}")
            print(f"   触发应用: {self.last_window.process_name}")
            print(f"   触发窗口: {self.last_window.window_title[:50]}")
        
        # 启动后台线程来监控文件对话框中选择的文件
        import threading
        monitor_thread = threading.Thread(
            target=self._monitor_file_dialog_selection,
            args=(window_data.window_handle,),
            daemon=True,
            name=f"FileDialogMonitor-{window_data.window_handle}"
        )
        monitor_thread.start()
    
    def _monitor_file_dialog_selection(self, dialog_handle):
        """后台监控文件对话框，捕获用户选择的文件路径"""
        try:
            import ctypes
            from ctypes import wintypes
            import time as time_module
            
            user32 = ctypes.windll.user32
            
            # 持续监控对话框直到它关闭
            start_time = time_module.time()
            max_wait = 60  # 最多等待60秒
            last_file_path = None
            
            while time_module.time() - start_time < max_wait:
                # 检查对话框是否还存在
                if not user32.IsWindow(dialog_handle):
                    break
                
                # 尝试读取文件名编辑框的内容
                file_path = self._read_file_dialog_path(dialog_handle)
                if file_path and file_path != last_file_path:
                    last_file_path = file_path
                    print(f"📎 检测到文件选择: {file_path}")
                
                time_module.sleep(0.3)  # 每300ms检查一次
            
            # 对话框关闭了，记录最终选择的文件
            if last_file_path:
                self._record_file_selection(last_file_path)
                
        except Exception as e:
            print(f"[WARN] 监控文件对话框失败: {e}")
    
    def _read_file_dialog_path(self, dialog_handle) -> Optional[str]:
        """从文件对话框读取当前选择的文件路径"""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Windows 文件对话框中，文件名编辑框的控件 ID 通常是 1148 (ComboBox) 或 1152 (Edit)
            # 地址栏的控件 ID 通常是 1001 (ToolbarWindow32 -> Breadcrumb)
            
            # 方法1: 尝试读取 Edit 控件 (控件ID 1148 的子控件)
            # 文件对话框结构: Dialog -> ComboBoxEx32 (1148) -> ComboBox -> Edit
            combo_box = user32.GetDlgItem(dialog_handle, 1148)
            if combo_box:
                # 获取 ComboBox 内的 Edit 控件
                edit_handle = user32.FindWindowExW(combo_box, None, "Edit", None)
                if not edit_handle:
                    edit_handle = user32.FindWindowExW(combo_box, None, "ComboBox", None)
                    if edit_handle:
                        edit_handle = user32.FindWindowExW(edit_handle, None, "Edit", None)
                
                if edit_handle:
                    # 读取编辑框文本
                    length = user32.SendMessageW(edit_handle, 0x000E, 0, 0)  # WM_GETTEXTLENGTH
                    if length > 0:
                        buffer = ctypes.create_unicode_buffer(length + 1)
                        user32.SendMessageW(edit_handle, 0x000D, length + 1, buffer)  # WM_GETTEXT
                        filename = buffer.value.strip()
                        
                        if filename and not filename.startswith("http"):
                            # 尝试获取当前目录
                            current_dir = self._get_file_dialog_current_dir(dialog_handle)
                            if current_dir:
                                import os
                                full_path = os.path.join(current_dir, filename)
                                if os.path.isfile(full_path):
                                    return full_path
                            # 如果是完整路径
                            if len(filename) > 2 and filename[1] == ':':
                                return filename
                            return filename
            
            return None
        except Exception as e:
            return None
    
    def _get_file_dialog_current_dir(self, dialog_handle) -> Optional[str]:
        """获取文件对话框的当前目录"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            
            # 尝试通过 CDM_GETFOLDERPATH 消息获取
            CDM_FIRST = 0x0400 + 100
            CDM_GETFOLDERPATH = CDM_FIRST + 2
            
            buffer = ctypes.create_unicode_buffer(260)
            result = user32.SendMessageW(dialog_handle, CDM_GETFOLDERPATH, 260, buffer)
            
            if result > 0:
                return buffer.value
            
            return None
        except:
            return None
    
    def _record_file_selection(self, file_path: str):
        """记录用户选择的文件"""
        import os
        
        if not file_path:
            return
        
        # 获取触发应用信息
        app_name = ""
        window_title = ""
        process_name = ""
        
        if self.dialog_trigger_window:
            process_name = self.dialog_trigger_window.process_name
            window_title = self.dialog_trigger_window.window_title
            # 规范化应用名称
            pname = process_name
            if pname.lower().endswith('.exe'):
                pname = pname[:-4]
            app_map = {
                "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox",
                "qq": "QQ", "wechat": "微信", "weixin": "微信"
            }
            app_name = app_map.get(pname.lower(), pname)
        
        # 构建文件选择事件
        basename = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        drive = os.path.splitdrive(file_path)[0]
        
        try:
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
        except:
            file_size = 0
        
        event = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "event_type": "file_selected",
            "file_path": file_path,
            "file_name": basename,
            "file_size": file_size,
            "file_extension": ext,
            "process_info": {
                "pid": str(self.dialog_trigger_window.process_id) if self.dialog_trigger_window else "",
                "process_name": process_name,
                "process_path": self.dialog_trigger_window.process_path if self.dialog_trigger_window else "",
                "cmdline": ""
            },
            "window_info": {
                "window_handle": "",
                "window_title": window_title,
                "window_class": ""
            },
            "user_info": {
                "username": os.environ.get("USERNAME", "Unknown"),
                "hostname": os.environ.get("COMPUTERNAME", "Unknown")
            },
            "disk_info": {
                "drive_letter": drive,
                "disk_type": "Fixed"
            },
            "app_name": app_name,
            "extra": {
                "raw_operation": "file_selected",
                "category": "文件上传",
                "source": "file_dialog_monitor",
                "detection_method": "file_dialog"
            }
        }
        
        # 更新 last_dialog_info
        if self.last_dialog_info:
            self.last_dialog_info['selected_file'] = file_path
        
        print(f"✅ 捕获文件选择: {file_path}")
        print(f"   目标应用: {app_name}")
        print(f"   窗口: {window_title[:50]}")
        
        # 发送到日志
        self._handle_monitor_event(event)

    
    def _infer_upload_action(self, window_data: WindowData):
        """推断文件上传操作"""
        # 确保对话框是最近打开的（10秒内，增加窗口以便捕获）
        if not self.last_dialog_info:
            return
        
        time_since_dialog = (datetime.now() - self.last_dialog_info['opened_at']).total_seconds()
        if time_since_dialog > 10.0:  # 超过10秒，认为对话框已过期
            self.last_dialog_info = None
            self.dialog_trigger_window = None
            return
        
        # 检查是否回到了邮件或聊天网站
        if self._is_mail_or_chat_website(window_data):
            # 优先使用从文件对话框直接捕获的文件
            suspected_file = self.last_dialog_info.get('selected_file')
            
            # 如果没有从对话框捕获到，尝试从事件缓冲区查找
            if not suspected_file:
                suspected_file = self._find_recent_accessed_file()
            
            # 提取网站名称
            website_name = self._extract_website_name(window_data.window_title)
            
            # 生成上传推理事件
            upload_event = {
                'action': 'suspected_file_upload',
                'confidence': 'high' if suspected_file else 'medium',
                'evidence': 'file_dialog_sequence',
                'website': website_name,
                'browser': window_data.process_name,
                'window_title': window_data.window_title,
                'dialog_time': self.last_dialog_info['opened_at'].isoformat(),
                'return_time': datetime.now().isoformat()
            }
            
            if suspected_file:
                upload_event['suspected_file'] = suspected_file
            
            # 记录推理日志
            self._log_upload_inference(upload_event)
            
            # 清除对话框状态
            self.last_dialog_info = None
            self.dialog_trigger_window = None
    

    def _is_browser_window(self, window_data: WindowData) -> bool:
        """判断是否是浏览器窗口"""
        browser_processes = self.config.get_browser_processes()
        return window_data.process_name.lower() in [b.lower() for b in browser_processes]
    
    def _is_mail_or_chat_website(self, window_data: WindowData) -> bool:
        """判断是否是邮件或聊天网站"""
        if not self._is_browser_window(window_data):
            return False
        
        # 检查窗口标题中的关键词
        mail_keywords = ['邮箱', 'mail', 'gmail', 'outlook', 'qq邮箱', '163邮箱', '126邮箱']
        chat_keywords = ['微信', 'wechat', 'qq', '钉钉', 'dingtalk', 'slack', 'teams', '飞书', 'lark']
        
        title_lower = window_data.window_title.lower()
        
        for keyword in mail_keywords + chat_keywords:
            if keyword in title_lower:
                return True
        
        return False
    
    def _extract_website_name(self, window_title: str) -> str:
        """从窗口标题中提取网站名称"""
        # QQ邮箱示例: "QQ邮箱 和另外 1 个页面 - 个人 - Microsoft​ Edge"
        # 提取第一个 - 之前的部分
        if ' - ' in window_title:
            parts = window_title.split(' - ')
            # 去掉 "和另外X个页面"
            name = parts[0].split(' 和另外')[0].strip()
            return name
        
        return window_title[:30]  # 返回前30个字符
    
    def _find_recent_accessed_file(self) -> Optional[str]:
        """查找最近访问的文件（从事件缓冲区查找）"""
        try:
            # 查找最近5秒内的用户文件事件
            recent_threshold = 5.0  # 秒
            now = datetime.now()
            
            # 用户文件扩展名（图片、文档等常见上传文件类型）
            user_file_exts = [
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',  # 图片
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # 文档
                '.zip', '.rar', '.7z', '.tar', '.gz',  # 压缩包
                '.txt', '.csv', '.json', '.xml',  # 文本
                '.mp3', '.mp4', '.avi', '.mov', '.wav',  # 多媒体
            ]
            
            # 从事件缓冲区反向查找（最近的在后面）
            with self.buffer_lock:
                for event in reversed(list(self.event_buffer)):
                    # 检查时间戳是否在阈值内
                    timestamp_str = event.get("timestamp", "")
                    if timestamp_str:
                        try:
                            event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            # 移除时区信息以便比较
                            if hasattr(event_time, 'replace'):
                                event_time = event_time.replace(tzinfo=None)
                            time_diff = (now - event_time).total_seconds()
                            if time_diff > recent_threshold:
                                continue  # 太旧了，跳过
                        except (ValueError, TypeError):
                            continue
                    
                    file_path = event.get("file_path", "")
                    file_ext = event.get("file_extension", "").lower()
                    event_type = event.get("event_type", "")
                    
                    # 只检查文件打开或创建事件
                    if event_type not in ["opened", "created", "read"]:
                        continue
                    
                    # 检查是否是用户文件类型
                    if file_ext not in user_file_exts:
                        continue
                    
                    # 过滤临时文件和缓存目录
                    temp_patterns = [
                        "\\Temp\\", "\\Cache\\", "\\AppData\\Local\\",
                        "\\AppData\\Roaming\\", "\\Windows\\", "\\Program Files"
                    ]
                    if any(pattern in file_path for pattern in temp_patterns):
                        continue
                    
                    # 找到符合条件的用户文件
                    print(f"📎 找到最近文件: {file_path}")
                    return file_path
            
            return None
        except Exception as e:
            print(f"[WARN] 查找最近文件失败: {e}")
            return None

    
    def _log_upload_inference(self, upload_event: dict):
        """记录上传推理日志"""
        # 打印到控制台
        print(f"\n🚀 推断文件上传操作:")
        print(f"   网站: {upload_event.get('website', 'Unknown')}")
        print(f"   置信度: {upload_event['confidence']}")
        if upload_event.get('suspected_file'):
            print(f"   疑似文件: {upload_event['suspected_file']}")
        print(f"   证据: {upload_event['evidence']}")
        
        # 写入日志文件
        if self.logger and self.logger.is_open():
            # 创建特殊的日志条目
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'inferred_upload',
                'confidence': upload_event['confidence'],
                'website': upload_event.get('website', ''),
                'browser': upload_event.get('browser', ''),
                'window_title': upload_event.get('window_title', ''),
                'suspected_file': upload_event.get('suspected_file', ''),
                'evidence': upload_event.get('evidence', ''),
                'dialog_opened_at': upload_event.get('dialog_time', ''),
                'returned_at': upload_event.get('return_time', '')
            }
            
            # 直接写入日志（不通过logger的标准流程）
            try:
                import json
                json_line = json.dumps(log_entry, ensure_ascii=False)
                if hasattr(self.logger, 'log_file') and self.logger.log_file:
                    self.logger.log_file.write(json_line + "\n")
                    self.logger.log_file.flush()
            except Exception as e:
                print(f"[ERROR] 写入上传推理日志失败: {e}")

