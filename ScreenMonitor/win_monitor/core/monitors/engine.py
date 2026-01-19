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

from .sensor import Sensor, WindowData
from .file_system_monitor import FileSystemMonitor
from .clipboard_monitor import ClipboardMonitor
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
        print(f"🚀 监控引擎已启动")
        print(f"   轮询间隔: {self.poll_interval}s")
        print(f"   最大录制时长: {self.MAX_RECORDING_DURATION // 60} 分钟")
        
        # 1. 启动所有监控器 (持续运行)
        self._start_monitors()
        
        # 2. 启动主循环线程
        self.monitor_thread = threading.Thread(target=self._main_loop, daemon=True)
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
        print("🛑 监控引擎已停止")
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
            print("📂 文件系统监控已启动 (watchdog)")
        except Exception as e:
            print(f"[ERROR] 启动 watchdog 监控失败: {e}")

        # 2. ETW 文件打开监控
        try:
            from .etw_file_monitor import ETWFileMonitor
            self.etw_monitor = ETWFileMonitor(event_callback=self._handle_monitor_event)
            self.etw_monitor.start()
            print("📂 ETW 文件监控已启动 (file open events)")
        except ImportError:
            print("[WARN] ETW 监控不可用 (pywintrace 未安装)")
        except Exception as e:
            print(f"[WARN] ETW 监控启动失败: {e}")
            
        # 3. 剪贴板监控
        try:
            self.clipboard_monitor = ClipboardMonitor(event_callback=self._handle_monitor_event)
            self.clipboard_monitor.start()
            print("📋 剪贴板监控已启动")
        except Exception as e:
            print(f"[ERROR] 启动剪贴板监控失败: {e}")

    def _stop_monitors(self):
        """停止所有底层监控器"""
        if self.file_monitor:
            try:
                self.file_monitor.stop()
            except Exception as e:
                print(f"[ERROR] 停止 watchdog 失败: {e}")
            self.file_monitor = None
            
        if hasattr(self, 'etw_monitor') and self.etw_monitor:
            try:
                self.etw_monitor.stop()
            except Exception as e:
                print(f"[ERROR] 停止 ETW 失败: {e}")
            self.etw_monitor = None
            
        if self.clipboard_monitor:
            try:
                self.clipboard_monitor.stop()
            except Exception as e:
                print(f"[ERROR] 停止剪贴板监控失败: {e}")
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
                print(f"[ERROR] 主循环异常: {e}")
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
    
    def _start_recording(self, match_result: MatchResult):
        """开始录制"""
        self.state = State.RECORDING
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_start_time = time.time()
        
        # 创建会话目录
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
                print(f"[ERROR] 启动屏幕录制失败: {e}")

    def _flush_event_buffer(self):
        """将缓存的事件写入日志文件"""
        count = 0
        with self.buffer_lock:
            print(f"📥 正在写入缓存的 {len(self.event_buffer)} 个历史事件...")
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
                print(f"[ERROR] 停止屏幕录制失败: {e}")
        
        # 生成INDEX.md
        self._generate_index(duration, event_count)
        
        self.current_session_id = None
        self.current_session_dir = None
        self.recording_start_time = None
    
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
        """处理文件对话框打开事件"""
        # 记录对话框信息
        self.last_dialog_info = {
            'opened_at': datetime.now(),
            'dialog_window': window_data
        }
        
        # 记录触发对话框的应用（上一个窗口）
        if self.last_window:
            self.dialog_trigger_window = self.last_window
            
            print(f"📂 检测到文件对话框:")
            print(f"   标题: {window_data.window_title}")
            print(f"   触发应用: {self.last_window.process_name}")
            print(f"   触发窗口: {self.last_window.window_title[:50]}")
    
    def _infer_upload_action(self, window_data: WindowData):
        """推断文件上传操作"""
        # 确保对话框是最近打开的（5秒内）
        if not self.last_dialog_info:
            return
        
        time_since_dialog = (datetime.now() - self.last_dialog_info['opened_at']).total_seconds()
        if time_since_dialog > 5.0:  # 超过5秒，认为对话框已过期
            self.last_dialog_info = None
            self.dialog_trigger_window = None
            return
        
        # 检查是否回到了邮件或聊天网站
        if self._is_mail_or_chat_website(window_data):
            # 尝试查找最近访问的文件
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
        browser_processes = ['msedge.exe', 'chrome.exe', 'firefox.exe', 'brave.exe', 'opera.exe']
        return window_data.process_name.lower() in browser_processes
    
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
        """查找最近访问的文件（从监控到的文件事件中查找）"""
        if not self.file_monitor:
            return None
        
        try:
            # 从 file_monitor 获取最近的文件事件
            # 查找最近3秒内打开的 .zip, .docx, .pdf 等文件
            recent_threshold = 3.0  # 秒
            current_time = datetime.now()
            
            # 这里需要file_monitor提供一个方法来获取最近的文件
            # 暂时先返回 None，后续可以增强
            # TODO: 实现文件监控器的 get_recent_files 方法
            
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

