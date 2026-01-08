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
    
    录制策略：
    - 检测到黑名单应用时开始录制
    - 持续录制直到用户停止监控或超过1小时
    """
    
    # 日志缓冲区最大容量
    MAX_LOG_BUFFER = 500
    
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
        
        # 文件系统监控器（watchdog）
        self.file_monitor: Optional[FileSystemMonitor] = None
        
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
        
        # 启动主循环线程
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
        
        self._add_log("info", "监控引擎已停止")
        print("🛑 监控引擎已停止")
        return True
    
    # Web UI 别名
    stop_monitoring = stop
    
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
                        
                        # 4. 日志记录（持续记录窗口切换）
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
        """状态机处理 - 简化逻辑：检测到风险就开始录制，直到停止或超时"""
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
        
        # 创建会话目录 (只使用 logs 和 video)
        self.current_session_dir = os.path.join(self.output_dir, f"session_{self.current_session_id}")
        os.makedirs(os.path.join(self.current_session_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.current_session_dir, "video"), exist_ok=True)
        
        # 启动日志记录
        log_path = os.path.join(self.current_session_dir, "logs", f"events_{self.current_session_id}.json")
        self.logger.open(log_path, time.time())
        
        self._add_log("info", f"开始录制 - 触发: {match_result.app_name} ({match_result.category})")
        print(f"🎬 开始录制 - 触发: {match_result.app_name} ({match_result.category})")
        print(f"   会话目录: {self.current_session_dir}")
        
        # 启动文件系统监控
        self._start_file_monitor()
        
        # 启动屏幕录制
        if self.recorder:
            video_path = os.path.join(self.current_session_dir, "video", f"recording_{self.current_session_id}.mp4")
            try:
                self.recorder.start(video_path)
            except Exception as e:
                print(f"[ERROR] 启动屏幕录制失败: {e}")
    
    def _start_file_monitor(self):
        """启动文件系统监控"""
        try:
            self.file_monitor = FileSystemMonitor(
                event_callback=self._on_file_event
            )
            self.file_monitor.start()
            print("📂 文件系统监控已启动")
        except Exception as e:
            print(f"[ERROR] 启动文件监控失败: {e}")
    
    def _on_file_event(self, event: dict):
        """处理文件系统事件"""
        if self.state != State.RECORDING:
            return
        
        # 记录到日志文件
        self.logger.log_file_event(event)
        
        # 添加到 Web UI 日志缓冲区
        event_type = event.get("event_type", "")
        file_name = event.get("file_name", "")
        self._add_log("file", f"[{event_type}] {file_name}", {
            "event_type": event_type,
            "file_path": event.get("file_path", ""),
            "file_name": file_name
        })
    
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
        """停止录制"""
        self.state = State.IDLE
        
        # 停止文件系统监控
        if self.file_monitor:
            try:
                self.file_monitor.stop()
                self.file_monitor = None
                print("📂 文件系统监控已停止")
            except Exception as e:
                print(f"[ERROR] 停止文件监控失败: {e}")
        
        # 关闭日志
        event_count = self.logger.get_event_count()
        self.logger.close()
        
        # 计算录制时长
        duration = 0
        if self.recording_start_time:
            duration = time.time() - self.recording_start_time
        
        self._add_log("info", f"录制已停止 (时长: {duration:.1f}s, 事件: {event_count})")
        print(f"🛑 录制已停止")
        print(f"   时长: {duration:.1f} 秒")
        print(f"   事件数: {event_count}")
        
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
            content = f"""# Recording Session Index

**Session ID**: {self.current_session_id}  
**Recording Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
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
