# -*- coding: utf-8 -*-
"""
engine.py - 监控引擎（主循环/状态机）
职责：协调各组件，实现主循环逻辑
- 从 Sensor 获取数据
- 用 RuleMatcher 判断是否命中
- 用 Logger 记录日志
- 控制 RecorderService 启停

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
from .rule_matcher import RuleMatcher, MatchResult
from .logger import Logger


class State(Enum):
    """引擎状态"""
    IDLE = "idle"  # 空闲，未录制
    RECORDING = "recording"  # 录制中
    COOLDOWN = "cooldown"  # 冷却期（风险消失后的缓冲）


class Engine:
    """
    监控引擎 - 状态机实现
    
    核心循环：
    1. Sensor.get_active_window() -> WindowData
    2. RuleMatcher.match(WindowData) -> MatchResult
    3. 状态机决策：是否开始/停止录制
    4. Logger.log() 记录事件
    """
    
    # 日志缓冲区最大容量
    MAX_LOG_BUFFER = 100
    
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
        
        # 录制服务（可选）
        self.recorder = recorder_service
        
        # 配置
        self.output_dir = output_dir
        self.poll_interval = config_loader.get_poll_interval_seconds()
        self.buffer_time = config_loader.get_buffer_time_seconds()
        
        # 状态机
        self.state = State.IDLE
        self.state_lock = threading.Lock()
        
        # 运行状态
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 窗口变化检测
        self.last_window: Optional[WindowData] = None
        
        # 冷却相关
        self.cooldown_start: Optional[float] = None
        
        # 会话信息
        self.current_session_id: Optional[str] = None
        
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
        
        self._add_log("info", f"监控引擎已启动 (轮询间隔: {self.poll_interval}s, 冷却时间: {self.buffer_time}s)")
        print(f"🚀 监控引擎已启动")
        print(f"   轮询间隔: {self.poll_interval}s")
        print(f"   冷却时间: {self.buffer_time}s")
        
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
        if self.state != State.IDLE:
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
                        
                        # 3. 日志记录（仅记录变化）
                        if self.state == State.RECORDING:
                            self.logger.log(window_data, match_result, time.time())
                        
                        # 4. 状态机决策
                        self._process_state(match_result, window_data)
                        
                        # 更新上次窗口
                        self.last_window = window_data
                
                # 处理冷却逻辑
                self._check_cooldown()
                
            except Exception as e:
                print(f"[ERROR] 主循环异常: {e}")
            
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
                elif self.state == State.COOLDOWN:
                    # COOLDOWN -> RECORDING（取消冷却）
                    print("🔄 冷却期间检测到风险，继续录制")
                    self.state = State.RECORDING
                    self.cooldown_start = None
            else:
                # 未命中规则
                if self.state == State.RECORDING:
                    # RECORDING -> COOLDOWN
                    self._enter_cooldown()
    
    def _start_recording(self, match_result: MatchResult):
        """开始录制"""
        self.state = State.RECORDING
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建会话目录
        session_dir = os.path.join(self.output_dir, f"session_{self.current_session_id}")
        os.makedirs(os.path.join(session_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "video"), exist_ok=True)
        
        # 启动日志记录
        log_path = os.path.join(session_dir, "logs", f"events_{self.current_session_id}.json")
        self.logger.open(log_path, time.time())
        
        print(f"🎬 开始录制 - 触发: {match_result.app_name} ({match_result.category})")
        
        # 启动屏幕录制
        if self.recorder:
            video_path = os.path.join(session_dir, "video", f"recording_{self.current_session_id}.mp4")
            try:
                self.recorder.start(video_path)
            except Exception as e:
                print(f"[ERROR] 启动屏幕录制失败: {e}")
    
    def _enter_cooldown(self):
        """进入冷却期"""
        self.state = State.COOLDOWN
        self.cooldown_start = time.time()
        print(f"⏳ 风险消失，进入 {self.buffer_time} 秒冷却期...")
    
    def _check_cooldown(self):
        """检查冷却是否结束"""
        with self.state_lock:
            if self.state == State.COOLDOWN and self.cooldown_start:
                elapsed = time.time() - self.cooldown_start
                if elapsed >= self.buffer_time:
                    self._stop_recording()
    
    def _stop_recording(self):
        """停止录制"""
        self.state = State.IDLE
        self.cooldown_start = None
        
        # 关闭日志
        self.logger.close()
        
        print("🛑 录制已停止")
        
        # 停止屏幕录制
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                print(f"[ERROR] 停止屏幕录制失败: {e}")
        
        self.current_session_id = None
    
    def get_status(self) -> dict:
        """获取当前状态"""
        with self.state_lock:
            status = {
                "state": self.state.value,
                "running": self.running,
                "session_id": self.current_session_id,
                "poll_interval": self.poll_interval,
                "buffer_time": self.buffer_time
            }
            
            if self.state == State.COOLDOWN and self.cooldown_start:
                elapsed = time.time() - self.cooldown_start
                status["cooldown_remaining"] = max(0, self.buffer_time - elapsed)
            
            if self.last_window:
                status["current_app"] = self.last_window.process_name
                status["current_title"] = self.last_window.window_title[:50]
            
            return status
