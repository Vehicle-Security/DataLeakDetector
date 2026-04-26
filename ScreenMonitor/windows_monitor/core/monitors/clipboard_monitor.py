# -*- coding: utf-8 -*-
"""
clipboard_monitor.py - 剪贴板持续监控模块

监控剪贴板变化，捕获文本和图片（含截图）事件。
长时间运行，独立于黑名单检测。
"""

import os
import time
import hashlib
import threading
import socket
from datetime import datetime
from typing import Optional, Callable, Dict, Any

from ..utils import app_logger

try:
    import win32clipboard
    import win32con
    import win32gui
    import win32api
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    app_logger.warning("[CLIPBOARD] Warning: pywin32 not installed, clipboard monitoring disabled")


class ClipboardMonitor:
    """
    剪贴板监控器
    
    监控剪贴板内容变化，区分文本和图片，记录来源进程。
    """
    
    def __init__(self, event_callback: Optional[Callable[[Dict], None]] = None):
        """
        Args:
            event_callback: 事件回调函数，接收剪贴板事件字典
        """
        self.event_callback = event_callback
        self.is_running = False
        self._monitor_thread = None
        self._last_text_hash = None
        self._last_image_hash = None
        self._poll_interval = 0.5  # 轮询间隔（秒）
        self._last_sequence_number = 0
        
        self.username = os.environ.get("USERNAME", "Unknown")
        self.hostname = socket.gethostname()
        
        # 尝试获取 GetClipboardSequenceNumber
        self._get_seq_num = None
        try:
            import ctypes
            self._get_seq_num = ctypes.windll.user32.GetClipboardSequenceNumber
        except:
            pass
    
    def start(self):
        """启动监控"""
        if not HAS_WIN32:
            app_logger.warning("[CLIPBOARD] Cannot start: pywin32 not available")
            return False
        
        if self.is_running:
            app_logger.warning("[CLIPBOARD] Already running")
            return False
        
        self.is_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="ClipboardMonitor")
        self._monitor_thread.start()
        app_logger.info("[CLIPBOARD] Monitor started")
        return True
    
    def stop(self):
        """停止监控"""
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        app_logger.info("[CLIPBOARD] Monitor stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._check_clipboard()
            except Exception as e:
                # 剪贴板可能被其他程序占用
                pass
            
            time.sleep(self._poll_interval)
    
    def _check_clipboard(self):
        """检查剪贴板内容"""
        # 优化: 使用 SequenceNumber 检测变化
        if self._get_seq_num:
            seq_num = self._get_seq_num()
            if seq_num == self._last_sequence_number:
                return  # 内容未变，跳过
            self._last_sequence_number = seq_num

        try:
            win32clipboard.OpenClipboard()
            
            # 只有内容变化时才获取进程信息，节省 CPU
            process_info = None
            
            # 检查文本
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                text_hash = self._hash_content(text)
                
                if text_hash != self._last_text_hash:
                    self._last_text_hash = text_hash
                    # 延迟获取进程信息
                    if not process_info:
                        process_info = self._get_active_process()
                        
                    self._emit_event("clipboard_text", {
                        "content_hash": text_hash,
                        "content_length": len(text),
                        "content_preview": text[:100] if len(text) > 100 else text,
                        "process_info": process_info
                    })
            
            # 检查图片（截图）
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_DIB):
                try:
                    image_data = win32clipboard.GetClipboardData(win32con.CF_DIB)
                    image_hash = self._hash_content(image_data)
                    
                    if image_hash != self._last_image_hash:
                        self._last_image_hash = image_hash
                        # 延迟获取进程信息
                        if not process_info:
                            process_info = self._get_active_process()
                            
                        self._emit_event("clipboard_image", {
                            "content_hash": image_hash,
                            "image_size": len(image_data),
                            "process_info": process_info
                        })
                except:
                    pass
            
            win32clipboard.CloseClipboard()
            
        except Exception as e:
            try:
                win32clipboard.CloseClipboard()
            except:
                pass
    
    def _get_active_process(self) -> Dict[str, str]:
        """获取当前活动窗口的进程信息"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            import psutil
            proc = psutil.Process(pid)
            
            return {
                "pid": str(pid),
                "process_name": proc.name(),
                "process_path": proc.exe() if proc.exe() else ""
            }
        except:
            return {
                "pid": "",
                "process_name": "",
                "process_path": ""
            }

    
    def _hash_content(self, content) -> str:
        """计算内容哈希"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.md5(content).hexdigest()[:16]
    
    def _emit_event(self, event_type: str, data: Dict):
        """发送事件"""
        event = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "event_type": event_type,
            "content_hash": data.get("content_hash", ""),
            "process_info": data.get("process_info", {}),
            "user_info": {
                "username": self.username,
                "hostname": self.hostname
            },
            "detection_method": "clipboard_monitor"
        }
        
        # 添加额外数据
        if event_type == "clipboard_text":
            event["content_preview"] = data.get("content_preview", "")
            event["content_length"] = data.get("content_length", 0)
            emoji = "📋"
        else:
            event["image_size"] = data.get("image_size", 0)
            emoji = "🖼️"
        
        proc_name = data.get("process_info", {}).get("process_name", "Unknown")
        app_logger.info(f"{emoji} [{event_type}] from {proc_name}")
        
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                app_logger.error(f"[CLIPBOARD] Callback error: {e}")



def get_clipboard_monitor(callback=None) -> ClipboardMonitor:
    """获取剪贴板监控器实例"""
    return ClipboardMonitor(event_callback=callback)


if __name__ == "__main__":
    # 测试
    def print_event(event):
        print(f"Event: {event['event_type']} - {event.get('content_hash', '')}")
    
    monitor = ClipboardMonitor(event_callback=print_event)
    monitor.start()
    
    try:
        print("Monitoring clipboard... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
