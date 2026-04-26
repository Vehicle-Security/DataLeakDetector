
import os
import sys
import ctypes
import win32api
import win32gui
import win32process
import psutil
from .base import PlatformBase
from core.detection.file_dialog_detector import get_file_dialog_detector

class WindowsPlatform(PlatformBase):
    def __init__(self):
        self._detector = None

    def is_admin(self) -> bool:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
            
    def get_drives(self) -> list:
        drives = []
        try:
            bitmask = win32api.GetLogicalDrives()
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if bitmask & 1:
                    drives.append(letter + ':\\')
                bitmask >>= 1
        except:
            drives = ['C:\\']
        return drives
        
    def get_user_name(self) -> str:
        try:
            return win32api.GetUserName()
        except:
            return os.environ.get('USERNAME', 'unknown')
            
    def get_active_window(self) -> dict:
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
            except:
                proc_name = "Unknown"
            return {"title": title, "process": proc_name, "pid": pid}
        except:
            return {"title": "", "process": "", "pid": 0}
            
    def start_file_dialog_monitor(self, callback):
        try:
            self._detector = get_file_dialog_detector(callback)
            self._detector.start_monitoring()
            return True
        except Exception as e:
            print(f"[PLATFORM] Failed to start UIA detector: {e}")
            return False

    def stop_file_dialog_monitor(self):
        if self._detector:
            self._detector.stop_monitoring()
