
import os
import sys
import subprocess
import psutil
from pathlib import Path
from .base import PlatformBase

class MacOSPlatform(PlatformBase):
    def is_admin(self) -> bool:
        return os.geteuid() == 0
        
    def get_drives(self) -> list:
        # Monitor root and user directory volumes
        paths = ["/"]
        # Can add /Volumes/* if needed
        return paths
        
    def get_user_name(self) -> str:
        return os.environ.get('USER', 'unknown')
        
    def get_active_window(self) -> dict:
        # Use AppleScript to get frontmost app
        script = 'tell application "System Events" to get name of first application process whose frontmost is true'
        try:
            proc_name = subprocess.check_output(['osascript', '-e', script]).decode().strip()
            # Title is hard to get without accessibility permissions, returning app name as title too
            return {"title": proc_name, "process": proc_name, "pid": 0}
        except:
            return {"title": "", "process": "", "pid": 0}

    def start_file_dialog_monitor(self, callback):
        # Initial version: No UIA equivalent monitoring
        # Relying on BrowserFileMonitor instead
        print("[PLATFORM] MacOS file dialog detection relies on File System Events (Watchdog)")
        return False

    def stop_file_dialog_monitor(self):
        pass
