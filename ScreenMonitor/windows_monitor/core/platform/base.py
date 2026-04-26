
import abc
import os
import sys

class PlatformBase(abc.ABC):
    """Platform-specific interface"""
    
    @abc.abstractmethod
    def is_admin(self) -> bool:
        """Check if running with highest privileges"""
        pass
        
    @abc.abstractmethod
    def get_drives(self) -> list:
        """Get list of mount points/drives to monitor"""
        pass
        
    @abc.abstractmethod
    def get_user_name(self) -> str:
        """Get current username"""
        pass
        
    @abc.abstractmethod
    def get_active_window(self) -> dict:
        """Get active window title and process name"""
        pass
    
    @abc.abstractmethod
    def start_file_dialog_monitor(self, callback):
        """Start monitoring file dialogs if supported"""
        pass

    @abc.abstractmethod
    def stop_file_dialog_monitor(self):
        """Stop file dialog monitor"""
        pass
        
    def is_windows(self):
        return sys.platform == 'win32'
