
import sys
from .base import PlatformBase

_current_platform = None

def get_platform() -> PlatformBase:
    global _current_platform
    if _current_platform:
        return _current_platform
        
    if sys.platform == 'win32':
        from .windows import WindowsPlatform
        _current_platform = WindowsPlatform()
    elif sys.platform == 'darwin':
        from .macos import MacOSPlatform
        _current_platform = MacOSPlatform()
    else:
        raise NotImplementedError(f"Platform {sys.platform} not supported")
        
    return _current_platform
