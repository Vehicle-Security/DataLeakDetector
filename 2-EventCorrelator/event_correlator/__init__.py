from .config import EventCorrelatorConfig
from .correlator import EventCorrelator
from .frontend import classify_frontend_app
from .service import InMemoryCorrelationService
from .windows import build_sensitive_windows

__all__ = [
    "EventCorrelator",
    "EventCorrelatorConfig",
    "InMemoryCorrelationService",
    "build_sensitive_windows",
    "classify_frontend_app",
]
