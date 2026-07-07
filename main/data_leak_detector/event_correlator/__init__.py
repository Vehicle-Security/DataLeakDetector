"""EventCorrelator 包边界。

本文件只暴露关联器及其少量支撑契约。下面的独立模块把分类、谱系、候选提取和事实生成
分开处理，从而便于独立演进。
"""

from __future__ import annotations

from .classification import classify_frontend_app
from .config import EventCorrelatorConfig
from .correlator import EventCorrelator
from .lineage import Lineage

__all__ = ["EventCorrelator", "EventCorrelatorConfig", "Lineage", "classify_frontend_app"]
