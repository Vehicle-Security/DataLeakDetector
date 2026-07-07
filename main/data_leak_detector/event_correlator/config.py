"""确定性事件关联的配置对象。

这些参数与关联器分离开来，这样测试和未来部署就可以调整时间窗口与默认置信度，而不用修改工作流代码。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventCorrelatorConfig:
    """用于确定性证据绑定的一组小参数。"""

    nearby_window_ms: int = 5 * 60 * 1000
    upload_confidence: float = 0.86
    transfer_confidence: float = 0.72
