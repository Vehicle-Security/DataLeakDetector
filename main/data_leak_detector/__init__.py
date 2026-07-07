"""重写版 DataLeakDetector 包的公共导入入口。

这里只导出稳定的入口。保持这个文件足够小，可以清楚地表明包根目录是一个契约边界，
而不是阶段实现的另一份拷贝。
"""

from __future__ import annotations

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
