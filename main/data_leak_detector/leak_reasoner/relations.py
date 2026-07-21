"""支持的符号关系与内部污点状态。

引擎把本模块作为自己的词汇表。把关系名称集中管理，可以让生成的事实、测试和未来的提示输出保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_RELATIONS = {
    "OpenFile",
    "TransferFile",
    "CrossProcessTransfer",
    "UploadBinding",
    "LeakFile",
    "SuspiciousBehavior",
    "ClipboardWrite",
    "ClipboardRead",
}


@dataclass(frozen=True)
class Taint:
    """符号传播过程中携带的内部污点状态。"""

    process: str
    data: str
    path: str
    timestamp: int
