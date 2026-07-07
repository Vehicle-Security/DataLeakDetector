"""未来 LLM 辅助事实提取的提示边界。

当前运行时并不依赖 LLM，但把提示构造放在一个小模块里，可以清楚地说明语言模型抽取会插入到哪里，
并且与确定性关联器使用同一组 Datalog 关系。
"""

from __future__ import annotations

from typing import Any


class PromptTemplates:
    """保持显式的提示边界，供未来的 LLM 事实提取使用。"""

    @staticmethod
    def get_messages(logs: list[dict[str, Any]], frame_observations: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Extract OpenFile, TransferFile, CrossProcessTransfer, and LeakFile facts from evidence.",
            },
            {
                "role": "user",
                "content": f"logs={len(logs)} frame_observations={len(frame_observations)}",
            },
        ]
