"""Prompt boundary for future LLM-assisted fact extraction.

The runtime does not depend on an LLM today, but keeping prompt construction in
one small module documents where language-model extraction would plug into the
same Datalog relations used by the deterministic correlator.
"""

from __future__ import annotations

from typing import Any


class PromptTemplates:
    """Prompt boundary kept explicit for future LLM fact extraction."""

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
