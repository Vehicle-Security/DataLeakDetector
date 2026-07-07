"""Supported symbolic relations and internal taint state.

The engine imports this module as its vocabulary. Centralizing relation names
keeps generated facts, tests, and future prompt output aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_RELATIONS = {
    "OpenFile",
    "TransferFile",
    "CrossProcessTransfer",
    "LeakFile",
    "ClipboardWrite",
    "ClipboardRead",
}


@dataclass(frozen=True)
class Taint:
    """Internal taint state carried during symbolic propagation."""

    process: str
    data: str
    path: str
    timestamp: int
