from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..models import DatalogFact, LeakPath


SUPPORTED_RELATIONS = {
    "OpenFile",
    "TransferFile",
    "CrossProcessTransfer",
    "LeakFile",
    "ClipboardWrite",
    "ClipboardRead",
}


@dataclass(frozen=True)
class _Taint:
    process: str
    data: str
    path: str
    timestamp: int


class DatalogEngine:
    """
    Small Python Datalog-style engine for taint propagation.

    Relations match the spec: OpenFile, TransferFile, CrossProcessTransfer and
    LeakFile. Clipboard facts are converted into cross-process transfers inside
    the engine so callers do not need a second reasoning pass.
    """

    def __init__(self, *_: Any, **__: Any):
        self.facts: dict[str, list[DatalogFact]] = {name: [] for name in SUPPORTED_RELATIONS}

    def add_fact(self, relation: str, *args: Any) -> None:
        if relation not in self.facts:
            raise ValueError(f"unknown relation: {relation}")
        self.facts[relation].append(DatalogFact(relation, tuple(args)))

    def add_clipboard_operation(
        self,
        write_op_id: str,
        write_proc: str,
        read_op_id: str,
        read_proc: str,
        data: str,
        write_ts: int,
        read_ts: int,
    ) -> None:
        self.add_fact("ClipboardWrite", write_op_id, write_proc, data, write_ts)
        self.add_fact("ClipboardRead", read_op_id, read_proc, data, read_ts)

    def query_leak(self) -> list[LeakPath]:
        tainted = self._propagate_taint()
        leaks: list[LeakPath] = []
        seen: set[tuple[str, str, str]] = set()

        for leak in self.facts["LeakFile"]:
            leak_id, process, file_path, channel, timestamp = _pad(leak.args, 5)
            for item in tainted:
                if item.process != str(process) or item.data != str(file_path):
                    continue
                key = (str(process), str(file_path), str(channel))
                if key in seen:
                    continue
                seen.add(key)
                leaks.append(
                    LeakPath(
                        start_op=item.path.split(" -> ")[0],
                        end_op=str(leak_id),
                        leaking_proc=str(process),
                        leaked_file=str(file_path),
                        full_path=f"{item.path} -> {leak_id}",
                        leak_channel=str(channel),
                        leak_timestamp=_int(timestamp),
                    )
                )
        return leaks

    def cleanup(self) -> None:
        self.facts = {name: [] for name in SUPPORTED_RELATIONS}

    def __enter__(self) -> "DatalogEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.cleanup()

    def _propagate_taint(self) -> set[_Taint]:
        transfers = list(self.facts["CrossProcessTransfer"]) + self._clipboard_transfers()
        tainted: set[_Taint] = set()
        best: dict[tuple[str, str], _Taint] = {}
        frontier: deque[_Taint] = deque()

        for fact in self.facts["OpenFile"]:
            op_id, process, file_path, timestamp = _pad(fact.args, 4)
            item = _Taint(str(process), str(file_path), str(op_id), _int(timestamp))
            _remember_taint(item, tainted, best, frontier)

        while frontier:
            current = frontier.popleft()
            for fact in self.facts["TransferFile"]:
                op_id, process, src, dst, timestamp = _pad(fact.args, 5)
                if current.process == str(process) and current.data == str(src):
                    _remember_taint(
                        _Taint(str(process), str(dst), f"{current.path} -> {op_id}", _int(timestamp)),
                        tainted,
                        best,
                        frontier,
                    )

            for fact in transfers:
                op_id, from_proc, to_proc, data, timestamp = _pad(fact.args, 5)
                if current.process == str(from_proc) and current.data == str(data):
                    _remember_taint(
                        _Taint(str(to_proc), str(data), f"{current.path} -> {op_id}", _int(timestamp)),
                        tainted,
                        best,
                        frontier,
                    )
        return tainted

    def _clipboard_transfers(self) -> list[DatalogFact]:
        derived: list[DatalogFact] = []
        for write in self.facts["ClipboardWrite"]:
            write_id, write_proc, data, write_ts = _pad(write.args, 4)
            for read in self.facts["ClipboardRead"]:
                read_id, read_proc, read_data, read_ts = _pad(read.args, 4)
                if data != read_data or write_proc == read_proc:
                    continue
                if _int(read_ts) <= _int(write_ts) or _int(read_ts) - _int(write_ts) > 300_000:
                    continue
                derived.append(
                    DatalogFact(
                        "CrossProcessTransfer",
                        (f"{write_id}_{read_id}", write_proc, read_proc, data, read_ts),
                    )
                )
        return derived


class PromptTemplates:
    """Minimal prompt boundary retained for callers that still import it."""

    @staticmethod
    def get_messages(logs: list[dict[str, Any]], video_frames: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "你是数据泄露分析助手，请把证据整理为 OpenFile/TransferFile/LeakFile 事实。",
            },
            {
                "role": "user",
                "content": f"logs={len(logs)} frames={len(video_frames)}",
            },
        ]


def _remember_taint(
    item: _Taint,
    tainted: set[_Taint],
    best: dict[tuple[str, str], _Taint],
    frontier: deque[_Taint],
) -> None:
    key = (item.process, item.data)
    existing = best.get(key)
    if existing is not None and len(existing.path) <= len(item.path):
        return
    if existing is not None:
        tainted.discard(existing)
    best[key] = item
    tainted.add(item)
    frontier.append(item)


def _pad(args: tuple[Any, ...], size: int) -> tuple[Any, ...]:
    return tuple(args) + tuple("" for _ in range(max(0, size - len(args))))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["DatalogEngine", "PromptTemplates", "LeakPath"]
