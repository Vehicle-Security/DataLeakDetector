"""用于已确认泄露路径的确定性污点传播引擎。

原始项目带有更重的推理脚手架。本重写把核心语义保留在纯 Python 中，以便单人迭代、测试和本地调试
保持快速，同时仍然建模源、传输、进程跨越、剪贴板和汇聚点关系。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..models import DatalogFact, LeakPath
from .relations import SUPPORTED_RELATIONS, Taint


class DatalogEngine:
    """
    小型 Datalog 风格的污点引擎。

    它用纯 Python 实现了根 README 中描述的关系，这样既能让单人迭代保持快速，也能让单元测试具备确定性。
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
        tainted = self._propagate()
        tainted_by_process_data = {(item.process, item.data): item for item in tainted}
        leaks: list[LeakPath] = []
        seen: set[tuple[str, str, str]] = set()

        for fact in self.facts["LeakFile"]:
            leak_id, process, data, channel, timestamp = _pad(fact.args, 5)
            item = tainted_by_process_data.get((str(process), str(data)))
            if item is None:
                continue
            key = (str(process), str(data), str(channel))
            if key in seen:
                continue
            seen.add(key)
            leaks.append(
                LeakPath(
                    start_op=item.path.split(" -> ")[0],
                    end_op=str(leak_id),
                    leaking_proc=str(process),
                    leaked_file=str(data),
                    leak_channel=str(channel),
                    leak_timestamp=_int(timestamp),
                    full_path=f"{item.path} -> {leak_id}",
                )
            )
        return leaks

    def cleanup(self) -> None:
        self.facts = {name: [] for name in SUPPORTED_RELATIONS}

    def __enter__(self) -> "DatalogEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.cleanup()

    def _propagate(self) -> set[Taint]:
        tainted: set[Taint] = set()
        best: dict[tuple[str, str], Taint] = {}
        frontier: deque[Taint] = deque()
        transfer_by_source: dict[tuple[str, str], list[DatalogFact]] = {}
        for fact in self.facts["TransferFile"]:
            _, process, src, _, _ = _pad(fact.args, 5)
            transfer_by_source.setdefault((str(process), str(src)), []).append(fact)

        for fact in self.facts["OpenFile"]:
            op_id, process, data, timestamp = _pad(fact.args, 4)
            _remember(Taint(str(process), str(data), str(op_id), _int(timestamp)), tainted, best, frontier)

        cross_process = list(self.facts["CrossProcessTransfer"]) + self._clipboard_transfers()
        cross_by_source: dict[tuple[str, str], list[DatalogFact]] = {}
        for fact in cross_process:
            _, from_proc, _, data, _ = _pad(fact.args, 5)
            cross_by_source.setdefault((str(from_proc), str(data)), []).append(fact)

        while frontier:
            current = frontier.popleft()
            for fact in transfer_by_source.get((current.process, current.data), ()):
                op_id, process, src, dst, timestamp = _pad(fact.args, 5)
                _remember(Taint(str(process), str(dst), f"{current.path} -> {op_id}", _int(timestamp)), tainted, best, frontier)

            for fact in cross_by_source.get((current.process, current.data), ()):
                op_id, from_proc, to_proc, data, timestamp = _pad(fact.args, 5)
                _remember(Taint(str(to_proc), str(data), f"{current.path} -> {op_id}", _int(timestamp)), tainted, best, frontier)
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
                derived.append(DatalogFact("CrossProcessTransfer", (f"{write_id}_{read_id}", write_proc, read_proc, data, read_ts)))
        return derived


def _remember(
    item: Taint,
    tainted: set[Taint],
    best: dict[tuple[str, str], Taint],
    frontier: deque[Taint],
) -> None:
    key = (item.process, item.data)
    current = best.get(key)
    if current is not None and len(current.path) <= len(item.path):
        return
    if current is not None:
        tainted.discard(current)
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
