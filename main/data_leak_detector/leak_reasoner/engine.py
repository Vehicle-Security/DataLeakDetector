"""用于已确认泄露路径的确定性污点传播引擎。

原始项目带有更重的推理脚手架。本重写把核心语义保留在纯 Python 中，以便单人迭代、测试和本地调试
保持快速，同时仍然建模源、传输、进程跨越、剪贴板和汇聚点关系。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..io import normalize_path
from ..models import DatalogFact, LeakPath
from .relations import SUPPORTED_RELATIONS, Taint


class DatalogEngine:
    """
    小型 Datalog 风格的污点引擎。

    它用纯 Python 实现了根 README 中描述的关系，这样既能让单人迭代保持快速，也能让单元测试具备确定性。
    """

    def __init__(self, *_: Any, case_id: str = "", **__: Any):
        self.case_id = str(case_id)
        self.facts: dict[str, list[DatalogFact]] = {name: [] for name in SUPPORTED_RELATIONS}

    def add_fact(self, relation: str, *args: Any, case_id: str | None = None) -> None:
        if relation not in self.facts:
            raise ValueError(f"unknown relation: {relation}")
        fact_case_id = str(case_id or self.case_id)
        if fact_case_id and not self.case_id:
            self.case_id = fact_case_id
            self.facts = {
                name: [
                    DatalogFact(fact.relation, fact.args, case_id=fact.case_id or fact_case_id)
                    for fact in facts
                ]
                for name, facts in self.facts.items()
            }
        elif self.case_id and fact_case_id and fact_case_id != self.case_id:
            raise ValueError(f"fact_case_mismatch: expected {self.case_id}, got {fact_case_id}")
        self.facts[relation].append(DatalogFact(relation, tuple(args), case_id=fact_case_id))

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
        seen: set[tuple[str, str, str, str]] = set()

        for fact in self.facts["LeakFile"]:
            leak_id, process, data, channel, timestamp, share_start = _pad(fact.args, 6)
            item = tainted_by_process_data.get((str(process), str(data)))
            leak_timestamp = _int(timestamp)
            screen_share_start = _int(share_start) if str(channel) == "screen_share" else 0
            bound_sources = self._bound_sources_for_leak(str(leak_id), str(data), leak_timestamp)
            complete_paths = self._complete_fact_paths(
                str(process),
                str(data),
                leak_timestamp,
                bound_sources=bound_sources,
                screen_share_start=screen_share_start,
            )
            if item is None or not _can_follow(item.timestamp, leak_timestamp):
                # For a confirmed active screen share, a file opened after the
                # share begins can still be exposed before sharing ends. The
                # visual observation supplies the leak evidence; the source
                # log is only used to bind its file identity.
                if not (screen_share_start and complete_paths):
                    continue
            if bound_sources and not complete_paths:
                continue
            path_candidates: list[tuple[str, list[DatalogFact]]] = []
            for complete_facts in complete_paths:
                operation_path = " -> ".join(
                    str(_pad(path_fact.args, 1)[0]) for path_fact in complete_facts
                )
                path_candidates.append((operation_path, [*complete_facts, fact]))
            if not path_candidates:
                path_candidates.append((item.path, self._facts_for_operation_path(item.path, fact)))

            for operation_path, path_facts in path_candidates:
                file_chain = _file_chain(path_facts)
                source_file = file_chain[0] if file_chain else ""
                key = (str(process), str(data), str(channel), _data_key(source_file))
                if key in seen:
                    continue
                seen.add(key)
                leaks.append(
                    LeakPath(
                        start_op=operation_path.split(" -> ")[0],
                        end_op=str(leak_id),
                        leaking_proc=str(process),
                        leaked_file=str(data),
                        leak_channel=str(channel),
                        leak_timestamp=leak_timestamp,
                        full_path=f"{operation_path} -> {leak_id}",
                        case_id=self.case_id or fact.case_id,
                        source_file=source_file,
                        file_chain=tuple(file_chain),
                        flow_steps=tuple(_flow_step(path_fact) for path_fact in path_facts),
                    )
                )
        return leaks

    def _complete_fact_paths(
        self,
        process: str,
        data: str,
        leak_timestamp: int,
        *,
        bound_sources: set[str] | None = None,
        screen_share_start: int = 0,
    ) -> list[list[DatalogFact]]:
        """Find the best acyclic, forward-time lineage path for each source."""

        roots: dict[tuple[str, str], list[DatalogFact]] = {}
        for fact in self.facts["OpenFile"]:
            _, root_process, root_data, _ = _pad(fact.args, 4)
            roots.setdefault((str(root_process), str(root_data)), []).append(fact)

        inbound: dict[tuple[str, str], list[tuple[tuple[str, str], DatalogFact]]] = {}
        for fact in self.facts["TransferFile"]:
            _, edge_process, source, derived, _ = _pad(fact.args, 5)
            inbound.setdefault((str(edge_process), str(derived)), []).append(
                ((str(edge_process), str(source)), fact)
            )
        for fact in [*self.facts["CrossProcessTransfer"], *self._clipboard_transfers()]:
            _, from_process, to_process, edge_data, _ = _pad(fact.args, 5)
            inbound.setdefault((str(to_process), str(edge_data)), []).append(
                ((str(from_process), str(edge_data)), fact)
            )

        candidates: list[list[DatalogFact]] = []

        def walk(
            state: tuple[str, str],
            next_timestamp: int,
            visited: frozenset[tuple[str, str]],
        ) -> None:
            for root in roots.get(state, ()):
                root_timestamp = _int(_pad(root.args, 4)[3])
                if _can_follow_in_leak_path(root_timestamp, next_timestamp, screen_share_start):
                    candidates.append([root])
            for previous_state, edge in inbound.get(state, ()):
                edge_timestamp = _int(_pad(edge.args, 5)[4])
                if previous_state in visited or not _can_follow_in_leak_path(
                    edge_timestamp, next_timestamp, screen_share_start
                ):
                    continue
                before = len(candidates)
                walk(previous_state, edge_timestamp or next_timestamp, visited | {previous_state})
                for index in range(before, len(candidates)):
                    candidates[index] = [*candidates[index], edge]

        target = (process, data)
        walk(target, leak_timestamp, frozenset({target}))
        if bound_sources:
            source_keys = {_data_key(item) for item in bound_sources}
            candidates = [
                path
                for path in candidates
                if _data_key(_path_source(path)) in source_keys
            ]
        if not candidates:
            return []
        best_by_source: dict[str, list[DatalogFact]] = {}
        for path in candidates:
            source_key = _data_key(_path_source(path))
            current = best_by_source.get(source_key)
            if current is None or _complete_path_rank(path) < _complete_path_rank(current):
                best_by_source[source_key] = path
        return sorted(best_by_source.values(), key=_complete_path_rank)

    def _complete_fact_path(self, process: str, data: str, leak_timestamp: int) -> list[DatalogFact]:
        """Compatibility wrapper returning the highest-ranked complete path."""

        paths = self._complete_fact_paths(process, data, leak_timestamp)
        return paths[0] if paths else []

    def _bound_sources_for_leak(self, leak_id: str, data: str, leak_timestamp: int) -> set[str]:
        sources: set[str] = set()
        for fact in self.facts["UploadBinding"]:
            _, bound_leak_id, source, bound_data, timestamp = _pad(fact.args, 5)
            if str(bound_leak_id) != leak_id or _data_key(bound_data) != _data_key(data):
                continue
            if not _can_follow(_int(timestamp), leak_timestamp):
                continue
            if str(source):
                sources.add(str(source))
        return sources

    def _facts_for_operation_path(self, path: str, leak_fact: DatalogFact) -> list[DatalogFact]:
        operation_ids = path.split(" -> ") if path else []
        lookup: dict[str, DatalogFact] = {}
        for relation in ("OpenFile", "TransferFile", "CrossProcessTransfer"):
            for fact in self.facts[relation]:
                lookup.setdefault(str(_pad(fact.args, 1)[0]), fact)
        for fact in self._clipboard_transfers():
            lookup.setdefault(str(_pad(fact.args, 1)[0]), fact)
        return [*(lookup[item] for item in operation_ids if item in lookup), leak_fact]

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
                transfer_timestamp = _int(timestamp)
                if not _can_follow(current.timestamp, transfer_timestamp):
                    continue
                _remember(
                    Taint(
                        str(process),
                        str(dst),
                        f"{current.path} -> {op_id}",
                        transfer_timestamp or current.timestamp,
                    ),
                    tainted,
                    best,
                    frontier,
                )

            for fact in cross_by_source.get((current.process, current.data), ()):
                op_id, from_proc, to_proc, data, timestamp = _pad(fact.args, 5)
                transfer_timestamp = _int(timestamp)
                if not _can_follow(current.timestamp, transfer_timestamp):
                    continue
                _remember(
                    Taint(
                        str(to_proc),
                        str(data),
                        f"{current.path} -> {op_id}",
                        transfer_timestamp or current.timestamp,
                    ),
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
                        case_id=self.case_id,
                    )
                )
        return derived


def _remember(
    item: Taint,
    tainted: set[Taint],
    best: dict[tuple[str, str], Taint],
    frontier: deque[Taint],
) -> None:
    key = (item.process, item.data)
    current = best.get(key)
    if current is not None and _taint_rank(current) <= _taint_rank(item):
        return
    if current is not None:
        tainted.discard(current)
    best[key] = item
    tainted.add(item)
    frontier.append(item)


def _taint_rank(item: Taint) -> tuple[int, int, int]:
    timestamp = item.timestamp if item.timestamp > 0 else -1
    return timestamp, item.path.count(" -> "), len(item.path)


def _complete_path_rank(path: list[DatalogFact]) -> tuple[int, int, int, str]:
    canonical_transfers = sum(
        fact.relation == "TransferFile" and str(_pad(fact.args, 5)[1]) == "case_lineage"
        for fact in path
    )
    file_transfers = sum(fact.relation == "TransferFile" for fact in path)
    return (
        -canonical_transfers,
        -file_transfers,
        -len(path),
        " -> ".join(str(_pad(fact.args, 1)[0]) for fact in path),
    )


def _file_chain(path: list[DatalogFact]) -> list[str]:
    files: list[str] = []
    for fact in path:
        if fact.relation == "OpenFile":
            value = str(_pad(fact.args, 4)[2])
            if value and not files:
                files.append(value)
        elif fact.relation == "TransferFile":
            _, _, source, derived, _ = _pad(fact.args, 5)
            for value in (str(source), str(derived)):
                if value and (not files or files[-1] != value):
                    files.append(value)
        elif fact.relation == "LeakFile" and not files:
            files.append(str(_pad(fact.args, 5)[2]))
    return files


def _path_source(path: list[DatalogFact]) -> str:
    for fact in path:
        if fact.relation == "OpenFile":
            return str(_pad(fact.args, 4)[2])
    return ""


def _data_key(value: Any) -> str:
    return normalize_path(value).lower()


def _flow_step(fact: DatalogFact) -> dict[str, Any]:
    if fact.relation == "OpenFile":
        op_id, process, data, timestamp = _pad(fact.args, 4)
        return {
            "relation": fact.relation,
            "op_id": str(op_id),
            "process": str(process),
            "file": str(data),
            "timestamp": _int(timestamp),
        }
    if fact.relation == "TransferFile":
        op_id, process, source, derived, timestamp = _pad(fact.args, 5)
        return {
            "relation": fact.relation,
            "op_id": str(op_id),
            "process": str(process),
            "source_file": str(source),
            "derived_file": str(derived),
            "timestamp": _int(timestamp),
        }
    if fact.relation == "CrossProcessTransfer":
        op_id, from_process, to_process, data, timestamp = _pad(fact.args, 5)
        return {
            "relation": fact.relation,
            "op_id": str(op_id),
            "from_process": str(from_process),
            "to_process": str(to_process),
            "file": str(data),
            "timestamp": _int(timestamp),
        }
    op_id, process, data, channel, timestamp, _ = _pad(fact.args, 6)
    return {
        "relation": fact.relation,
        "op_id": str(op_id),
        "process": str(process),
        "file": str(data),
        "channel": str(channel),
        "timestamp": _int(timestamp),
    }


def _can_follow(previous_timestamp: int, next_timestamp: int) -> bool:
    return previous_timestamp <= 0 or next_timestamp <= 0 or next_timestamp >= previous_timestamp


def _can_follow_in_leak_path(
    previous_timestamp: int,
    next_timestamp: int,
    screen_share_start: int = 0,
) -> bool:
    if _can_follow(previous_timestamp, next_timestamp):
        return True
    # Visual frames and desktop-monitor file events can be emitted in a
    # different order. A confirmed active share remains a state after it has
    # started, so file activity during that state is valid identity/lineage
    # evidence even when its monitor timestamp trails the visual frame.
    return bool(screen_share_start and previous_timestamp >= screen_share_start)


def _pad(args: tuple[Any, ...], size: int) -> tuple[Any, ...]:
    return tuple(args) + tuple("" for _ in range(max(0, size - len(args))))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
