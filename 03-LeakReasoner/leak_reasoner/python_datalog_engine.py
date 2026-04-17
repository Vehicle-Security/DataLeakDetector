from collections import deque
from typing import Dict, List, NamedTuple, Set, Tuple

try:
    from .legacy_datalog_types import LeakPath
except ImportError:
    from legacy_datalog_types import LeakPath


class OpenFileFact(NamedTuple):
    op_id: str
    process: str
    file: str
    timestamp: int


class TransferFileFact(NamedTuple):
    op_id: str
    process: str
    src: str
    dst: str
    timestamp: int


class CrossProcessTransferFact(NamedTuple):
    op_id: str
    from_process: str
    to_process: str
    shared_data: str
    timestamp: int


class LeakFileFact(NamedTuple):
    op_id: str
    process: str
    file: str
    leak_channel: str
    timestamp: int


class TaintedTuple(NamedTuple):
    process: str
    data: str
    path: str
    timestamp: int


class PythonDatalogEngine:
    def __init__(self):
        self.open_files: List[OpenFileFact] = []
        self.transfer_files: List[TransferFileFact] = []
        self.cross_process_transfers: List[CrossProcessTransferFact] = []
        self.leak_files: List[LeakFileFact] = []

    def add_fact(self, relation: str, *args):
        if relation == "OpenFile":
            self.open_files.append(OpenFileFact(*args))
        elif relation == "TransferFile":
            self.transfer_files.append(TransferFileFact(*args))
        elif relation == "CrossProcessTransfer":
            self.cross_process_transfers.append(CrossProcessTransferFact(*args))
        elif relation == "LeakFile":
            self.leak_files.append(LeakFileFact(*args))
        else:
            raise ValueError(f"Unknown relation: {relation}")

    def run_inference(self) -> List[LeakPath]:
        tainted: Set[TaintedTuple] = set()
        best_tainted: Dict[Tuple[str, str], TaintedTuple] = {}
        frontier = deque()

        for of in self.open_files:
            initial = TaintedTuple(
                process=of.process,
                data=of.file,
                path=of.op_id,
                timestamp=of.timestamp,
            )
            state_key = (initial.process, initial.data)
            if state_key in best_tainted:
                continue
            tainted.add(initial)
            best_tainted[state_key] = initial
            frontier.append(initial)

        while frontier:
            new_tainted: Set[TaintedTuple] = set()
            current_batch = list(frontier)
            frontier.clear()

            for t in current_batch:
                for tf in self.transfer_files:
                    if tf.process == t.process and tf.src == t.data:
                        candidate = TaintedTuple(
                            process=tf.process,
                            data=tf.dst,
                            path=f"{t.path} -> {tf.op_id}",
                            timestamp=tf.timestamp,
                        )
                        state_key = (candidate.process, candidate.data)
                        existing = best_tainted.get(state_key)
                        if existing is None or len(candidate.path) < len(existing.path):
                            if existing is not None:
                                tainted.discard(existing)
                            tainted.add(candidate)
                            best_tainted[state_key] = candidate
                            new_tainted.add(candidate)

            for t in current_batch:
                for cpt in self.cross_process_transfers:
                    if cpt.from_process == t.process and cpt.shared_data == t.data:
                        candidate = TaintedTuple(
                            process=cpt.to_process,
                            data=cpt.shared_data,
                            path=f"{t.path} -> {cpt.op_id}",
                            timestamp=cpt.timestamp,
                        )
                        state_key = (candidate.process, candidate.data)
                        existing = best_tainted.get(state_key)
                        if existing is None or len(candidate.path) < len(existing.path):
                            if existing is not None:
                                tainted.discard(existing)
                            tainted.add(candidate)
                            best_tainted[state_key] = candidate
                            new_tainted.add(candidate)

            if not new_tainted:
                break
            frontier.extend(new_tainted)

        leak_paths: List[LeakPath] = []
        seen_leaks: Set[Tuple[str, str, str]] = set()
        for t in tainted:
            for lf in self.leak_files:
                if lf.process == t.process and lf.file == t.data:
                    dedup_key = (lf.process, lf.file, lf.leak_channel)
                    if dedup_key in seen_leaks:
                        continue
                    seen_leaks.add(dedup_key)
                    leak_paths.append(
                        LeakPath(
                            start_op=t.path,
                            end_op=lf.op_id,
                            leaking_proc=lf.process,
                            leaked_file=lf.file,
                            full_path=f"{t.path} -> {lf.op_id}",
                            leak_channel=lf.leak_channel,
                            leak_timestamp=lf.timestamp,
                        )
                    )

        return leak_paths
