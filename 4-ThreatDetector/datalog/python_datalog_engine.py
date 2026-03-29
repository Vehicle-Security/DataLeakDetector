"""
纯 Python Datalog 推理引擎
完整复刻 taint_rules.dl 中的污点追踪推理逻辑
在 Souffle 不可用时（如 Windows 平台）作为降级方案使用

推理规则:
  1. OpenFile → 污染源头
  2. TransferFile → 同进程内污染传播
  3. CrossProcessTransfer → 跨进程污染传播
  4. TransferFile + "copy" → 副本传播
  5. Tainted × LeakFile → 泄露检测
  剪贴板: ClipboardWrite × ClipboardRead → CrossProcessTransfer
"""

from typing import List, Dict, Set, Tuple, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from datalog.datalog_engine import LeakPath


# ==================== 事实类型定义 ====================

class OpenFileFact(NamedTuple):
    """进程打开/读取敏感文件"""
    op_id: str
    process: str
    file: str
    timestamp: int


class TransferFileFact(NamedTuple):
    """文件/数据转移 (同进程内)"""
    op_id: str
    process: str
    src: str
    dst: str
    timestamp: int


class CrossProcessTransferFact(NamedTuple):
    """跨进程数据传输"""
    op_id: str
    from_process: str
    to_process: str
    shared_data: str
    timestamp: int


class LeakFileFact(NamedTuple):
    """数据外泄操作"""
    op_id: str
    process: str
    file: str
    leak_channel: str
    timestamp: int


class ClipboardWriteFact(NamedTuple):
    """剪贴板写入"""
    op_id: str
    process: str
    data: str
    timestamp: int


class ClipboardReadFact(NamedTuple):
    """剪贴板读取"""
    op_id: str
    process: str
    data: str
    timestamp: int


class TaintedTuple(NamedTuple):
    """污染追踪中间结果: (进程, 数据, 路径历史, 时间戳)"""
    process: str
    data: str
    path: str
    timestamp: int


# ==================== 纯 Python Datalog 推理引擎 ====================

class PythonDatalogEngine:
    """
    纯 Python 实现的 Datalog 推理引擎
    使用不动点迭代 (fixed-point iteration) 算法
    """

    def __init__(self):
        self.open_files: List[OpenFileFact] = []
        self.transfer_files: List[TransferFileFact] = []
        self.cross_process_transfers: List[CrossProcessTransferFact] = []
        self.leak_files: List[LeakFileFact] = []
        self.clipboard_writes: List[ClipboardWriteFact] = []
        self.clipboard_reads: List[ClipboardReadFact] = []

    def add_fact(self, relation: str, *args):
        """添加事实，与 DatalogEngine.add_fact 参数格式完全一致"""
        if relation == "OpenFile":
            self.open_files.append(OpenFileFact(*args))
        elif relation == "TransferFile":
            self.transfer_files.append(TransferFileFact(*args))
        elif relation == "CrossProcessTransfer":
            self.cross_process_transfers.append(CrossProcessTransferFact(*args))
        elif relation == "LeakFile":
            self.leak_files.append(LeakFileFact(*args))
        elif relation == "ClipboardWrite":
            self.clipboard_writes.append(ClipboardWriteFact(*args))
        elif relation == "ClipboardRead":
            self.clipboard_reads.append(ClipboardReadFact(*args))
        else:
            raise ValueError(f"Unknown relation: {relation}")

    def _derive_clipboard_transfers(self):
        """
        剪贴板规则:
        CrossProcessTransfer(cat(write_id, "_", read_id), p_write, p_read, data, read_ts) :-
            ClipboardWrite(write_id, p_write, data, write_ts),
            ClipboardRead(read_id, p_read, data, read_ts),
            read_ts > write_ts,
            read_ts - write_ts < 300000,  // 5分钟内
            p_write != p_read.
        """
        for cw in self.clipboard_writes:
            for cr in self.clipboard_reads:
                if (cr.data == cw.data and
                        cr.timestamp > cw.timestamp and
                        cr.timestamp - cw.timestamp < 300000 and
                        cw.process != cr.process):
                    derived_id = f"{cw.op_id}_{cr.op_id}"
                    derived = CrossProcessTransferFact(
                        op_id=derived_id,
                        from_process=cw.process,
                        to_process=cr.process,
                        shared_data=cr.data,
                        timestamp=cr.timestamp
                    )
                    self.cross_process_transfers.append(derived)

    def run_inference(self) -> 'List[LeakPath]':
        """
        执行完整的不动点推理

        Returns:
            检测到的泄露路径列表
        """
        from datalog.datalog_engine import LeakPath

        print("\n[INFO] \u5f00\u59cb Python Datalog \u63a8\u7406...")

        # 第0步: 从剪贴板操作派生 CrossProcessTransfer
        self._derive_clipboard_transfers()

        # 第1步: 规则1 — 初始化污染源头
        #   Tainted(p, f, id, timestamp) :- OpenFile(id, p, f, timestamp).
        tainted: Set[TaintedTuple] = set()
        for of in self.open_files:
            tainted.add(TaintedTuple(
                process=of.process,
                data=of.file,
                path=of.op_id,
                timestamp=of.timestamp
            ))

        print(f"   规则1 (污染源头): {len(tainted)} 条初始污点")

        # 第2-4步: 不动点迭代 — 反复应用传播规则直到集合不再增长
        iteration = 0
        while True:
            iteration += 1
            new_tainted: Set[TaintedTuple] = set()

            # 规则2: 同进程内文件传播
            #   Tainted(p, dst, cat(history, " -> ", id), new_ts) :-
            #       Tainted(p, src, history, _),
            #       TransferFile(id, p, src, dst, new_ts).
            for t in tainted:
                for tf in self.transfer_files:
                    if tf.process == t.process and tf.src == t.data:
                        new_path = f"{t.path} -> {tf.op_id}"
                        candidate = TaintedTuple(
                            process=tf.process,
                            data=tf.dst,
                            path=new_path,
                            timestamp=tf.timestamp
                        )
                        if candidate not in tainted:
                            new_tainted.add(candidate)

            # 规则3: 跨进程污染传播
            #   Tainted(p2, shared_data, cat(history, " -> ", id), new_ts) :-
            #       Tainted(p1, shared_data, history, _),
            #       CrossProcessTransfer(id, p1, p2, shared_data, new_ts).
            for t in tainted:
                for cpt in self.cross_process_transfers:
                    if cpt.from_process == t.process and cpt.shared_data == t.data:
                        new_path = f"{t.path} -> {cpt.op_id}"
                        candidate = TaintedTuple(
                            process=cpt.to_process,
                            data=cpt.shared_data,
                            path=new_path,
                            timestamp=cpt.timestamp
                        )
                        if candidate not in tainted:
                            new_tainted.add(candidate)

            # 规则4: 含 "copy" 的副本传播
            #   Tainted(p, dst, cat(history, " -> ", id), new_ts) :-
            #       Tainted(p, src, history, _),
            #       TransferFile(id, p, src, dst, new_ts),
            #       contains("copy", id).
            for t in tainted:
                for tf in self.transfer_files:
                    if (tf.process == t.process and
                            tf.src == t.data and
                            "copy" in tf.op_id):
                        new_path = f"{t.path} -> {tf.op_id}"
                        candidate = TaintedTuple(
                            process=tf.process,
                            data=tf.dst,
                            path=new_path,
                            timestamp=tf.timestamp
                        )
                        if candidate not in tainted:
                            new_tainted.add(candidate)

            # 检查不动点
            if not new_tainted:
                break

            tainted.update(new_tainted)
            print(f"   迭代 {iteration}: 新增 {len(new_tainted)} 条污点，总计 {len(tainted)} 条")

        print(f"   不动点收敛: 共 {iteration} 轮迭代，最终 {len(tainted)} 条污点记录")

        # 第5步: 规则5 — 泄露检测
        #   SearchLeak(history, leak_id, p, f, cat(history, " -> ", leak_id), channel, leak_ts) :-
        #       Tainted(p, f, history, _),
        #       LeakFile(leak_id, p, f, channel, leak_ts).
        leak_paths: List[LeakPath] = []
        seen_leaks: Set[Tuple[str, str, str, str]] = set()  # 去重

        for t in tainted:
            for lf in self.leak_files:
                if lf.process == t.process and lf.file == t.data:
                    dedup_key = (t.path, lf.op_id, lf.process, lf.file)
                    if dedup_key not in seen_leaks:
                        seen_leaks.add(dedup_key)
                        full_path = f"{t.path} -> {lf.op_id}"
                        leak_paths.append(LeakPath(
                            start_op=t.path,
                            end_op=lf.op_id,
                            leaking_proc=lf.process,
                            leaked_file=lf.file,
                            full_path=full_path,
                            leak_channel=lf.leak_channel,
                            leak_timestamp=lf.timestamp
                        ))

        print(f"\n[OK] \u53d1\u73b0 {len(leak_paths)} \u6761\u6cc4\u9732\u8def\u5f84")
        return leak_paths
