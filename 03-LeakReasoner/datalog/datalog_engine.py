"""
Datalog 推理引擎核心模块
支持 Souffle 引擎 和 纯 Python 降级引擎
运行时自动检测: Souffle 可用则使用 Souffle，否则自动降级为 Python 引擎
"""

import os
import subprocess
import csv
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import time


@dataclass
class LeakPath:
    """泄露路径结构"""
    start_op: str
    end_op: str
    leaking_proc: str
    leaked_file: str
    full_path: str
    leak_channel: str
    leak_timestamp: int
    
    def to_dict(self) -> dict:
        return {
            'start_op': self.start_op,
            'end_op': self.end_op,
            'leaking_proc': self.leaking_proc,
            'leaked_file': self.leaked_file,
            'full_path': self.full_path,
            'leak_channel': self.leak_channel,
            'leak_timestamp': self.leak_timestamp,
            'path_steps': self.full_path.split(' -> ')
        }


@dataclass
class DatalogFact:
    """Datalog 事实结构"""
    relation: str
    args: Tuple
    
    def to_datalog_string(self) -> str:
        formatted_args = []
        for arg in self.args:
            if isinstance(arg, str):
                escaped = arg.replace('"', '\\"')
                formatted_args.append(f'"{escaped}"')
            else:
                formatted_args.append(str(arg))
        return f"{self.relation}({', '.join(formatted_args)})."


class DatalogEngine:
    """
    Datalog 推理引擎封装
    
    自动检测 Souffle 是否可用:
      - Souffle 可用 → 使用原生 Souffle 引擎（高性能）
      - Souffle 不可用 → 自动降级为纯 Python 引擎（跨平台兼容）
    
    对外接口完全一致: add_fact() / query_leak() / cleanup()
    """
    
    def __init__(self, rules_file: str = None, work_dir: str = None):
        self.rules_file = rules_file or os.path.join(
            os.path.dirname(__file__), 
            'taint_rules.dl'
        )
        
        self.work_dir = work_dir or tempfile.mkdtemp(prefix='datalog_')
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.facts: Dict[str, List[DatalogFact]] = {
            'OpenFile': [],
            'TransferFile': [],
            'CrossProcessTransfer': [],
            'LeakFile': [],
            'ClipboardWrite': [],
            'ClipboardRead': []
        }
        
        # 自动检测引擎
        self.use_souffle = False
        self.souffle_bin = None
        self._python_engine = None
        
        try:
            self.souffle_bin = self._find_souffle()
            self.use_souffle = True
            print("[OK] Datalog 引擎初始化成功 (Souffle)")
        except RuntimeError:
            # Souffle 不可用，降级为 Python 引擎
            from datalog.python_datalog_engine import PythonDatalogEngine
            self._python_engine = PythonDatalogEngine()
            print("[WARN] Souffle 未找到，自动切换为 Python Datalog 引擎")
            print("[OK] Datalog 引擎初始化成功 (Python 降级模式)")
    
    def _find_souffle(self) -> str:
        candidates = ['souffle', '/usr/local/bin/souffle', '/opt/homebrew/bin/souffle']
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, '--version'], 
                    capture_output=True, 
                    timeout=5,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                continue
        raise RuntimeError("Souffle not found")
    
    def add_fact(self, relation: str, *args):
        if relation not in self.facts:
            raise ValueError(f"Unknown relation: {relation}")
        fact = DatalogFact(relation, args)
        self.facts[relation].append(fact)
    
    def add_clipboard_operation(self, write_op_id: str, write_proc: str, 
                                read_op_id: str, read_proc: str, 
                                data: str, write_ts: int, read_ts: int):
        self.add_fact('ClipboardWrite', write_op_id, write_proc, data, write_ts)
        self.add_fact('ClipboardRead', read_op_id, read_proc, data, read_ts)
    
    def _write_facts_to_files(self):
        for relation, facts in self.facts.items():
            fact_file = os.path.join(self.work_dir, f"{relation}.facts")
            with open(fact_file, 'w', newline='', encoding='utf-8') as f:
                if facts:
                    writer = csv.writer(f, delimiter='\t')
                    for fact in facts:
                        writer.writerow(fact.args)
                    print(f"   写入 {len(facts)} 条 {relation} 事实")
                # 空文件也会被创建，满足 Souffle 的要求
    
    def query_leak(self) -> List[LeakPath]:
        """执行推理并返回泄露路径"""
        if self.use_souffle:
            return self._souffle_query_leak()
        else:
            return self._python_query_leak()
    
    def _souffle_query_leak(self) -> List[LeakPath]:
        """使用 Souffle 引擎执行推理"""
        print("\n[INFO] 开始 Datalog 推理 (Souffle)...")
        self._write_facts_to_files()
        
        try:
            run_cmd = [
                self.souffle_bin,
                self.rules_file,
                '-F', self.work_dir,
                '-D', self.work_dir
            ]
            
            print(f"   执行: {' '.join(run_cmd)}")
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60, cwd=self.work_dir)
            
            if result.returncode != 0:
                raise RuntimeError(f"Souffle 执行失败:\n{result.stderr}")
            
            print("   [OK] 推理完成")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Souffle 执行超时")
        
        leak_paths = self._parse_leak_results()
        print(f"\n[OK] 发现 {len(leak_paths)} 条泄露路径")
        return leak_paths
    
    def _python_query_leak(self) -> List[LeakPath]:
        """使用纯 Python 引擎执行推理"""
        # 将累积的事实传入 Python 引擎
        for relation, facts in self.facts.items():
            for fact in facts:
                self._python_engine.add_fact(relation, *fact.args)
        
        # 执行推理
        return self._python_engine.run_inference()
    
    def _parse_leak_results(self) -> List[LeakPath]:
        result_file = os.path.join(self.work_dir, 'leak_results.csv')
        if not os.path.exists(result_file):
            return []
        
        leak_paths = []
        with open(result_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) >= 7:
                    leak_path = LeakPath(
                        start_op=row[0].strip(),
                        end_op=row[1].strip(),
                        leaking_proc=row[2].strip(),
                        leaked_file=row[3].strip(),
                        full_path=row[4].strip(),
                        leak_channel=row[5].strip(),
                        leak_timestamp=int(row[6].strip()) if row[6].strip().isdigit() else 0
                    )
                    leak_paths.append(leak_path)
        return leak_paths
    
    def cleanup(self):
        if hasattr(self, 'work_dir') and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
