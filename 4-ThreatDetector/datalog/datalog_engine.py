"""
Datalog 推理引擎核心模块
支持 Souffle 引擎，封装污点追踪推理逻辑
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
    """Souffle Datalog 推理引擎封装"""
    
    def __init__(self, rules_file: str = None, work_dir: str = None):
        self.rules_file = rules_file or os.path.join(
            os.path.dirname(__file__), 
            'taint_rules.dl'
        )
        
        if not os.path.exists(self.rules_file):
            raise FileNotFoundError(f"Datalog rules file not found: {self.rules_file}")
        
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
        
        self.souffle_bin = self._find_souffle()
        print(f"✅ Datalog 引擎初始化成功")
    
    def _find_souffle(self) -> str:
        candidates = ['souffle', '/usr/local/bin/souffle', '/opt/homebrew/bin/souffle']
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, '--version'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return candidate
            except:
                continue
        raise RuntimeError("Souffle not found. Install: brew install souffle")
    
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
        print("\n🔍 开始 Datalog 推理...")
        self._write_facts_to_files()
        
        try:
            # 使用解释模式直接运行（不编译C++）
            run_cmd = [
                self.souffle_bin,
                self.rules_file,
                '-F', self.work_dir,  # 输入目录
                '-D', self.work_dir   # 输出目录
            ]
            
            print(f"   执行: {' '.join(run_cmd)}")
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60, cwd=self.work_dir)
            
            if result.returncode != 0:
                raise RuntimeError(f"Souffle 执行失败:\n{result.stderr}")
            
            print("   ✅ 推理完成")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Souffle 执行超时")
        
        leak_paths = self._parse_leak_results()
        print(f"\n✅ 发现 {len(leak_paths)} 条泄露路径")
        return leak_paths
    
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
