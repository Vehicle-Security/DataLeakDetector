# -*- coding: utf-8 -*-
"""
detector.py - 主检测引擎

协调日志转换、Soufflé 执行和结果解析。
"""

import os
import subprocess
import csv
import json
from typing import Dict, List, Any
from datetime import datetime

from log_to_facts import LogToFactsConverter


class AttackDetector:
    """基于 Datalog 的攻击检测器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.facts_dir = os.path.join(output_dir, "facts")
        self.results_dir = os.path.join(output_dir, "results")
        self.rules_dir = "rules"
        
        os.makedirs(self.facts_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 检测结果
        self.detection_results = {}
    
    def convert_logs(self, log_path: str) -> int:
        """转换日志为 Datalog 事实"""
        print(f"\n{'='*60}")
        print(f"[1/3] 转换日志为 Datalog 事实")
        print(f"{'='*60}")
        
        converter = LogToFactsConverter(output_dir=self.facts_dir)
        count = converter.convert_log_file(log_path)
        converter.write_facts()
        
        return count
    
    def _check_souffle(self) -> bool:
        """检查 Soufflé 是否可用"""
        try:
            result = subprocess.run(
                ['souffle', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"[SOUFFLE] {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            print("[WARNING] Soufflé 未安装，将使用模拟模式")
            return False
        except Exception as e:
            print(f"[WARNING] Soufflé 检查失败: {e}")
            return False
    
    def run_souffle(self, rule_file: str = "rules/advanced.dl") -> bool:
        """运行 Soufflé 查询"""
        print(f"\n{'='*60}")
        print(f"[2/3] 执行 Datalog 规则推理")
        print(f"{'='*60}")
        
        if not self._check_souffle():
            # 模拟模式：生成示例输出
            return self._run_simulation()
        
        # 构建 Soufflé 命令
        cmd = [
            'souffle',
            '-F', self.facts_dir,  # 事实文件目录
            '-D', self.results_dir,  # 输出目录
            rule_file
        ]
        
        try:
            print(f"[SOUFFLE] 执行: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"[ERROR] Soufflé 执行失败:")
                print(result.stderr)
                return False
            
            print("[SOUFFLE] 规则推理完成")
            return True
            
        except subprocess.TimeoutExpired:
            print("[ERROR] Soufflé 执行超时")
            return False
        except Exception as e:
            print(f"[ERROR] Soufflé 执行错误: {e}")
            return False
    
    def _run_simulation(self) -> bool:
        """模拟模式：基于简单规则生成检测结果"""
        print("[SIMULATION] 使用 Python 模拟 Datalog 推理")
        
        # 读取事实文件
        facts = self._load_facts()
        
        # 简单规则模拟
        results = []
        
        # 规则1: 检测敏感文件打开后的浏览器访问
        open_files = facts.get('open_file', [])
        browser_accesses = facts.get('browser_access', [])
        
        for of in open_files:
            for ba in browser_accesses:
                # 时间窗口检查
                try:
                    t1 = int(of[2]) if len(of) > 2 else 0
                    t2 = int(ba[2]) if len(ba) > 2 else 0
                    
                    if t2 > t1 and t2 - t1 < 300:
                        # 检查是否访问上传站点
                        url = ba[1].strip('"') if len(ba) > 1 else ""
                        if any(kw in url.lower() for kw in ['doubao', 'chatgpt', 'claude', 'kimi']):
                            results.append({
                                'type': 'potential_upload',
                                'file': of[1].strip('"') if len(of) > 1 else "",
                                'url': url,
                                't_open': t1,
                                't_access': t2,
                                'risk': 'HIGH'
                            })
                except:
                    continue
        
        # 保存模拟结果
        self._save_simulation_results(results)
        
        return True
    
    def _load_facts(self) -> Dict[str, List]:
        """加载事实文件"""
        facts = {}
        
        for filename in os.listdir(self.facts_dir):
            if filename.endswith('.facts'):
                name = filename[:-6]  # 移除 .facts
                path = os.path.join(self.facts_dir, filename)
                
                with open(path, 'r', encoding='utf-8') as f:
                    facts[name] = [line.strip().split('\t') for line in f if line.strip()]
        
        return facts
    
    def _save_simulation_results(self, results: List[Dict]):
        """保存模拟结果"""
        # 保存为 CSV
        if results:
            csv_path = os.path.join(self.results_dir, "potential_upload.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['file', 'url', 't_open', 't_access', 'risk', 'type'])
                writer.writeheader()
                writer.writerows(results)
        
        self.detection_results['potential_upload'] = results
    
    def parse_results(self) -> Dict[str, Any]:
        """解析检测结果"""
        print(f"\n{'='*60}")
        print(f"[3/3] 解析检测结果")
        print(f"{'='*60}")
        
        results = {
            'potential_upload': [],
            'rename_evasion': [],
            'attack_chain': [],
            'suspicious_switch': [],
            'high_risk_event': []
        }
        
        # 读取 Soufflé 输出的 CSV 文件
        for result_type in results.keys():
            csv_path = os.path.join(self.results_dir, f"{result_type}.csv")
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    results[result_type] = [row for row in reader]
        
        # 合并模拟结果
        if self.detection_results:
            for key, value in self.detection_results.items():
                if key in results:
                    results[key].extend(value) if isinstance(value, list) else None
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成检测报告"""
        report_lines = [
            "=" * 60,
            "       复杂攻击检测报告",
            f"       生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # 统计
        total_alerts = sum(len(v) for v in results.values() if isinstance(v, list))
        report_lines.append(f"📊 总计检测到 {total_alerts} 个可疑事件\n")
        
        # 高风险事件
        high_risk = results.get('high_risk_event', []) + results.get('attack_chain', [])
        if high_risk:
            report_lines.append("🚨 高风险事件:")
            report_lines.append("-" * 40)
            for event in high_risk[:10]:  # 最多显示10个
                if isinstance(event, dict):
                    report_lines.append(f"  • [{event.get('risk', 'HIGH')}] {event.get('type', 'UNKNOWN')}")
                    report_lines.append(f"    文件: {event.get('file', 'N/A')}")
                    report_lines.append(f"    目标: {event.get('url', 'N/A')}")
                elif isinstance(event, list) and len(event) >= 2:
                    report_lines.append(f"  • {event[0]}: {event[1]}")
            report_lines.append("")
        
        # 潜在上传
        uploads = results.get('potential_upload', [])
        if uploads:
            report_lines.append("📤 潜在数据泄露:")
            report_lines.append("-" * 40)
            for upload in uploads[:5]:
                if isinstance(upload, dict):
                    report_lines.append(f"  • 文件: {upload.get('file', 'N/A')}")
                    report_lines.append(f"    目标URL: {upload.get('url', 'N/A')}")
                    report_lines.append(f"    风险等级: {upload.get('risk', 'MEDIUM')}")
                    report_lines.append("")
            report_lines.append("")
        
        # 可疑应用切换
        switches = results.get('suspicious_switch', [])
        if switches:
            report_lines.append("🔄 可疑应用切换:")
            report_lines.append("-" * 40)
            for switch in switches[:5]:
                if isinstance(switch, list) and len(switch) >= 3:
                    report_lines.append(f"  • {switch[0]} -> {switch[1]} @ {switch[2]}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        report_lines.append("报告结束")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n[REPORT] 报告已保存: {report_path}")
        
        return report
    
    def detect(self, log_path: str) -> Dict[str, Any]:
        """完整检测流程"""
        print("\n" + "=" * 60)
        print("   复杂攻击检测引擎 v1.0")
        print("   基于 Datalog (Soufflé) 的间接关联分析")
        print("=" * 60)
        
        # Step 1: 转换日志
        event_count = self.convert_logs(log_path)
        print(f"[FACTS] 共转换 {event_count} 个事件")
        
        # Step 2: 运行规则推理
        self.run_souffle()
        
        # Step 3: 解析结果
        results = self.parse_results()
        
        # Step 4: 生成报告
        self.generate_report(results)
        
        return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complex Attack Detection using Datalog')
    parser.add_argument('--input', '-i', required=True, help='Input JSON log file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--rules', '-r', default='rules/advanced.dl', help='Datalog rules file')
    
    args = parser.parse_args()
    
    detector = AttackDetector(output_dir=args.output)
    detector.detect(args.input)


if __name__ == '__main__':
    main()
