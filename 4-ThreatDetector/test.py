"""
ThreatDetector 测试文件
完整流程: Mock数据 → LLM分析 → Datalog推理 → 威胁检测
"""

import os
import sys
import json
import re
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 导入实际模块
from datalog.datalog_engine import DatalogEngine, LeakPath
from prompts import PromptTemplates

# ==================== 数据结构（复用demo.py的定义）====================

@dataclass
class DatalogFact:
    """Datalog 事实"""
    relation: str
    operation_id: str
    process: str
    file: str
    dst_file: str = None
    timestamp: str = None
    description: str = None
    # CrossProcessTransfer 特有字段
    from_process: str = None
    to_process: str = None
    shared_data: str = None
    
    def to_souffle_args(self):
        """转换为Souffle引擎需要的参数格式"""
        ts = self._parse_timestamp()
        
        if self.relation == "OpenFile":
            # OpenFile(id, proc, file, timestamp)
            return (self.operation_id, self.process, self.file, ts)
            
        elif self.relation == "TransferFile":
            # TransferFile(id, proc, src, dst, timestamp)
            return (self.operation_id, self.process, self.file, self.dst_file, ts)
            
        elif self.relation == "CrossProcessTransfer":
            # CrossProcessTransfer(id, from_proc, to_proc, shared_data, timestamp)
            from_proc = self.from_process or self.process
            to_proc = self.to_process or "Unknown"
            data = self.shared_data or self.file
            return (self.operation_id, from_proc, to_proc, data, ts)
            
        elif self.relation == "LeakFile":
            # LeakFile(id, proc, file, channel, timestamp)
            channel = "network"  # 默认渠道
            return (self.operation_id, self.process, self.file, channel, ts)
            
        return None
    
    def _parse_timestamp(self):
        """解析时间戳为毫秒"""
        if self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
            except:
                pass
        return 0



# ==================== LLM规则生成器 ====================

class LLMRuleGenerator:
    """使用LLM从日志和视频帧生成Datalog规则"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
        self.base_url = os.getenv("LLM_BASE_URL")
        
        if not self.api_key:
            raise ValueError("❌ 未配置 LLM_API_KEY，请在.env文件中设置")
        
        try:
            from openai import OpenAI
            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                print(f"✅ LLM 规则生成器初始化成功 ({self.model_name})")
            else:
                self.client = OpenAI(api_key=self.api_key)
                print(f"✅ LLM 规则生成器初始化成功 (OpenAI)")
        except ImportError:
            raise ImportError("❌ 请安装 openai 库: pip install openai")
        except Exception as e:
            raise RuntimeError(f"❌ LLM 初始化失败: {e}")
    
    def generate_facts(self, logs: List[Dict], video_frames: List[Dict]) -> List[DatalogFact]:
        """分析日志和视频帧，生成Datalog事实"""
        print("\n🤖 调用 LLM 分析日志和视频帧...")
        print(f"   发送请求到 LLM ({self.model_name})...")
        
        # 使用 prompts.py 的标准模板
        messages = PromptTemplates.get_messages(logs, video_frames)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            print("   ✅ LLM 返回结果")
            
            result_text = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            
            if not json_match:
                raise ValueError("LLM未返回有效JSON格式")
            
            facts_data = json.loads(json_match.group())
            
            # 转换为DatalogFact对象
            facts = []
            for fd in facts_data:
                # 根据relation类型使用不同的字段
                if fd['relation'] == 'CrossProcessTransfer':
                    fact = DatalogFact(
                        relation=fd['relation'],
                        operation_id=fd['operation_id'],
                        process=fd.get('process', 'Unknown'),
                        from_process=fd.get('from_process'),
                        to_process=fd.get('to_process'),
                        shared_data=fd.get('shared_data'),
                        file=fd.get('shared_data', ''),
                        timestamp=fd.get('timestamp'),
                        description=fd.get('description')
                    )
                else:
                    fact = DatalogFact(
                        relation=fd['relation'],
                        operation_id=fd['operation_id'],
                        process=fd.get('process', 'Unknown'),
                        file=fd.get('file', ''),
                        dst_file=fd.get('dst_file'),
                        timestamp=fd.get('timestamp'),
                        description=fd.get('description')
                    )
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            raise RuntimeError(f"LLM分析失败: {e}")



# ==================== Mock 测试数据 ====================

SCENARIO_CLIPBOARD_LEAK = {
    "name": "剪贴板泄露",
    "logs": [
        {
            "timestamp": "2026-01-10T10:00:00.000",
            "event_type": "opened",
            "file_path": "/Users/admin/Documents/机密工资表.xlsx",
            "process_info": {"process_name": "Excel", "pid": "1234"},
            "description": "用户使用Excel打开了机密工资表文件"
        },
        {
            "timestamp": "2026-01-10T10:00:15.000",
            "event_type": "clipboard_copy",
            "file_path": "/Users/admin/Documents/机密工资表.xlsx",
            "process_info": {"process_name": "Excel", "pid": "1234"},
            "description": "用户从Excel复制了工资表内容到剪贴板"
        },
        {
            "timestamp": "2026-01-10T10:00:20.000",
            "event_type": "clipboard_paste",
            "file_path": "",
            "process_info": {"process_name": "WeChat", "pid": "5678"},
            "description": "用户将剪贴板内容粘贴到微信聊天窗口"
        },
        {
            "timestamp": "2026-01-10T10:00:25.000",
            "event_type": "network_send",
            "file_path": "",
            "process_info": {"process_name": "WeChat", "pid": "5678"},
            "description": "微信将消息发送到网络"
        }
    ],
    "frames": [
        {
            "timestamp": "2026-01-10T10:00:00.000",
            "app_name": "Excel",
            "operation_type": "文件打开",
            "behavior_category": "正常操作",
            "description": "Excel打开机密工资表，显示员工薪资数据"
        },
        {
            "timestamp": "2026-01-10T10:00:15.000",
            "app_name": "Excel",
            "operation_type": "复制数据",
            "behavior_category": "潜在风险",
            "description": "用户选中数据并Ctrl+C复制"
        },
        {
            "timestamp": "2026-01-10T10:00:20.000",
            "app_name": "WeChat",
            "operation_type": "粘贴数据",
            "behavior_category": "高风险",
            "description": "微信窗口激活，Ctrl+V粘贴"
        },
        {
            "timestamp": "2026-01-10T10:00:25.000",
            "app_name": "WeChat",
            "operation_type": "发送消息",
            "behavior_category": "数据泄露",
            "description": "微信显示消息已发送"
        }
    ]
}


# ==================== 主测试函数 ====================

def test_complete_workflow():
    """测试完整工作流：Mock数据 → LLM → Datalog → 检测"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           🧪 ThreatDetector 完整工作流测试                            ║
║                                                                      ║
║   流程:                                                              ║
║   1. Mock 日志 + 视频帧                                              ║
║   2. LLM 分析生成 Datalog 事实                                        ║
║   3. Souffle Datalog 推理                                            ║
║   4. 泄露路径检测                                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # 阶段1: 准备数据
    print("=" * 80)
    print("📋 阶段1: 准备测试数据")
    print("=" * 80)
    
    logs = SCENARIO_CLIPBOARD_LEAK["logs"]
    frames = SCENARIO_CLIPBOARD_LEAK["frames"]
    
    print(f"\n✅ 已加载测试场景: {SCENARIO_CLIPBOARD_LEAK['name']}")
    print(f"   - 系统日志: {len(logs)} 条")
    print(f"   - 视频帧: {len(frames)} 帧")
    
    # 阶段2: LLM分析
    print("\n" + "=" * 80)
    print("🤖 阶段2: LLM 分析生成 Datalog 事实")
    print("=" * 80)
    
    rule_generator = LLMRuleGenerator()
    facts = rule_generator.generate_facts(logs, frames)
    
    print(f"\n✅ 生成了 {len(facts)} 条 Datalog 事实:")
    for i, fact in enumerate(facts, 1):
        if fact.relation == "CrossProcessTransfer":
            print(f"   {i}. {fact.relation}({fact.operation_id}, {fact.from_process} → {fact.to_process}, {fact.shared_data})")
        elif fact.dst_file:
            print(f"   {i}. {fact.relation}({fact.operation_id}, {fact.process}, {fact.file} → {fact.dst_file})")
        else:
            print(f"   {i}. {fact.relation}({fact.operation_id}, {fact.process}, {fact.file})")
    
    # 阶段3: Datalog推理
    print("\n" + "=" * 80)
    print("⚖️  阶段3: Datalog 推理引擎")
    print("=" * 80)
    
    engine = DatalogEngine()
    
    # 将 DatalogFact 转换为 Souffle 引擎格式并添加
    print("\n📝 添加事实到 Souffle 引擎...")
    for fact in facts:
        args = fact.to_souffle_args()
        if args:
            engine.add_fact(fact.relation, *args)
    
    # 执行推理
    leak_paths = engine.query_leak()
    
    # 阶段4: 结果展示
    print("\n" + "=" * 80)
    print("📊 阶段4: 检测结果")
    print("=" * 80)
    
    print(f"\n{'🚨' if leak_paths else '✅'} 检测到 {len(leak_paths)} 条泄露路径\n")
    
    for i, leak in enumerate(leak_paths, 1):
        print(f"泄露路径 #{i}:")
        print(f"  📁 泄露文件: {leak.leaked_file}")
        print(f"  📤 泄露进程: {leak.leaking_proc}")
        print(f"  🌐 泄露渠道: {leak.leak_channel}")
        print(f"  🛤️  完整路径: {leak.full_path}")
        print()
    
    # 总结
    print("=" * 80)
    print("✅ 测试完成")
    print("=" * 80)
    print(f"\n统计:")
    print(f"  - 输入日志: {len(logs)} 条")
    print(f"  - 生成事实: {len(facts)} 条")
    print(f"  - 检测泄露: {len(leak_paths)} 条")
    
    if leak_paths:
        print(f"\n⚠️  发现数据泄露风险！")
    else:
        print(f"\n✅ 未发现数据泄露")
    
    # 清理
    engine.cleanup()
    
    return len(leak_paths) > 0


if __name__ == "__main__":
    has_leak = test_complete_workflow()
    sys.exit(1 if has_leak else 0)
