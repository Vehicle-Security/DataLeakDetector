"""
DataLeakDetector Full E2E Pipeline
完整流程: 日志 + 视频 → 模块2(FileTracker) → 模块3(RiskHunter) → 模块4(ThreatDetector) → 证据报告

Pipeline 调用链:
  模块2 (FileTracker): WorklistManager + BehaviorAnalysisGraph
      ↓ 调用
  模块1 (FrameAnalyzer): 视频帧分析
      ↓ 返回
  模块2: 识别隐藏行为，更新敏感文件列表
      ↓
  模块3 (RiskHunter): 判断黑名单，生成报警
      ↓
  模块4 (ThreatDetector): Datalog推理，完整证据链
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("E2E-Pipeline")

# 添加模块路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, '1-FrameAnalyzer'))
sys.path.insert(0, os.path.join(PROJECT_DIR, '2-FileTracker'))
sys.path.insert(0, os.path.join(PROJECT_DIR, '3-RiskHunter'))
sys.path.insert(0, os.path.join(PROJECT_DIR, '4-ThreatDetector'))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_DIR, '4-ThreatDetector', '.env'))
except ImportError:
    pass

# ==================== 配置 ====================

DATA_DIR = os.path.join(PROJECT_DIR, 'video&log')

# 默认敏感文件关键词 (从 config.yaml 提取)
DEFAULT_SENSITIVE_KEYWORDS = [
    "机密", "绝密", "合同", "协议", "工资", "财务",
    "密码", "核心", "客户", "AccessKey", "credential"
]


# ==================== 模块导入 ====================

def import_modules():
    """延迟导入模块，避免启动时的依赖问题"""
    global WorklistManager, load_log_from_json, SensitiveFileEvent
    global BehaviorAnalysisGraph, analyze_sensitive_event_behavior
    global UploadDetectionConfig
    global DatalogEngine, PromptTemplates
    
    try:
        from worklist_manager import WorklistManager, load_log_from_json, SensitiveFileEvent
        from behavior_analysis_graph import BehaviorAnalysisGraph, analyze_sensitive_event_behavior
        print("✅ 模块2 (FileTracker + BehaviorAnalysis) 加载成功")
    except ImportError as e:
        logger.warning(f"模块2导入失败: {e}")
        WorklistManager = None
        BehaviorAnalysisGraph = None
    
    try:
        from upload_detection_config import UploadDetectionConfig
        print("✅ 模块3 (RiskHunter) 加载成功")
    except ImportError as e:
        logger.warning(f"模块3导入失败: {e}")
        UploadDetectionConfig = None
    
    try:
        from datalog.datalog_engine import DatalogEngine
        from prompts import PromptTemplates
        print("✅ 模块4 (ThreatDetector) 加载成功")
    except ImportError as e:
        logger.warning(f"模块4导入失败: {e}")
        DatalogEngine = None


# ==================== 辅助函数 ====================

def load_logs(log_file: str) -> List[Dict]:
    """加载日志文件，支持 JSON Lines 和 JSON Array 格式"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 尝试 JSON Array 格式
    if content.startswith('['):
        return json.loads(content)
    
    # JSON Lines 格式（每行一个 JSON 对象）
    logs = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def get_recording_start_time(logs: List[Dict]) -> str:
    """从第一条日志获取录屏开始时间"""
    if logs:
        first_ts = logs[0].get('timestamp', '')
        return first_ts.replace('T', ' ').split('.')[0]
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_sensitive_files_from_logs(logs: List[Dict], keywords: List[str]) -> List[str]:
    """从日志中提取敏感文件列表"""
    sensitive_files = set()
    
    for log in logs:
        file_path = log.get('file_path', '')
        file_name = log.get('file_name', '')
        
        # 检查文件名是否包含敏感关键词
        for keyword in keywords:
            if keyword.lower() in file_name.lower() or keyword.lower() in file_path.lower():
                if file_path:
                    sensitive_files.add(file_path)
                break
        
        # 检查 upload_detection 中的原始文件
        upload_info = log.get('upload_detection', {})
        if upload_info.get('is_upload'):
            orig_file = upload_info.get('original_file', '')
            if orig_file:
                for keyword in keywords:
                    if keyword.lower() in orig_file.lower():
                        sensitive_files.add(orig_file)
                        break
    
    return list(sensitive_files)


def create_index_file(rec_start: str, output_dir: str) -> str:
    """创建临时 INDEX.md 文件"""
    index_content = f"**Recording Time**: {rec_start}\n"
    index_path = os.path.join(output_dir, "INDEX.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(index_path, 'w') as f:
        f.write(index_content)
    return index_path


# ==================== Datalog 数据结构 ====================

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
    from_process: str = None
    to_process: str = None
    shared_data: str = None
    
    def to_souffle_args(self):
        ts = self._parse_timestamp()
        if self.relation == "OpenFile":
            return (self.operation_id, self.process, self.file, ts)
        elif self.relation == "TransferFile":
            return (self.operation_id, self.process, self.file, self.dst_file, ts)
        elif self.relation == "CrossProcessTransfer":
            return (self.operation_id, self.from_process or self.process, 
                    self.to_process or "Unknown", self.shared_data or self.file, ts)
        elif self.relation == "LeakFile":
            return (self.operation_id, self.process, self.file, "network", ts)
        return None
    
    def _parse_timestamp(self):
        if self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
            except:
                pass
        return 0


# ==================== 模块2: FileTracker ====================

def run_file_tracker(logs: List[Dict], video_path: str, index_path: str, 
                     sensitive_files: List[str]) -> Dict[str, Any]:
    """
    运行模块2: FileTracker
    
    功能:
    1. 使用 WorklistManager 管理敏感文件事件
    2. 调用 BehaviorAnalysisGraph 分析视频帧
    3. 识别隐藏行为（重命名、格式转换等）
    4. 更新敏感文件映射关系
    
    Returns:
        包含处理结果的字典
    """
    print("\n" + "=" * 80)
    print("📂 模块2: FileTracker - 敏感文件追踪")
    print("=" * 80)
    
    if WorklistManager is None:
        print("   ⚠️ WorklistManager 未加载")
        return {"events_processed": 0, "hidden_behaviors": [], "file_mappings": {}}
    
    # 初始化 WorklistManager
    manager = WorklistManager(sensitive_files=sensitive_files)
    print(f"   初始敏感文件: {len(sensitive_files)} 个")
    
    # 扫描日志构建 worklist
    added_count = manager.scan_and_build_worklist(logs)
    print(f"   Worklist 构建: {added_count} 个敏感事件")
    
    # 收集结果
    all_hidden_behaviors = []
    all_frame_analyses = []
    events_processed = 0
    
    # 首先从日志追踪文件重命名/移动操作
    for log in logs:
        event_type = log.get('event_type', '')
        file_path = log.get('file_path', '')
        dest_path = log.get('destination_path', '')
        
        if event_type in ['renamed', 'moved'] and dest_path:
            manager.update_file_mapping(file_path, dest_path)
            manager.add_sensitive_file(dest_path)
            all_hidden_behaviors.append({
                "operation_type": event_type,
                "original_file": file_path,
                "new_file": dest_path,
                "timestamp": log.get('timestamp', '')
            })
            print(f"   🔍 发现重命名: {os.path.basename(file_path)} → {os.path.basename(dest_path)}")
    
    # 如果 BehaviorAnalysisGraph 可用，使用视频分析
    if 'BehaviorAnalysisGraph' in dir() and BehaviorAnalysisGraph is not None:
        print(f"\n   🎬 启动视频帧分析...")
        while not manager.is_empty():
            event = manager.get_next_event()
            if not event:
                break
            
            events_processed += 1
            print(f"   [{events_processed}] 处理: {os.path.basename(event.current_file)}")
            
            try:
                result = analyze_sensitive_event_behavior(
                    event=event,
                    index_path=index_path,
                    video_path=video_path,
                    worklist_manager=manager,
                    log_events=logs
                )
                
                frame_result = result.get("frame_analysis_result")
                if frame_result and frame_result.get("events"):
                    all_frame_analyses.extend(frame_result.get("events", []))
                
                if result.get("has_hidden_behavior"):
                    hidden_ops = result.get("hidden_operations", [])
                    all_hidden_behaviors.extend(hidden_ops)
                    print(f"       🔍 发现 {len(hidden_ops)} 个隐藏行为")
            except Exception as e:
                logger.warning(f"视频分析失败: {e}")
                continue
    else:
        # 仅处理 worklist
        while not manager.is_empty():
            event = manager.get_next_event()
            if not event:
                break
            events_processed += 1
    
    print(f"\n   ✅ FileTracker 完成: 处理 {events_processed} 个事件")
    print(f"   📊 发现 {len(all_hidden_behaviors)} 个隐藏行为")
    print(f"   📊 文件映射: {len(manager.file_mapping)} 个")
    print(f"   📊 帧分析结果: {len(all_frame_analyses)} 个")
    
    return {
        "events_processed": events_processed,
        "hidden_behaviors": all_hidden_behaviors,
        "file_mappings": dict(manager.file_mapping),
        "frame_analyses": all_frame_analyses,
        "sensitive_files": list(manager.sensitive_files),
        "worklist_manager": manager
    }


# ==================== 模块3: RiskHunter ====================

def run_risk_hunter(logs: List[Dict], file_tracker_result: Dict, 
                    video_path: str, index_path: str) -> Dict[str, Any]:
    """
    运行模块3: RiskHunter
    
    功能:
    1. 分析上传/外发行为
    2. 判断应用黑白名单
    3. 生成报警事件
    
    Returns:
        包含报警结果的字典
    """
    print("\n" + "=" * 80)
    print("🔍 模块3: RiskHunter - 风险检测")
    print("=" * 80)
    
    if UploadDetectionConfig is None:
        print("   ⚠️ 模块3未加载，跳过")
        return {"alert_events": [], "info_events": []}
    
    config = UploadDetectionConfig()
    
    alert_events = []
    info_events = []
    
    # 分析日志中的上传检测
    for log in logs:
        upload_info = log.get('upload_detection', {})
        if not upload_info.get('is_upload'):
            continue
        
        app_name = upload_info.get('app_name', '未知')
        original_file = upload_info.get('original_file', '')
        upload_type = upload_info.get('upload_type', '')
        
        # 判断应用类别
        app_category = config.get_app_category(app_name)
        should_alert, alert_level = config.should_alert(app_category, "直接外发")
        
        event_info = {
            "timestamp": log.get('timestamp'),
            "file_path": original_file,
            "file_name": os.path.basename(original_file),
            "app_name": app_name,
            "app_category": app_category,
            "upload_type": upload_type,
            "should_alert": should_alert,
            "alert_level": alert_level,
            "process_info": log.get('process_info', {})
        }
        
        if should_alert:
            alert_events.append(event_info)
            print(f"   🚨 报警: {os.path.basename(original_file)} → {app_name} ({app_category})")
        else:
            info_events.append(event_info)
            print(f"   ℹ️ 记录: {os.path.basename(original_file)} → {app_name} ({app_category})")
    
    # 也分析帧分析结果中的外发行为
    frame_analyses = file_tracker_result.get("frame_analyses", [])
    for frame_event in frame_analyses:
        behavior_category = frame_event.get("behavior_category", "")
        if "外发" in behavior_category or "上传" in behavior_category:
            app_name = frame_event.get("app_name", "未知")
            app_category = config.get_app_category(app_name)
            should_alert, alert_level = config.should_alert(app_category, behavior_category)
            
            event_info = {
                "timestamp": frame_event.get("time_range", ""),
                "app_name": app_name,
                "app_category": app_category,
                "operation_type": frame_event.get("operation_type", ""),
                "behavior_category": behavior_category,
                "description": frame_event.get("description", ""),
                "should_alert": should_alert,
                "alert_level": alert_level
            }
            
            if should_alert:
                alert_events.append(event_info)
    
    print(f"\n   ✅ RiskHunter 完成")
    print(f"   🚨 报警事件: {len(alert_events)} 个")
    print(f"   ℹ️ 信息事件: {len(info_events)} 个")
    
    return {
        "alert_events": alert_events,
        "info_events": info_events
    }


# ==================== 模块4: ThreatDetector ====================

def run_threat_detector(logs: List[Dict], file_tracker_result: Dict,
                        risk_hunter_result: Dict) -> Dict[str, Any]:
    """
    运行模块4: ThreatDetector
    
    功能:
    1. 使用 LLM 生成 Datalog 事实
    2. 运行 Souffle 推理
    3. 检测泄露路径
    """
    print("\n" + "=" * 80)
    print("⚖️ 模块4: ThreatDetector - Datalog 推理")
    print("=" * 80)
    
    if DatalogEngine is None:
        print("   ⚠️ 模块4未加载，跳过")
        return {"leak_paths": [], "datalog_facts": []}
    
    # 合并帧分析结果
    frame_analyses = file_tracker_result.get("frame_analyses", [])
    
    # 如果没有帧分析，从日志模拟
    if not frame_analyses:
        print("   使用日志模拟视频帧...")
        for log in logs:
            if log.get('upload_detection', {}).get('is_upload'):
                frame_analyses.append({
                    "timestamp": log.get('timestamp', ''),
                    "app_name": log.get('process_info', {}).get('process_name', ''),
                    "operation_type": "上传",
                    "behavior_category": "数据外发",
                    "description": f"文件 {log.get('file_name', '')} 被上传"
                })
    
    # 使用 LLM 生成 Datalog 事实
    print("\n   🤖 调用 LLM 分析...")
    facts = generate_datalog_facts(logs, frame_analyses)
    print(f"   ✅ 生成 {len(facts)} 条 Datalog 事实")
    
    for i, fact in enumerate(facts, 1):
        if fact.relation == "CrossProcessTransfer":
            print(f"   {i}. {fact.relation}({fact.from_process} → {fact.to_process})")
        elif fact.dst_file:
            print(f"   {i}. {fact.relation}({fact.process}, {fact.file[:30]}... → {fact.dst_file})")
        else:
            print(f"   {i}. {fact.relation}({fact.process}, {fact.file[:30]}...)")
    
    # 运行 Datalog 推理
    print("\n   🔍 运行 Souffle 推理...")
    engine = DatalogEngine()
    
    for fact in facts:
        args = fact.to_souffle_args()
        if args:
            engine.add_fact(fact.relation, *args)
    
    leak_paths = engine.query_leak()
    engine.cleanup()
    
    print(f"\n   ✅ ThreatDetector 完成")
    print(f"   🚨 泄露路径: {len(leak_paths)} 条")
    
    return {
        "leak_paths": leak_paths,
        "datalog_facts": facts
    }


def generate_datalog_facts(logs: List[Dict], video_frames: List[Dict]) -> List[DatalogFact]:
    """使用 LLM 生成 Datalog 事实"""
    import re
    from openai import OpenAI
    
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    base_url = os.getenv("LLM_BASE_URL")
    
    if not api_key:
        raise ValueError("❌ 未配置 LLM_API_KEY")
    
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    
    messages = PromptTemplates.get_messages(logs, video_frames)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1
    )
    
    result_text = response.choices[0].message.content
    json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
    
    if not json_match:
        raise ValueError("LLM未返回有效JSON格式")
    
    facts_data = json.loads(json_match.group())
    
    facts = []
    for fd in facts_data:
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


# ==================== 证据报告 ====================

def generate_evidence_report(log_file: str, video_file: str,
                             file_tracker_result: Dict,
                             risk_hunter_result: Dict,
                             threat_detector_result: Dict,
                             output_dir: str = None) -> str:
    """生成完整的泄露证据报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    leak_paths = threat_detector_result.get("leak_paths", [])
    
    report = {
        "report_id": f"full_evidence_{timestamp}",
        "generated_at": datetime.now().isoformat(),
        "input": {
            "log_file": log_file,
            "video_file": video_file
        },
        "summary": {
            "module2_events_processed": file_tracker_result.get("events_processed", 0),
            "module2_hidden_behaviors": len(file_tracker_result.get("hidden_behaviors", [])),
            "module3_alert_events": len(risk_hunter_result.get("alert_events", [])),
            "module3_info_events": len(risk_hunter_result.get("info_events", [])),
            "module4_datalog_facts": len(threat_detector_result.get("datalog_facts", [])),
            "module4_leak_paths": len(leak_paths)
        },
        "file_tracker": {
            "sensitive_files": file_tracker_result.get("sensitive_files", []),
            "file_mappings": file_tracker_result.get("file_mappings", {}),
            "hidden_behaviors": file_tracker_result.get("hidden_behaviors", [])
        },
        "risk_hunter": {
            "alert_events": risk_hunter_result.get("alert_events", []),
            "info_events": risk_hunter_result.get("info_events", [])
        },
        "threat_detector": {
            "datalog_facts": [
                {
                    "relation": f.relation,
                    "operation_id": f.operation_id,
                    "process": f.process,
                    "file": f.file,
                    "description": f.description
                }
                for f in threat_detector_result.get("datalog_facts", [])
            ],
            "leak_paths": [lp.to_dict() for lp in leak_paths]
        },
        "conclusion": "🚨 发现数据泄露风险！" if leak_paths else "✅ 未发现数据泄露"
    }
    
    report_file = os.path.join(output_dir, f"full_evidence_{timestamp}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report_file


# ==================== 主流程 ====================

def run_full_e2e_pipeline(log_file: str, video_file: str, 
                           sensitive_keywords: List[str] = None):
    """运行完整的 E2E 流程"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        🔍 DataLeakDetector Full E2E Pipeline                         ║
║                                                                      ║
║   完整流程:                                                          ║
║   1. 加载日志 → 提取敏感文件                                         ║
║   2. 模块2 FileTracker → 追踪隐藏行为                                ║
║   3. 模块3 RiskHunter → 风险报警                                     ║
║   4. 模块4 ThreatDetector → Datalog 推理                             ║
║   5. 输出证据报告                                                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # 导入模块
    import_modules()
    
    # 阶段1: 加载数据
    print("\n" + "=" * 80)
    print("📂 阶段1: 加载数据")
    print("=" * 80)
    
    logs = load_logs(log_file)
    rec_start = get_recording_start_time(logs)
    print(f"   日志文件: {os.path.basename(log_file)}")
    print(f"   日志事件: {len(logs)} 条")
    print(f"   录屏开始时间: {rec_start}")
    
    # 提取敏感文件
    keywords = sensitive_keywords or DEFAULT_SENSITIVE_KEYWORDS
    sensitive_files = extract_sensitive_files_from_logs(logs, keywords)
    print(f"   敏感文件: {len(sensitive_files)} 个")
    for f in sensitive_files[:5]:
        print(f"      - {os.path.basename(f)}")
    if len(sensitive_files) > 5:
        print(f"      ... 还有 {len(sensitive_files) - 5} 个")
    
    # 创建临时 INDEX.md
    output_dir = os.path.join(PROJECT_DIR, 'output', 'temp')
    index_path = create_index_file(rec_start, output_dir)
    
    # 阶段2: 模块2 FileTracker
    file_tracker_result = run_file_tracker(
        logs=logs,
        video_path=video_file,
        index_path=index_path,
        sensitive_files=sensitive_files
    )
    
    # 阶段3: 模块3 RiskHunter
    risk_hunter_result = run_risk_hunter(
        logs=logs,
        file_tracker_result=file_tracker_result,
        video_path=video_file,
        index_path=index_path
    )
    
    # 阶段4: 模块4 ThreatDetector
    threat_detector_result = run_threat_detector(
        logs=logs,
        file_tracker_result=file_tracker_result,
        risk_hunter_result=risk_hunter_result
    )
    
    # 阶段5: 结果汇总
    print("\n" + "=" * 80)
    print("📊 阶段5: 检测结果汇总")
    print("=" * 80)
    
    leak_paths = threat_detector_result.get("leak_paths", [])
    alert_events = risk_hunter_result.get("alert_events", [])
    
    if leak_paths:
        print(f"\n🚨 检测到 {len(leak_paths)} 条泄露路径:\n")
        for i, leak in enumerate(leak_paths, 1):
            print(f"泄露路径 #{i}:")
            print(f"  📁 泄露文件: {leak.leaked_file}")
            print(f"  📤 泄露进程: {leak.leaking_proc}")
            print(f"  🌐 泄露渠道: {leak.leak_channel}")
            print(f"  🛤️  完整路径: {leak.full_path}")
            print()
    
    if alert_events:
        print(f"\n⚠️ 黑名单报警 {len(alert_events)} 个:")
        for evt in alert_events[:3]:
            print(f"   - {evt.get('file_name', '?')} → {evt.get('app_name', '?')}")
    
    # 生成报告
    report_file = generate_evidence_report(
        log_file, video_file,
        file_tracker_result, risk_hunter_result, threat_detector_result
    )
    print(f"\n📄 证据报告: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ Full E2E Pipeline 完成")
    print("=" * 80)
    
    if leak_paths or alert_events:
        print(f"\n⚠️ 发现数据泄露风险！")
    else:
        print(f"\n✅ 未发现数据泄露")
    
    return len(leak_paths) > 0 or len(alert_events) > 0


def main():
    parser = argparse.ArgumentParser(description='DataLeakDetector Full E2E Pipeline')
    parser.add_argument('--log', '-l', type=str, required=True, help='日志文件路径')
    parser.add_argument('--video', '-v', type=str, required=True, help='视频文件路径')
    parser.add_argument('--keywords', '-k', type=str, nargs='*', help='敏感关键词列表')
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.log):
        print(f"❌ 日志文件不存在: {args.log}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"❌ 视频文件不存在: {args.video}")
        sys.exit(1)
    
    has_risk = run_full_e2e_pipeline(
        log_file=args.log,
        video_file=args.video,
        sensitive_keywords=args.keywords
    )
    
    sys.exit(1 if has_risk else 0)


if __name__ == "__main__":
    main()
