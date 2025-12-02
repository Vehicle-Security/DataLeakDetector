"""
N3侦探引擎 - Detective Engine
基于ReAct框架的多轮次推理系统
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import logging
import uuid

from ..core.event_bus import Event, EventBus
from ..core.memory import Memory, Node, Relationship
from ..core.toolbox import Toolbox, ToolResult

logger = logging.getLogger(__name__)


class InvestigationState(Enum):
    """侦查状态"""
    IDLE = "idle"  # 空闲（休眠）
    REASONING = "reasoning"  # 推理中
    ACTING = "acting"  # 行动中
    WAITING = "waiting"  # 等待结果
    ALERTING = "alerting"  # 告警
    COMPLETED = "completed"  # 完成


@dataclass
class Investigation:
    """
    侦查任务
    表示N3对一个可疑事件的完整侦查过程
    """
    inv_id: str  # 侦查ID
    trigger_event: Event  # 触发侦查的事件
    state: InvestigationState = InvestigationState.IDLE
    hypothesis: Optional[str] = None  # 当前假设
    evidence: List[Any] = field(default_factory=list)  # 收集的证据
    actions_taken: List[Dict] = field(default_factory=list)  # 已采取的行动
    iterations: int = 0  # 迭代次数
    max_iterations: int = 10  # 最大迭代次数
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    threat_confirmed: bool = False  # 威胁是否确认
    threat_chain: List[str] = field(default_factory=list)  # 威胁链节点
    
    def add_evidence(self, evidence: Any):
        """添加证据"""
        self.evidence.append(evidence)
        self.updated_at = datetime.now()
    
    def record_action(self, action: str, result: Any):
        """记录行动"""
        self.actions_taken.append({
            "action": action,
            "result": result,
            "timestamp": datetime.now()
        })
        self.updated_at = datetime.now()


class DetectiveEngine:
    """
    N3侦探引擎 - Agent的推理大脑
    
    核心功能：
    1. ReAct循环（推理-行动）
    2. 假设生成与验证
    3. 证据收集与链接
    4. 污点传播决策
    5. 威胁确认与告警
    
    工作流程（以"文本转音频"为例）：
    
    迭代1：
        Trigger: userA (tainted) 打开 unknown_website.com
        Reason: "持有敏感数据的用户打开未知网站，假设这是数据外泄跳板"
        Act: 调用video_search工具分析网站UI
        Wait: 等待N1工具返回结果
    
    迭代2：
        Trigger: N1返回 found_UI=['text_to_audio_button']
        Reason: "假设被证实！这是跨模态伪装工具"
        Act: 监控.mp3/.wav文件创建
        Wait: 进入休眠，等待监控结果
    
    迭代3：
        Trigger: 监控到 output_audio.mp3 创建
        Reason: "这是转换产物，需要传播污点"
        Act: 标记output_audio.mp3为污点
        Wait: 继续监控该文件的后续操作
    
    迭代4：
        Trigger: userA尝试上传 output_audio.mp3
        Reason: "完整的STAC链条已闭环：DB复制->文本转音频->上传"
        Act: 立即告警！
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        memory: Memory,
        toolbox: Toolbox
    ):
        self.event_bus = event_bus
        self.memory = memory
        self.toolbox = toolbox
        
        # 当前进行中的侦查任务
        self.active_investigations: Dict[str, Investigation] = {}
        
        # 已完成的侦查任务（用于审计和学习）
        self.completed_investigations: List[Investigation] = []
        
        # 统计信息
        self.stats = {
            "total_investigations": 0,
            "threats_detected": 0,
            "false_positives": 0
        }
        
        logger.info("侦探引擎已初始化")
    
    def activate(self, event: Event):
        """
        激活N3引擎处理可疑事件
        这是从Triage系统接收到可疑事件的入口
        
        Args:
            event: 可疑事件
        """
        logger.warning(f"🕵️  N3已激活处理事件: {event}")
        
        # 创建新的侦查任务
        investigation = Investigation(
            inv_id=str(uuid.uuid4()),
            trigger_event=event
        )
        
        self.active_investigations[investigation.inv_id] = investigation
        self.stats["total_investigations"] += 1
        
        # 启动ReAct循环
        self._react_loop(investigation)
    
    def _react_loop(self, investigation: Investigation):
        """
        ReAct循环主体
        
        Args:
            investigation: 侦查任务
        """
        while investigation.iterations < investigation.max_iterations:
            investigation.iterations += 1
            logger.info(f"侦查 {investigation.inv_id}: 第 {investigation.iterations} 次迭代")
            
            # 步骤1: 推理（Reason）
            investigation.state = InvestigationState.REASONING
            reasoning_result = self._reason(investigation)
            
            if reasoning_result.get("conclusion") == "threat_confirmed":
                # 威胁确认，立即告警
                self._alert(investigation)
                break
            
            if reasoning_result.get("conclusion") == "no_threat":
                # 确认无威胁，结束侦查
                logger.info(f"侦查 {investigation.inv_id}: 未检测到威胁")
                investigation.state = InvestigationState.COMPLETED
                break
            
            # 步骤2: 行动（Act）
            investigation.state = InvestigationState.ACTING
            action_result = self._act(investigation, reasoning_result)
            
            if action_result.get("wait_required"):
                # 需要等待（如监控任务、工具异步结果）
                investigation.state = InvestigationState.WAITING
                logger.info(f"侦查 {investigation.inv_id}: 进入等待状态")
                # 进入休眠，等待新事件唤醒
                break
            
            # 继续下一轮迭代
        
        # 检查是否达到最大迭代次数
        if investigation.iterations >= investigation.max_iterations:
            logger.warning(f"侦查 {investigation.inv_id}: 已达到最大迭代次数")
            investigation.state = InvestigationState.COMPLETED
        
        # 将完成的侦查任务移到历史记录
        if investigation.state == InvestigationState.COMPLETED:
            self._finalize_investigation(investigation)
    
    def _reason(self, investigation: Investigation) -> Dict[str, Any]:
        """
        推理步骤：分析当前情况，生成假设
        
        Args:
            investigation: 侦查任务
            
        Returns:
            推理结果
        """
        logger.info(f"🧠 侦查 {investigation.inv_id} 推理中")
        
        # 构建上下文
        context = self._build_context(investigation)
        
        # 调用LLM进行推理
        llm_result = self.toolbox.call(
            "llm_reasoning",
            context=context,
            question="Analyze the threat potential of this behavior chain"
        )
        
        if not llm_result.success:
            logger.error(f"LLM推理失败: {llm_result.error}")
            return {"conclusion": "no_threat"}
        
        reasoning_data = llm_result.data
        
        # 更新假设
        if "hypothesis" in reasoning_data:
            investigation.hypothesis = reasoning_data["hypothesis"]
            logger.info(f"假设: {investigation.hypothesis}")
        
        # 决策：是否已经可以得出结论
        threat_level = reasoning_data.get("threat_level", "low")
        
        if threat_level == "high" and len(investigation.evidence) >= 2:
            # 高威胁 + 足够证据 = 确认威胁
            return {
                "conclusion": "threat_confirmed",
                "threat_type": reasoning_data.get("hypothesis", "unknown"),
                "confidence": reasoning_data.get("confidence", 0.5)
            }
        
        if threat_level == "low" and investigation.iterations > 3:
            # 低威胁 + 多轮迭代无进展 = 无威胁
            return {"conclusion": "no_threat"}
        
        # 需要继续收集证据
        return {
            "conclusion": "continue",
            "suggested_actions": reasoning_data.get("suggested_actions", []),
            "threat_level": threat_level
        }
    
    def _act(self, investigation: Investigation, reasoning_result: Dict) -> Dict[str, Any]:
        """
        行动步骤：根据推理结果，调用工具收集证据
        
        Args:
            investigation: 侦查任务
            reasoning_result: 推理结果
            
        Returns:
            行动结果
        """
        logger.info(f"🎬 侦查 {investigation.inv_id} 执行行动")
        
        suggested_actions = reasoning_result.get("suggested_actions", [])
        
        for action_desc in suggested_actions:
            # 解析行动描述，选择合适的工具
            action_result = self._execute_action(investigation, action_desc)
            investigation.record_action(action_desc, action_result)
            
            # 如果行动需要等待，立即返回
            if action_result.get("wait_required"):
                return {"wait_required": True}
        
        return {"wait_required": False}
    
    def _execute_action(self, investigation: Investigation, action_desc: str) -> Dict:
        """
        执行具体的行动
        
        Args:
            investigation: 侦查任务
            action_desc: 行动描述
            
        Returns:
            执行结果
        """
        action_lower = action_desc.lower()
        event = investigation.trigger_event
        
        # 行动1：调用视频搜索工具
        if "video" in action_lower or "ui" in action_lower or "analyze" in action_lower:
            result = self.toolbox.call(
                "video_search",
                prompt="upload convert audio text"
            )
            
            if result.success and result.data.get("found_ui_elements"):
                # 发现可疑UI元素
                investigation.add_evidence({
                    "type": "ui_analysis",
                    "elements": result.data["found_ui_elements"],
                    "confidence": result.data.get("confidence", 0)
                })
                
                # 在记忆中标记该网站的假设
                if event.target:
                    self.memory.set_hypothesis(event.target, "Modal_Jump_Tool")
            
            return {"success": True, "wait_required": False}
        
        # 行动2：监控文件创建
        elif "monitor" in action_lower and "file" in action_lower:
            file_types = []
            if "audio" in action_lower or "mp3" in action_lower:
                file_types = [".mp3", ".wav", ".m4a"]
            
            result = self.toolbox.call(
                "monitor_process",
                action="start",
                user=event.actor,
                file_type=file_types
            )
            
            # 监控任务启动后，需要等待
            return {"success": True, "wait_required": True, "monitor_task": result.data}
        
        # 行动3：标记污点
        elif "taint" in action_lower or "mark" in action_lower:
            if event.target:
                self.memory.mark_tainted(event.target, taint_source=event.actor)
                investigation.threat_chain.append(event.target)
            
            return {"success": True, "wait_required": False}
        
        # 行动4：追踪污点传播
        elif "track" in action_lower or "propagate" in action_lower:
            # 查找污点传播路径
            if event.target and event.actor:
                paths = self.memory.find_paths(event.actor, event.target)
                if paths:
                    investigation.add_evidence({
                        "type": "taint_chain",
                        "paths": paths
                    })
                    investigation.threat_chain.extend(paths[0] if paths else [])
            
            return {"success": True, "wait_required": False}
        
        # 默认：记录为通用证据
        return {"success": True, "wait_required": False}
    
    def _build_context(self, investigation: Investigation) -> str:
        """
        构建推理上下文
        
        Args:
            investigation: 侦查任务
            
        Returns:
            上下文字符串
        """
        event = investigation.trigger_event
        
        # 查询相关历史事件
        related_events = self.event_bus.query_history(
            actor=event.actor,
            limit=10
        )
        
        # 查询污点链
        taint_chain = []
        if self.memory.check_taint(event.actor):
            taint_chain = self.memory.get_taint_chain(event.actor)
        
        context = f"""
Current Event: {event}
Actor: {event.actor} (Tainted: {self.memory.check_taint(event.actor)})
Target: {event.target} (Tainted: {self.memory.check_taint(event.target) if event.target else False})

Current Hypothesis: {investigation.hypothesis or 'None'}
Iterations: {investigation.iterations}
Evidence Collected: {len(investigation.evidence)}

Related History:
{chr(10).join(f"  - {e}" for e in related_events[-5:])}

Taint Chain:
{chr(10).join(f"  - {node.node_id} ({node.node_type})" for node in taint_chain)}

Evidence:
{chr(10).join(f"  - {e}" for e in investigation.evidence)}
"""
        return context.strip()
    
    def _alert(self, investigation: Investigation):
        """
        发出威胁告警
        
        Args:
            investigation: 侦查任务
        """
        investigation.state = InvestigationState.ALERTING
        investigation.threat_confirmed = True
        self.stats["threats_detected"] += 1
        
        # 构建告警信息
        alert_msg = f"""
🚨 THREAT DETECTED! 🚨

Investigation ID: {investigation.inv_id}
Trigger Event: {investigation.trigger_event}
Hypothesis: {investigation.hypothesis}
Threat Chain: {' -> '.join(investigation.threat_chain)}

Evidence:
{chr(10).join(f"  - {e}" for e in investigation.evidence)}

Actions Taken:
{chr(10).join(f"  - {a['action']}" for a in investigation.actions_taken)}

Total Iterations: {investigation.iterations}
Detection Time: {(investigation.updated_at - investigation.created_at).total_seconds():.2f} seconds
"""
        
        logger.critical(alert_msg)
        
        # 在实际应用中，这里应该：
        # 1. 发送到SIEM系统
        # 2. 触发自动响应（如阻断、隔离）
        # 3. 通知安全运营人员
        
        # 保存完整的威胁链到记忆
        self._save_threat_chain(investigation)
        
        investigation.state = InvestigationState.COMPLETED
    
    def _save_threat_chain(self, investigation: Investigation):
        """
        将确认的威胁链保存到记忆系统
        
        Args:
            investigation: 侦查任务
        """
        for i in range(len(investigation.threat_chain) - 1):
            source = investigation.threat_chain[i]
            target = investigation.threat_chain[i + 1]
            
            self.memory.add_relationship(Relationship(
                source_id=source,
                target_id=target,
                rel_type="threat_chain",
                properties={
                    "investigation_id": investigation.inv_id,
                    "confirmed": True
                }
            ))
        
        logger.info(f"威胁链已保存到记忆: {' -> '.join(investigation.threat_chain)}")
    
    def _finalize_investigation(self, investigation: Investigation):
        """
        完成侦查任务
        
        Args:
            investigation: 侦查任务
        """
        # 从活跃列表移除
        if investigation.inv_id in self.active_investigations:
            del self.active_investigations[investigation.inv_id]
        
        # 添加到历史记录
        self.completed_investigations.append(investigation)
        
        logger.info(f"侦查 {investigation.inv_id} 已完成: 威胁={investigation.threat_confirmed}")
    
    def resume_investigation(self, inv_id: str, new_event: Event):
        """
        恢复（唤醒）一个处于等待状态的侦查任务
        当监控工具返回结果时调用
        
        Args:
            inv_id: 侦查ID
            new_event: 新的触发事件
        """
        if inv_id not in self.active_investigations:
            logger.warning(f"侦查 {inv_id} 未找到或已完成")
            return
        
        investigation = self.active_investigations[inv_id]
        
        if investigation.state != InvestigationState.WAITING:
            logger.warning(f"侦查 {inv_id} 不在等待状态")
            return
        
        logger.info(f"🔄 恢复侦查 {inv_id}，新事件: {new_event}")
        
        # 将新事件添加为证据
        investigation.add_evidence({
            "type": "monitored_event",
            "event": new_event
        })
        
        # 更新触发事件
        investigation.trigger_event = new_event
        
        # 继续ReAct循环
        self._react_loop(investigation)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            **self.stats,
            "active_investigations": len(self.active_investigations),
            "completed_investigations": len(self.completed_investigations)
        }
