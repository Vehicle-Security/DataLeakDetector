"""
分诊系统 - Triage
决定事件由N2战术手册引擎处理，还是激活N3侦探引擎
"""

from typing import Optional, Callable, List
from enum import Enum
import logging
from .event_bus import Event, EventBus
from .memory import Memory

logger = logging.getLogger(__name__)


class TriageDecision(Enum):
    """分诊决策"""
    N2_HANDLE = "n2_handle"  # 交给N2战术手册引擎处理
    N3_ACTIVATE = "n3_activate"  # 激活N3侦探引擎
    IGNORE = "ignore"  # 忽略（正常行为）


class PlaybookEngine:
    """
    N2战术手册引擎的模拟接口
    
    在实际应用中，这应该对接需求2的完整战术手册系统
    """
    
    def __init__(self):
        # 已知的行为模式（90+个剧本）
        self.playbooks = {
            "file_copy_to_usb": ["file_write", "usb_device"],
            "screenshot_capture": ["screenshot", "save_image"],
            "email_attachment": ["email_open", "file_attach"],
            "browser_download": ["browser", "file_download"],
            # ... 其他90+个已知剧本
        }
        logger.info(f"战术手册引擎已初始化，包含 {len(self.playbooks)} 个剧本")
    
    def check_match(self, event: Event) -> bool:
        """
        检查事件是否匹配任何已知剧本
        
        Args:
            event: 待检查的事件
            
        Returns:
            True如果匹配，False如果未知
        """
        # 简化逻辑：检查事件类型是否在任何剧本中
        for playbook_name, triggers in self.playbooks.items():
            if event.event_type in triggers:
                logger.debug(f"事件 {event.event_type} 匹配到剧本: {playbook_name}")
                return True
        
        logger.debug(f"事件 {event.event_type} 未被任何剧本识别")
        return False
    
    def handle_event(self, event: Event):
        """
        N2引擎处理已知事件
        执行固定的战术动作
        """
        logger.info(f"N2处理事件: {event}")
        # N2的快速响应逻辑
        # 在实际应用中，这里会执行具体的剧本动作


class TaintChecker:
    """
    污点检查器
    检查事件是否触碰到被标记为污点的实体
    """
    
    def __init__(self, memory: Memory):
        self.memory = memory
    
    def is_suspicious(self, event: Event) -> bool:
        """
        判断事件是否可疑
        
        可疑的定义：
        1. 事件的actor（执行者）被标记为污点
        2. 事件的target（目标）被标记为污点
        3. 事件涉及的数据（如剪贴板内容）被标记为污点
        
        Args:
            event: 待检查的事件
            
        Returns:
            True如果可疑，False如果正常
        """
        # 检查actor
        if self.memory.check_taint(event.actor):
            logger.info(f"事件可疑: 执行者 '{event.actor}' 被污染")
            return True
        
        # 检查target
        if event.target and self.memory.check_taint(event.target):
            logger.info(f"事件可疑: 目标 '{event.target}' 被污染")
            return True
        
        # 检查metadata中的关键实体
        for key, value in event.metadata.items():
            if isinstance(value, str) and self.memory.check_taint(value):
                logger.info(f"事件可疑: 元数据 '{key}' 的值 '{value}' 被污染")
                return True
        
        return False


class TriageSystem:
    """
    分诊系统 - 高性能事件路由
    
    核心流程：
    1. 接收新事件
    2. 首先让N2战术手册引擎检查
    3. 如果N2不认识，检查是否触碰污点
    4. 只有当(未命中N2) AND (触碰污点)时，才激活N3
    
    这个机制确保了N3的"昂贵大脑"只在必要时被激活
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        memory: Memory,
        n2_engine: Optional[PlaybookEngine] = None
    ):
        self.event_bus = event_bus
        self.memory = memory
        self.n2_engine = n2_engine or PlaybookEngine()
        self.taint_checker = TaintChecker(memory)
        
        # N3激活回调
        self.n3_activation_callback: Optional[Callable[[Event], None]] = None
        
        # 统计信息
        self.stats = {
            "total_events": 0,
            "n2_handled": 0,
            "n3_activated": 0,
            "ignored": 0
        }
        
        logger.info("分诊系统已初始化")
    
    def set_n3_callback(self, callback: Callable[[Event], None]):
        """
        设置N3激活回调函数
        
        Args:
            callback: N3引擎的事件处理函数
        """
        self.n3_activation_callback = callback
        logger.info("N3激活回调已注册")
    
    def start(self):
        """
        启动分诊系统
        订阅事件总线的所有事件
        """
        self.event_bus.subscribe("*", self.process_event)
        logger.info("分诊系统已启动并订阅事件总线")
    
    def process_event(self, event: Event) -> TriageDecision:
        """
        处理单个事件，做出分诊决策
        
        Args:
            event: 待处理的事件
            
        Returns:
            TriageDecision
        """
        self.stats["total_events"] += 1
        
        logger.debug(f"分诊事件: {event}")
        
        # 步骤1：让N2战术手册引擎检查
        if self.n2_engine.check_match(event):
            # N2认识这个行为，交给N2处理
            self.n2_engine.handle_event(event)
            self.stats["n2_handled"] += 1
            logger.info(f"决策: N2处理 {event}")
            return TriageDecision.N2_HANDLE
        
        # 步骤2：N2不认识，检查是否触碰污点
        if self.taint_checker.is_suspicious(event):
            # 触碰污点 -> 激活N3
            logger.warning(f"决策: 激活N3 {event}")
            self.stats["n3_activated"] += 1
            
            # 回调N3引擎
            if self.n3_activation_callback:
                try:
                    self.n3_activation_callback(event)
                except Exception as e:
                    logger.error(f"N3回调错误: {e}", exc_info=True)
            
            return TriageDecision.N3_ACTIVATE
        
        # 步骤3：既不匹配N2，也不触碰污点 -> 正常行为，忽略
        self.stats["ignored"] += 1
        logger.debug(f"决策: 忽略 {event}")
        return TriageDecision.IGNORE
    
    def get_stats(self) -> dict:
        """获取分诊统计信息"""
        total = self.stats["total_events"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "n2_percentage": round(self.stats["n2_handled"] / total * 100, 2),
            "n3_percentage": round(self.stats["n3_activated"] / total * 100, 2),
            "ignored_percentage": round(self.stats["ignored"] / total * 100, 2)
        }
