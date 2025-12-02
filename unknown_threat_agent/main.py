"""
Unknown Threat Agent - 未知威胁侦探系统主程序

基于"分诊-推理"机制的Agent核心框架
支持N2（战术手册）和N3（侦探引擎）协同工作
"""

import logging
import logging.config
import signal
import sys
import time
from typing import Optional

from .config import LOG_CONFIG, AGENT_CONFIG
from .core import EventBus, Memory, Toolbox, TriageSystem
from .engines import DetectiveEngine
from .utils import TaintTracker


class UnknownThreatAgent:
    """
    未知威胁侦探Agent主类
    
    整合所有组件，提供统一的启动和管理接口
    """
    
    def __init__(self):
        # 配置日志
        logging.config.dictConfig(LOG_CONFIG)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("=" * 80)
        self.logger.info("正在初始化未知威胁侦探Agent...")
        self.logger.info("=" * 80)
        
        # 初始化共享组件
        self.event_bus = EventBus(
            max_history=AGENT_CONFIG["event_bus"]["max_history"]
        )
        
        self.memory = Memory(
            db_path=AGENT_CONFIG["memory"]["db_path"]
        )
        
        self.toolbox = Toolbox()
        
        # 初始化分诊系统
        self.triage = TriageSystem(
            event_bus=self.event_bus,
            memory=self.memory
        )
        
        # 初始化N3侦探引擎
        self.detective = DetectiveEngine(
            event_bus=self.event_bus,
            memory=self.memory,
            toolbox=self.toolbox
        )
        
        # 初始化污点追踪器（可选）
        self.taint_tracker: Optional[TaintTracker] = None
        if AGENT_CONFIG["taint_tracker"]["enable"]:
            self.taint_tracker = TaintTracker(
                event_bus=self.event_bus,
                memory=self.memory
            )
        
        # 连接组件
        self._connect_components()
        
        # 运行状态
        self.running = False
        
        self.logger.info("未知威胁侦探Agent初始化成功")
    
    def _connect_components(self):
        """连接各个组件"""
        # 将N3引擎连接到分诊系统
        self.triage.set_n3_callback(self.detective.activate)
        
        self.logger.info("组件连接完成")
    
    def start(self):
        """启动Agent系统"""
        if self.running:
            self.logger.warning("Agent已经在运行中")
            return
        
        self.logger.info("正在启动未知威胁侦探Agent...")
        
        # 启动事件总线
        self.event_bus.start()
        
        # 启动分诊系统
        self.triage.start()
        
        # 启动污点追踪器
        if self.taint_tracker:
            self.taint_tracker.start()
        
        self.running = True
        
        self.logger.info("✅ 未知威胁侦探Agent已启动运行")
        self.logger.info("-" * 80)
        self._print_status()
    
    def stop(self):
        """停止Agent系统"""
        if not self.running:
            return
        
        self.logger.info("正在停止未知威胁侦探Agent...")
        
        # 停止事件总线
        self.event_bus.stop()
        
        # 保存记忆到磁盘
        self.memory.save_to_disk()
        
        self.running = False
        
        self.logger.info("未知威胁侦探Agent已停止")
    
    def inject_event(self, event_type: str, actor: str, target: str = None, **metadata):
        """
        手动注入事件（用于测试或集成外部系统）
        
        Args:
            event_type: 事件类型
            actor: 执行者
            target: 目标
            **metadata: 元数据
        """
        from .core.event_bus import Event
        
        event = Event(
            event_type=event_type,
            actor=actor,
            target=target,
            metadata=metadata
        )
        
        self.event_bus.publish(event)
        self.logger.debug(f"已注入事件: {event}")
    
    def get_status(self) -> dict:
        """获取系统状态"""
        return {
            "running": self.running,
            "event_bus": self.event_bus.get_stats(),
            "memory": self.memory.get_stats(),
            "triage": self.triage.get_stats(),
            "detective": self.detective.get_stats(),
            "taint_tracker": self.taint_tracker.get_taint_summary() if self.taint_tracker else None
        }
    
    def _print_status(self):
        """打印系统状态"""
        status = self.get_status()
        
        self.logger.info("系统状态:")
        self.logger.info(f"  事件总线: 已处理 {status['event_bus']['total_events']} 个事件")
        self.logger.info(f"  记忆系统: {status['memory']['total_nodes']} 个节点, {status['memory']['total_relationships']} 个关系")
        self.logger.info(f"  污点节点: {status['memory']['tainted_nodes']} 个")
        self.logger.info(f"  分诊系统: N2处理={status['triage']['n2_handled']}, N3激活={status['triage']['n3_activated']}")
        self.logger.info(f"  侦探引擎: 检测到 {status['detective']['threats_detected']} 个威胁")
        self.logger.info("-" * 80)


def signal_handler(sig, frame):
    """信号处理器"""
    print("\n收到中断信号，正在关闭...")
    sys.exit(0)


def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建并启动Agent
    agent = UnknownThreatAgent()
    agent.start()
    
    # 演示：注入一些测试事件
    print("\n" + "=" * 80)
    print("演示: 模拟'文本转音频'威胁场景")
    print("=" * 80 + "\n")
    
    time.sleep(1)
    
    # 场景1：用户从数据库复制敏感数据
    print("步骤1: 用户访问敏感数据库...")
    agent.inject_event(
        event_type="database_query",
        actor="userC",
        target="customer_db",
        sensitive=True,
        query="SELECT * FROM customers WHERE level='VIP'"
    )
    time.sleep(2)
    
    # 场景2：用户打开一个未知网站
    print("步骤2: 用户打开未知网站...")
    agent.inject_event(
        event_type="url_open",
        actor="userC",
        target="unknown_website.com",
        url="https://unknown_website.com/text-to-audio"
    )
    time.sleep(3)
    
    # 场景3：检测到文件创建（音频文件）
    print("步骤3: 音频文件已创建...")
    agent.inject_event(
        event_type="file_create",
        actor="userC",
        target="output_audio.mp3",
        file_path="/tmp/output_audio.mp3",
        file_size=1024000
    )
    time.sleep(2)
    
    # 场景4：用户尝试上传文件
    print("步骤4: 用户尝试上传文件...")
    agent.inject_event(
        event_type="file_upload",
        actor="userC",
        target="output_audio.mp3",
        destination="external_server",
        url="https://file-share.example.com/upload"
    )
    time.sleep(2)
    
    print("\n" + "=" * 80)
    print("演示完成。请查看日志了解详细分析。")
    print("=" * 80 + "\n")
    
    # 打印最终状态
    agent._print_status()
    
    # 保持运行
    try:
        print("Agent正在运行中。按 Ctrl+C 停止...\n")
        while True:
            time.sleep(10)
            # 定期打印状态
            agent._print_status()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
