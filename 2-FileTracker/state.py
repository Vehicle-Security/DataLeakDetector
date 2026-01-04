# state.py
import operator
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class SensitiveOperation(TypedDict):
    """
    单个敏感操作片段
    """
    operation_id: str              # 操作唯一标识
    operation_type: str            # 操作类型（如：文件访问、上传、复制、重命名等）
    resource_name: str             # 操作的资源名称（文件名、应用名等）
    app_name: str                  # 执行操作的应用名称
    start_time: str                # 操作开始时间
    end_time: Optional[str]        # 操作结束时间
    keyframes: List[str]           # 相关关键帧路径
    raw_description: str           # RiskSieve 输出的原始文字描述


class Resource(TypedDict):
    """
    被追踪的敏感资源
    """
    resource_id: str               # 资源唯一标识
    resource_name: str             # 资源名称
    resource_type: str             # 资源类型（文件、文本片段、截图等）
    original_source: Optional[str] # 原始来源（如果是派生资源）
    first_seen_time: str           # 首次发现时间
    last_seen_time: str            # 最后出现时间
    related_operations: List[str]  # 相关操作ID列表
    attributes: Dict[str, Any]     # 资源属性（文件大小、格式、内容特征等）


class EvidenceChain(TypedDict):
    """
    证据链路
    """
    chain_id: str                  # 链路唯一标识
    root_resource: Resource        # 起始资源
    derived_resources: List[Resource]  # 派生资源列表
    operation_sequence: List[SensitiveOperation]  # 操作序列
    relationships: List[Dict[str, str]]  # 资源间的关系（parent, child, type）
    risk_indicators: List[str]     # 风险指标（如：跨应用转移、格式转换等）


class TrackerState(TypedDict):
    """
    EvidenceTracer 的运行状态
    """
    # LLM 对话消息
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 输入：来自 RiskSieve 的敏感操作片段
    input_operations: List[SensitiveOperation]
    
    # 当前正在分析的操作索引
    current_operation_index: int
    
    # 已识别的敏感资源池
    tracked_resources: Dict[str, Resource]  # key: resource_id
    
    # 证据链路列表
    evidence_chains: List[EvidenceChain]
    
    # 当前工具输出
    current_tool_output: Optional[str]
    
    # 中间分析结果
    analysis_results: List[Dict[str, Any]]
    
    # 最终输出
    final_output: Optional[Dict[str, Any]]
    
    # 是否完成所有操作的分析
    is_complete: bool
