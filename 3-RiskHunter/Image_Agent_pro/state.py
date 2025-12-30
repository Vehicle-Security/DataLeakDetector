# state.py
import operator
from typing import Annotated, List, TypedDict, Optional, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Agent 的运行状态
    """
    
    messages: Annotated[List[BaseMessage], operator.add]
    
    
    frame_paths: List[str]     
    timestamps: List[str]      
    
    # 系统日志相关字段
    system_logs: Optional[List[dict]]  # 系统日志列表
    
    
    current_tool_output: Optional[str]  
    
    
    final_output: Optional[dict]        