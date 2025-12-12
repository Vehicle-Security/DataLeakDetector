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
    
    
    current_tool_output: Optional[str]  
    
    
    final_output: Optional[dict]        