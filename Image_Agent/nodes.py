import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from tools import run_vlm_analysis


reasoning_llm = ChatOpenAI(
    model="qwen2-vl-72b-instruct", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-api-key", 
    temperature=0.01, 
    streaming=False
)

SYSTEM_PROMPT = """
你是一名数据防泄漏安全分析专家。你的任务是基于提供的“关键帧列表”分析用户的敏感操作。
你必须严格遵循 ReAct (Reasoning -> Acting -> Observing) 循环来工作。

【可用工具】
- analyze_gui_frames: 调用视觉模型看图。
  - 参数格式: {"frame_indices": [...], "query": "..."} (注意: indices必须是整数列表)

【输出格式规范】
1. Thought: 思考当前已知什么，还需要通过看图确认什么（应用名？文件名？操作类型？）。
2. Action: analyze_gui_frames
3. Action Input: {"frame_indices": [...], "query": "..."}
   (注意：Action Input 必须是纯 JSON 字符串，不要使用 Markdown 代码块)
4. Observation: (此处留空，等待工具返回)

当你确认了所有关键信息（应用、文件名、操作时间、类型）后，输出最终结果：
Final Answer: {
    "app_name": "应用名", 
    "resource_name": "资源文件名(若无则填无)", 
    "operation_time": "发生操作的时间戳", 
    "operation_type": "如: 发送文件/截图/复制", 
    "reasoning": "一句话判断依据",
} 

注意：要输出最终结果的时候必须按照Final_answer的格式
"""

def reason_node(state: AgentState):
    """
    思考节点：决定下一步行动或输出结果
    """
    messages = state["messages"]
    
    
    if len(messages) == 0:
        frames_info = "【输入关键帧序列】\n"
        for i, (path, ts) in enumerate(zip(state["frame_paths"], state["timestamps"])):
            frames_info += f"Index {i}: {path} (Time: {ts})\n"
        
        initial_prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=frames_info + "\n请开始分析。")
        ]
        
        print("\n🧠 [Reasoning] Qwen 正在思考...")
        response = reasoning_llm.invoke(initial_prompt)
        
        
        return {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=frames_info), response]}
    
    
    print("\n🧠 [Reasoning] Qwen 正在基于 Observation 思考...")
    response = reasoning_llm.invoke(messages)
    return {"messages": [response]}


def act_node(state: AgentState):
    """
    行动节点：解析 LLM 指令并执行 Python 函数
    """
    last_message = state["messages"][-1]
    content = last_message.content
     
    
    print(f"\n--- [Act Node] 解析指令 ---")
    
    # 1. 正则提取 Action 和 Action Input
    
    action_match = re.search(r"Action:\s*(\w+)", content, re.DOTALL)
    input_match = re.search(r"Action Input:\s*(\{.*\})", content, re.DOTALL)
    
   
    
    tool_output = "错误：无法解析 Action Input，请确保使用严格的 JSON 格式。"
    
    if action_match and input_match:
        try:
            
            json_str = input_match.group(1).strip()
            
            
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            

            
            if "'" in json_str and '"' not in json_str:
                json_str = json_str.replace("'", '"')
            
            args = json.loads(json_str)
            indices = args.get("frame_indices", [])
            query = args.get("query", "")
            
            
            all_paths = state["frame_paths"]
            target_paths = []
            valid_indices = []
            
            for idx in indices:
                if 0 <= idx < len(all_paths):
                    target_paths.append(all_paths[idx])
                    valid_indices.append(idx)
            
            if not target_paths:
                tool_output = "错误：提供的帧索引越界或无效。"
            else:
                
                print(f">> 执行 VLM 分析，Indices: {valid_indices}")
                vlm_result = run_vlm_analysis(target_paths, query)
                tool_output = f"工具返回结果: {vlm_result}"
                
        except json.JSONDecodeError:
            tool_output = f"错误：Action Input JSON 解析失败。原始内容: {json_str}"
        except Exception as e:
            tool_output = f"执行异常: {str(e)}"
    
    else:
        
        print("   [Act Warning] 未匹配到 Action 模式，可能是 Final Answer 或格式错误。")
        tool_output = "系统提示：未检测到标准 Action 格式。如果你已经有了结论，请输出 Final Answer。"

    return {"current_tool_output": tool_output}


def observation_node(state: AgentState):
    """
    观察节点：将工具结果反馈给 Agent
    """
    tool_output = state["current_tool_output"]
    print(f"--- [Observation Node] ---")
    
    # 构造 Observation 消息
    
    obs_text = f"Observation: {tool_output}"
    
    return {
        "messages": [HumanMessage(content=obs_text)],
        "current_tool_output": None 
    }