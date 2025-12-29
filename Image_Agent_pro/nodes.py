import re
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from tools import run_vlm_analysis, analyze_system_logs

# 加载环境变量
load_dotenv()

reasoning_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen2-vl-72b-instruct"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", "0.01")),
    streaming=False
)

SYSTEM_PROMPT = """
你是一名数据防泄漏安全分析专家。你的任务是基于提供的"关键帧列表"和"系统日志"分析用户敏感操作的关键信息。
你必须严格遵循 ReAct (Reasoning -> Acting -> Observing) 循环来工作。

【可用工具】
- analyze_gui_frames: 调用视觉模型看图, 识别GUI界面信息。
  - 参数格式: {"frame_indices": [...], "query": "..."} (注意: indices必须是整数列表)
  
- analyze_system_logs: 分析系统日志，获取精确的文件操作、进程、应用信息。
  - 参数格式: {"time_range": "开始时间-结束时间(可选)", "query": "查询目标"}
  - 示例: {"time_range": "92.7s-114.0s", "query": "文件操作信息"}

【分析策略】
- 优先使用系统日志获取精确的文件名、应用名、进程信息等结构化数据
- 使用视觉分析补充GUI界面细节和用户交互行为
- 结合两种数据源进行交叉验证，提高准确性

【输出格式规范】
阶段1：工具调用阶段（分析过程中）。此时，输出必须按照以下格式输出：
1. Thought: 思考当前已知什么，还需要通过什么工具确认什么信息（优先考虑日志分析）。
2. Action: analyze_system_logs
3. Action Input: {"time_range": "...", "query": "..."}
   (注意:Action Input 必须是纯 JSON 字符串，不要使用 Markdown 代码块)
4. Observation: (此处留空，等待工具返回)
**或**
1. Thought: 思考当前已知什么，还需要通过什么工具确认什么信息（如日志无法确认，则使用视觉分析）。
2. Action: analyze_gui_frames
3. Action Input: {"frame_indices": [...], "query": "..."}
    (注意:Action Input 必须是纯 JSON 字符串，不要使用 Markdown 代码块)
4. Observation: (此处留空，等待工具返回)
**注意**: 在这一阶段，不要输出 Final Answer。

阶段2：当且仅当你认为已经收集了所有必要信息（应用、文件名、操作时间、类型）后，才输出最终结果（格式必须严格一致）：
Final Answer: {
    "app_name": "应用名", 
    "resource_name": "资源文件名(若无则填无)", 
    "operation_time": "发生操作的时间戳(可以从帧文件名或日志中提取)", 
    "operation_type": "如: 发送文件/截图/复制", 
    "reasoning": "结合日志和视觉分析的综合判断依据",
} 

注意:要输出最终结果的时候必须严格按照Final_Answer的格式输出, 在输出Final Answer之前，不要再写Action和Action Input。
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
        
        # 添加系统日志信息
        logs_info = ""
        if state.get("system_logs"):
            logs_info = "\n【输入系统日志序列】\n"
            logs_info += f"共有 {len(state['system_logs'])} 条系统日志记录可供分析。\n"
            logs_info += "日志包含：文件操作、进程信息、窗口信息等详细数据。\n"
        
        initial_prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=frames_info + logs_info + "\n请开始分析。")
        ]
        
        print("\n🧠 [Reasoning] Qwen 正在思考...")
        response = reasoning_llm.invoke(initial_prompt)
        
        
        return {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=frames_info + logs_info), response]}
    
    
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
    
    action_name = action_match.group(1) if action_match else None
    
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
            
            # 根据不同的action调用不同的工具
            if action_name == "analyze_system_logs":
                # 调用日志分析工具
                time_range = args.get("time_range", None)  # TODO: 将时间范围更改为时间戳序列
                query = args.get("query", "")
                
                logs = state.get("system_logs", [])

                if not logs:
                    tool_output = "错误：未提供系统日志数据。"
                else:
                    print(f">> 执行系统日志分析，Time Range: {time_range}")
                    log_result = analyze_system_logs(logs, time_range, query)
                    tool_output = f"工具返回结果: {log_result}"
                    
            elif action_name == "analyze_gui_frames":
                # 调用视觉分析工具
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
                    print(f">> 执行 VLM 关键帧分析，Indices: {valid_indices}")
                    vlm_result = run_vlm_analysis(target_paths, query)
                    tool_output = f"工具返回结果: {vlm_result}"
            else:
                tool_output = f"错误：未知的工具名称 '{action_name}'。可用工具: analyze_system_logs, analyze_gui_frames"
                
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
    print(f"\n--- [Observation Node] ---")
    
    # 构造 Observation 消息
    
    obs_text = f"Observation: {tool_output}"
    
    return {
        "messages": [HumanMessage(content=obs_text)],
        "current_tool_output": None 
    }
