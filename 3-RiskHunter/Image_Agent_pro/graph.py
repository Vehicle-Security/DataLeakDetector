# graph.py
import json
import re
import os
import glob
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import reason_node, act_node, observation_node

# 加载环境变量
load_dotenv()

def should_continue(state: AgentState):
    """
    条件边逻辑：判断是继续调用工具，还是结束
    """
    last_message = state["messages"][-1]
    content = last_message.content
    
    
    if "Final Answer" in content:
        return "end"
    
    elif "Action:" in content:
        return "act"
    
    return "act" 

def extract_final_json(state: AgentState):
    """
    辅助函数：从最后的对话中提取 JSON
    """
    last_msg = state["messages"][-1].content
    try:
        
        json_part = last_msg.split("Final Answer:")[-1].strip()
        
        json_part = json_part.replace("```json", "").replace("```", "").strip()

        json_part = re.sub(r',\s*}', '}', json_part)
        json_part = re.sub(r',\s*]', ']', json_part)

        return json.loads(json_part)
    except:
        return {"error": "无法解析最终 JSON", "raw": last_msg}


# --- 构建 Graph ---
workflow = StateGraph(AgentState)


workflow.add_node("reason", reason_node)
workflow.add_node("act", act_node)
workflow.add_node("observe", observation_node)


workflow.set_entry_point("reason")


workflow.add_conditional_edges(
    "reason",
    should_continue,
    {
        "act": "act",
        "end": END
    }
)
workflow.add_edge("act", "observe")
workflow.add_edge("observe", "reason")


app = workflow.compile()

def load_frames_from_folder(folder_path):
    """
    从指定文件夹自动加载关键帧并提取时间戳
    Args:
        folder_path: 关键帧文件夹路径
    Returns:
        tuple: (frame_paths列表, timestamps列表)
    """
    # 获取所有关键帧图片文件
    frame_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    # 按文件名排序
    frame_files.sort()
    
    if not frame_files:
        print(f"⚠️  警告：在 {folder_path} 中未找到图片文件")
        return [], []
    
    frame_paths = []
    timestamps = []
    
    # 正则表达式匹配时间戳：time后跟数字和s
    time_pattern = re.compile(r'time([\d.]+)s', re.IGNORECASE)
    
    for frame_file in frame_files:
        frame_paths.append(frame_file)
        
        # 从文件名中提取时间戳
        filename = os.path.basename(frame_file)
        match = time_pattern.search(filename)
        if match:
            timestamp = match.group(1) + "s"
            timestamps.append(timestamp)
        else:
            # 如果无法提取时间戳，使用索引作为标识
            timestamps.append(f"frame_{len(timestamps)+1}")
            print(f"⚠️  警告：无法从 {filename} 提取时间戳，使用默认值：索引")
    
    print(f"✅ 成功加载 {len(frame_paths)} 个关键帧")
    print(f"   时间范围: {timestamps[0]} ~ {timestamps[-1]}\n")
    
    return frame_paths, timestamps

# --- 主程序运行入口 ---
if __name__ == "__main__":

    # 从环境变量读取配置
    keyframe_folder = os.getenv("KEYFRAME_FOLDER", "../KeyFrame/output/scene_keyframes/scene_02_豆包_上传文件")
    log_file_path = os.getenv("LOG_FILE_PATH", "../logs/log_1.json")
    
    # 自动加载关键帧和时间戳
    frame_paths, timestamps = load_frames_from_folder(keyframe_folder)
    if not frame_paths:
        print("❌ 错误：未找到关键帧文件，程序退出")
        exit(1)

    # 加载系统日志
    system_logs = None
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            system_logs = json.load(f)
        print(f"✅ 成功加载系统日志: {len(system_logs)} 条记录\n")
    except Exception as e:
        print(f"⚠️  未能加载系统日志: {e}\n")

    input_data = {
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "system_logs": system_logs,
        "messages": [] 
    }

    print("🚀 启动数据泄露敏感行为分析 Agent...\n")
    
    # 运行图
    final_state = None
    for event in app.stream(input_data):
        for key, value in event.items():
            if key == "reason":
                print(f"\n🧠 [Reasoning] \n{value['messages'][-1].content}")
            
            final_state = value 
            
    # 提取结果
    if final_state and "messages" in final_state:
        
        print("\n✅ 分析结束。最终结果:")
        result_json = extract_final_json({"messages": [final_state['messages'][-1]]})
        print(json.dumps(result_json, indent=4, ensure_ascii=False))

    graph_png = app.get_graph().draw_mermaid_png()
    with open("agent_workflow_graph.png", "wb") as f:
        f.write(graph_png)