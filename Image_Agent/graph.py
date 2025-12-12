# graph.py
import json
import re
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import reason_node, act_node, observation_node

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

# --- 主程序运行入口 ---
if __name__ == "__main__":
    
    input_data = {
        "frame_paths": [
            "../first_stage/output/scene_keyframes/scene_01_百度网盘_上传文件/frame_001_index300_time5.0s.jpg", 
            "../first_stage/output/scene_keyframes/scene_01_百度网盘_上传文件/frame_002_index480_time8.0s.jpg", 
            "../first_stage/output/scene_keyframes/scene_01_百度网盘_上传文件/frame_003_index540_time9.0s.jpg",
            "../first_stage/output/scene_keyframes/scene_01_百度网盘_上传文件/frame_004_index660_time11.0s.jpg",
            "../first_stage/output/scene_keyframes/scene_01_百度网盘_上传文件/frame_005_index840_time14.0s.jpg",
            "../first_stage/output/scene_keyframes/scene_01_百度网盘_上传文件/frame_006_index1680_time28.0s.jpg"
            
        ],
        "timestamps": [
            "5.0s", 
            "8.0s", 
            "9.0s",
            "11.0s",
            "14.0s",
            "28.0s"
        ],
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