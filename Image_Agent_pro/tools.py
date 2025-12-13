# tools.py
import base64
import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


vlm_llm = ChatOpenAI(
    model="qwen2-vl-72b-instruct",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-ff8d074c45cc4863869b230d1aec51f1", 
    temperature=0.1,
    streaming=False,
)

def encode_image(image_path: str) -> str:
    """将本地图片转换为 Base64 编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败: {image_path}, {e}")
        return ""

def run_vlm_analysis(target_paths: list[str], query: str) -> str:
    """
    核心工具：调用 Qwen2-VL 分析图片列表
    """
    
    if not target_paths or not os.path.exists(target_paths[0]):
        print(f"   [VLM Warning] 图片路径不存在，进入模拟模式: {target_paths}")
        
        return "视觉识别结果：屏幕显示用户点击了发送按钮。"
    # -----------------------------------------------------

    print(f"   [VLM] 正在调用 Qwen2-VL 分析 {len(target_paths)} 张图片... Query: {query}")

    content_parts = []
    
    # 1. 添加 Prompt
    content_parts.append({
        "type": "text", 
        "text": f"请仔细观察这些关键帧序列。以下是你需要识别的目标：{query}。请直接识别出任务目标，不要进行推测。"
    })

    # 2. 添加图片
    for path in target_paths:
        b64_img = encode_image(path)
        if b64_img:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

    message = HumanMessage(content=content_parts)
    system_prompt = SystemMessage(content="你是一个专业的GUI界面审计专家。")

    try:
        response = vlm_llm.invoke([system_prompt, message])
        print("视觉大模型识别结果: ", response.content)
        return response.content
    except Exception as e:
        return f"视觉分析调用失败: {str(e)}"


def analyze_system_logs(logs: list[dict], time_range: str = None, query: str = "") -> str:
    """
    分析系统日志，提取指定时间范围内的操作信息
    Args:
        logs: 系统日志列表
        time_range: 时间范围，格式如 "92.7s-114.0s" 或单个时间点 "100s"
        query: 查询目标，如 "文件操作"、"应用信息" 等
    Returns:
        格式化的日志分析结果
    """
    if not logs:
        return "未提供系统日志数据。"
    
    print(f"   [LogAnalyzer] 正在分析系统日志... Query: {query}, Time Range: {time_range}")
    
    # 解析时间范围
    filtered_logs = logs
    if time_range:
        try:
            if "-" in time_range:
                start_str, end_str = time_range.split("-")
                start_sec = float(start_str.replace("s", ""))
                end_sec = float(end_str.replace("s", ""))
                
                #TODO: 这里假设日志中的timestamp可以与视频时间对应
                # 实际应用中可能需要更复杂的时间匹配逻辑
                filtered_logs = logs  # 简化处理，返回所有日志
            else:
                target_sec = float(time_range.replace("s", ""))
                filtered_logs = logs
        except:
            pass
    
    # 格式化日志信息
    result_parts = []
    result_parts.append(f"找到 {len(filtered_logs)} 条相关日志记录：\n")
    
    for i, log in enumerate(filtered_logs, 1):
        log_info = f"记录 {i}:\n"
        log_info += f"  - 时间: {log.get('timestamp', 'N/A')}\n"
        log_info += f"  - 操作类型: {log.get('event_type', 'N/A')}\n"
        log_info += f"  - 文件路径: {log.get('file_path', 'N/A')}\n"
        log_info += f"  - 文件名: {log.get('file_name', 'N/A')}\n"
        log_info += f"  - 文件大小: {log.get('file_size', 'N/A')} bytes\n"
        log_info += f"  - 文件扩展名: {log.get('file_extension', 'N/A')}\n"
        
        process_info = log.get('process_info', {})
        log_info += f"  - 进程名: {process_info.get('process_name', 'N/A')}\n"
        log_info += f"  - 进程路径: {process_info.get('process_path', 'N/A')}\n"
        
        window_info = log.get('window_info', {})
        log_info += f"  - 窗口标题: {window_info.get('window_title', 'N/A')}\n"
        
        result_parts.append(log_info)
        # TODO: 日志分析llm 可以在这里添加更多的日志字段信息
    
    return "\n".join(result_parts)