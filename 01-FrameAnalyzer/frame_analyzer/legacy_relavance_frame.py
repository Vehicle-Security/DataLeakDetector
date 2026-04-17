# relavance_frame.py
"""
模块1入口函数封装，供模块2调用

这个文件封装对 VideoFileOperationAgent 的调用，提供 analyze_video_behavior 接口
"""

from typing import List, Dict, Any
from .legacy_agent import VideoFileOperationAgent


def analyze_video_behavior(
    rec_start_time_str: str,
    search_start_time_str: str,
    search_end_time_str: str,
    target_keywords: List[str],
    video_path: str
) -> Dict[str, Any]:
    """
    分析视频行为的入口函数，供模块2调用
    
    Args:
        rec_start_time_str: 录屏开始时间 (格式: "YYYY-MM-DD HH:MM:SS")
        search_start_time_str: 搜索开始时间
        search_end_time_str: 搜索结束时间
        target_keywords: 目标关键词列表
        video_path: 视频文件路径
    
    Returns:
        分析结果字典，包含:
        - search_range: 搜索时间范围
        - total_events: 事件总数
        - events: 事件列表，每个事件包含:
            - app_name: 应用名称
            - behavior_category: 行为类别 (正常操作/潜在隐藏行为/直接外发)
            - operation_type: 操作类型
            - original_filename: 原始文件名
            - modified_filename: 修改后文件名
            - time_range: 时间范围
            - involved_timestamps: 涉及的时间戳列表
            - description: 行为描述
    """
    try:
        agent = VideoFileOperationAgent()
        
        result = agent.run({
            "video_path": video_path,
            "keywords": target_keywords,
            "rec_start": rec_start_time_str,
            "search_start": search_start_time_str,
            "search_end": search_end_time_str
        })
        
        return result
        
    except Exception as e:
        print(f"❌ analyze_video_behavior 失败: {e}")
        return {
            "search_range": {
                "start": search_start_time_str,
                "end": search_end_time_str
            },
            "total_events": 0,
            "events": [],
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    # 测试
    result = analyze_video_behavior(
        rec_start_time_str="2025-12-28 18:41:28",
        search_start_time_str="2025-12-28 18:41:53",
        search_end_time_str="2025-12-28 18:42:10",
        target_keywords=["项目2需求分析"],
        video_path="../video/42.mp4"
    )
    print(result)
