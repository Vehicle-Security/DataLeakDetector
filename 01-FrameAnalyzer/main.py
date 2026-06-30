import json
import logging
from agent import VideoFileOperationAgent

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    agent = VideoFileOperationAgent()
    result = agent.run({
        "video_path": "../video/42.mp4",
        "keywords": ["项目2需求分析"],
        "rec_start": "2025-12-28 18:41:28",
        "search_start": "2025-12-28 18:41:53",
        "search_end": "2025-12-28 18:42:10"
    })
    
    print(json.dumps(result, indent=4, ensure_ascii=False))
    
    events = result.get("events", [])
    if events:
        first_event = events[0]
        print(f"\n✅ 成功捕获事件！")
        print(f"事件的应用名称是: {first_event.get('app_name', '未识别到应用名称')}")
        print(f"具体行为: {first_event.get('behavior_category', '未知')}")
        print(f"操作描述: {first_event.get('description', '无')}")
        print(f"第一个事件的操作类型是: {first_event.get('operation_type', '未知')}")
        print(f"变更前文件名是: {first_event.get('original_filename', '未知')}")
        print(f"变更后文件名是: {first_event.get('modified_filename', '未知')}")
    else:
        print("\n⚠️ 未能识别到任何有效事件。")