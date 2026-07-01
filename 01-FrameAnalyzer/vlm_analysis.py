import cv2
import base64
import json
import re
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import easyocr
import torch
from prompt_loader import PROMPTS

load_dotenv(find_dotenv())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
reader = easyocr.Reader(['ch_sim', 'en'], gpu=(DEVICE == 'cuda'))

TOKEN_PLAN_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def _first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def analyze_scene_deep_dive(filtered_frames, output_path="./spec/output/deep_report.json"):
    if not filtered_frames:
        print("⚠️ 输入帧序列为空，跳过分析。")
        return None

    print(f"🕵️ 启动全量溯源分析 (共 {len(filtered_frames)} 帧)...")

  
    table_rows = []
    processed_frames = []
    for i, f_info in enumerate(filtered_frames):
        
        raw_input = f_info.get('frame')
        img = cv2.imread(raw_input)
        texts = reader.readtext(img, detail=0)
        ocr_snippet = ", ".join(texts[:15])
        table_rows.append(
            f"{i+1:^5} | {f_info.get('frame_index', 'N/A'):^8} | {f_info.get('timestamp', 0.0):^10.1f} | {ocr_snippet[:50]}"
        )
        processed_frames.append(img)
    
    table_str = "序号 | 帧索引 | 时间戳(秒) | OCR 辅助关键词\n" + "-"*70 + "\n" + "\n".join(table_rows)

    
    final_prompt = PROMPTS.SCENE_DEEP_DIVE_PROMPT.format(
        frame_count=len(filtered_frames),
        frame_info_table=table_str
    )

    contents = [{"type": "text", "text": final_prompt}]
    
    for img in processed_frames:
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    
    llm = ChatOpenAI(
        model=os.getenv("VL_MODEL_NAME", "qwen3.7-plus"),
        base_url=_first_env("OPENAI_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "VL_BASE_URL") or TOKEN_PLAN_BASE_URL,
        api_key=_first_env("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "VL_API_KEY"),
        streaming=False,
    )

    try:
        print("🤖 正在请求 Qwen2.5-VL 推理意图...")
        messages = [HumanMessage(content=contents)]
        resp = llm.invoke(messages)
        llm_res = resp.content
        
        
        clean_text = re.sub(r'```json\n?|```', '', llm_res).strip()
        result_json = json.loads(clean_text)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 分析报告已保存至: {output_path}")
        return result_json

    except Exception as e:
        print(f"❌ 深度分析失败: {e}")
        
        with open("vlm_error_log.txt", "w", encoding="utf-8") as f_err:
            f_err.write(str(e))
        return None
    
if __name__ == "__main__":
    
    test_data = [
        {'frame': './spec/output/frame_002773.jpg', 'frame_index': 2773, 'timestamp': 46.2},
        {'frame': './spec/output/frame_002832.jpg', 'frame_index': 2832, 'timestamp': 47.2},
        {'frame': './spec/output/frame_002891.jpg', 'frame_index': 2891, 'timestamp': 48.2},
        {'frame': './spec/output/frame_003422.jpg', 'frame_index': 3422, 'timestamp': 57.0},
        
    ]
    
    final_report = analyze_scene_deep_dive(
        filtered_frames=test_data,
        output_path="./spec/output/security_deep_dive.json"
    )

    
