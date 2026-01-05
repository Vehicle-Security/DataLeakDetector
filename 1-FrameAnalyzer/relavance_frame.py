import os
import shutil
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import easyocr
import base64
import json
import re
import prompts  
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_model = nn.Sequential(*list(resnet_base.children())[:-1]).to(DEVICE).eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
reader = easyocr.Reader(['ch_sim', 'en'], gpu=(DEVICE == 'cuda'))


def analyze_video_behavior(target_keywords, video_path, output_dir, similarity_threshold=0.98, sample_interval=1.0):
    """
    针对视频进行行为检索与分析
    :param target_keywords: 关键词列表，如 ['AAA公司员工守则']
    :param video_path: 视频文件路径
    :param output_dir: 结果输出目录
    :param similarity_threshold: 画面变化阈值
    :param sample_interval: 采样间隔（秒）
    """
    
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * sample_interval)
    filtered_frames_data = []

    
    candidate_indices = []
    prev_feature = None
    pbar_feat = tqdm(total=total_frames, desc="🚀 阶段 1/3: 画面特征提取")
    
    curr_idx = 0
    while curr_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
        ret, frame = cap.read()
        if not ret: break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = resnet_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            curr_feat = feature_model(img_t).flatten()
            
        if prev_feature is None:
            candidate_indices.append(curr_idx)
            prev_feature = curr_feat
        else:
            sim = torch.nn.functional.cosine_similarity(prev_feature.unsqueeze(0), curr_feat.unsqueeze(0)).item()
            if sim < similarity_threshold:
                candidate_indices.append(curr_idx)
                prev_feature = curr_feat
        
        curr_idx += step
        pbar_feat.update(step)
    pbar_feat.close()

    
    pbar_ocr = tqdm(total=len(candidate_indices), desc="🔍 阶段 2/3: 关键词文本过滤")
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        results = reader.readtext(frame, detail=0)
        found = any(any(kw in text for kw in target_keywords) for text in results)
        
        if found:
            save_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(save_path, frame)
            filtered_frames_data.append({
                'frame': frame,
                'frame_index': idx,
                'timestamp': idx / fps
            })
        pbar_ocr.update(1)
    pbar_ocr.close()
    cap.release()

    print(f"\n🎯 过滤完成！保留了 {len(filtered_frames_data)} 帧关键画面。")

   
    if not filtered_frames_data:
        print("📭 未发现相关关键帧，停止 VLM 分析。")
        return

    print(f"🧠 启动阶段 3/3: Qwen2.5-VL 多模态行为溯源...")
    
    llm = ChatOpenAI(
        model="qwen2.5-vl-72b-instruct",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-62823ac2c480482084d040855d2e5a15",
        streaming=False,
    )

    all_events = []

    for start_idx in range(0, len(filtered_frames_data), 5):
        batch = filtered_frames_data[start_idx : start_idx + 5]
        print(f"📦 分析批次 {start_idx//5 + 1}...")

        
        table_rows = [f"{i+1:^8} | {f['frame_index']:^10} | {f['timestamp']:^10.1f}" for i, f in enumerate(batch)]
        table_str = "输入顺序 | 原始帧索引 | 时间戳(秒)\n" + "-"*40 + "\n" + "\n".join(table_rows)

        # 2. 填充 Prompt
        final_prompt = prompts.RETRIEVE_FRAMES_PROMPT.format(
            target_keywords=", ".join(target_keywords),
            frame_count=len(batch),
            frame_info_table=table_str
        )

        
        contents = [{"type": "text", "text": final_prompt}]
        for f in batch:
            _, buffer = cv2.imencode('.jpg', f['frame'], [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        
        try:
            resp = llm.invoke([HumanMessage(content=contents)])
            clean_text = re.sub(r'```json\n?|```', '', resp.content).strip()
            batch_data = json.loads(clean_text)
            if "events" in batch_data:
                all_events.extend(batch_data["events"])
        except Exception as e:
            print(f"  ⚠️ 批次 {start_idx//5 + 1} 分析失败: {e}")

    
    report_file = os.path.join(output_dir, "behavior_analysis_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({"total_events": len(all_events), "events": all_events}, f, ensure_ascii=False, indent=4)
    
    print(f"\n✨ 任务完成！分析报告已保存至: {report_file}")


if __name__ == "__main__":
    # 使用示例
    analyze_video_behavior(
        target_keywords=['AAA公司员工守则'],
        video_path="../video/1.mov",
        output_dir="./output_final/"
    )