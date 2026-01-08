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
from datetime import datetime, timedelta
import logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("analysis.log", encoding='utf-8')  
    ]
)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_model = nn.Sequential(*list(resnet_base.children())[:-1]).to(DEVICE).eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
reader = easyocr.Reader(['ch_sim', 'en'], gpu=(DEVICE == 'cuda'))


def analyze_video_behavior(rec_start_time_str, search_start_time_str, search_end_time_str, target_keywords, video_path, similarity_threshold=0.98, sample_interval=1.0):
    """
    针对视频进行行为检索与分析
    :param rec_start_time_str: 录屏开始的时刻 (格式: "2026-01-06 17:00:00")
    :param search_start_time_str: 想要开始搜索的时刻 (格式: "2026-01-06 17:30:00")
    :param search_end_time_str: 搜索范围-结束时刻
    :param target_keywords: 目标关键词列表
    :param video_path: 视频文件路径
    :param output_dir: 结果输出目录
    :param similarity_threshold: 画面变化阈值
    :param sample_interval: 采样间隔（秒）
    """
    video_filename = os.path.basename(video_path) 
    video_name_only = os.path.splitext(video_filename)[0] 
    output_dir = f"./output_{video_name_only}/"
    
    t_fmt = "%Y-%m-%d %H:%M:%S"
    t_rec = datetime.strptime(rec_start_time_str, t_fmt)
    t_s_start = datetime.strptime(search_start_time_str, t_fmt)
    t_s_end = datetime.strptime(search_end_time_str, t_fmt)
    
   
    start_offset = (t_s_start - t_rec).total_seconds()
    end_offset = (t_s_end - t_rec).total_seconds()

    if start_offset < 0:
        logger.warning(f"搜索开始时间早于录屏开始时间，重置为 0s。")
        start_offset = 0
    
    if end_offset <= start_offset:
        logger.error("错误：搜索结束时间必须晚于搜索开始时间。")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    start_frame_idx = int(start_offset * fps)
    end_frame_idx = min(int(end_offset * fps), total_frames)

    if start_frame_idx >= total_frames:
        logger.error(f"定位失败：搜索起始点 {search_start_time_str} 已超过视频总长度。")
        return

    logger.info(f"📍 范围定位：从 {search_start_time_str} 到 {search_end_time_str} (持续 {end_offset - start_offset} 秒)")

    step = int(fps * sample_interval)
    filtered_frames_data = []

    
    candidate_indices = []
    prev_feature = None
    pbar_feat = tqdm(total=end_frame_idx - start_frame_idx, desc="🚀 阶段 1/3: 范围特征提取")
    
    curr_idx = start_frame_idx
    while curr_idx < end_frame_idx: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
        ret, frame = cap.read()
        if not ret: break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = resnet_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            curr_feat = feature_model(img_t).flatten()
            
        if prev_feature is None or torch.nn.functional.cosine_similarity(prev_feature.unsqueeze(0), curr_feat.unsqueeze(0)).item() < similarity_threshold:
            candidate_indices.append(curr_idx)
            prev_feature = curr_feat
        
        curr_idx += step
        pbar_feat.update(step)
    pbar_feat.close()

    
    
    filtered_frames_data = []
    last_hit_idx = -1  
    pbar_ocr = tqdm(total=len(candidate_indices), desc="🔍 阶段 2/3: 关键词文本过滤")
    
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        results = reader.readtext(frame, detail=0)
        found = False
        for text in results:
            text_cleaned = text.replace(" ", "") 
            if any(kw in text_cleaned for kw in target_keywords):
                found = True
                break
        
        if found:
            last_hit_idx = idx 
            current_wall_time = (t_rec + timedelta(seconds=idx / fps)).strftime(t_fmt)
            save_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(save_path, frame)
            filtered_frames_data.append({'frame': frame, 'frame_index': idx, 'wall_clock_time': current_wall_time})
        pbar_ocr.update(1)
    pbar_ocr.close()

    
    if last_hit_idx != -1:
        logger.info(f"⏳ 检测到最后匹配帧 {last_hit_idx}，正在延伸 3 秒上下文...")
        
        extension_seconds = [3, 8, 13, 18]
        
        for sec in extension_seconds:
            ext_idx = last_hit_idx + int(sec * fps)
            
            
            if ext_idx >= total_frames:
                logger.warning(f"延伸帧索引 {ext_idx} 已超过视频总长度，跳过。")
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, ext_idx)
            ret, frame = cap.read()
            if ret:
                ext_wall_time = (t_rec + timedelta(seconds=ext_idx / fps)).strftime(t_fmt)
                
                save_path = os.path.join(output_dir, f"frame_ext_{sec}s_{ext_idx:06d}.jpg")
                cv2.imwrite(save_path, frame)
                
                filtered_frames_data.append({
                    'frame': frame, 
                    'frame_index': ext_idx, 
                    'wall_clock_time': ext_wall_time,
                    'is_extended': True  
                })
    
    cap.release()

    if not filtered_frames_data:
        logger.warning(f"📭 在时间范围内未发现关键词。")
        return

    
    
    filtered_frames_data.sort(key=lambda x: x['frame_index'])
    
    logger.info(f"🧠 阶段 3/3: 全量分析开始，共识别到 {len(filtered_frames_data)} 帧关键画面")
    
    llm = ChatOpenAI(
        model="qwen2.5-vl-72b-instruct",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-62823ac2c480482084d040855d2e5a15",
    )

    table_rows = [f"{i+1:^8} | {f['frame_index']:^10} | {f['wall_clock_time']:^20}" for i, f in enumerate(filtered_frames_data)]
    table_str = "输入顺序 | 原始帧索引 | 现实时间戳 (%Y-%m-%d %H:%M:%S)\n" + "-"*60 + "\n" + "\n".join(table_rows)

    final_prompt = prompts.RETRIEVE_FRAMES_PROMPT.format(
        target_keywords=", ".join(target_keywords),
        frame_count=len(filtered_frames_data),
        frame_info_table=table_str
    )

    contents = [{"type": "text", "text": final_prompt}]
    for f in filtered_frames_data:
        _, buffer = cv2.imencode('.jpg', f['frame'], [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    all_events = []
    try:
        logger.info(f"🚀 正在向模型发送全量数据（共 {len(filtered_frames_data)} 张图片）...")
        resp = llm.invoke([HumanMessage(content=contents)])
        clean_text = re.sub(r'```json\n?|```', '', resp.content).strip()
        batch_data = json.loads(clean_text)
        if "events" in batch_data:
            all_events = batch_data["events"]
    except Exception as e:
        logger.error(f"❌ 全量分析失败: {e}")

    result_data = {
        "search_range": {"start": search_start_time_str, "end": search_end_time_str}, 
        "total_events": len(all_events), 
        "events": all_events
    }
    
    report_file = os.path.join(output_dir, "behavior_analysis_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    
    logger.info(f"✨ 任务完成！分析报告已保存至: {report_file}")
    return result_data


if __name__ == "__main__":
    # 使用示例

    # 1. 43:文件压缩
    logger.info("开始分析视频 43.mp4 的行为...")
    analysis_results = analyze_video_behavior(
        rec_start_time_str='2025-12-28 18:55:36',    # 录屏开始时间
        search_start_time_str='2025-12-28 18:55:46', # 搜索起始时间点
        search_end_time_str='2025-12-28 18:56:05',   # 搜索结束时间点
        target_keywords=['项目1详细规划'],
        video_path="../video/43.mp4"
    )
    events = analysis_results.get("events", [])
    first_event = events[0]
    logger.info(f"第一个事件的应用名称是: {first_event.get('app_name', '未知')}")
    logger.info(f"第一个事件的操作类型是: {first_event.get('operation_type', '未知')}")
    logger.info(f"第一个事件的行为类别是: {first_event.get('behavior_category', '未知')}")
    logger.info(f"第一个事件的变更前文件名是: {first_event.get('original_filename', '未知')}")
    logger.info(f"第一个事件的变更后文件名是: {first_event.get('modified_filename', '未知')}")
    logger.info(f"第一个事件的描述是: {first_event.get('description', '无')}")

    # 2. 42: 文件格式转换
    logger.info("开始分析视频 42.mp4 的行为...")
    analysis_results = analyze_video_behavior(
        rec_start_time_str='2025-12-28 18:41:28',    
        search_start_time_str='2025-12-28 18:41:53', 
        search_end_time_str='2025-12-28 18:42:10',   
        target_keywords=['项目2需求分析'],
        video_path="../video/42.mp4"
    )
    events = analysis_results.get("events", [])
    first_event = events[0]
    logger.info(f"第一个事件的应用名称是: {first_event.get('app_name', '未知')}")
    logger.info(f"第一个事件的操作类型是: {first_event.get('operation_type', '未知')}")
    logger.info(f"第一个事件的行为类别是: {first_event.get('behavior_category', '未知')}")
    logger.info(f"第一个事件的变更前文件名是: {first_event.get('original_filename', '未知')}")
    logger.info(f"第一个事件的变更后文件名是: {first_event.get('modified_filename', '未知')}")
    logger.info(f"第一个事件的描述是: {first_event.get('description', '无')}")

    # 3. 60: 文件重命名
    logger.info("开始分析视频 60.mp4 的行为...")
    analysis_results = analyze_video_behavior(
        rec_start_time_str='2026-01-05 15:45:08',    
        search_start_time_str='2026-01-05 15:45:18', 
        search_end_time_str='2026-01-05 15:45:50',   
        target_keywords=['项目3prd设计'],
        video_path="../video/60.mp4"
    )
    events = analysis_results.get("events", [])
    first_event = events[0]
    logger.info(f"第一个事件的应用名称是: {first_event.get('app_name', '未知')}")
    logger.info(f"第一个事件的操作类型是: {first_event.get('operation_type', '未知')}")
    logger.info(f"第一个事件的行为类别是: {first_event.get('behavior_category', '未知')}")
    logger.info(f"第一个事件的变更前文件名是: {first_event.get('original_filename', '未知')}")
    logger.info(f"第一个事件的变更后文件名是: {first_event.get('modified_filename', '未知')}")
    logger.info(f"第一个事件的描述是: {first_event.get('description', '无')}")



