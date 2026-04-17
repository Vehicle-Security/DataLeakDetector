import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import easyocr
import base64
import json
import re
import logging
from datetime import timedelta
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from .legacy_schema import AgentState
from thefuzz import fuzz
from .legacy_prompt_loader import PROMPTS

load_dotenv(find_dotenv())
logger = logging.getLogger("VideoAgent")


def _resolve_debug_frame_dir() -> str | None:
    enabled = str(os.getenv("FRAME_ANALYZER_SAVE_DEBUG_FRAMES", "") or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    target_dir = str(os.getenv("FRAME_ANALYZER_DEBUG_FRAME_DIR", "output/vlm_debug_frames") or "").strip()
    return target_dir or "output/vlm_debug_frames"


class VideoFileOperationAgent:
    def __init__(self, model_name=os.getenv("VL_MODEL_NAME", "gpt-5")):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_models()
        self.llm_model = model_name

    def _init_models(self):
        resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_model = nn.Sequential(*list(resnet_base.children())[:-1]).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=(self.device == 'cuda'))

    def vision_preprocessing_node(self, state: AgentState) -> AgentState:
        logger.info("Step 1: 正在通过视觉特征提取进行关键帧初步筛选...")
        cap = cv2.VideoCapture(state.video_path)
        state.fps = cap.get(cv2.CAP_PROP_FPS)
        state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_idx = int((state.time_range['search_start'] - state.time_range['rec_start']).total_seconds() * state.fps)
        end_idx = min(int((state.time_range['search_end'] - state.time_range['rec_start']).total_seconds() * state.fps), state.total_frames)

        prev_feat = None
        step = int(state.fps)

        for curr_idx in range(start_idx, end_idx, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
            ret, frame = cap.read()
            if not ret: break

            img_t = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                curr_feat = self.feature_model(img_t).flatten()

            if prev_feat is None or torch.nn.functional.cosine_similarity(prev_feat.unsqueeze(0), curr_feat.unsqueeze(0)).item() < 0.985:
                state.candidate_frames.append({'idx': curr_idx, 'frame': frame})
                prev_feat = curr_feat
        
        cap.release()
        return state

    def keyword_filter_node(self, state: AgentState) -> AgentState:
        is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)
        scene_markers = ["共享", "屏幕", "会议", "腾讯会议", "zoom", "meeting", "share", "screen"]
        is_scene_mode = any(
            any(marker.lower() in kw.lower() for marker in scene_markers)
            for kw in state.target_keywords
        )

        if is_long_text_mode:
            target_text = "".join(state.target_keywords).replace(" ", "").lower()
            logger.info(f"Step 2: long-text fuzzy match mode: {target_text[:20]}...")
        elif is_scene_mode:
            logger.info(f"Step 2: scene mode keyword filtering: {state.target_keywords}")
        else:
            logger.info(f"Step 2: OCR keyword filtering: {state.target_keywords}")

        last_hit_idx = -1
        for item in state.candidate_frames:
            results = self.reader.readtext(item['frame'], detail=0)
            text_blob = "".join(results).replace(" ", "").lower()

            is_match = False
            if is_long_text_mode:
                score = fuzz.partial_ratio(target_text, text_blob)
                if score >= 65:
                    is_match = True
                    logger.debug(f"frame {item['idx']} fuzzy hit: {score}")
            elif is_scene_mode:
                if any(kw.lower() in text_blob for kw in state.target_keywords if kw):
                    is_match = True
                elif len(state.candidate_frames) <= 180:
                    is_match = True
            else:
                if any(kw.lower() in text_blob for kw in state.target_keywords):
                    is_match = True

            if is_match:
                wall_time = (state.time_range['rec_start'] + timedelta(seconds=item['idx'] / state.fps)).strftime("%Y-%m-%d %H:%M:%S")
                state.hit_frames.append({
                    'idx': item['idx'],
                    'frame': item['frame'],
                    'time': wall_time,
                    'type': 'key_event'
                })
                last_hit_idx = item['idx']

        state.final_report['_last_hit_idx'] = last_hit_idx
        return state

    def context_extension_node(self, state: AgentState) -> AgentState:
        last_idx = state.final_report.get('_last_hit_idx', -1)
        if last_idx == -1: return state
        logger.info("Step 3: 正在针对命中帧提取后续上下文时间节点的补充画面...")
        cap = cv2.VideoCapture(state.video_path)
        for sec in [3, 8, 15]: 
            ext_idx = last_idx + int(sec * state.fps)
            if ext_idx < state.total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ext_idx)
                ret, frame = cap.read()
                if ret:
                    wall_time = (state.time_range['rec_start'] + timedelta(seconds=ext_idx / state.fps)).strftime("%Y-%m-%d %H:%M:%S")
                    state.hit_frames.append({
                        'idx': ext_idx, 'frame': frame, 'time': wall_time, 'type': 'context'
                    })
        cap.release()
        return state

    def behavior_analysis_node(self, state: AgentState) -> AgentState:
        if not state.hit_frames:
            state.final_report = {
                "search_range": {
                    "start": state.time_range['search_start'].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": state.time_range['search_end'].strftime("%Y-%m-%d %H:%M:%S")
                },
                "total_events": 0,
                "events": [],
                "status": "no_hits_found"
            }
            return state

        logger.info("Step 4: sending selected frames to VLM for behavior analysis...")
        save_dir = _resolve_debug_frame_dir()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        state.hit_frames.sort(key=lambda x: x['idx'])
        scene_markers = ["meeting", "share", "screen", "zoom", "共享", "屏幕", "会议", "腾讯会议"]
        is_scene_mode = any(
            any(marker.lower() in kw.lower() for marker in scene_markers)
            for kw in state.target_keywords
        )
        selected_hit_frames = state.hit_frames[:10] if is_scene_mode and len(state.hit_frames) > 10 else state.hit_frames

        llm = ChatOpenAI(
            model=self.llm_model,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        table_rows = [
            f"{i+1:^8} | {f['idx']:^10} | {f['time']:^20} | {f['type']:^10}"
            for i, f in enumerate(selected_hit_frames)
        ]
        table_str = "InputOrder | FrameIdx | WallTime | Type\n" + "-" * 75 + "\n" + "\n".join(table_rows)

        final_prompt = PROMPTS.RETRIEVE_FRAMES_PROMPT.format(
            frame_count=len(selected_hit_frames),
            frame_info_table=table_str
        )

        contents = [{"type": "text", "text": final_prompt}]
        for f in selected_hit_frames:
            if save_dir:
                safe_time = f['time'].replace(":", "-").replace(" ", "_")
                file_name = f"frame_{f['idx']}_{safe_time}_{f['type']}.jpg"
                save_path = os.path.join(save_dir, file_name)
                cv2.imwrite(save_path, f['frame'])
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
            raw_events = batch_data["events"] if isinstance(batch_data, dict) and "events" in batch_data else batch_data

            is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)
            if is_long_text_mode or is_scene_mode:
                final_events = raw_events
            else:
                final_events = []
                for event in raw_events:
                    original_name = event.get("original_filename", "").lower()
                    if any(kw.lower() in original_name for kw in state.target_keywords):
                        final_events.append(event)

            state.final_report = {
                "search_range": {
                    "start": state.time_range['search_start'].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": state.time_range['search_end'].strftime("%Y-%m-%d %H:%M:%S")
                },
                "total_events": len(final_events),
                "events": final_events,
                "status": "success"
            }
        except Exception as e:
            state.final_report = {"error": str(e), "status": "failed"}

        return state

    def run(self, config: Dict):
        from datetime import datetime
        t_fmt = "%Y-%m-%d %H:%M:%S"
        state = AgentState(
            video_path=config['video_path'],
            target_keywords=config['keywords'],
            time_range={
                'rec_start': datetime.strptime(config['rec_start'], t_fmt),
                'search_start': datetime.strptime(config['search_start'], t_fmt),
                'search_end': datetime.strptime(config['search_end'], t_fmt),
            }
        )
        pipeline = [
            self.vision_preprocessing_node,
            self.keyword_filter_node,
            self.context_extension_node,
            self.behavior_analysis_node
        ]
        for node in pipeline:
            state = node(state)
        return state.final_report
