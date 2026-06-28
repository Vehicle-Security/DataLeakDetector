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
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from schema import AgentState
from thefuzz import fuzz
from prompt_loader import PROMPTS

load_dotenv(find_dotenv())
logger = logging.getLogger("VideoAgent")

class VideoFileOperationAgent:
    def __init__(self, model_name=os.getenv("VL_MODEL_NAME", "gpt-5")):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_models()
        self.llm_model = model_name

    @staticmethod
    def _get_int_env(name: str, default: int, minimum: int = 1) -> int:
        try:
            return max(minimum, int(os.getenv(name, str(default))))
        except ValueError:
            return default

    @staticmethod
    def _ocr_snippet(text: str, limit: int = 80) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text[:limit]

    @staticmethod
    def _dedupe_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for frame in sorted(frames, key=lambda item: item.get("idx", 0)):
            idx = frame.get("idx")
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(frame)
        return deduped

    @staticmethod
    def _pick_evenly(frames: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if count <= 0 or not frames:
            return []
        if len(frames) <= count:
            return list(frames)
        if count == 1:
            return [max(frames, key=lambda item: item.get("ocr_score", 0))]

        chosen = {}
        for pos in range(count):
            idx = round(pos * (len(frames) - 1) / (count - 1))
            frame = frames[idx]
            chosen[frame.get("idx")] = frame
        return [chosen[key] for key in sorted(chosen)]

    def _select_vlm_frames(self, hit_frames: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        max_frames = self._get_int_env("DLD_VLM_MAX_FRAMES", 6)
        deduped = self._dedupe_frames(hit_frames)
        key_frames = [item for item in deduped if item.get("type") == "key_event"]
        context_frames = [item for item in deduped if item.get("type") != "key_event"]

        selected_context = context_frames[-3:]
        key_budget = max_frames - len(selected_context)
        if key_budget < 1 and key_frames:
            selected_context = selected_context[-max(0, max_frames - 1):]
            key_budget = max_frames - len(selected_context)

        selected = self._pick_evenly(key_frames, key_budget) + selected_context
        if len(selected) > max_frames:
            selected = selected[:max_frames]
        selected = self._dedupe_frames(selected)

        meta = {
            "candidate_hit_frames": len(hit_frames),
            "deduped_hit_frames": len(deduped),
            "vlm_sent_frames": len(selected),
            "max_vlm_frames": max_frames,
            "selection_strategy": "ocr_hits_evenly_sampled_plus_tail_context",
        }
        return selected, meta

    def _resize_for_vlm(self, frame):
        max_edge = self._get_int_env("DLD_VLM_IMAGE_MAX_EDGE", 1280)
        height, width = frame.shape[:2]
        largest = max(height, width)
        if largest <= max_edge:
            return frame
        scale = max_edge / largest
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

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
        # 判断是否需要执行“长文本模糊匹配”模式
        # 逻辑：如果 keywords 列表中有任何一个元素长度 > 10，则视为粘贴的长文本
        is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)
        
        if is_long_text_mode:
            # 将所有关键词拼接成一个目标长文本进行比对
            target_text = "".join(state.target_keywords).replace(" ", "").lower()
            logger.info(f"Step 2: 检测到长文本，正在执行模糊相似度匹配 (目标: {target_text[:20]}...)")
        else:
            logger.info(f"Step 2: 正在执行 OCR 文本识别并匹配关键词: {state.target_keywords}...")

        last_hit_idx = -1
        for item in state.candidate_frames:
            results = self.reader.readtext(item['frame'], detail=0)
            raw_ocr_text = " ".join(results)
            text_blob = raw_ocr_text.replace(" ", "").lower()
            
            is_match = False
            score = 0
            if is_long_text_mode:
                score = fuzz.partial_ratio(target_text, text_blob)
                if score >= 65:
                    is_match = True
                    logger.debug(f"帧 {item['idx']} 模糊匹配成功，得分: {score}")
            else:
                if any(kw.lower() in text_blob for kw in state.target_keywords):
                    is_match = True
                    score = 100

            if is_match:
                wall_time = (state.time_range['rec_start'] + timedelta(seconds=item['idx'] / state.fps)).strftime("%Y-%m-%d %H:%M:%S")
                state.hit_frames.append({
                    'idx': item['idx'], 
                    'frame': item['frame'], 
                    'time': wall_time,
                    'type': 'key_event',
                    'ocr_text': raw_ocr_text,
                    'ocr_score': score,
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
                "status": "no_hits_found",
                "vlm_optimization": {
                    "candidate_frames": len(state.candidate_frames),
                    "ocr_hit_frames": 0,
                    "vlm_sent_frames": 0,
                    "reason": "no_ocr_hits",
                }
            }
            return state

        logger.info(f"Step 4: 正在发送关键帧至 VLM 模型进行深层行为意图分析...")
        # --- 新增：创建保存图片的目录 ---
        save_dir = "vlm_debug_frames"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # ---------------------------
        state.hit_frames.sort(key=lambda x: x['idx'])
        vlm_frames, optimization_meta = self._select_vlm_frames(state.hit_frames)
        optimization_meta.update({
            "candidate_frames": len(state.candidate_frames),
            "ocr_hit_frames": len([item for item in state.hit_frames if item.get("type") == "key_event"]),
            "image_max_edge": self._get_int_env("DLD_VLM_IMAGE_MAX_EDGE", 1280),
            "jpeg_quality": self._get_int_env("DLD_VLM_JPEG_QUALITY", 65, minimum=30),
        })
        logger.info(
            "Step 4 token optimization: "
            f"OCR hits={optimization_meta['ocr_hit_frames']}, "
            f"VLM frames={optimization_meta['vlm_sent_frames']}/"
            f"{optimization_meta['candidate_hit_frames']}"
        )
        llm = ChatOpenAI(
            model=self.llm_model,
            base_url=os.getenv("OPENAI_BASE_URL"),#"https://www.DMXapi.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),#os.getenv("DMX_API_KEY"), 
        )

        table_rows = [
            (
                f"{i+1:^8} | {f['idx']:^10} | {f['time']:^20} | "
                f"{f['type']:^10} | {self._ocr_snippet(f.get('ocr_text', ''))}"
            )
            for i, f in enumerate(vlm_frames)
        ]
        table_str = "输入顺序 | 原始帧索引 | 现实时间戳(%Y-%m-%d %H:%M:%S) | 类型 | OCR摘要\n" + "-"*95 + "\n" + "\n".join(table_rows)

        final_prompt = PROMPTS.RETRIEVE_FRAMES_PROMPT.format(
            frame_count=len(vlm_frames),
            frame_info_table=table_str
        )

        contents = [{"type": "text", "text": final_prompt}]
        jpeg_quality = optimization_meta["jpeg_quality"]
        for f in vlm_frames:
            safe_time = f['time'].replace(":", "-").replace(" ", "_")
            file_name = f"frame_{f['idx']}_{safe_time}_{f['type']}.jpg"
            save_path = os.path.join(save_dir, file_name)
            vlm_frame = self._resize_for_vlm(f['frame'])
            cv2.imwrite(save_path, vlm_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            logger.debug(f"已保存 VLM 输入帧至: {save_path}")
            _, buffer = cv2.imencode('.jpg', vlm_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
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

            if is_long_text_mode:
                
                logger.info("Long text mode: using all VLM behavior events without filename filtering")
                final_events = raw_events
            else:
                logger.info(f"🔍 普通模式：正在匹配文件名关键词: {state.target_keywords}")
                final_events = []
                for event in raw_events:
                    original_name = event.get("original_filename", "").lower()
                    final_events.append(event)
                    if any(kw.lower() in original_name for kw in state.target_keywords):
                        final_events.append(event)
                    else:
                        logger.info(f"🗑️ 剔除无关文件名: {original_name}")

            # 2. 生成最终报告
            state.final_report = {
                "search_range": {
                    "start": state.time_range['search_start'].strftime("%Y-%m-%d %H:%M:%S"), 
                    "end": state.time_range['search_end'].strftime("%Y-%m-%d %H:%M:%S")
                },
                "total_events": len(final_events),
                "events": final_events,
                "status": "success",
                "vlm_optimization": optimization_meta,
            }
        except Exception as e:
            state.final_report = {
                "error": str(e),
                "status": "failed",
                "vlm_optimization": optimization_meta,
            }

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
