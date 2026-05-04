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
import unicodedata
from datetime import timedelta
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from schema import AgentState
from thefuzz import fuzz
from prompt_loader import PROMPTS

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover - optional runtime dependency
    repair_json = None

load_dotenv(find_dotenv())
logger = logging.getLogger("VideoAgent")


def _normalize_match_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text)


def _compact_match_text(value) -> str:
    text = _normalize_match_text(value)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _keyword_variants(keywords: List[str]) -> List[str]:
    seen = set()
    variants = []
    for keyword in keywords or []:
        text = _normalize_match_text(keyword)
        if not text:
            continue
        basename = text.replace("\\", "/").rsplit("/", 1)[-1]
        stem = os.path.splitext(basename)[0]
        candidates = [text, basename, stem]
        candidates.extend(part for part in re.split(r"[\s._\-()（）【】\[\]{}]+", stem) if len(part) >= 2)
        for candidate in candidates:
            compact = _compact_match_text(candidate)
            if compact and compact not in seen:
                seen.add(compact)
                variants.append(compact)
    return variants


def _sample_evenly(items: List[Dict], limit: int) -> List[Dict]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indexes = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in indexes]


def _dedupe_events(events: List[Dict]) -> List[Dict]:
    deduped = []
    seen = set()
    for event in events:
        key = (
            event.get("time_range", ""),
            event.get("app_name", ""),
            event.get("behavior_category", ""),
            event.get("operation_type", ""),
            event.get("original_filename", ""),
            event.get("modified_filename", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _parse_vlm_json(content: str):
    clean_text = re.sub(r"```json\n?|```", "", str(content or "")).strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        if repair_json is None:
            raise
        match = re.search(r"\{.*\}", clean_text, re.S)
        candidate = match.group(0) if match else clean_text
        return json.loads(repair_json(candidate))

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

        if not state.fps or state.fps <= 0:
            cap.release()
            state.final_report["_diagnostics"] = {"vision_error": "invalid_fps"}
            return state

        start_idx = max(0, int((state.time_range['search_start'] - state.time_range['rec_start']).total_seconds() * state.fps))
        end_idx = min(int((state.time_range['search_end'] - state.time_range['rec_start']).total_seconds() * state.fps), state.total_frames)

        prev_feat = None
        sample_fps = float(os.getenv("FRAME_ANALYZER_SAMPLE_FPS", "2.0"))
        step = max(1, int(round(state.fps / max(sample_fps, 0.1))))
        similarity_threshold = float(os.getenv("FRAME_ANALYZER_VISUAL_SIM_THRESHOLD", "0.992"))

        for curr_idx in range(start_idx, end_idx, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
            ret, frame = cap.read()
            if not ret: break

            img_t = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                curr_feat = self.feature_model(img_t).flatten()

            if prev_feat is None or torch.nn.functional.cosine_similarity(prev_feat.unsqueeze(0), curr_feat.unsqueeze(0)).item() < similarity_threshold:
                state.candidate_frames.append({'idx': curr_idx, 'frame': frame})
                prev_feat = curr_feat
        
        cap.release()
        state.final_report["_diagnostics"] = {
            "candidate_frames": len(state.candidate_frames),
            "sample_fps": sample_fps,
            "visual_similarity_threshold": similarity_threshold,
        }
        return state

    def keyword_filter_node(self, state: AgentState) -> AgentState:
        # 判断是否需要执行“长文本模糊匹配”模式
        # 逻辑：如果 keywords 列表中有任何一个元素长度 > 10，则视为粘贴的长文本
        is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)
        
        if is_long_text_mode:
            # 将所有关键词拼接成一个目标长文本进行比对
            target_text = _compact_match_text("".join(state.target_keywords))
            logger.info(f"Step 2: 检测到长文本，正在执行模糊相似度匹配 (目标: {target_text[:20]}...)")
        else:
            logger.info(f"Step 2: 正在执行 OCR 文本识别并匹配关键词: {state.target_keywords}...")

        last_hit_idx = -1
        keyword_variants = _keyword_variants(state.target_keywords)
        fuzzy_threshold = int(os.getenv("FRAME_ANALYZER_OCR_FUZZY_THRESHOLD", "62"))
        ocr_scale = float(os.getenv("FRAME_ANALYZER_OCR_SCALE", "1.6"))
        debug_ocr = os.getenv("FRAME_ANALYZER_DEBUG_OCR", "").strip().lower() in {"1", "true", "yes", "on"}
        for item in state.candidate_frames:
            frame_for_ocr = item['frame']
            if ocr_scale > 1:
                h, w = frame_for_ocr.shape[:2]
                frame_for_ocr = cv2.resize(
                    frame_for_ocr,
                    (max(1, int(w * ocr_scale)), max(1, int(h * ocr_scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            results = self.reader.readtext(frame_for_ocr, detail=0)
            text_blob = _compact_match_text("".join(results))
            
            is_match = False
            if is_long_text_mode:
                score = fuzz.partial_ratio(target_text, text_blob)
                if score >= 65:
                    is_match = True
                    logger.debug(f"帧 {item['idx']} 模糊匹配成功，得分: {score}")
            else:
                direct_hit = any(kw in text_blob for kw in keyword_variants)
                fuzzy_hit = any(
                    len(kw) >= 4 and fuzz.partial_ratio(kw, text_blob) >= fuzzy_threshold
                    for kw in keyword_variants
                )
                if direct_hit or fuzzy_hit:
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
            elif debug_ocr:
                logger.debug(f"OCR miss frame={item['idx']} text={text_blob[:300]}")

        state.final_report['_last_hit_idx'] = last_hit_idx
        diagnostics = state.final_report.setdefault("_diagnostics", {})
        diagnostics["ocr_hit_frames"] = len([f for f in state.hit_frames if f.get("type") == "key_event"])
        diagnostics["keyword_variants"] = keyword_variants[:50]
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
            allow_fallback = os.getenv("FRAME_ANALYZER_ALLOW_VLM_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
            fallback_limit = int(os.getenv("FRAME_ANALYZER_FALLBACK_VLM_IMAGES", os.getenv("FRAME_ANALYZER_MAX_VLM_IMAGES", "4")))
            if allow_fallback and state.candidate_frames:
                for item in _sample_evenly(state.candidate_frames, fallback_limit):
                    wall_time = (state.time_range['rec_start'] + timedelta(seconds=item['idx'] / state.fps)).strftime("%Y-%m-%d %H:%M:%S")
                    state.hit_frames.append({
                        'idx': item['idx'],
                        'frame': item['frame'],
                        'time': wall_time,
                        'type': 'fallback_context'
                    })
                diagnostics = state.final_report.setdefault("_diagnostics", {})
                diagnostics["vlm_fallback_used"] = True
                diagnostics["vlm_fallback_frames"] = len(state.hit_frames)
            else:
                state.final_report = {
                    "search_range": {
                        "start": state.time_range['search_start'].strftime("%Y-%m-%d %H:%M:%S"),
                        "end": state.time_range['search_end'].strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "total_events": 0,
                    "events": [],
                    "status": "no_hits_found",
                    "diagnostics": state.final_report.get("_diagnostics", {})
                }
                return state

        logger.info(f"Step 4: 正在发送关键帧至 VLM 模型进行深层行为意图分析...")
        # --- 新增：创建保存图片的目录 ---
        save_dir = "vlm_debug_frames"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # ---------------------------
        state.hit_frames.sort(key=lambda x: x['idx'])
        max_vlm_images = int(os.getenv("FRAME_ANALYZER_MAX_VLM_IMAGES", "8"))
        key_frames = [f for f in state.hit_frames if f.get('type') == 'key_event']
        context_frames = [f for f in state.hit_frames if f.get('type') != 'key_event']
        if max_vlm_images > 0 and len(state.hit_frames) > max_vlm_images:
            if len(key_frames) >= max_vlm_images:
                state.hit_frames = _sample_evenly(key_frames, max_vlm_images)
            else:
                state.hit_frames = key_frames + _sample_evenly(context_frames, max_vlm_images - len(key_frames))
            state.hit_frames.sort(key=lambda x: x['idx'])

        llm = ChatOpenAI(
            model=self.llm_model,
            base_url=os.getenv("OPENAI_BASE_URL"),#"https://www.DMXapi.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),#os.getenv("DMX_API_KEY"), 
        )

        table_rows = [
            f"{i+1:^8} | {f['idx']:^10} | {f['time']:^20} | {f['type']:^10}" 
            for i, f in enumerate(state.hit_frames)
        ]
        table_str = "输入顺序 | 原始帧索引 | 现实时间戳 (%Y-%m-%d %H:%M:%S) | 类型\n" + "-"*75 + "\n" + "\n".join(table_rows)

        final_prompt = PROMPTS.RETRIEVE_FRAMES_PROMPT.format(
            frame_count=len(state.hit_frames),
            frame_info_table=table_str
        )

        contents = [{"type": "text", "text": final_prompt}]
        max_side = int(os.getenv("FRAME_ANALYZER_VLM_MAX_SIDE", "640"))
        for f in state.hit_frames:
            safe_time = f['time'].replace(":", "-").replace(" ", "_")
            file_name = f"frame_{f['idx']}_{safe_time}_{f['type']}.jpg"
            save_path = os.path.join(save_dir, file_name)
            frame_for_vlm = f['frame']
            h, w = frame_for_vlm.shape[:2]
            if max_side > 0 and max(h, w) > max_side:
                scale = max_side / max(h, w)
                frame_for_vlm = cv2.resize(
                    frame_for_vlm,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(save_path, frame_for_vlm)
            logger.debug(f"已保存 VLM 输入帧至: {save_path}")
            _, buffer = cv2.imencode('.jpg', frame_for_vlm, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        try:
            resp = llm.invoke([HumanMessage(content=contents)])
            batch_data = _parse_vlm_json(resp.content)
            raw_events = batch_data["events"] if isinstance(batch_data, dict) and "events" in batch_data else batch_data
            

            is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)

            if is_long_text_mode:
                
                logger.info("📝 长文本模式：已获取 VLM 识别的所有行为序列，跳过文件名过滤")
                final_events = raw_events
            else:
                logger.info(f"🔍 普通模式：正在匹配文件名关键词: {state.target_keywords}")
                final_events = []
                keyword_variants = _keyword_variants(state.target_keywords)
                for event in raw_events:
                    original_name = _compact_match_text(event.get("original_filename", ""))
                    final_events.append(event)
                    if not any(kw in original_name for kw in keyword_variants):
                        logger.info(f"🗑️ 剔除无关文件名: {original_name}")

            final_events = _dedupe_events(final_events)
            diagnostics = state.final_report.get("_diagnostics", {})
            diagnostics["vlm_input_frames"] = len(state.hit_frames)
            diagnostics["vlm_max_side"] = max_side

            # 2. 生成最终报告
            state.final_report = {
                "search_range": {
                    "start": state.time_range['search_start'].strftime("%Y-%m-%d %H:%M:%S"), 
                    "end": state.time_range['search_end'].strftime("%Y-%m-%d %H:%M:%S")
                },
                "total_events": len(final_events),
                "events": final_events,
                "status": "success",
                "diagnostics": diagnostics
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
