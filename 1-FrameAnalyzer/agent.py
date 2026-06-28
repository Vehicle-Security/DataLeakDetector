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

    @staticmethod
    def _is_long_text_mode(keywords: List[str]) -> bool:
        return any(len(str(kw)) > 10 for kw in keywords)

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    @staticmethod
    def _event_text(event: Dict[str, Any]) -> str:
        fields = [
            "app_name",
            "behavior_category",
            "operation_type",
            "original_filename",
            "modified_filename",
            "description",
            "visual_evidence",
            "detected_content",
        ]
        return " ".join(str(event.get(field, "")) for field in fields)

    def _parse_vlm_response_content(self, content: str) -> Any:
        cleaned = re.sub(r"```(?:json)?|```", "", str(content or ""), flags=re.IGNORECASE).strip()
        decoder = json.JSONDecoder()
        for pos, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                data, _ = decoder.raw_decode(cleaned[pos:])
                return data
            except json.JSONDecodeError:
                continue
        raise ValueError("VLM response did not contain valid JSON")

    @staticmethod
    def _coerce_event_list(batch_data: Any) -> List[Dict[str, Any]]:
        if isinstance(batch_data, dict):
            raw_events = batch_data.get("events", [])
        else:
            raw_events = batch_data
        if not isinstance(raw_events, list):
            return []
        return [event for event in raw_events if isinstance(event, dict)]

    def _event_matches_keywords(self, event: Dict[str, Any], keywords: List[str], is_long_text_mode: bool) -> bool:
        compact_event = self._compact_text(self._event_text(event))
        compact_keywords = [self._compact_text(keyword) for keyword in keywords if str(keyword).strip()]
        if not compact_keywords:
            return True

        if is_long_text_mode:
            target_text = "".join(compact_keywords)
            if fuzz.partial_ratio(target_text, compact_event) >= 55:
                return True
            risk_tokens = [
                "paste",
                "copy",
                "clipboard",
                "send",
                "upload",
                "\u7c98\u8d34",
                "\u590d\u5236",
                "\u526a\u8d34\u677f",
                "\u53d1\u9001",
                "\u4e0a\u4f20",
                "\u5916\u53d1",
            ]
            return any(token in compact_event for token in risk_tokens)

        return any(keyword in compact_event for keyword in compact_keywords)

    @staticmethod
    def _event_dedup_key(event: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        return (
            str(event.get("time_range", "")).strip(),
            str(event.get("app_name", "")).strip().lower(),
            str(event.get("operation_type", "")).strip().lower(),
            str(event.get("original_filename", "")).strip().lower(),
            str(event.get("modified_filename", "")).strip().lower(),
        )

    def _filter_vlm_events(self, raw_events: List[Dict[str, Any]], target_keywords: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        is_long_text_mode = self._is_long_text_mode(target_keywords)
        filtered = []
        seen = set()
        dropped = 0

        for event in raw_events:
            if not self._event_matches_keywords(event, target_keywords, is_long_text_mode):
                dropped += 1
                continue
            key = self._event_dedup_key(event)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            filtered.append(event)

        meta = {
            "vlm_raw_events": len(raw_events),
            "vlm_kept_events": len(filtered),
            "vlm_dropped_events": dropped,
            "vlm_filter_mode": "long_text" if is_long_text_mode else "keyword",
        }
        return filtered, meta

    @staticmethod
    def _qwen_guardrail_prompt() -> str:
        return """

### Robustness guardrails
- Return one JSON object only: {"events": [...]}. Do not add markdown or explanations.
- If the screen only shows an app/webpage being opened, and no target file/text is pasted, uploaded, attached, sent, copied, renamed, compressed, converted, screenshotted, or recorded, return {"events": []}.
- Do not create duplicate events for the same action. Merge consecutive frames into one event.
- Prefer visible evidence from the title bar, upload dialog, file picker, message composer, attachment chip, clipboard menu, or OCR table.
- If a filename is uncertain, write "未知"; do not invent filenames.
- For "直接外发", require visible target data/object leaving the local trusted context. A chat window alone is not enough.
"""

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
        logger.info("Step 1: 姝ｅ湪閫氳繃瑙嗚鐗瑰緛鎻愬彇杩涜鍏抽敭甯у垵姝ョ瓫閫?..")
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
        # 鍒ゆ柇鏄惁闇€瑕佹墽琛屸€滈暱鏂囨湰妯＄硦鍖归厤鈥濇ā寮?        # 閫昏緫锛氬鏋?keywords 鍒楄〃涓湁浠讳綍涓€涓厓绱犻暱搴?> 10锛屽垯瑙嗕负绮樿创鐨勯暱鏂囨湰
        is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)
        
        if is_long_text_mode:
            # 灏嗘墍鏈夊叧閿瘝鎷兼帴鎴愪竴涓洰鏍囬暱鏂囨湰杩涜姣斿
            target_text = "".join(state.target_keywords).replace(" ", "").lower()
            logger.info(f"Step 2: 妫€娴嬪埌闀挎枃鏈紝姝ｅ湪鎵ц妯＄硦鐩镐技搴﹀尮閰?(鐩爣: {target_text[:20]}...)")
        else:
            logger.info(f"Step 2: 姝ｅ湪鎵ц OCR 鏂囨湰璇嗗埆骞跺尮閰嶅叧閿瘝: {state.target_keywords}...")

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
                    logger.debug(f"甯?{item['idx']} 妯＄硦鍖归厤鎴愬姛锛屽緱鍒? {score}")
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
        logger.info("Step 3: 姝ｅ湪閽堝鍛戒腑甯ф彁鍙栧悗缁笂涓嬫枃鏃堕棿鑺傜偣鐨勮ˉ鍏呯敾闈?..")
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

        logger.info(f"Step 4: 姝ｅ湪鍙戦€佸叧閿抚鑷?VLM 妯″瀷杩涜娣卞眰琛屼负鎰忓浘鍒嗘瀽...")
        # --- 鏂板锛氬垱寤轰繚瀛樺浘鐗囩殑鐩綍 ---
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
        table_str = "杈撳叆椤哄簭 | 鍘熷甯х储寮?| 鐜板疄鏃堕棿鎴?%Y-%m-%d %H:%M:%S) | 绫诲瀷 | OCR鎽樿\n" + "-"*95 + "\n" + "\n".join(table_rows)

        final_prompt = PROMPTS.RETRIEVE_FRAMES_PROMPT.format(
            frame_count=len(vlm_frames),
            frame_info_table=table_str
        ) + self._qwen_guardrail_prompt()

        contents = [{"type": "text", "text": final_prompt}]
        jpeg_quality = optimization_meta["jpeg_quality"]
        for f in vlm_frames:
            safe_time = f['time'].replace(":", "-").replace(" ", "_")
            file_name = f"frame_{f['idx']}_{safe_time}_{f['type']}.jpg"
            save_path = os.path.join(save_dir, file_name)
            vlm_frame = self._resize_for_vlm(f['frame'])
            cv2.imwrite(save_path, vlm_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            logger.debug(f"宸蹭繚瀛?VLM 杈撳叆甯ц嚦: {save_path}")
            _, buffer = cv2.imencode('.jpg', vlm_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        try:
            resp = llm.invoke([HumanMessage(content=contents)])
            batch_data = self._parse_vlm_response_content(resp.content)
            raw_events = self._coerce_event_list(batch_data)
            final_events, filter_meta = self._filter_vlm_events(raw_events, state.target_keywords)
            optimization_meta.update(filter_meta)

            state.final_report = {
                "search_range": {
                    "start": state.time_range['search_start'].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": state.time_range['search_end'].strftime("%Y-%m-%d %H:%M:%S"),
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
