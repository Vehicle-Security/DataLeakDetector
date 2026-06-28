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
    def _keyword_concept_groups() -> List[Tuple[str, ...]]:
        return [
            ("\u85aa\u8d44", "\u5de5\u8d44", "\u85aa\u916c", "\u6708\u85aa", "salary", "payroll"),
            ("\u5ba2\u6237", "\u5ba2\u6237\u540d\u5355", "customer", "client"),
            ("\u9884\u7b97", "\u6210\u672c", "\u8d22\u52a1", "\u8463\u4e8b\u4f1a", "budget", "finance", "cost"),
            ("\u8d26\u53f7", "\u8d26\u6237", "\u94f6\u884c", "\u6536\u6b3e", "account", "bank"),
            ("\u5408\u540c", "\u534f\u8bae", "contract", "agreement"),
            ("\u6218\u7565", "\u89c4\u5212", "\u8def\u7ebf\u56fe", "strategy", "roadmap"),
            ("\u5e76\u8d2d", "\u8c08\u5224", "\u4f1a\u8bae\u7eaa\u8981", "merger", "m&a", "minutes"),
        ]

    @classmethod
    def _shared_sensitive_concept(cls, keyword_text: str, event_text: str) -> bool:
        compact_keyword = cls._compact_text(keyword_text)
        compact_event = cls._compact_text(event_text)
        for group in cls._keyword_concept_groups():
            keyword_hit = any(cls._compact_text(token) in compact_keyword for token in group)
            event_hit = any(cls._compact_text(token) in compact_event for token in group)
            if keyword_hit and event_hit:
                return True
        return False

    @staticmethod
    def _risk_tokens() -> List[str]:
        return [
            "paste",
            "copy",
            "clipboard",
            "send",
            "upload",
            "attach",
            "share",
            "screenshot",
            "screenrecord",
            "record",
            "qr",
            "download",
            "export",
            "\u7c98\u8d34",
            "\u590d\u5236",
            "\u526a\u8d34\u677f",
            "\u53d1\u9001",
            "\u4e0a\u4f20",
            "\u9644\u4ef6",
            "\u5206\u4eab",
            "\u5916\u53d1",
            "\u622a\u56fe",
            "\u5f55\u5c4f",
            "\u5c4f\u5e55\u5171\u4eab",
            "\u5171\u4eab\u5c4f\u5e55",
            "\u4e8c\u7ef4\u7801",
            "\u751f\u6210",
            "\u4e0b\u8f7d",
            "\u5bfc\u51fa",
            "\u8f6c\u53d1",
            "\u7f16\u7801",
        ]

    @classmethod
    def _event_has_risk_signal(cls, event: Dict[str, Any]) -> bool:
        compact = cls._compact_text(cls._event_text(event))
        return any(cls._compact_text(token) in compact for token in cls._risk_tokens())

    @staticmethod
    def _event_text(event: Dict[str, Any]) -> str:
        fields = [
            "app_name",
            "app",
            "application",
            "behavior_category",
            "operation_type",
            "operation",
            "action",
            "event_type",
            "original_filename",
            "original_file",
            "source_filename",
            "source_file",
            "file_name",
            "filename",
            "modified_filename",
            "target_filename",
            "target_file",
            "description",
            "summary",
            "evidence",
            "visual_evidence",
            "detected_content",
            "content",
            "ocr_text",
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
            return self._event_has_risk_signal(event)

        if any(keyword in compact_event for keyword in compact_keywords):
            return True

        keyword_text = " ".join(str(keyword) for keyword in keywords)
        if self._event_has_risk_signal(event) and self._shared_sensitive_concept(keyword_text, self._event_text(event)):
            return True

        filenameish_event = re.sub(r"[._\\/\-()\[\]\s]+", "", compact_event)
        for keyword in compact_keywords:
            filenameish_keyword = re.sub(r"[._\\/\-()\[\]\s]+", "", keyword)
            if filenameish_keyword and fuzz.partial_ratio(filenameish_keyword, filenameish_event) >= 82:
                return True
        return False

    @staticmethod
    def _event_dedup_key(event: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        return (
            str(event.get("time_range", "")).strip(),
            str(event.get("app_name", "")).strip().lower(),
            str(event.get("operation_type", "")).strip().lower(),
            str(event.get("original_filename", "")).strip().lower(),
            str(event.get("modified_filename", "")).strip().lower(),
        )

    @staticmethod
    def _normalize_vlm_event(event: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(event)
        normalized.setdefault("time_range", "")
        normalized.setdefault("involved_timestamps", [])
        normalized.setdefault("app_name", "\u672a\u77e5")
        normalized.setdefault("app_type", "\u672a\u77e5")
        normalized.setdefault("behavior_category", "\u672a\u77e5")
        normalized.setdefault(
            "operation_type",
            event.get("operation") or event.get("action") or event.get("event_type") or "\u672a\u77e5",
        )
        normalized.setdefault(
            "original_filename",
            event.get("original_file")
            or event.get("source_filename")
            or event.get("source_file")
            or event.get("file_name")
            or event.get("filename")
            or "\u672a\u77e5",
        )
        normalized.setdefault(
            "modified_filename",
            event.get("target_filename")
            or event.get("target_file")
            or event.get("output_file")
            or "\u672a\u77e5",
        )
        normalized.setdefault("description", "")
        return normalized

    def _is_low_value_normal_event(self, event: Dict[str, Any]) -> bool:
        compact = self._compact_text(self._event_text(event))
        label_text = self._compact_text(
            f"{event.get('behavior_category', '')} {event.get('operation_type', '')}"
        )
        risk_tokens = self._risk_tokens() + [
            "rename",
            "compress",
            "convert",
            "\u91cd\u547d\u540d",
            "\u538b\u7f29",
            "\u8f6c\u6362",
            "\u9690\u85cf",
        ]
        normal_tokens = [
            "open",
            "read",
            "view",
            "scroll",
            "\u6253\u5f00",
            "\u9605\u8bfb",
            "\u6d4f\u89c8",
            "\u6eda\u52a8",
            "\u6b63\u5e38\u64cd\u4f5c",
        ]
        if any(token in label_text for token in normal_tokens) and not any(
            token in label_text for token in risk_tokens
        ):
            return True
        if any(token in compact for token in risk_tokens):
            return False
        return any(token in compact for token in normal_tokens)

    def _filter_vlm_events(self, raw_events: List[Dict[str, Any]], target_keywords: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        is_long_text_mode = self._is_long_text_mode(target_keywords)
        filtered = []
        seen = set()
        dropped = 0

        for event in raw_events:
            event = self._normalize_vlm_event(event)
            if self._is_low_value_normal_event(event):
                dropped += 1
                continue
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
        # 判断是否需要执行“长文本模糊匹配”模式。
        # 如果 keywords 中有任意元素长度 > 10，则视为粘贴的长文本。
        is_long_text_mode = any(len(kw) > 10 for kw in state.target_keywords)
        
        if is_long_text_mode:
            # 将所有关键词拼接成一个目标长文本进行比对。
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

        logger.info("Step 4: 正在发送关键帧至 VLM 模型进行深层行为意图分析...")
        # 创建保存图片的目录。
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
        ) + self._qwen_guardrail_prompt()

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
