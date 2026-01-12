from core_analyzer import BaseVideoAnalyzer, logger
import cv2
import os
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import prompts
import torch

class ContentRetriever(BaseVideoAnalyzer):
    def analyze(self, rec_start_str, s_start_str, s_end_str, video_path, 
                target_keywords=None, target_text=None, similarity_threshold=0.98):
        
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"./output_{video_name}/"
        os.makedirs(output_dir, exist_ok=True)
        
        t_fmt = "%Y-%m-%d %H:%M:%S"
        t_rec = datetime.strptime(rec_start_str, t_fmt)
        t_s_start = datetime.strptime(s_start_str, t_fmt)
        t_s_end = datetime.strptime(s_end_str, t_fmt)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_idx = int((t_s_start - t_rec).total_seconds() * fps)
        end_idx = int((t_s_end - t_rec).total_seconds() * fps)

        
        candidate_indices = []
        prev_feat = None
        for curr in range(start_idx, end_idx, int(fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr)
            ret, frame = cap.read()
            if not ret: break
            feat = self.get_frame_feature(frame)
            if prev_feat is None or torch.nn.functional.cosine_similarity(prev_feat.unsqueeze(0), feat.unsqueeze(0)).item() < similarity_threshold:
                candidate_indices.append(curr)
                prev_feat = feat

        
        filtered_frames = []
        last_hit = -1
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            ocr_res = self.reader.readtext(frame, detail=0)
            combined = "".join(ocr_res).replace(" ", "").lower()
            
            is_match = False
            if target_keywords:
                is_match = any(kw.lower() in combined for kw in target_keywords)
            elif target_text:
                score = fuzz.partial_ratio(target_text.replace(" ", "").lower(), combined)
                is_match = score >= 65

            if is_match:
                last_hit = idx
                wall_time = (t_rec + timedelta(seconds=idx/fps)).strftime(t_fmt)
                filtered_frames.append({'frame': frame, 'frame_index': idx, 'wall_clock_time': wall_time})

        
        if last_hit != -1 and not target_text:
            for sec in [3, 8]:
                ext_idx = last_hit + int(sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, ext_idx)
                ret, frame = cap.read()
                if ret:
                    filtered_frames.append({
                        'frame': frame, 
                        'frame_index': ext_idx, 
                        'wall_clock_time': (t_rec + timedelta(seconds=ext_idx/fps)).strftime(t_fmt)
                    })

        cap.release()
        
        
        if not filtered_frames: 
            return {"events": []}
        
        
        if target_text:
            prompt = prompts.COPY_PASTE_ANALYSIS_PROMPT.format(
                target_text=target_text, 
                frame_count=len(filtered_frames), 
                frame_info_table=""
            )
        else:
            prompt = prompts.RETRIEVE_FRAMES_PROMPT.format(
                target_keywords=",".join(target_keywords), 
                frame_count=len(filtered_frames), 
                frame_info_table=""
            )
        
        
        llm_response = self.call_llm(prompt, filtered_frames)
        all_events = llm_response.get("events", [])

        
        result_data = {
            "search_range": {"start": s_start_str, "end": s_end_str},
            "analysis_type": "long_text" if target_text else "keywords",
            "target_query": target_text if target_text else target_keywords,
            "total_events": len(all_events),
            "events": all_events
        }

        
        self.save_report(output_dir, "behavior_analysis_report.json", result_data)
        
        return result_data