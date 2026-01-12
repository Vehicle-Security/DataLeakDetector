from core_analyzer import BaseVideoAnalyzer, logger
import cv2
import os
import numpy as np
from datetime import datetime, timedelta
import prompts

class BlacklistAnalyzer(BaseVideoAnalyzer):
    def analyze_blacklist(self, rec_start, s_start, s_end, video_path, batch_size=6, max_samples=18):
        """
        专门用于检测疑似黑名单/套壳AI应用的逻辑，并保存 JSON 报告
        """
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"./blacklist_output_{video_name}/"
        
        t_fmt = "%Y-%m-%d %H:%M:%S"
        t_rec = datetime.strptime(rec_start, t_fmt)
        t_s_start = datetime.strptime(s_start, t_fmt)
        t_s_end = datetime.strptime(s_end, t_fmt)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_idx = int((t_s_start - t_rec).total_seconds() * fps)
        end_idx = int((t_s_end - t_rec).total_seconds() * fps)

        
        indices = range(start_idx, end_idx, int(fps))
        if len(indices) > max_samples:
            sampled_indices = [indices[i] for i in np.linspace(0, len(indices)-1, max_samples, dtype=int)]
        else:
            sampled_indices = indices

        frames_to_analyze = []
        for idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                wall_time = (t_rec + timedelta(seconds=idx/fps)).strftime(t_fmt)
                frames_to_analyze.append({
                    'frame': frame, 
                    'idx': idx, 
                    'time': wall_time
                })
        cap.release()

        
        all_detected_events = []
        for i in range(0, len(frames_to_analyze), batch_size):
            batch = frames_to_analyze[i : i + batch_size]
            logger.info(f"🚀 正在分析黑名单批次 {i//batch_size + 1}...")
            
            table_rows = [f"{f['idx']} | {f['time']}" for f in batch]
            table_str = "Frame_Index | Timestamp\n" + "-"*30 + "\n" + "\n".join(table_rows)

            prompt = prompts.BLACKLIST_WRAPPER_DETECTION_PROMPT.format(
                frame_count=len(batch), 
                frame_info_table=table_str
            )

            res = self.call_llm(prompt, batch)
            events = res.get("events", [])
            if events:
                all_detected_events.extend(events)

        
        result_data = {
            "search_range": {"start": s_start, "end": s_end}, 
            "total_violations": len(all_detected_events), 
            "events": all_detected_events
        }
        
        
        self.save_report(output_dir, "blacklist_report.json", result_data)
        
        return result_data