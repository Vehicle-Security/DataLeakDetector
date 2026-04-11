from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

@dataclass
class AgentState:
    video_path: str
    target_keywords: List[str]
    time_range: Dict[str, datetime]
    fps: float = 0.0
    total_frames: int = 0
    candidate_frames: List[Dict] = field(default_factory=list)
    hit_frames: List[Dict] = field(default_factory=list)
    final_report: Dict = field(default_factory=dict)