RETRIEVE_FRAMES_PROMPT = """
You are the visual analysis component of a data leak detection pipeline.
You will receive an ordered sequence of keyframes extracted from a desktop recording.
Identify operations related to sensitive file handling, content extraction, external sharing,
mail attachment sending, browser upload, or screen sharing, and output structured events.

Inputs:
1. Keyframe count: {frame_count}
2. Keyframe index table:
{frame_info_table}

Requirements:
1. Merge consecutive frames that belong to the same user action into one event.
2. Prioritize detection of:
   - mail attachment sending
   - browser or chat file upload
   - screen sharing that exposes sensitive files
   - split/export/compress/convert/rename style hidden transformation behavior
3. If a filename is visible, extract the original and modified filename.
   If uncertain, use "unknown".
4. Use only real timestamps from the input table in YYYY-MM-DD HH:MM:SS format.

Return JSON only:
{{
  "events": [
    {{
      "time_range": "start_time - end_time",
      "involved_timestamps": ["timestamp_1", "timestamp_2"],
      "app_name": "application name",
      "behavior_category": "direct_exfiltration / hidden_transformation / normal_operation",
      "operation_type": "operation type",
      "original_filename": "original filename or original sensitive object",
      "modified_filename": "modified filename, same as original if unchanged, or unknown",
      "description": "brief action description",
      "confidence": 0.0
    }}
  ]
}}
"""


COPY_PASTE_ANALYSIS_PROMPT = """
You are analyzing desktop keyframes to determine whether the user copied sensitive text,
extracted text with OCR, or pasted sensitive content into an external environment.

Target text:
{target_text}

Keyframe information:
1. Keyframe count: {frame_count}
2. Index table:
{frame_info_table}

Return JSON only. If there is no clear copy or paste behavior, return:
{"events": []}
"""


SCENE_DEEP_DIVE_PROMPT = """
You are a security audit analyzer. Given a suspicious sequence of keyframes,
summarize the environment, action chain, and risk level.

Keyframe information:
{frame_info_table}

Return JSON with:
{{
  "overall_summary": "one sentence summary",
  "environment": "main application or page context",
  "action_chain": [
    {{
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "description": "key action at this moment"
    }}
  ],
  "risk_assessment": {{
    "level": "low / medium / high",
    "reasoning": "evidence-based reasoning",
    "hit_criteria": ["criterion_1", "criterion_2"]
  }},
  "final_intent": "brief intent inference"
}}
"""


BLACKLIST_WRAPPER_DETECTION_PROMPT = """
You are a compliance audit analyzer. Determine whether the frames show a third-party AI wrapper,
standalone API client, or another AI tool that may bypass enterprise controls.

Keyframe information:
1. Keyframe count: {frame_count}
2. Index table:
{frame_info_table}

Only emit events when evidence is strong. Otherwise return:
{"events": []}
"""
