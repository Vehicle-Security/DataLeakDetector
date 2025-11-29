import json
import base64
import re
import time
from tqdm import tqdm
from frame_processor import compress_frame
from vlm_inference import api_inference_video

def batch_analyze_frames_with_vlm(frames, model_name="qwen2-vl-72b-instruct"):
    """每5帧一组一次性输入大模型进行敏感操作识别和ROI检测"""
    all_frame_details = []
    
    print(f"🖼️  开始每5帧一组分析，总帧数: {len(frames)}")
    
    for group_idx in range(0, len(frames), 5):
        group_frames = frames[group_idx:group_idx+5]
        print(f"📦 处理第 {group_idx//5 + 1} 组，包含 {len(group_frames)} 帧")
        
        frame_details_table_data = []
        frame_base64_list = []
        frame_info_list = []
        
        for i, frame_info in enumerate(group_frames):
            buffer, original_shape = compress_frame(frame_info['frame'])
            frame_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            frame_base64_list.append(frame_base64)
            frame_info_list.append(frame_info)
            
            frame_details_table_data.append({
                'input_order': group_idx + i + 1,
                'original_frame_index': frame_info['frame_index'],
                'timestamp': frame_info['timestamp'],
                'description': f"第{group_idx + i + 1}张图片"
            })
        
        frame_info_table = "帧详细信息表:\n"
        frame_info_table += "输入顺序 | 原始帧索引 | 时间戳(秒) | 描述\n"
        frame_info_table += "-" * 50 + "\n"
        for detail in frame_details_table_data:
            frame_info_table += f"{detail['input_order']:^8} | {detail['original_frame_index']:^10} | {detail['timestamp']:^10.1f} | {detail['description']}\n"
        
        contents = [
            {"type": "text", "text": f"""
你将会看到{len(group_frames)}张按时间顺序排列的屏幕截图，这些图片来自同一个视频的不同时间点。

{frame_info_table}

重要说明：
- 「输入顺序」表示图片在你接收到的顺序中的位置
- 「原始帧索引」表示该帧在原始视频中的实际帧编号
- 「时间戳」表示该帧在原始视频中的时间位置（秒）

请完成敏感操作识别任务，并严格按照要求的 JSON 格式输出。

返回 JSON 格式：
{{
  "frame_details": [
    {{
      "frame_index": 原始帧索引,
      "timestamp": 时间戳,
      "is_sensitive": true/false,
      "app_name": "应用名称",
      "operation_type": "操作类型",
      "description": "该帧的详细情况和判断描述",
      "roi_bbox": [x_min, y_min, x_max, y_max] // 敏感帧必须填写，非敏感帧必须填 null
    }}
  ]
}}
        """}
        ]
        
        for img in frame_base64_list:
            contents.append({
                "type": "image_url",
                "image_url": img
            })
        
        try:
            print(f"  🤖 调用VLM API分析第 {group_idx//5 + 1} 组 ({len(group_frames)} 帧)...")
            response = api_inference_video(model_name=model_name, contents=contents)
            
            group_result = parse_group_vlm_response(response, frame_info_list)
            if group_result and 'frame_details' in group_result:
                all_frame_details.extend(group_result['frame_details'])
                print(f"  ✅ 第 {group_idx//5 + 1} 组分析成功，获得 {len(group_result['frame_details'])} 个结果")
            else:
                print(f"  ❌ 第 {group_idx//5 + 1} 组分析失败，使用默认结果")
                for frame_info in group_frames:
                    default_result = {
                        "frame_index": frame_info['frame_index'],
                        "timestamp": frame_info['timestamp'],
                        "is_sensitive": False,
                        "app_name": "未知应用",
                        "operation_type": "无操作",
                        "description": "分析失败，默认非敏感",
                        "roi_bbox": None
                    }
                    all_frame_details.append(default_result)
                    
        except Exception as e:
            print(f"  ❌ 第 {group_idx//5 + 1} 组分析异常: {e}")
            for frame_info in group_frames:
                default_result = {
                    "frame_index": frame_info['frame_index'],
                    "timestamp": frame_info['timestamp'],
                    "is_sensitive": False,
                    "app_name": "未知应用",
                    "operation_type": "无操作",
                    "description": "分析异常，默认非敏感",
                    "roi_bbox": None
                }
                all_frame_details.append(default_result)
        
        if group_idx + 5 < len(frames):
            print("  ⏳ 等待1秒后处理下一组...")
            time.sleep(1)
    
    final_result = {
        "frame_details": all_frame_details
    }
    
    print(f"✅ 每5帧一组分析完成，总共分析了 {len(all_frame_details)} 帧")
    return final_result

def parse_group_vlm_response(response, frame_info_list):
    """解析每组VLM分析的响应"""
    try:
        if response is None:
            return None
            
        print(f"VLM原始响应 : {response}...")
        
        def clean_json_string(json_str):
            json_str = re.sub(r'^\s*```json\s*', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'\s*```\s*$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'//.*', '', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'(,\s*)}', r'}', json_str)
            json_str = re.sub(r'(,\s*)]', r']', json_str)
            
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                 json_str = json_str[start:end+1]
            
            return json_str.strip()
        
        cleaned_json = clean_json_string(response)
        
        if not cleaned_json:
            print("❌ 无法从响应中提取有效的JSON结构")
            return None
            
        try:
            result = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            return None
        
        if 'frame_details' not in result:
            print("⚠️ VLM输出JSON缺少 'frame_details' 字段")
            return None
        
        result = validate_and_fix_frame_details(result, frame_info_list)
        return result
            
    except Exception as e:
        print(f"❌ 解析组VLM响应出错: {e}")
        return None

def validate_and_fix_frame_details(result, frame_info_list):
    """验证和修复 frame_details 列表中的帧索引和时间戳"""
    original_frame_map = {frame_info['frame_index']: frame_info for frame_info in frame_info_list}
    
    if 'frame_details' in result:
        valid_details = []
        for frame in result['frame_details']:
            frame_index = frame.get('frame_index')
            
            if frame_index in original_frame_map:
                original_frame = original_frame_map[frame_index]
                frame['timestamp'] = original_frame['timestamp']
                valid_details.append(frame)
            else:
                print(f"  ⚠️ 无效帧索引: {frame_index}，跳过该帧")
        
        result['frame_details'] = valid_details
    
    return result

def cluster_sensitive_operations(frame_details):
    """对原子级的帧识别结果进行聚类"""
    sensitive_frames = [
        f for f in frame_details 
        if f.get('is_sensitive') is True
    ]
    
    if not sensitive_frames:
        return []

    sensitive_frames.sort(key=lambda x: x['frame_index'])
    
    clustered_operations = []
    current_group = None
    group_id_counter = 1
    
    for i, frame in enumerate(sensitive_frames):
        app_name = frame.get('app_name')
        
        if current_group is None:
            pass
        elif app_name != current_group['app_name']:
            clustered_operations.append(current_group)
            current_group = None

        if current_group is None:
            current_group = {
                "group_id": group_id_counter,
                "app_name": app_name,
                "operation_type": frame.get('operation_type', '未知操作'),
                "frames": []
            }
            group_id_counter += 1
            
        frame_data = {
            "frame_index": frame['frame_index'],
            "timestamp": frame['timestamp'],
            "description": frame.get('description', ''),
            "roi_bbox": frame.get('roi_bbox', [0, 0, 0, 0])
        }
        current_group['frames'].append(frame_data)
        
    if current_group is not None:
        clustered_operations.append(current_group)
        
    print(f"✅ 成功聚类 {len(clustered_operations)} 个敏感操作组。")
    return clustered_operations