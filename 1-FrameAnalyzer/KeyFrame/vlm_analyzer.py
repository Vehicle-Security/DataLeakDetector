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

请完成以下任务，并严格按照要求的 JSON 格式输出：

### 任务：原子级识别 (Frame-level Analysis)

#### 🛑 **最高优先级规则：精确应用名称识别** 🛑

**1. 应用名称 (`app_name`) 精确识别：**
* **浏览器处理：** 严禁输出 `Chrome`、`Edge` 等通用名称，**必须**识别网页内容的具体服务名称（如 "GitHub", "Kimi", "QQ邮箱"）。
* **指定敏感应用列表 (必须精准识别以下或同类应用):**
    * **AI 大模型/客户端:** ChatGPT, Kimi, 文心一言, 通义千问, 豆包, 元宝(及网页版), Cherry Studio, Chatbox, DeepSeek.
    * **即时通讯/会议:** 微信(及网页版), QQ, 钉钉, 飞书, 腾讯会议, 钉钉会议, Zoom.
    * **开发/技术社区:** GitHub, CSDN.
    * **云存储/笔记/邮箱:** 百度网盘, 夸克网盘, 有道云笔记, 网易邮箱, QQ邮箱.
    * **其他工具:** 文本转语音网页, 企业内部系统.

**2. 敏感操作 (`is_sensitive`) 判定标准 (核心逻辑)：**
分析每张图片，如果包含以下任一行为，**必须**标记为 `is_sensitive: true`：

* **A. 数据外发与传输 (最高风险):**
    * **文件操作：** 打开文件管理器/选择框、拖拽文件、上传/下载文件、**任何文件选择对话框**。
    * **图片操作：** 上传图片、选择图片、图片预览、图片编辑。
    * **即时通讯发送：** 在 QQ/微信/飞书/钉钉 等软件中，**点击发送按钮**、**分享链接**、**发送图片/文件**。
    * **AI 交互：** 在 AI 应用（如 Kimi, Chatbox, Cherry Studio）中**上传文件**、或对话气泡中明显显示**正在分析/已接收的文件/图片**。

* **B. 内容发布与公开:**
    * **技术社区发布：** 在 **GitHub** (Push代码, Create Repo, Issue)、**CSDN** (发布文章/博客) 等平台进行**内容发布、提交或保存**的操作。
    * **笔记同步：** 在有道云笔记/网易邮箱等平台保存或发送包含内容的笔记/邮件。

* **C. 会议屏幕共享泄露:**
    * **场景特征：** 界面上显示会议控制栏（如腾讯会议/Zoom 的"正在共享屏幕"提示、绿色边框、悬浮条）。
    * **敏感行为：** 在共享屏幕的状态下，**打开了本地文件**（Word, Excel, PDF等）、**浏览敏感文件夹**、或**切换到了即时通讯软件的私人聊天界面**。

* **D. 敏感数据处理:**
    * **剪贴板操作：** 画面显示右键菜单点击"**复制**"、"**粘贴**"，或出现剪贴板历史记录窗口。
    * **敏感内容输入：** 在输入框中输入长文本、代码块、或粘贴了图片/文件。

**3. ROI 区域检测 (仅针对敏感帧):**
对于敏感帧，必须返回 `roi_bbox` (归一化 [0, 1000])，框选规则如下：

* **文件选择对话框：** 框选整个文件选择窗口区域，包括文件列表和确认按钮。
* **图片上传界面：** 框选图片预览区域或文件选择区域。
* **常规输入/发送：** 框选输入框、发送按钮、或刚发送的消息气泡。
* **文件交互：** 框选文件选择窗口、正在拖拽的文件图标、或 AI 对话中的文件卡片。
* **发布/提交：** 框选编辑器的主要区域或"发布/Commit/Submit"按钮。
* **会议共享泄露：** 框选**被打开的文件窗口区域**或**暴露的敏感聊天窗口区域**（不要只框选会议控制条，要框选泄露的内容）。

**4. 时间顺序准确性要求：**
* **必须严格按照提供的帧索引和时间戳进行分析**
* **确保操作描述的连续性**，避免时间逻辑错误
* **仔细核对每个帧的实际内容**，不要基于推测判断

**5. 每帧详细输出：**
* `operation_type`: 例如 "上传文件", "选择文件", "发送消息", "发布博客", "会议中打开文件", "粘贴内容"。
* `description`: 详细描述操作，例如 "用户在QQ邮箱中打开了文件选择对话框，正在选择要上传的文件"。

**6. 输出规定：**
* 对网易邮箱，只允许输出为"网易邮箱"，不允许输出为"163邮箱"
---

返回 JSON 格式：
{{
  // 仅输出每一帧的原子级识别结果（敏感/非敏感）
  "frame_details": [
    {{
      "frame_index": 原始帧索引, // 必须使用上表中的「原始帧索引」
      "timestamp": 时间戳,       // 必须使用上表中的「时间戳」
      "is_sensitive": true/false, // 识别结果：是否是敏感操作
      "app_name": "应用名称",
      "operation_type": "操作类型" // 如果是非敏感，可填 "浏览" 或 "无操作"
      "description": "该帧的详细情况和判断描述",
      "roi_bbox": [x_min, y_min, x_max, y_max] // 敏感帧必须填写，非敏感帧必须填 null
    }}
    // ... 当前组其他所有帧的详细信息
  ]
}}

**关于ROI边界框的说明：**
- 坐标系统归一化到 [0, 1000] 范围
- 格式: [x_min, y_min, x_max, y_max]
- 识别与场景内容主题相关的关键操作区域，特别是输入区域：

ROI选择原则：
- 选择用户实际进行输入操作的核心区域
- 框选完整的输入组件，包括可见的文本内容
- 对于对话框操作，框选整个对话框区域
- **对于文件选择操作，ROI应框选整个文件选择对话框，包括文件列表区域和操作按钮。**
- **对于图片上传操作，ROI应框选图片预览区域或文件选择界面。**
- 确保ROI能够反映当前的操作状态

**⚠️ 关键改进点和纠正要求：**
1. **严格时间顺序准确性**：必须按照提供的帧索引和时间戳准确分析
2. **文件操作精确识别**：任何文件选择对话框、上传界面都必须准确识别
3. **避免内容混淆**：仔细区分不同帧的实际内容，不要张冠李戴
4. **动态ROI调整**：根据每帧实际界面调整ROI坐标
5. **操作连续性检查**：确保相邻帧的操作描述逻辑连贯

重要要求：
1. 对于同一操作组的不同帧，ROI坐标应该根据每帧的实际内容动态调整
2. 不要对所有帧返回相同的ROI坐标，要根据界面变化调整
3. 确保在返回的JSON中使用原始的时间戳值
4. ROI坐标要精确反映当前帧的输入区域位置和大小

⚠️ **极其重要格式要求：**
- **返回的内容必须是纯JSON格式，不要包含任何注释、额外文本或Markdown代码块标记**
- **直接返回JSON对象，不要用 ```json ``` 包裹**
- 确保JSON格式完全正确，包括所有引号、逗号和括号
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