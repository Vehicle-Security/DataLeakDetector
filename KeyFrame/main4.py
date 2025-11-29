import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import json
from vlm_inference import api_inference_video
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import glob
import tempfile
import subprocess


# === 配置参数 ===
SSIM_THRESHOLD = 0.99
SAMPLE_INTERVAL = 1
FINAL_FRAMES_COUNT = 12  
JPEG_QUALITY = 90  # 降低JPEG质量
SCALE_FACTOR = 0.77  # 图片缩放比例
MAX_SEGMENT_DURATION = 240  # 每个分段的最大时长（秒），4分钟

# 边界检测参数
TAU_LOW = 0.88  # 全局相似度低阈值
K_STABLE_FRAMES = 3  # 局部稳定所需的连续帧数
ADJACENT_SIMILARITY_THRESHOLD = 0.98  # 局部相邻相似度极高阈值
MIN_PROGRESS_FRAMES = 1  # 至少需要经过的帧数，才开始判断稳定性


class OperationBoundaryDetector:
    """操作边界检测器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.preprocess = self._get_preprocess()
        self.feature_cache = {}
        
    def _load_model(self):
        """加载预训练的ResNet-50模型"""
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model
    
    def _get_preprocess(self):
        """获取图像预处理流程"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame, roi_coords=None):
        """
        提取图像特征向量，支持ROI裁剪
        
        Args:
            frame: 图像帧 (numpy数组)
            roi_coords: VLM提供的标准化ROI坐标 [x1, y1, x2, y2]
            
        Returns:
            numpy数组: 特征向量 (2048维)
        """
        cache_key = (frame.tobytes(), tuple(roi_coords) if roi_coords else None)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # 转换为PIL图像
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if roi_coords is not None:
            # 转换为像素坐标并裁剪
            height, width = frame.shape[:2]
            x1 = int(roi_coords[0] * width / 1000)
            y1 = int(roi_coords[1] * height / 1000)
            x2 = int(roi_coords[2] * width / 1000)
            y2 = int(roi_coords[3] * height / 1000)
            
            # 确保坐标在有效范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
        
        # 预处理
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(input_tensor)
            feature_vector = features.squeeze().cpu().numpy().flatten()
            
        self.feature_cache[cache_key] = feature_vector
        return feature_vector
        
    def calculate_similarity(self, feature1, feature2):
        """计算两个特征向量的余弦相似度"""
        if feature1 is None or feature2 is None:
            return 0.0
            
        feature1 = feature1.flatten().astype(np.float64)
        feature2 = feature2.flatten().astype(np.float64)
        
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(feature1, feature2) / (norm1 * norm2)
        
        return float(similarity)
        
    def find_operation_boundaries(self, ref_frames, all_frames, roi_coords=None):
        """
        寻找操作的开始和结束帧
        
        Args:
            ref_frames: 基准帧列表 (包含frame, frame_index, timestamp)
            all_frames: 所有帧列表
            roi_coords: VLM提供的标准化ROI坐标
            
        Returns:
            start_frame_info, end_frame_info: 开始和结束帧信息
        """
        print(f"🔍 寻找操作边界...")
        print(f"基准帧数量: {len(ref_frames)}, ROI: {roi_coords}")

        # --- 配置阈值 ---
        # 骤降阈值：比如 0.97 -> 0.91 (差值 0.06)，
        SIMILARITY_DROP_THRESHOLD = 0.04  
        
        

        # 1. 提取平均全局特征
        ref_features_list = [self.extract_features(frame_info['frame']) for frame_info in ref_frames]
        ref_feature_global = np.mean(ref_features_list, axis=0)

        # 获取所有帧的索引
        all_frame_indices = [f['frame_index'] for f in all_frames]

        # 找到基准帧的索引范围
        ref_frame_indices = [f['frame_index'] for f in ref_frames]
        ref_start_index = all_frame_indices.index(min(ref_frame_indices))
        ref_end_index = all_frame_indices.index(max(ref_frame_indices))

        # --- 2. 向前搜索（寻找开始帧 T_start）---
        print("\n向前搜索开始帧 (T_start)...")
        start_frame_index = ref_start_index
        
        # [修改点 1] 初始化 last_similarity
        # 我们从基准帧开始往前倒推，所以初始的 "last" (其实是时间轴后一帧) 应该是基准帧的相似度(近似1.0)
        # 为了精确，我们计算 ref_start_index 这一帧的实际相似度
        start_feat = self.extract_features(all_frames[ref_start_index]['frame'])
        last_similarity = self.calculate_similarity(ref_feature_global, start_feat)
        
        low_similarity_counter = 0
        REQUIRED_LOW_SIMILARITY_FRAMES = 2 

        # i 从 ref_start_index - 1 递减到 0 (避免第一帧自己减自己)
        for i in range(ref_start_index - 1, -1, -1):
            current_frame_info = all_frames[i]
            current_feature = self.extract_features(current_frame_info['frame'])
            
            # 计算当前相似度
            current_similarity = self.calculate_similarity(ref_feature_global, current_feature)
            
            # [修改点 2] 计算相对骤降 (Last - Current)
            # 因为是向前倒推，如果 Current 突然变低，说明这里是“悬崖”
            # sim_diff = last_similarity - current_similarity

            # print(f"  帧 {current_frame_info['frame_index']} (全局): Sim={current_similarity:.4f} | Diff={sim_diff:.4f}")

            # # 逻辑 A: 检查骤降 (优先级最高，通常意味着硬切换)
            # if sim_diff > SIMILARITY_DROP_THRESHOLD:
            #     print(f"  >>> 帧 {current_frame_info['frame_index']}: 检测到相似度骤降 ({last_similarity:.4f} -> {current_similarity:.4f}, 跌幅 {sim_diff:.4f})")
            #     # 既然当前帧导致了骤降，说明操作是从它的"后面"（也就是 i+1）开始的
            #     start_frame_index = i + 1
            #     break

            # 逻辑 B: 检查绝对阈值 (原有的逻辑，用于处理非骤降的低相似度)
            if current_similarity < TAU_LOW:
                low_similarity_counter += 1
                if low_similarity_counter >= REQUIRED_LOW_SIMILARITY_FRAMES:
                    start_frame_index = i + REQUIRED_LOW_SIMILARITY_FRAMES
                    start_frame_index = min(start_frame_index, ref_start_index)
                    print(f"  帧 {current_frame_info['frame_index']}: 连续 {REQUIRED_LOW_SIMILARITY_FRAMES} 帧低于绝对阈值，T_start确定为 {all_frames[start_frame_index]['frame_index']}")
                    break
            else:
                low_similarity_counter = 0 
                # 如果既没有骤降，也没有低于绝对阈值，更新 last_similarity 继续往前找
                start_frame_index = i
            
            # [修改点 3] 更新 last_similarity 用于下一次循环
            last_similarity = current_similarity
        
        # 循环结束边界处理
        if start_frame_index < 0: start_frame_index = 0
        if low_similarity_counter > 0 and i == -1: start_frame_index = 0
             

        # --- 3. 向后搜索（寻找结束帧 T_end）---
        print("\n向后搜索结束帧 (T_end)...")
        end_frame_index = ref_end_index

        # [修改点 4] 初始化 last_similarity
        end_feat = self.extract_features(all_frames[ref_end_index]['frame'])
        last_similarity = self.calculate_similarity(ref_feature_global, end_feat)

        current_idx = ref_end_index + 1
        stable_frame_counter = 0 

        # 初始前一帧ROI特征
        prev_roi_feature = self.extract_features(all_frames[ref_end_index]['frame'], roi_coords)

        frames_passed_since_ref_end = 0

        while current_idx < len(all_frames):
            current_frame_info = all_frames[current_idx]
            current_frame_index = current_frame_info['frame_index']
            current_feature_global = self.extract_features(current_frame_info['frame'])

            frames_passed_since_ref_end += 1

            # 1. 全局相似度检查
            global_similarity = self.calculate_similarity(current_feature_global, ref_feature_global)
            
            # [修改点 5] 计算相对骤降
            #sim_diff = last_similarity - global_similarity
            
            # 只有当差值显著时才打印详细 Diff 信息，避免刷屏
            
            #print(f"  帧 {current_frame_index} (全局): Sim={global_similarity:.4f} | Diff={sim_diff:.4f}")

            # # 边界条件 A1: 相对骤降 (新加逻辑)
            # if sim_diff > SIMILARITY_DROP_THRESHOLD:
            #     end_frame_index = current_idx - 1
            #     print(f"  >>> 帧 {current_frame_index}: 全局相似度骤降，T_end确定为 {all_frames[end_frame_index]['frame_index']}")
            #     break
            
            # 边界条件 A2: 绝对值过低 (原有逻辑)
            if global_similarity < TAU_LOW:
                end_frame_index = current_idx - 1
                print(f"  帧 {current_frame_index}: 全局相似度低于绝对阈值，T_end确定为 {all_frames[end_frame_index]['frame_index']}")
                break

            # 更新 last_similarity
            last_similarity = global_similarity

            # 2. 局部相邻相似度检查 (ROI 检查逻辑保持不变)
            if frames_passed_since_ref_end < MIN_PROGRESS_FRAMES:
                # print(f"  帧 {current_frame_index} (跳过稳定性检查)...") 
                current_roi_feature = self.extract_features(current_frame_info['frame'], roi_coords)
                prev_roi_feature = current_roi_feature
            else:
                current_roi_feature = self.extract_features(current_frame_info['frame'], roi_coords)
                s_adj_c = self.calculate_similarity(current_roi_feature, prev_roi_feature)

                # print(f"  帧 {current_frame_index} (局部相邻): 相似度 = {s_adj_c:.4f}")

                if s_adj_c >= ADJACENT_SIMILARITY_THRESHOLD: 
                    stable_frame_counter += 1
                    if stable_frame_counter >= K_STABLE_FRAMES:
                        end_frame_index = current_idx - K_STABLE_FRAMES 
                        print(f"  帧 {current_frame_index}: 局部连续 {K_STABLE_FRAMES} 帧稳定，T_end确定为 {all_frames[end_frame_index]['frame_index']}")
                        break
                else:
                    stable_frame_counter = 0 

                prev_roi_feature = current_roi_feature

            current_idx += 1

        if current_idx == len(all_frames):
            end_frame_index = len(all_frames) - 1

        # --- 4. 结果整合 ---
        start_frame_info = all_frames[start_frame_index]
        end_frame_info = all_frames[end_frame_index]

        print(f"\n=== 操作边界检测结果 ===")
        print(f"开始帧: {start_frame_info['frame_index']}, 时间: {start_frame_info['timestamp']:.2f}s")
        print(f"结束帧: {end_frame_info['frame_index']}, 时间: {end_frame_info['timestamp']:.2f}s")

        return start_frame_info, end_frame_info

def resize_frame(frame, scale_factor=SCALE_FACTOR):
    """缩放帧图片到指定比例"""
    if scale_factor == 1.0:
        return frame
        
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame

def compress_frame(frame):
    """压缩帧图片以减少大小"""
#     # 先缩放
#     frame_resized = resize_frame(frame, SCALE_FACTOR)
    
#     # 编码为JPEG并控制质量
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
#     _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
    
#     return buffer, frame_resized.shape[:2]

    # 强制调整为固定尺寸 (Width, Height)
    target_size = (720, 720)
    
    # 直接缩放，不按比例
    frame_resized = cv2.resize(frame, target_size)
    
    # 编码为JPEG并控制质量
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
    
    return buffer, frame_resized.shape[:2]

def get_video_duration(video_path):
    """获取视频总时长（秒）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return duration

def split_video_segments(video_path, max_duration=MAX_SEGMENT_DURATION):
    """
    将视频分割成多个不超过指定时长的分段
    
    Args:
        video_path: 视频文件路径
        max_duration: 每个分段的最大时长（秒）
        
    Returns:
        segments: 分段信息列表，每个元素为 (start_time, end_time, segment_id)
    """
    total_duration = get_video_duration(video_path)
    print(f"📊 视频总时长: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    
    if total_duration <= max_duration:
        print("✅ 视频时长不超过4分钟，无需分段")
        return [(0, total_duration, "full_video")]
    
    segments = []
    num_segments = int(np.ceil(total_duration / max_duration))
    
    for i in range(num_segments):
        start_time = i * max_duration
        end_time = min((i + 1) * max_duration, total_duration)
        segment_id = f"segment_{i+1}"
        segments.append((start_time, end_time, segment_id))
        
        print(f"  分段 {i+1}: {start_time:.1f}s - {end_time:.1f}s (时长: {end_time-start_time:.1f}s)")
    
    print(f"📁 视频被分割为 {num_segments} 个分段")
    return segments

def keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.95):
    """流式处理：提取整个视频的关键帧"""
    segment_output_dir = os.path.join(output_dir, "keyframes")
    if os.path.exists(segment_output_dir):
        shutil.rmtree(segment_output_dir)
    os.makedirs(segment_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算起始和结束帧索引 - 整个视频
    start_frame = 0
    end_frame = total_frames
    
    step = int(fps * SAMPLE_INTERVAL)
    
    print(f"  提取整个视频关键帧: 帧 0 - {end_frame}")

    # 初始化特征提取器
    feature_extractor = OperationBoundaryDetector()
    prev_feature = None
    kept_count = 0
    kept_frames = []  # 保存帧数据和索引

    pbar = tqdm(total=end_frame-start_frame, desc=f"关键帧提取")
    frame_idx = start_frame

    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break

        # 提取当前帧的ResNet50特征
        current_feature = feature_extractor.extract_features(frame)

        # 特征去重
        if prev_feature is not None:
            similarity = feature_extractor.calculate_similarity(prev_feature, current_feature)
            if similarity >= similarity_threshold:
                frame_idx += step
                pbar.update(step)
                continue
        
        # 保存不相似的帧
        prev_feature = current_feature
        kept_count += 1
        
        # 保存帧数据和索引
        frame_info = {
            'frame': frame.copy(),
            'frame_index': frame_idx,
            'timestamp': frame_idx / fps,
        }
        kept_frames.append(frame_info)
        
        save_path = os.path.join(segment_output_dir, f"keyframe_{frame_idx:06d}.jpg")
        cv2.imwrite(save_path, frame)

        frame_idx += step
        pbar.update(step)

    pbar.close()
    cap.release()
    print(f"  ✅ 保存了 {kept_count} 张去重后的关键帧")
    return kept_frames

def select_uniform_frames(frames, step=3):
    """从帧列表中按固定步长选择帧"""
    total_frames = len(frames)
    if total_frames == 0:
        return []
    
    # 按固定步长选择帧
    indices = list(range(0, total_frames, step))
    selected_frames = [frames[i] for i in indices]
    
    print(f"📊 固定步长选择: 从 {total_frames} 帧中选择 {len(selected_frames)} 帧")
    print(f"  固定步长: {step}")
    print(f"  选择帧索引: {[frames[i]['frame_index'] for i in indices]}")
    
    return selected_frames

def batch_analyze_frames_with_vlm(frames, model_name="qwen2-vl-72b-instruct"):
    """
    每5帧一组一次性输入大模型进行敏感操作识别和ROI检测。
    """
    all_frame_details = []
    
    print(f"🖼️  开始每5帧一组分析，总帧数: {len(frames)}")
    
    # 按每5帧一组进行处理
    for group_idx in range(0, len(frames), 5):
        group_frames = frames[group_idx:group_idx+5]
        print(f"📦 处理第 {group_idx//5 + 1} 组，包含 {len(group_frames)} 帧")
        
        # 构建当前组的帧信息表
        frame_details_table_data = []
        frame_base64_list = []
        frame_info_list = []
        
        # 压缩图片并构建帧信息
        for i, frame_info in enumerate(group_frames):
            # 压缩当前帧
            buffer, original_shape = compress_frame(frame_info['frame'])
            frame_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            frame_base64_list.append(frame_base64)
            frame_info_list.append(frame_info)
            
            # 构建帧详细信息
            frame_details_table_data.append({
                'input_order': group_idx + i + 1,
                'original_frame_index': frame_info['frame_index'],
                'timestamp': frame_info['timestamp'],
                'description': f"第{group_idx + i + 1}张图片"
            })
        
        # 构建详细的帧信息字符串
        frame_info_table = "帧详细信息表:\n"
        frame_info_table += "输入顺序 | 原始帧索引 | 时间戳(秒) | 描述\n"
        frame_info_table += "-" * 50 + "\n"
        for detail in frame_details_table_data:
            frame_info_table += f"{detail['input_order']:^8} | {detail['original_frame_index']:^10} | {detail['timestamp']:^10.1f} | {detail['description']}\n"
        
        # 构建提示词（针对5帧组）
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
        
        # 添加当前组的所有图片
        for img in frame_base64_list:
            contents.append({
                "type": "image_url",
                "image_url": img
            })
        
        # ✅ 调用 API 分析当前5帧组
        try:
            print(f"  🤖 调用VLM API分析第 {group_idx//5 + 1} 组 ({len(group_frames)} 帧)...")
            response = api_inference_video(model_name=model_name, contents=contents)
            
            # 解析当前组的响应
            group_result = parse_group_vlm_response(response, frame_info_list)
            if group_result and 'frame_details' in group_result:
                all_frame_details.extend(group_result['frame_details'])
                print(f"  ✅ 第 {group_idx//5 + 1} 组分析成功，获得 {len(group_result['frame_details'])} 个结果")
            else:
                print(f"  ❌ 第 {group_idx//5 + 1} 组分析失败，使用默认结果")
                # 为当前组的所有帧添加默认结果
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
            # 为当前组的所有帧添加默认结果
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
        
        # 每组处理完成后稍微延迟，避免API限制
        if group_idx + 5 < len(frames):
            print("  ⏳ 等待1秒后处理下一组...")
            import time
            time.sleep(1)
    
    # 返回完整结果
    final_result = {
        "frame_details": all_frame_details
    }
    
    print(f"✅ 每5帧一组分析完成，总共分析了 {len(all_frame_details)} 帧")
    return final_result
    
def parse_group_vlm_response(response, frame_info_list):
    """
    解析每组VLM分析的响应。
    预期结构: {"frame_details": [...]}
    """
    try:
        if response is None:
            return None
            
        print(f"VLM原始响应 : {response}...")
        
        import re
        
        # 清理JSON字符串
        def clean_json_string(json_str):
            # 移除Markdown代码块标记 (```json ... ```)
            json_str = re.sub(r'^\s*```json\s*', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'\s*```\s*$', '', json_str, flags=re.MULTILINE)
            # 移除单行注释 (// ...)
            json_str = re.sub(r'//.*', '', json_str)
            # 移除多行注释 (/* ... */)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # 尝试移除JSON中可能存在的尾随逗号
            json_str = re.sub(r'(,\s*)}', r'}', json_str)
            json_str = re.sub(r'(,\s*)]', r']', json_str)
            
            # 确保移除包裹在JSON外层的多余文本
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                 json_str = json_str[start:end+1]
            
            return json_str.strip()
        
        # 提取并清理JSON部分
        cleaned_json = clean_json_string(response)
        
        if not cleaned_json:
            print("❌ 无法从响应中提取有效的JSON结构")
            return None
            
        try:
            result = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"失败位置附近: {cleaned_json[max(0, e.pos-20):min(len(cleaned_json), e.pos+20)]}")
            return None
        
        # 验证结果结构
        if 'frame_details' not in result:
            print("⚠️ VLM输出JSON缺少 'frame_details' 字段")
            return None
        
        # 验证和修复帧索引
        result = validate_and_fix_frame_details(result, frame_info_list)
        
        return result
            
    except Exception as e:
        print(f"❌ 解析组VLM响应出错: {e}")
        return None
            
    except Exception as e:
        print(f"❌ 解析批量VLM响应出错: {e}")
        return None

def validate_and_fix_frame_details(result, frame_info_list):
    """验证和修复 frame_details 列表中的帧索引和时间戳"""
    # 创建原始帧索引映射
    original_frame_map = {frame_info['frame_index']: frame_info for frame_info in frame_info_list}
    
    if 'frame_details' in result:
        valid_details = []
        for frame in result['frame_details']:
            frame_index = frame.get('frame_index')
            
            if frame_index in original_frame_map:
                original_frame = original_frame_map[frame_index]
                # 确保时间戳正确
                frame['timestamp'] = original_frame['timestamp']
                valid_details.append(frame)
            else:
                print(f"  ⚠️ 无效帧索引: {frame_index}，跳过该帧")
        
        result['frame_details'] = valid_details
    
    return result

# ====================================================================
# 实现基于 AppName 连续一致性的敏感操作聚类
# ====================================================================

def cluster_sensitive_operations(frame_details):
    """
    对原子级的帧识别结果进行聚类。
    
    聚类规则: 如果下一帧的 app_name 和当前帧的 app_name 一致，则为同一组。
    
    Args:
        frame_details (list): VLM输出的 frame_details 列表。
        
    Returns:
        list: 聚类后的敏感操作列表，结构与原 sensitive_operations 字段一致。
    """
    
    # 1. 过滤出敏感帧
    sensitive_frames = [
        f for f in frame_details 
        if f.get('is_sensitive') is True
    ]
    
    if not sensitive_frames:
        return []

    # 2. 按时间戳或 frame_index 确保顺序
    sensitive_frames.sort(key=lambda x: x['frame_index'])
    
    clustered_operations = []
    current_group = None
    group_id_counter = 1
    
    for i, frame in enumerate(sensitive_frames):
        app_name = frame.get('app_name')
        
        # 检查是否需要开启新组
        if current_group is None:
            # 开启第一组
            pass
        elif app_name != current_group['app_name']:
            # app_name 不一致，结束当前组，开启新组
            clustered_operations.append(current_group)
            current_group = None # 准备开启新组

        # 开启新组的逻辑
        if current_group is None:
            current_group = {
                "group_id": group_id_counter,
                "app_name": app_name,
                "operation_type": frame.get('operation_type', '未知操作'),
                "frames": []
            }
            group_id_counter += 1
            
        # 将帧添加到当前组
        frame_data = {
            "frame_index": frame['frame_index'],
            "timestamp": frame['timestamp'],
            "description": frame.get('description', ''),
            "roi_bbox": frame.get('roi_bbox', [0, 0, 0, 0])
        }
        current_group['frames'].append(frame_data)

        # 保持 operation_type 为组内第一帧的类型，除非后续操作类型变化较大（这里简化为保持第一帧的类型）
        
    # 3. 添加最后一组
    if current_group is not None:
        clustered_operations.append(current_group)
        
    print(f"✅ 成功聚类 {len(clustered_operations)} 个敏感操作组。")
    return clustered_operations

# ====================================================================
# 重新定义 extend_operation_boundaries 和 merge_segment_results
# ====================================================================

def extend_operation_boundaries(sensitive_operations, all_frames):
    """
    为每个敏感操作组扩展时间边界。
    参数由 VLM 原始输出结果改为聚类后的敏感操作列表。
    """
    detector = OperationBoundaryDetector()
    extended_operations = []
    
    if not sensitive_operations:
        return extended_operations
    
    for group in sensitive_operations: # 注意这里是直接迭代聚类后的列表
        print(f"\n🎯 处理操作组: {group['app_name']} - {group['operation_type']}")
        
        # 获取该组的基准帧
        ref_frames = []
        for frame_info in group['frames']:
            # 在all_frames中找到对应的帧
            original_frame = next(
                (f for f in all_frames if f['frame_index'] == frame_info['frame_index']), 
                None
            )
            if original_frame:
                ref_frames.append(original_frame)
        
        if not ref_frames:
            print(f"⚠️ 未找到基准帧，跳过该组")
            continue
        
        # 获取ROI坐标（使用第一个非空ROI）
        roi_coords = None
        for frame_info in group['frames']:
            # 检查 ROI 是否存在且不为 null 或 [0, 0, 0, 0]
            if frame_info.get('roi_bbox') not in ([0, 0, 0, 0], None):
                roi_coords = frame_info['roi_bbox']
                break
        
        try:
            # 寻找操作边界
            start_frame_info, end_frame_info = detector.find_operation_boundaries(
                ref_frames=ref_frames,
                all_frames=all_frames,
                roi_coords=roi_coords
            )
            
            # 创建扩展后的操作信息
            extended_operation = {
                'group_id': group['group_id'],
                'app_name': group['app_name'],
                'operation_type': group['operation_type'],
                'original_frames': group['frames'],  # 原始聚类检测的帧
                'extended_start_frame': start_frame_info['frame_index'],
                'extended_end_frame': end_frame_info['frame_index'],
                'extended_start_time': start_frame_info['timestamp'],
                'extended_end_time': end_frame_info['timestamp'],
                'extended_duration': end_frame_info['timestamp'] - start_frame_info['timestamp'],
                'roi_coords': roi_coords
            }
            
            extended_operations.append(extended_operation)
            
        except Exception as e:
            print(f"❌ 边界检测失败: {e}")
            # 如果边界检测失败，使用原始范围
            original_timestamps = [f.get('timestamp', 0) for f in group['frames']]
            extended_operation = {
                'group_id': group['group_id'],
                'app_name': group['app_name'],
                'operation_type': group['operation_type'],
                'original_frames': group['frames'],
                'extended_start_frame': min([f['frame_index'] for f in group['frames']]),
                'extended_end_frame': max([f['frame_index'] for f in group['frames']]),
                'extended_start_time': min(original_timestamps),
                'extended_end_time': max(original_timestamps),
                'extended_duration': max(original_timestamps) - min(original_timestamps),
                'roi_coords': roi_coords,
                'note': '边界检测失败，使用原始范围'
            }
            extended_operations.append(extended_operation)
    
    return extended_operations

def merge_segment_results(all_segment_results):
    """
    合并所有分段的已处理结果（不再重新聚类或重新计算边界）。
    逻辑：
    1. 收集所有分段的操作结果。
    2. 按时间排序。
    3. 如果相邻两个操作是同一个APP且时间重叠/接续，则合并它们。
    
    Args:
        all_segment_results: 所有分段的结果列表，每个元素为 (VLM_result, extended_ops)
        
    Returns:
        merged_ops: 合并重叠区间后的最终操作列表
    """
    print("\n🔄 合并分段结果 (区间合并模式)...")
    
    # 1. 收集所有分段的操作
    all_ops = []
    for _, segment_ops in all_segment_results:
        all_ops.extend(segment_ops)
        
    if not all_ops:
        return [], {}

    # 2. 按开始时间排序
    all_ops.sort(key=lambda x: x['extended_start_time'])
    
    merged_ops = []
    
    # 定义合并的时间容差（秒），例如两个操作间隔小于1.5秒且是同APP，视为同一个连续操作
    MERGE_TOLERANCE = 1.5 

    for op in all_ops:
        if not merged_ops:
            merged_ops.append(op)
            continue
            
        last_op = merged_ops[-1]
        
        # 判断是否应该合并
        # 条件1: 应用名称相同
        is_same_app = last_op['app_name'] == op['app_name']
        
        # 条件2: 时间重叠 或 间隔非常短 (当前开始时间 <= 上一个结束时间 + 容差)
        is_time_connected = op['extended_start_time'] <= (last_op['extended_end_time'] + MERGE_TOLERANCE)
        
        if is_same_app and is_time_connected:
            # === 执行合并 ===
            # print(f"  🔗 合并操作: {last_op['app_name']} ({last_op['extended_end_time']:.1f}s) + ({op['extended_start_time']:.1f}s)")
            
            # 1. 更新结束时间和帧索引
            last_op['extended_end_time'] = max(last_op['extended_end_time'], op['extended_end_time'])
            last_op['extended_end_frame'] = max(last_op['extended_end_frame'], op['extended_end_frame'])
            
            # 2. 更新时长
            last_op['extended_duration'] = last_op['extended_end_time'] - last_op['extended_start_time']
            
            # 3. 合并原始帧列表 (并去重，虽然理论上分段不会有重复帧)
            existing_indices = {f['frame_index'] for f in last_op['original_frames']}
            for frame in op['original_frames']:
                if frame['frame_index'] not in existing_indices:
                    last_op['original_frames'].append(frame)
                    existing_indices.add(frame['frame_index'])
            
            # 确保帧列表有序
            last_op['original_frames'].sort(key=lambda x: x['frame_index'])
            
            # 4. ROI处理：如果之前的ROI为空，尝试使用当前的ROI
            if not last_op.get('roi_coords') and op.get('roi_coords'):
                last_op['roi_coords'] = op['roi_coords']
                
        else:
            # === 不合并，作为新操作添加 ===
            merged_ops.append(op)

    # 3. 重新生成 Group ID
    for i, op in enumerate(merged_ops):
        op['group_id'] = i + 1

    print(f"✅ 合并完成: 原有 {len(all_ops)} 个片段 -> 合并为 {len(merged_ops)} 个独立操作")
    
    return merged_ops, {}

def extract_video_segment(video_path, start_time, end_time, output_path):
    """
    从原视频中截取指定时间段的视频片段
    
    Args:
        video_path: 原视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出视频路径
    """
    try:
        # 计算持续时间
        duration = end_time - start_time
        
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # 开始时间
            '-i', video_path,        # 输入文件
            '-t', str(duration),     # 持续时间
            '-c', 'copy',           # 流复制，快速且无损
            '-avoid_negative_ts', 'make_zero',
            '-y',                   # 覆盖输出文件
            output_path
        ]
        
        print(f"  截取视频: {start_time:.1f}s - {end_time:.1f}s (时长: {duration:.1f}s)")
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ 成功保存: {output_path}")
            return True
        else:
            print(f"  ❌ 截取失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ 截取异常: {e}")
        return False
def save_scene_keyframes(merged_extended_operations, all_frames, output_dir):
    """
    将每个场景的关键帧存入不同的文件夹中
    
    Args:
        merged_extended_operations: 合并后的扩展操作列表
        all_frames: 所有关键帧列表
        output_dir: 输出目录
        
    Returns:
        str: 场景关键帧目录路径
    """
    print(f"\n{'='*60}")
    print("🖼️  开始保存每个场景的关键帧")
    print(f"{'='*60}")
    
    # 创建场景关键帧目录
    scene_keyframes_dir = os.path.join(output_dir, "scene_keyframes")
    os.makedirs(scene_keyframes_dir, exist_ok=True)
    
    # 按frame_index创建所有帧的映射
    frame_map = {frame['frame_index']: frame for frame in all_frames}
    
    saved_scenes = 0
    
    for i, operation in enumerate(merged_extended_operations):
        print(f"\n📁 处理第 {i+1} 个场景:")
        print(f"   应用: {operation['app_name']}")
        print(f"   操作: {operation['operation_type']}")
        print(f"   时间区间: {operation['extended_start_time']:.1f}s - {operation['extended_end_time']:.1f}s")
        
        # 创建场景文件夹（清理文件名中的特殊字符）
        app_name_clean = "".join(c for c in operation['app_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        op_type_clean = "".join(c for c in operation['operation_type'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        scene_folder_name = f"scene_{i+1:02d}_{app_name_clean}_{op_type_clean}"
        scene_folder_path = os.path.join(scene_keyframes_dir, scene_folder_name)
        
        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.makedirs(scene_folder_path)
        
        # 获取该场景时间范围内的所有关键帧
        start_frame = operation['extended_start_frame']
        end_frame = operation['extended_end_frame']
        
        scene_frames = []
        for frame_index in range(start_frame, end_frame + 1):
            if frame_index in frame_map:
                scene_frames.append(frame_map[frame_index])
        
        # 如果没有找到帧，尝试使用原始检测的帧
        if not scene_frames and 'original_frames' in operation:
            for frame_info in operation['original_frames']:
                frame_index = frame_info['frame_index']
                if frame_index in frame_map:
                    scene_frames.append(frame_map[frame_index])
        
        # 保存关键帧到场景文件夹
        if scene_frames:
            print(f"   📸 找到 {len(scene_frames)} 个关键帧")
            
            for j, frame_info in enumerate(scene_frames):
                frame = frame_info['frame']
                frame_index = frame_info['frame_index']
                timestamp = frame_info['timestamp']
                
                # 生成文件名
                filename = f"frame_{j+1:03d}_index{frame_index}_time{timestamp:.1f}s.jpg"
                filepath = os.path.join(scene_folder_path, filename)
                
                # 保存帧
                cv2.imwrite(filepath, frame)
            
            # 创建场景信息文件
            info_filepath = os.path.join(scene_folder_path, "scene_info.txt")
            with open(info_filepath, 'w', encoding='utf-8') as f:
                f.write(f"场景ID: {operation['group_id']}\n")
                f.write(f"应用名称: {operation['app_name']}\n")
                f.write(f"操作类型: {operation['operation_type']}\n")
                f.write(f"开始时间: {operation['extended_start_time']:.2f}s\n")
                f.write(f"结束时间: {operation['extended_end_time']:.2f}s\n")
                f.write(f"持续时间: {operation['extended_duration']:.2f}s\n")
                f.write(f"开始帧: {operation['extended_start_frame']}\n")
                f.write(f"结束帧: {operation['extended_end_frame']}\n")
                f.write(f"关键帧数量: {len(scene_frames)}\n")
                if operation.get('roi_coords'):
                    f.write(f"ROI坐标: {operation['roi_coords']}\n")
            
            saved_scenes += 1
            print(f"   ✅ 成功保存到: {scene_folder_name}")
        else:
            print(f"   ⚠️ 未找到该场景的关键帧")
    
    print(f"\n📊 场景关键帧保存完成:")
    print(f"   总场景数: {len(merged_extended_operations)}")
    print(f"   成功保存: {saved_scenes}")
    print(f"   保存位置: {scene_keyframes_dir}")
    
    return scene_keyframes_dir


def main():
    # 步骤1: 直接处理整个视频
    video_path = "../../video/26.mov"
    output_dir = "./output"
    
    if os.path.exists(output_dir):
        print(f"🧹 检测到旧输出目录，正在清理: {output_dir} ...")
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"⚠️ 清理目录失败 (可能是文件被占用): {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎬 开始处理整个视频...")
    
    # 提取整个视频的关键帧
    all_frames = keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.97)
    
    if not all_frames:
        print("❌ 没有提取到任何关键帧")
        return
    
    # 均匀选择帧用于VLM分析
    selected_frames = select_uniform_frames(all_frames, step=3)
    
    # 批量VLM分析
    print(f"🤖 开始VLM敏感操作原子级分析...")
    batch_result = batch_analyze_frames_with_vlm(selected_frames)
    
    if batch_result is None or 'frame_details' not in batch_result:
        print("❌ VLM分析失败或结构错误")
        return
    
    # 对原子结果进行聚类
    clustered_ops = cluster_sensitive_operations(batch_result['frame_details'])
    
    # 扩展操作边界
    print(f"\n🔄 开始操作边界扩展...")
    extended_operations = extend_operation_boundaries(clustered_ops, all_frames)
    
    # 输出结果
    print(f"\n🎯 敏感操作时间区间:")
    print("=" * 80)
    
    for operation in extended_operations:
        print(f"组 {operation['group_id']}. 应用: {operation['app_name']}")
        print(f"   操作: {operation['operation_type']}")
        print(f"   时间: {operation['extended_start_time']:.1f}s - {operation['extended_end_time']:.1f}s")
        print(f"   时长: {operation['extended_duration']:.1f}s")
        if operation['roi_coords']:
            print(f"   ROI坐标: {operation['roi_coords']}")
        print()
    
    # 保存场景关键帧
    scene_keyframes_dir = save_scene_keyframes(extended_operations, all_frames, output_dir)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("🎬 处理完成总结")
    print(f"{'='*60}")
    print(f"总识别敏感操作: {len(extended_operations)} 个")
    
    # 显示场景关键帧文件夹
    print("\n📁 生成的场景关键帧文件夹:")
    for folder in os.listdir(scene_keyframes_dir):
        folder_path = os.path.join(scene_keyframes_dir, folder)
        if os.path.isdir(folder_path):
            jpg_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            print(f"  📂 {folder} ({jpg_count} 张关键帧)")


if __name__ == "__main__":
    main()