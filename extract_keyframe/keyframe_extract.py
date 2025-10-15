import os
import cv2
import shutil
import multiprocessing as mp
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import re

TEMPLATE_IMG = './templates/file_manager.png'
THRESHOLD = 0.50
SKIP_FPS = 1
SSIM_THRESHOLD = 0.90


def _process_chunk(args):
    """子进程任务：处理视频的一段帧区间"""
    video_path, start_frame, end_frame, fps, template, tH, tW, skip_frames, threshold = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, _ = cv2.minMaxLoc(res)
            if maxVal >= threshold:
                time_sec = frame_idx / fps
                results.append((frame, time_sec, maxVal, frame_idx))

        frame_idx += 1
    cap.release()
    return results


def keyframe_extract(video_path, output_dir, sensitivity_threshold=0.09, num_workers=None):
    """主函数：并行提取关键帧"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    template = cv2.imread(TEMPLATE_IMG, 0)
    if template is None:
        raise FileNotFoundError(f"模板图像 {TEMPLATE_IMG} 无法打开")
    tH, tW = template.shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, int(fps * SKIP_FPS))
    cap.release()

    # 分块：每个进程处理一段
    num_workers = num_workers or mp.cpu_count()
    chunk_size = total_frames // num_workers
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = total_frames if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((video_path, start, end, fps, template, tH, tW, skip_frames, THRESHOLD))

    print(f"🧠 启动 {num_workers} 个进程进行关键帧提取...")
    with mp.Pool(processes=num_workers) as pool:
        all_results = list(tqdm(pool.imap_unordered(_process_chunk, tasks), total=len(tasks), desc="提取中"))

    # 合并所有进程结果
    frames = []
    for chunk_result in all_results:
        for frame, time_sec, maxVal, frame_idx in chunk_result:
            out_path = os.path.join(output_dir, f'关键帧_{time_sec:.2f}s_{frame_idx}.jpg')
            cv2.imwrite(out_path, frame)
            frames.append((out_path, maxVal))


    print(f'✅ 提取完成，共检测到 {len(frames)} 张候选关键帧')
    group_and_select(frames, SSIM_THRESHOLD, output_dir)


def calculate_ssim(image_path1, image_path2):
    img1 = imread(image_path1, as_gray=True)
    img2 = imread(image_path2, as_gray=True)
    return ssim(img1, img2, data_range=img1.max() - img1.min())


def group_and_select(frames, ssim_threshold, output_dir):
    """去重 + 保留最优关键帧"""
    if not frames:
        print("⚠️ 没有检测到任何帧")
        return

    def extract_time_from_filename(path):
        """从文件名中提取秒数（支持关键帧_97.00s_5880.jpg 这种格式）"""
        name = os.path.basename(path)
        match = re.search(r'_(\d+\.\d+)s', name)
        if match:
            return float(match.group(1))
        return 0.0  # fallback

    frames.sort(key=lambda x: extract_time_from_filename(x[0]))
    groups = []
    current_group = [frames[0]]

    for i in range(1, len(frames)):
        current_frame = frames[i]
        previous_frame = frames[i - 1]
        ssim_value = calculate_ssim(current_frame[0], previous_frame[0])
        if ssim_value > ssim_threshold:
            current_group.append(current_frame)
        else:
            groups.append(current_group)
            current_group = [current_frame]
    if current_group:
        groups.append(current_group)

    selected_frames = [max(group, key=lambda x: x[1]) for group in groups]
    non_selected_dir = os.path.join(os.path.dirname(output_dir), 'non_selected')
    os.makedirs(non_selected_dir, exist_ok=True)

    for path, maxVal in frames:
        if (path, maxVal) not in selected_frames:
            base_name = os.path.basename(path)
            dest_path = os.path.join(non_selected_dir, base_name)

            # 如果目标文件已存在，则加后缀避免覆盖
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(base_name)
                dest_path = os.path.join(non_selected_dir, f"{name}_dup{ext}")

            shutil.move(path, dest_path)

    print(f"保留的关键帧数量：{len(selected_frames)}")


def extract_frames_around_keyframes(output_dir, video_path, keyframe_paths):
    """提取关键帧前后 5s、3s、当前帧（共 5 张）"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for keyframe_path in tqdm(keyframe_paths, desc='提取上下文帧'):
        filename = os.path.basename(keyframe_path)

        # ✅ 正确提取时间戳
        match = re.search(r'_(\d+\.\d+)s_', filename)
        if not match:
            print(f"⚠️ 文件名格式不匹配，跳过: {filename}")
            continue

        keyframe_time = float(match.group(1))
        keyframe_idx = int(keyframe_time * fps)

        # ✅ 合理命名上下文目录
        context_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        os.makedirs(context_dir, exist_ok=True)

        # ✅ 提取附近帧索引（5秒前、3秒前、当前、3秒后、5秒后）
        nearby = [
            max(0, int(keyframe_idx - 5 * fps)),
            max(0, int(keyframe_idx - 3 * fps)),
            int(keyframe_idx),
            min(total_frames - 1, int(keyframe_idx + 3 * fps)),
            min(total_frames - 1, int(keyframe_idx + 5 * fps)),
        ]

        extracted = 0
        for idx in nearby:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(context_dir, f'frame_{idx:06d}.jpg')
                cv2.imwrite(frame_path, frame)
                extracted += 1

        if extracted == 0:
            print(f"⚠️ 未成功提取任何帧: {filename}")
        else:
            print(f"✅ {filename} → 提取 {extracted} 张帧")

    cap.release()
    print('🎯 上下文帧提取完成')