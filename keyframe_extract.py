import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import shutil
from collections import defaultdict
from tqdm import tqdm
from natsort import natsorted

# ---------- 参数区 ----------
TEMPLATE_IMG = '/Users/tujiali/Desktop/templates/file_manager.png'     # 模板图
THRESHOLD    = 0.50               # 匹配模板的相似度阈值
SKIP_FPS     = 1                 # 每 SKIP_FPS 秒抽一帧
SSIM_THRESHOLD = 0.90            # SSIM 相似度阈值，用于去重
# ----------------------------


def keyframe_extract(video_path, output_dir, sensitivity_threshold=0.09):
    os.makedirs(output_dir, exist_ok=True)

    # 读取模板并转灰度
    template = cv2.imread(TEMPLATE_IMG, 0)  # 直接灰度
    tH, tW = template.shape

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'视频 {video_path} 无法打开')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(fps * SKIP_FPS)

    frame_idx = 0
    saved = 0

    pbar = tqdm(total=total_frames, desc='检测中')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 只处理指定间隔帧
        if frame_idx % skip_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

            _, maxVal, _, _ = cv2.minMaxLoc(res)
            if maxVal >= THRESHOLD:
                time_sec = frame_idx / fps
                out_path = os.path.join(output_dir, f'关键帧_{time_sec:.2f}s.jpg')
                print(f'检测到关键帧{out_path}\n')
                cv2.imwrite(out_path, frame)
                saved += 1
                frames.append((out_path, maxVal))  # 保存路径和相似度
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f'检测完成！共保存 {saved} 张关键帧到 {output_dir}/')
    group_and_select(frames, SSIM_THRESHOLD,output_dir)


def group_and_select(frames, ssim_threshold,output_dir):
    # 初始化分组
    groups = []
    current_group = [frames[0]]

    # 从第一帧开始，逐帧向后比较
    for i in range(1, len(frames)):
        current_frame = frames[i]
        previous_frame = frames[i-1]
        ssim_value = calculate_ssim(current_frame[0], previous_frame[0])
        if ssim_value > ssim_threshold:
            current_group.append(current_frame)
        else:
            groups.append(current_group)
            current_group = [current_frame]

    # 添加最后一组
    if current_group:
        groups.append(current_group)

    # 从每个组中选择与模板最相似的关键帧
    selected_frames = []
    for group in groups:
        if len(group) > 1:
            # 计算组内帧之间的 SSIM，选择与模板最相似的帧
            selected_frame = max(group, key=lambda x: x[1])
        else:
            selected_frame = group[0]
        selected_frames.append(selected_frame)

    # 移动未被选择的帧到一个单独的目录
    non_selected_dir = os.path.join(os.path.dirname(output_dir), 'non_selected')
    os.makedirs(non_selected_dir, exist_ok=True)
    for path, maxVal in frames:
        if (path, maxVal) not in selected_frames:
            shutil.move(path, non_selected_dir)
            print(f"移动未被选择的帧：{path} 到 {non_selected_dir}/")

    print(f"保留的关键帧数量：{len(selected_frames)}")


def calculate_ssim(image_path1, image_path2):
    """计算两幅图像的结构相似性指数"""
    img1 = imread(image_path1, as_gray=True)
    img2 = imread(image_path2, as_gray=True)
    # 指定 data_range 为图像数据的可能取值范围
    return ssim(img1, img2, data_range=img1.max() - img1.min())


# # 指定视频路径
# video_path = '/Users/tujiali/Desktop/v2.mov'
#keyframe_extract(video_path)


def extract_frames_around_keyframes(output_dir, video_path,  keyframe_paths):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'视频 {video_path} 无法打开')
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 遍历每个关键帧
    for keyframe_path in tqdm(keyframe_paths, desc='处理关键帧'):
        # 从文件名中提取时间（秒）
        time_str = os.path.splitext(os.path.basename(keyframe_path))[0].split('_')[-1].replace('s', '')
        keyframe_time = float(time_str)
        keyframe_idx = int(keyframe_time * fps)

        context_dir = os.path.join(output_dir, os.path.basename(keyframe_path).replace('.', '_'))
        os.makedirs(context_dir, exist_ok=True)

        # 计算前后 5 秒和 3 秒的时间索引
        two_seconds_before = max(0, keyframe_idx - 5 * fps)
        one_second_before = max(0, keyframe_idx - 3 *fps)
        current_frame_idx = keyframe_idx
        one_second_after = min(total_frames - 1, keyframe_idx + 3*fps)
        two_seconds_after = min(total_frames - 1, keyframe_idx + 5 * fps)

        # 提取并保存帧
        for frame_idx in [two_seconds_before, one_second_before, current_frame_idx, one_second_after,
                          two_seconds_after]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(frame_idx)
                frame_path = os.path.join(context_dir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(frame_path, frame)
                print(f'保存帧: {frame_path}')

    cap.release()
    print('处理完成。')

#keyframe_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 调用函数
# extract_frames_around_keyframes(output_dir, video_path,  keyframe_paths)