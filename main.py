from extract_keyframe.keyframe_extract import keyframe_extract, extract_frames_around_keyframes
import os
from Frames_inference.frame_recongnize import process_directory
from Frames_inference.final_decision import result

# 指定视频路径
video_path = './videos/v2.mov'
# 关键帧保存目录
output_dir   = 'output2'
model_name = "qwen2.5-vl-72b-instruct"
use_api = True
# step1: 获取与模版匹配的帧
keyframe_extract(video_path, output_dir)
keyframe_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# step2： 对单帧前后扩充，丰富上下文， 所有的帧序列都保存在output_dir下
extract_frames_around_keyframes(output_dir, video_path,  keyframe_paths)

# step3：对每组帧，依次调用llm来挖掘数据操作
res = process_directory(output_dir,model_name,use_api)

# step4: 把前面的响应组合给llm来综合判定
res_all = result(res,model_name,use_api)



