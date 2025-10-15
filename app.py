import os
import shutil
import gradio as gr
from extract_keyframe.keyframe_extract import keyframe_extract, extract_frames_around_keyframes
from Frames_inference.frame_recongnize import process_directory
from Frames_inference.final_decision import result

# 配置常量
OUTPUT_DIR = 'output2'
#MODEL_NAME = "qwen2.5-vl-72b-instruct"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
USE_API = False


def clean_output_dir(directory):
    """清理输出目录"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def process_video(video_path, progress=gr.Progress()):
    """
    处理视频并生成关键帧

    Args:
        video_path: 视频文件路径
        progress: Gradio进度条对象

    Returns:
        tuple: (关键帧路径列表, 分析结果文本)
    """
    if video_path is None:
        return None, "❌ 错误: 请先上传视频文件"

    try:
        # 清理输出目录
        progress(0, desc="🧹 清理输出目录...")
        clean_output_dir(OUTPUT_DIR)

        # Step 1: 提取关键帧
        progress(0.2, desc="🎯 正在提取关键帧...")
        keyframe_extract(video_path, OUTPUT_DIR)

        # 获取关键帧路径
        keyframe_paths = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not keyframe_paths:
            return None, "⚠️ 警告: 未检测到任何关键帧，请检查视频内容"

        # Step 2: 扩充上下文帧
        progress(0.4, desc="📸 正在扩充上下文帧...")
        extract_frames_around_keyframes(OUTPUT_DIR, video_path, keyframe_paths)

        # Step 3: 分析每组帧
        progress(0.6, desc="🤖 正在进行AI分析...")
        analysis_results = process_directory(OUTPUT_DIR, MODEL_NAME, USE_API)

        # Step 4: 综合判定
        progress(0.9, desc="📊 生成最终报告...")
        result_text = result(analysis_results, MODEL_NAME, USE_API)

        progress(1.0, desc="✅ 处理完成!")

        # 格式化输出结果
        formatted_result = f"""
✅ 处理成功!

📊 检测到关键帧数量: {len(keyframe_paths)}

🔍 分析结果:
{result_text}

⏰ 处理完成时间: {os.popen('date').read().strip()}
        """.strip()

        return keyframe_paths, formatted_result

    except FileNotFoundError as e:
        error_msg = f"❌ 文件错误: {str(e)}\n请检查文件路径是否正确"
        return None, error_msg

    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}\n请检查视频格式或联系管理员"
        print(f"Error details: {e}")  # 服务端日志
        import traceback
        traceback.print_exc()
        return None, error_msg


# 创建Gradio界面
with gr.Blocks(
        title="数据泄漏检测系统",
        theme=gr.themes.Soft(),
        css="""
    .gradio-container {
        max-width: 1400px !important;
    }
    """
) as demo:
    # 标题区域
    gr.Markdown(
        """
        # 🎬 数据泄漏检测与分析系统

        自动提取关键帧并进行深度识别数据操作
        """
    )

    with gr.Row():
        # 左侧列：视频上传和预览
        with gr.Column(scale=1):
            gr.Markdown("### 📹 视频上传")
            video_input = gr.Video(
                label="上传视频文件",
                height=400,
                format="mp4"
            )

            with gr.Row():
                generate_btn = gr.Button(
                    "🚀 开始分析",
                    variant="primary",
                    size="lg",
                    scale=3
                )
                clear_btn = gr.Button(
                    "🗑️ 清空",
                    variant="secondary",
                    size="lg",
                    scale=1
                )

        # 右侧列：结果展示
        with gr.Column(scale=1):
            gr.Markdown("### 📊 分析结果")
            text_output = gr.Textbox(
                label="分析报告",
                placeholder="上传视频并点击'开始分析'按钮，AI分析结果将显示在这里...",
                lines=12,
                max_lines=20,
                show_copy_button=True
            )

    # 底部：关键帧展示区域
    gr.Markdown("### 🖼️ 关键帧预览")
    keyframes_gallery = gr.Gallery(
        label="提取的关键帧",
        show_label=False,
        columns=4,
        rows=2
    )

    # 绑定事件
    generate_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[keyframes_gallery, text_output],
        api_name="analyze_video"
    )

    # 清空按钮事件
    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=[],
        outputs=[video_input, keyframes_gallery, text_output]
    )

    # 使用说明
    with gr.Accordion("📝 使用说明", open=False):
        gr.Markdown(
            """
            ### 操作步骤

            1. **上传视频**: 点击上方区域选择或拖拽视频文件 (支持常见视频格式)
            2. **预览确认**: 视频上传后会自动预览，确认无误后继续
            3. **开始分析**: 点击「🚀 开始分析」按钮，系统将自动完成以下步骤:
               - 🎯 提取视频关键帧
               - 📸 扩充上下文帧序列
               - 🤖 AI深度分析
               - 📊 生成综合报告
            4. **查看结果**: 
               - 关键帧将在下方画廊中展示
               - 详细分析结果显示在右侧文本框
               - 可使用复制按钮保存结果

            ### 注意事项

            - ⚠️ 视频文件大小建议不超过 500MB
            - ⏱️ 处理时间取决于视频长度，请耐心等待
            - 🔄 处理过程中请勿刷新页面
            - 💾 结果仅临时保存，建议及时导出

            ### 技术支持

            遇到问题? 请联系技术支持或查看日志文件
            """
        )

    # 页脚信息
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            💡 Powered by Qwen2.5-VL | 🔒 数据安全保护 | 📧 技术支持
        </div>
        """
    )

# 启动应用
if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 启动应用
    demo.launch(
    share=False,
    server_port=7860,
    show_error=True,
    quiet=False
)