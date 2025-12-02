"""
变形图片搜索系统 - Gradio前端界面
"""
import gradio as gr
import os
import sys

# 设置 Gradio 临时目录到项目目录下，避免权限问题
GRADIO_TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradio_temp')
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR
print(f"Gradio 临时目录设置为: {GRADIO_TEMP_DIR}")

# 全局引擎变量，延迟初始化
engine = None

def get_engine():
    """延迟初始化处理引擎"""
    global engine # 定义全局变量，以便在函数内修改
    if engine is None:
        print("正在初始化系统...")
        try:
            from processing_engine import ProcessingEngine
            engine = ProcessingEngine(
                base_dir=os.path.dirname(os.path.abspath(__file__)),
                # top_k=100,  # 增加到100个候选帧，确保长视频中不遗漏正确匹配
                extraction_fps=10  # 保持10fps,符合甲方要求
            )
            print("系统初始化完成！")
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    return engine


def process_search(query_image, video_file, threshold, progress=gr.Progress()):
    """
    处理搜索请求
    
    Args:
        query_image: 上传的查询图像
        video_file: 上传的视频文件
        threshold: 匹配阈值
        progress: Gradio进度条
        
    Returns:
        结果组件列表: [summary] + [row1_visible, info1, img1, row2_visible, info2, img2, ...]
    """
    if query_image is None:
        return [gr.update(value="请上传查询图像！", visible=True)] + [gr.update(visible=False)] * 30
    
    if video_file is None:
        return [gr.update(value="请上传视频文件！", visible=True)] + [gr.update(visible=False)] * 30
    
    # 进度回调函数
    def update_progress(message, percentage=None):
        if percentage is not None:
            progress(percentage / 100, desc=message)
    
    try:
        # 获取引擎实例
        eng = get_engine()
        
        # 执行处理
        result = eng.process_query(
            query_image,
            video_file,
            threshold=threshold,
            progress_callback=update_progress
        )
        
        if result['status'] == 'error':
            return [gr.update(value=f"错误: {result['message']}", visible=True)] + [gr.update(visible=False)] * 15
        
        # 构建摘要信息
        summary = f"""
## 🎉 搜索完成！

**基本信息：**
- 查询图像: {result['query_image']}
- 视频文件: {result['video_file']}
- 匹配阈值: {result['threshold']}

**处理统计：**
- 总帧数: {result['total_frames']}
- 筛选候选: {result['candidates_screened']}
- 找到匹配: {result['matches_found']}

**所有匹配结果分数：**
"""
        
        # 添加所有分数列表
        if 'all_scores' in result and len(result['all_scores']) > 0:
            for score_info in result['all_scores'][:20]:  # 最多显示前20个
                summary += f"\n{score_info['rank']}. **时间 {score_info['timestamp_sec']}s** | "
                summary += f"分数: {score_info['score']:.2f} | "
                summary += f"全局相似度: {score_info['global_similarity']:.3f} | "
                summary += f"内点: {score_info['inliers']}/{score_info['total_matches']}"
            
            if len(result['all_scores']) > 20:
                summary += f"\n\n*...还有 {len(result['all_scores']) - 20} 个结果未显示*"
        
        # 准备返回结果
        updates = [gr.update(value=summary, visible=True)]
        
        # 最多显示10个匹配结果
        max_display = 10
        
        if result['matches_found'] > 0:
            import cv2
            import numpy as np
            
            for idx in range(max_display):
                if idx < len(result['results']):
                    match = result['results'][idx]
                    
                    # 构建该匹配的信息
                    match_info = f"""
### 匹配 {idx + 1}

**⏱️ 时间戳:** {match['timestamp_sec']}秒  
**📊 匹配分数:** {match['score']:.2f}  
**🌐 全局相似度:** {match['global_similarity']:.3f}  
**🎯 内点数:** {match['inliers']} / {match['total_matches']}
"""
                    
                    # 读取图像
                    try:
                        with open(match['image_url'], 'rb') as f:
                            img_array = np.frombuffer(f.read(), dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                # 显示Row、更新info和image
                                updates.append(gr.update(visible=True))  # Row可见
                                updates.append(gr.update(value=match_info, visible=True))  # info
                                updates.append(gr.update(value=img_rgb, visible=True))  # image
                            else:
                                # 隐藏整行
                                updates.append(gr.update(visible=False))  # Row
                                updates.append(gr.update(visible=False))  # info
                                updates.append(gr.update(visible=False))  # image
                    except Exception as e:
                        print(f"加载图像失败: {e}")
                        updates.append(gr.update(visible=False))  # Row
                        updates.append(gr.update(visible=False))  # info
                        updates.append(gr.update(visible=False))  # image
                else:
                    # 没有更多结果，隐藏这个匹配框
                    updates.append(gr.update(visible=False))  # Row
                    updates.append(gr.update(visible=False))  # info
                    updates.append(gr.update(visible=False))  # image
        else:
            # 没有匹配结果，隐藏所有匹配框
            for _ in range(max_display * 3):  # 每个匹配3个组件
                updates.append(gr.update(visible=False))
        
        return updates
    
    except Exception as e:
        import traceback
        error_msg = f"处理过程中发生错误:\n{str(e)}\n\n{traceback.format_exc()}"
        return [gr.update(value=error_msg, visible=True)] + [gr.update(visible=False)] * 30  # *30的意思是30个组件分别为10个匹配结果的3个组件 使之全部隐藏不可见


# 创建Gradio界面
def create_interface():
    # 自定义 CSS 样式 - 只保留滚动条样式
    custom_css = """
    .result-summary {
        overflow-y: auto !important;
    }
    """
    
    with gr.Blocks(title="变形图片搜索系统", css=custom_css) as demo: 
        gr.Markdown("<h1 style='text-align: center;'>🔍 变形图片搜索系统</h1>")
        gr.Markdown("<p style='text-align: center;'>该系统能够从视频中检索与查询图像匹配的帧，即使图像经过了复杂的变换。</p>")
        
        with gr.Row(equal_height=True):
            # 左侧：输入
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入")
                query_image = gr.Image(label="变形查询图像", type="filepath")
                video_file = gr.Video(label="视频文件")
                threshold_slider = gr.Slider(
                    minimum=0, maximum=100, value=65, step=1,
                    label="匹配阈值 (推荐60-80)"
                )
                search_btn = gr.Button("🚀 开始搜索", variant="primary", size="lg")
            
            # 右侧：输出摘要
            with gr.Column(scale=1):
                gr.Markdown("### 📤 处理结果摘要")
                result_summary = gr.Markdown(
                    value="等待处理...",
                    elem_classes=["result-summary"]
                )
        
        # 匹配结果区域 - 每个结果一行，左边信息右边图片
        gr.Markdown("---")
        gr.Markdown("## 🎯 匹配详情")
        
        # 创建10个匹配结果的显示区域
        match_outputs = []
        for i in range(10):
            with gr.Row(visible=False) as match_row:
                with gr.Column(scale=1):
                    match_info = gr.Markdown(f"匹配 {i+1}")
                with gr.Column(scale=2):
                    match_image = gr.Image(label=f"匹配 {i+1} 可视化", type="numpy")
            # 将Row、info和image都加入输出列表
            match_outputs.extend([match_row, match_info, match_image])
        
        # 绑定事件 - 输出包括摘要和所有匹配结果
        all_outputs = [result_summary] + match_outputs
        
        search_btn.click(
            fn=process_search,
            inputs=[query_image, video_file, threshold_slider],
            outputs=all_outputs
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
