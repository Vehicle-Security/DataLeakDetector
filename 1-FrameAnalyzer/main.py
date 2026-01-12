import logging
import os
from unified_retriever import ContentRetriever
from blacklist_analyzer import BlacklistAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainWorkflow")

def run_analysis():
    
    retriever = ContentRetriever()
    bl_analyzer = BlacklistAnalyzer()

    # ========================================================
    # 场景 1: 文件名称检索分析
    # ========================================================
    logger.info("🚀 [阶段 1/3] 开始执行：文件名称检索分析...")
    kw_results = retriever.analyze(
        rec_start_str='2025-12-28 18:55:36',
        s_start_str='2025-12-28 18:55:46',
        s_end_str='2025-12-28 18:56:05',
        target_keywords=['项目1详细规划'],
        video_path="../video/43.mp4"
    )
    
    if kw_results and kw_results.get("events"):
        logger.info(f"✅ 场景 1 完成：检测到 {len(kw_results['events'])} 个行为事件。")
        events = kw_results.get("events", [])
        first_event = events[0]
        logger.info(f"第一个事件的应用名称是: {first_event.get('app_name', '未知')}")
        logger.info(f"第一个事件的操作类型是: {first_event.get('operation_type', '未知')}")
        logger.info(f"第一个事件的行为类别是: {first_event.get('behavior_category', '未知')}")
        logger.info(f"第一个事件的变更前文件名是: {first_event.get('original_filename', '未知')}")
        logger.info(f"第一个事件的变更后文件名是: {first_event.get('modified_filename', '未知')}")
        logger.info(f"第一个事件的描述是: {first_event.get('description', '无')}")
    else:
        logger.warning("⚠️ 场景 1 未发现匹配事件。")

    # ========================================================
    # 场景 2: 长文本匹配分析
    # ========================================================
    logger.info("\n🚀 [阶段 2/3] 开始执行：长文本匹配分析...")
    long_text_content = """我总结了当前四类主流防护工具的局限性：
        EPR/IPS：只能记录系统调用，粒度过粗
        DLP：依赖内容匹配，能识别的场景少。
        UAM/UEBA：侧重行为元数据分析，但无法理解用户操作的具体内容和意图。
        总的来说，现有方法对非结构化视觉内容（如图片、视频）和跨应用场景的覆盖严重不足。

        因此，本研究提出三个核心研究问题：
        如何从视频流中高效捕捉关键操作事件？
        如何识别用户是否在执行敏感操作？
        如何追踪隐私数据的传播路径？"""
    
    text_results = retriever.analyze(
        rec_start_str='2025-12-28 10:28:00',
        s_start_str='2025-12-28 10:28:00',
        s_end_str='2025-12-28 10:28:40',
        target_text=long_text_content, 
        video_path="../video/paste.mp4"
    )
    
    if text_results and text_results.get("events"):
        logger.info(f"✅ 场景 2 完成：已识别长文本传播路径。")
        
    else:
        logger.warning("⚠️ 场景 2 未匹配到目标文本。")

    # ========================================================
    # 场景 3: 黑名单应用/套壳检测（会自动保存 JSON，分批处理）
    # ========================================================
    logger.info("\n🚀 [阶段 3/3] 开始执行：黑名单应用扫描...")
    bl_results = bl_analyzer.analyze_blacklist(
        rec_start='2026-01-05 10:00:00',
        s_start='2026-01-05 10:00:00',
        s_end='2026-01-05 10:01:20',
        video_path="../video/wrapped_app.mp4",
        batch_size=6
    )
    
    if bl_results and bl_results.get("events"):
        logger.warning(f"🚨 场景 3 警告：发现 {len(bl_results['events'])} 个违规应用事件！")
    else:
        logger.info("✅ 场景 3 安全：未发现黑名单应用。")

if __name__ == "__main__":
    
    if not os.path.exists("../video/"):
        logger.error("错误：找不到视频目录 '../video/'，请检查路径设置。")
    else:
        try:
            run_analysis()
            logger.info("\n✨✨ 所有分析任务已执行完毕，请在各自输出目录下查看 JSON 报告。 ✨✨")
        except Exception as e:
            logger.error(f"❌ 程序运行崩溃: {e}", exc_info=True)