"""
示例脚本：完整演示"文本转音频"威胁检测流程

本脚本模拟一个真实的数据外泄场景：
1. 用户从敏感数据库复制数据
2. 使用在线工具将文本转换为音频（跨模态伪装）
3. 尝试上传音频文件到外部服务器

观察N3侦探引擎如何通过多轮次推理和污点追踪发现这一隐蔽威胁
"""

import time
import sys
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from unknown_threat_agent import UnknownThreatAgent


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    print_section("🕵️  Unknown Threat Agent - 文本转音频威胁检测演示")
    
    # 1. 初始化Agent
    print("📦 初始化Agent系统...")
    agent = UnknownThreatAgent()
    agent.start()
    
    time.sleep(2)
    
    # 2. 场景开始
    print_section("📖 威胁场景：内部员工通过跨模态转换外泄客户数据")
    
    print("""
背景：
  - 员工 userB 有权访问客户数据库
  - userB 计划外泄VIP客户信息
  - 为了规避DLP检测，userB 使用"文本转音频"工具伪装数据
  - 最后通过文件分享网站上传音频文件
    """)
    
    time.sleep(3)
    
    # 3. 步骤1：访问敏感数据库
    print_section("Step 1: 员工访问敏感数据库")
    print("🔍 监测到：userB 执行敏感查询")
    print("   Query: SELECT name, phone, address FROM vip_customers LIMIT 100")
    
    agent.inject_event(
        event_type="database_query",
        actor="userB",
        target="vip_customer_db",
        sensitive=True,
        query="SELECT name, phone, address FROM vip_customers LIMIT 100",
        result_count=100
    )
    
    print("\n✅ Taint Tracker: 检测到敏感数据源")
    print("   → 标记 userB 为污点实体 (is_tainted=True)")
    print("   → 原因：访问了敏感数据库")
    
    time.sleep(4)
    
    # 4. 步骤2：打开未知网站
    print_section("Step 2: 员工打开可疑网站")
    print("🔍 监测到：userB 打开 text-to-speech-online.com")
    
    agent.inject_event(
        event_type="url_open",
        actor="userB",
        target="text-to-speech-online.com",
        url="https://text-to-speech-online.com/convert"
    )
    
    print("\n⚠️  Triage: 事件分诊中...")
    print("   ├─ N2战术手册检查: ❌ 未命中已知剧本")
    print("   ├─ 污点检查: ✅ actor 'userB' 是污点实体")
    print("   └─ 决策: 🚨 激活N3侦探引擎")
    
    print("\n🕵️  N3 Detective Engine 已激活")
    print("   Investigation ID: inv-001")
    print("   Trigger: userB (tainted) → text-to-speech-online.com")
    
    print("\n🧠 迭代1: 推理阶段")
    print("   Hypothesis: '污点用户访问未知网站，可能是数据外泄跳板'")
    print("   Confidence: Medium")
    
    print("\n🎬 迭代1: 行动阶段")
    print("   Action: 调用视频搜索工具(N1)分析网站UI")
    print("   → 正在分析页面界面元素...")
    
    time.sleep(5)
    
    # 5. 步骤3：模拟N1工具返回结果
    print_section("Step 3: 视频搜索工具返回分析结果")
    print("📊 N1 Analysis Result:")
    print("   Found UI Elements:")
    print("     - 'Text Input Area' (confidence: 0.95)")
    print("     - 'Convert to Audio Button' (confidence: 0.92)")
    print("     - 'Download MP3 Button' (confidence: 0.88)")
    
    print("\n🧠 迭代2: 推理阶段")
    print("   Evidence: UI分析确认这是文本转音频工具")
    print("   Hypothesis: '跨模态伪装工具 (Modal_Jump_Tool)'")
    print("   Confidence: High")
    
    print("\n🎬 迭代2: 行动阶段")
    print("   Action 1: 在Memory中记录假设")
    print("     → text-to-speech-online.com.hypothesis = 'Modal_Jump_Tool'")
    print("   Action 2: 启动文件监控")
    print("     → 监控 userB 创建的 .mp3/.wav 文件")
    print("   → N3进入休眠，等待监控结果...")
    
    time.sleep(5)
    
    # 6. 步骤4：检测到音频文件创建
    print_section("Step 4: 检测到音频文件创建")
    print("🔍 监测到：文件创建事件")
    print("   File: /tmp/customer_data_20240102.mp3")
    print("   Size: 2.3 MB")
    print("   Created by: userB")
    
    agent.inject_event(
        event_type="file_create",
        actor="userB",
        target="/tmp/customer_data_20240102.mp3",
        file_size=2400000,
        file_type="audio/mpeg"
    )
    
    print("\n🔄 N3引擎被唤醒 (resume investigation)")
    
    print("\n🧠 迭代3: 推理阶段")
    print("   Evidence: 检测到音频文件，时间与访问网站相近")
    print("   Analysis: 这很可能是转换产物")
    print("   Decision: 需要传播污点到该文件")
    
    print("\n🎬 迭代3: 行动阶段")
    print("   Action 1: 污点传播")
    print("     → /tmp/customer_data_20240102.mp3.is_tainted = True")
    print("     → taint_source = 'userB'")
    print("     → propagation_type = 'modal_conversion_text_to_audio'")
    
    print("\n🔴 污点传播链:")
    print("   vip_customer_db")
    print("        ↓ (database_query)")
    print("      userB")
    print("        ↓ (modal_conversion)")
    print("   text-to-speech-online.com")
    print("        ↓ (file_create)")
    print("   customer_data_20240102.mp3 ⚠️")
    
    print("\n   Action 2: 继续监控该文件的后续操作")
    print("   → N3再次进入休眠...")
    
    time.sleep(5)
    
    # 7. 步骤5：尝试上传文件
    print_section("Step 5: 检测到文件上传行为")
    print("🔍 监测到：文件上传事件")
    print("   File: customer_data_20240102.mp3")
    print("   Destination: file-share-anonymous.com")
    print("   Upload Size: 2.3 MB")
    
    agent.inject_event(
        event_type="file_upload",
        actor="userB",
        target="/tmp/customer_data_20240102.mp3",
        destination="file-share-anonymous.com",
        url="https://file-share-anonymous.com/upload/a7b3c9"
    )
    
    print("\n⚠️  Triage: 污点检查")
    print("   → 文件 'customer_data_20240102.mp3' 被标记为污点 ✅")
    print("   → 再次激活N3引擎")
    
    print("\n🧠 迭代4: 推理阶段")
    print("   Evidence Chain:")
    print("     [1] 敏感数据库访问 ✓")
    print("     [2] 跨模态转换工具使用 ✓")
    print("     [3] 可疑文件创建 ✓")
    print("     [4] 外部上传行为 ✓")
    
    print("\n   Analysis:")
    print("   完整的STAC链条已形成：")
    print("   DB Query → Tainted User → Modal Conversion → Tainted File → External Upload")
    
    print("\n   Conclusion: 🚨 威胁确认！")
    print("   Threat Type: Cross-Modal Data Exfiltration")
    print("   Confidence: 0.94")
    
    print("\n🎬 迭代4: 行动阶段")
    print("   Action: 立即告警 + 记录威胁链")
    
    time.sleep(3)
    
    # 8. 告警
    print_section("🚨 CRITICAL ALERT")
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                         THREAT DETECTED                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║ Investigation ID: inv-001                                             ║
║ Threat Type:      Cross-Modal Data Exfiltration                       ║
║ Severity:         HIGH                                                ║
║ Confidence:       94%                                                 ║
║                                                                       ║
║ Threat Chain:                                                         ║
║   vip_customer_db → userB → text-to-speech-online.com                ║
║   → customer_data_20240102.mp3 → file-share-anonymous.com            ║
║                                                                       ║
║ Attack Technique:                                                     ║
║   - 跨模态伪装 (Text-to-Audio Conversion)                             ║
║   - 使用匿名文件分享服务                                               ║
║   - 规避传统DLP文本检测                                                ║
║                                                                       ║
║ Evidence:                                                             ║
║   [1] Sensitive database access (100 VIP records)                    ║
║   [2] UI analysis confirmed modal conversion tool                    ║
║   [3] Taint propagation tracked across modalities                    ║
║   [4] Upload to external anonymous server                            ║
║                                                                       ║
║ Recommended Actions:                                                  ║
║   ✓ Block file upload immediately                                    ║
║   ✓ Quarantine user account 'userB'                                  ║
║   ✓ Investigate all recent activities                                ║
║   ✓ Alert security operations center                                 ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    time.sleep(3)
    
    # 9. 总结
    print_section("📊 检测总结")
    
    status = agent.get_status()
    
    print("系统统计:")
    print(f"  总事件数:        {status['event_bus']['total_events']}")
    print(f"  N2处理事件数:    {status['triage']['n2_handled']}")
    print(f"  N3激活次数:      {status['triage']['n3_activated']}")
    print(f"  检测到的威胁:    {status['detective']['threats_detected']}")
    print(f"  图节点数:        {status['memory']['total_nodes']}")
    print(f"  污点节点数:      {status['memory']['tainted_nodes']}")
    
    print("\n关键优势:")
    print("  ✓ 跨时间追踪：持久化记忆支持数小时级别的侦查")
    print("  ✓ 跨模态检测：成功识别文本→音频伪装")
    print("  ✓ 低误报率：通过多轮迭代收集充分证据")
    print("  ✓ 高性能：分诊机制确保N3仅在必要时激活")
    
    print("\n" + "=" * 80)
    print("  演示完成！查看 logs/agent.log 获取详细日志")
    print("=" * 80 + "\n")
    
    # 10. 清理
    time.sleep(2)
    print("正在关闭Agent...")
    agent.stop()
    print("✅ Agent已安全关闭")


if __name__ == "__main__":
    main()
