from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


OUT = Path(__file__).resolve().parent
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc" if Path(r"C:\Windows\Fonts\msyhbd.ttc").exists() else FONT


def ft(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


C = {
    "ink": "#102033",
    "muted": "#52616F",
    "line": "#4B5F76",
    "border": "#8EA3BA",
    "blue": "#DCEBFF",
    "sky": "#E0F7FF",
    "green": "#DCFCE7",
    "mint": "#D9FBEA",
    "yellow": "#FFF5C7",
    "amber": "#FFE8B5",
    "purple": "#EEE7FF",
    "red": "#FFD8D8",
    "gray": "#F5F7FA",
    "white": "#FFFFFF",
}


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for para in str(text).split("\n"):
        if not para:
            lines.append("")
            continue
        buf = ""
        for ch in para:
            candidate = buf + ch
            if draw.textlength(candidate, font=font) <= width or not buf:
                buf = candidate
            else:
                lines.append(buf)
                buf = ch
        if buf:
            lines.append(buf)
    return lines


def label_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.FreeTypeFont) -> None:
    x1, y1, x2, y2 = box
    lines = wrap(draw, text, font, x2 - x1 - 36)
    line_h = font.size + 7
    y = y1 + (y2 - y1 - len(lines) * line_h) / 2
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, fill=C["ink"], font=font)
        y += line_h


def box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    fill: str,
    title: str | None = None,
    font: ImageFont.FreeTypeFont | None = None,
    align: str = "center",
) -> None:
    font = font or ft(28)
    draw.rounded_rectangle(rect, radius=22, fill=fill, outline=C["border"], width=3)
    x1, y1, x2, y2 = rect
    if title:
        draw.text((x1 + 24, y1 + 14), title, fill=C["ink"], font=ft(25, True))
        y1 += 56
    lines = wrap(draw, text, font, x2 - x1 - 42)
    line_h = font.size + 7
    y = y1 + (y2 - y1 - len(lines) * line_h) / 2
    for line in lines:
        if align == "left":
            x = x1 + 24
        else:
            w = draw.textlength(line, font=font)
            x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, fill=C["ink"], font=font)
        y += line_h


def diamond(draw: ImageDraw.ImageDraw, cx: int, cy: int, w: int, h: int, text: str, fill: str) -> None:
    pts = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(pts, fill=fill, outline=C["border"])
    draw.line(pts + [pts[0]], fill=C["border"], width=3)
    label_center(draw, (cx - w // 3, cy - h // 3, cx + w // 3, cy + h // 3), text, ft(26, True))


def arrow(
    draw: ImageDraw.ImageDraw,
    a: tuple[int, int],
    b: tuple[int, int],
    label: str | None = None,
    color: str | None = None,
) -> None:
    color = color or C["line"]
    ax, ay = a
    bx, by = b
    draw.line((ax, ay, bx, by), fill=color, width=5)
    angle = math.atan2(by - ay, bx - ax)
    size = 18
    p1 = (bx - size * math.cos(angle - 0.5), by - size * math.sin(angle - 0.5))
    p2 = (bx - size * math.cos(angle + 0.5), by - size * math.sin(angle + 0.5))
    draw.polygon([b, p1, p2], fill=color)
    if label:
        f = ft(20)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        tw = draw.textlength(label, font=f)
        draw.rounded_rectangle((mx - tw / 2 - 10, my - 18, mx + tw / 2 + 10, my + 16), radius=8, fill="white")
        draw.text((mx - tw / 2, my - 16), label, fill=color, font=f)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((55, 42), title, fill=C["ink"], font=ft(44, True))
    draw.text((58, 100), subtitle, fill=C["muted"], font=ft(24))


def footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((60, 1435), text, fill=C["muted"], font=ft(20))


def save(img: Image.Image, name: str) -> None:
    img.save(OUT / name, quality=95)


def system_architecture() -> None:
    img = Image.new("RGB", (2400, 1500), "white")
    d = ImageDraw.Draw(img)
    header(d, "DataLeakDetector 总体架构", "日志优先缩小搜索空间，视觉模型确认完成态，符号推理输出可审计泄露路径")
    for x, label, fill in [
        (70, "证据输入", C["blue"]),
        (520, "稳定入口", C["sky"]),
        (910, "三大分析模块", C["yellow"]),
        (1700, "推理与报告", C["green"]),
    ]:
        d.rounded_rectangle((x, 150, x + 350, 1350), radius=30, fill=fill, outline="#C7D2E1", width=2)
        d.text((x + 25, 172), label, fill=C["ink"], font=ft(30, True))

    box(d, (110, 270, 380, 420), "文件 / 窗口 / 剪贴板 / 网络日志", C["white"], title="系统日志")
    box(d, (110, 500, 380, 650), "屏幕录制视频\nmp4 / mov", C["white"], title="录屏")
    box(d, (110, 730, 380, 880), "敏感文件\n黑白名单\n策略配置", C["white"], title="策略")
    box(d, (560, 300, 830, 500), "统一编排\n加载日志、视频与配置\n串联三阶段", C["white"], title="main/run_e2e.py")
    box(d, (560, 650, 830, 850), "稳定导入适配层\n隐藏历史目录重排", C["white"], title="data_leak_detector")
    box(d, (965, 240, 1320, 430), "LogFirstDetector\n确定性日志规则\nVLM fallback gate\n代表帧抽样", C["white"], title="01-FrameAnalyzer")
    box(d, (965, 520, 1320, 710), "前台应用识别\n敏感窗口构建\n多跳 lineage 绑定", C["white"], title="02-EventCorrelator")
    box(d, (965, 800, 1320, 990), "Datalog facts\nOpen / Transfer / CrossProcess / Leak", C["white"], title="03-LeakReasoner")
    box(d, (1385, 385, 1645, 560), "qwen3.7-plus\n视觉完成态确认", C["amber"], title="Live VLM")
    box(d, (1385, 675, 1645, 850), "外发候选\n证据引用\n对象绑定", C["purple"], title="证据包")
    box(d, (1750, 320, 2130, 520), "泄露路径\n完整证据链\n可解释结论", C["white"], title="证据报告")
    box(d, (1750, 650, 2130, 850), "NAS 全量评测\n分诊 / 确定性 / 最终口径\n逐例 TP / FP / TN / FN 日志", C["white"], title="评测")
    box(d, (1750, 980, 2130, 1180), "Typst 报告\nJSON 结果\n图表与复现实验", C["white"], title="报告产物")
    box(d, (110, 1100, 720, 1295), "蓝色：原始证据与输入\n黄色：日志/视觉分析\n绿色：符号推理与可审计输出\n紫色：中间证据包与评测数据", C["gray"], title="图例", align="left")

    arrow(d, (380, 345), (560, 380), "日志")
    arrow(d, (380, 575), (965, 335), "视频")
    arrow(d, (380, 805), (560, 745), "策略")
    arrow(d, (830, 400), (965, 335), "编排")
    arrow(d, (830, 745), (965, 615), "统一导入")
    arrow(d, (1320, 335), (1385, 470), "可疑窗口")
    arrow(d, (1320, 615), (1385, 760), "lineage")
    arrow(d, (1645, 470), (1645, 760), "视觉证据")
    arrow(d, (1645, 760), (1750, 420), "候选事实")
    arrow(d, (1320, 895), (1750, 420), "Datalog")
    arrow(d, (1940, 850), (1940, 980), "汇总")
    footer(d, "对应 introduce.typ：总体架构图。重点展示重构后的 01/02/03 模块职责与 main/data_leak_detector 稳定入口。")
    save(img, "system_architecture.png")


def evidence_graph() -> None:
    img = Image.new("RGB", (2400, 1500), "white")
    d = ImageDraw.Draw(img)
    header(d, "轻量证据图模型", "把日志、窗口、文件派生、视觉证据和 Datalog facts 统一成可追踪的审计链")
    box(d, (90, 260, 390, 430), "进程、窗口、文件系统、剪贴板、网络事件", C["blue"], title="原始日志事件")
    box(d, (90, 560, 390, 720), "OCR/VLM 看到的页面、按钮、附件、聊天内容、远端列表", C["yellow"], title="视觉片段")
    box(d, (520, 180, 820, 330), "原始敏感文件\n配置命中 / 日志推断", C["green"], title="SensitiveFile")
    box(d, (520, 430, 820, 580), "派生文件\n导出、压缩、重命名、复制", C["mint"], title="DerivedArtifact")
    box(d, (520, 680, 820, 830), "内容对象\n剪贴板文本、截图、OCR 表格", C["amber"], title="ContentArtifact")
    box(d, (950, 260, 1250, 430), "前台应用\n邮箱 / AI / 网盘 / 会议 / VM", C["sky"], title="FrontendApp")
    box(d, (950, 560, 1250, 720), "复核时间窗\nanchor + post buffer\n候选事件", C["sky"], title="AnalysisWindow")
    box(d, (1380, 180, 1700, 345), "对象绑定\n源对象 -> 当前对象 -> 外部汇", C["purple"], title="外发候选")
    box(d, (1380, 470, 1700, 635), "Datalog 事实\nOpen / Transfer\nCrossProcess / Leak", C["gray"], title="推理事实")
    box(d, (1380, 760, 1700, 925), "逐例评测记录\n期望结果 / 最终结果\nTP FP TN FN", C["red"], title="样本判定")
    box(d, (1830, 360, 2240, 560), "最终解释\n泄露文件、外发进程、外发渠道、完整路径", C["green"], title="泄露路径与报告")
    arrow(d, (390, 345), (520, 255), "打开/命中")
    arrow(d, (390, 345), (950, 335), "窗口分类")
    arrow(d, (820, 255), (520, 505), "派生")
    arrow(d, (820, 505), (520, 755), "内容化")
    arrow(d, (390, 640), (950, 640), "支撑复核")
    arrow(d, (1250, 335), (950, 640), "可疑上下文")
    arrow(d, (1250, 640), (1380, 262), "绑定")
    arrow(d, (820, 505), (1380, 262), "当前对象")
    arrow(d, (820, 255), (1380, 262), "源对象")
    arrow(d, (1700, 262), (1700, 552), "事实注入")
    arrow(d, (1700, 552), (1830, 460), "推理")
    arrow(d, (1700, 842), (1830, 500), "统计")
    box(d, (120, 1050, 690, 1265), "确定性证据：日志可直接连接敏感对象与上传/外部传输时，不依赖 VLM。", C["gray"], title="Deterministic", align="left")
    box(d, (825, 1050, 1395, 1265), "视觉证据：日志只能提示可疑窗口时，VLM 判断是否已经完成发送、上传、共享或复制。", C["gray"], title="VLM Completion", align="left")
    box(d, (1530, 1050, 2100, 1265), "审计闭环：每条最终判断都保留对象绑定、证据来源、帧选择和推理路径。", C["gray"], title="Audit Trail", align="left")
    footer(d, "对应 introduce.typ：轻量证据图。强调不是单点告警，而是从对象、上下文、视觉证据到 Datalog 的证据链。")
    save(img, "evidence_graph.png")


def vlm_gate_flow() -> None:
    img = Image.new("RGB", (2400, 1500), "white")
    d = ImageDraw.Draw(img)
    header(d, "VLM 触发门控与最终判定", "先用日志规则保证精度，再把无法确定但可疑的样本送入视觉复核")
    box(d, (120, 160, 440, 300), "NAS 样本\n日志 + 视频 + 标注", C["blue"])
    box(d, (580, 160, 900, 300), "LogFirstDetector\n敏感锚点、派生映射、上传事件", C["sky"])
    diamond(d, 1080, 230, 260, 160, "日志能否确认外发?", C["amber"])
    box(d, (1260, 130, 1570, 305), "日志直接命中\n进入最终结果\n通常精度高", C["green"])
    box(d, (580, 470, 900, 650), "Fallback Gate\nAI/邮箱/网盘/IM/会议\n剪贴板/截图/VM/远程桌面", C["sky"])
    diamond(d, 1080, 560, 260, 160, "是否需要 VLM?", C["amber"])
    box(d, (1260, 470, 1570, 650), "跳过 VLM\n无敏感上下文或纯噪声\n按日志结果收束", C["gray"])
    box(d, (240, 830, 560, 1010), "构造复核窗口\nanchor 前后文\npost buffer", C["blue"])
    box(d, (720, 830, 1040, 1010), "代表帧选择\n候选帧 + 场景变化\n首/中/尾上下文", C["yellow"])
    box(d, (1200, 830, 1520, 1010), "Live VLM\n确认完成态\n过滤草稿/取消/未完成", C["amber"])
    box(d, (1680, 830, 2000, 1010), "EventCorrelator\n视觉片段 -> 外发候选", C["purple"])
    box(d, (1680, 1140, 2050, 1300), "最终分类\nTP / FP / TN / FN\n实时写入日志", C["red"])
    arrow(d, (440, 230), (580, 230))
    arrow(d, (900, 230), (950, 230))
    arrow(d, (1210, 230), (1260, 220), "是")
    arrow(d, (1080, 310), (740, 470), "否")
    arrow(d, (900, 560), (950, 560))
    arrow(d, (1210, 560), (1260, 560), "否")
    arrow(d, (1080, 640), (400, 830), "是")
    arrow(d, (560, 920), (720, 920))
    arrow(d, (1040, 920), (1200, 920))
    arrow(d, (1520, 920), (1680, 920))
    arrow(d, (1840, 1010), (1840, 1140))
    arrow(d, (1570, 220), (1840, 1140), "det=1")
    arrow(d, (1570, 560), (1840, 1140), "skip")
    box(d, (120, 1120, 620, 1320), "分诊阶段 = 日志已命中 或 需要 VLM 复核。目标是尽量不漏掉风险样本。", C["gray"], title="召回优先", align="left")
    box(d, (720, 1120, 1220, 1320), "确定性阶段 = 只靠日志规则直接判断，通常精确率高但召回率较低。", C["gray"], title="精度优先", align="left")
    box(d, (1320, 1120, 1660, 1320), "最终口径 = 如果开启 live VLM，则结合 VLM/Correlator 后给出最终分类。", C["gray"], title="最终口径", align="left")
    footer(d, "对应 introduce.typ：VLM 触发门控。新增 VM/远程桌面视觉上下文与逐 case 日志，但主图关注整体判定流程。")
    save(img, "vlm_gate_flow.png")


def frame_sampling_strategy() -> None:
    img = Image.new("RGB", (2400, 1500), "white")
    d = ImageDraw.Draw(img)
    header(d, "代表帧抽样策略（补充图）", "只在 VLM 复核窗口内抽少量高价值帧，减少调用成本并保留完成态证据")
    box(d, (100, 200, 410, 360), "VLM 复核窗口\n来自 fallback meta", C["blue"])
    box(d, (540, 200, 850, 360), "候选时间点\n按窗口时长分配\n默认 max(24, 6x)", C["sky"])
    box(d, (980, 200, 1290, 360), "稀疏解码\n只 seek 候选帧\n不全量扫视频", C["yellow"])
    box(d, (1420, 120, 1740, 280), "96x54 灰度缩略图\n低成本视觉签名", C["yellow"])
    box(d, (1420, 360, 1740, 520), "场景变化分数\nmean(absdiff)/255", C["amber"])
    box(d, (1860, 220, 2220, 430), "代表帧集合\nwindow_start / mid / end\n+ scene_change", C["green"])
    box(d, (760, 760, 1080, 940), "JPEG/base64\nmax_edge + quality", C["green"])
    box(d, (1200, 760, 1520, 940), "VLM payload\n时间、源帧号\n选择原因、分数", C["purple"])
    box(d, (1640, 760, 1980, 940), "JSON 审计字段\nframe_selection[]", C["red"])
    arrow(d, (410, 280), (540, 280))
    arrow(d, (850, 280), (980, 280))
    arrow(d, (1290, 280), (1420, 200), "候选帧")
    arrow(d, (1580, 280), (1580, 360))
    arrow(d, (1740, 440), (1860, 325))
    arrow(d, (2040, 430), (920, 760), "入选帧")
    arrow(d, (1080, 850), (1200, 850))
    arrow(d, (1520, 850), (1640, 850))
    arrow(d, (2040, 430), (1810, 760), "审计")
    box(d, (160, 1100, 760, 1300), "为什么不是均匀抽帧？\n完成态 UI 往往很短，均匀点容易落在等待、编辑或静止页面。", C["gray"], align="left")
    box(d, (900, 1100, 1500, 1300), "为什么不用重模型？\n日志已经限制了窗口，用 OpenCV 缩略图差异即可近似 scene change。", C["gray"], align="left")
    box(d, (1640, 1100, 2240, 1300), "如何调参？\n增加 DLD_VLM_REVIEW_CANDIDATE_FRAMES 可提高短暂状态命中率。", C["gray"], align="left")
    footer(d, "补充图：不替代 introduce.typ 的主三图，可用于 frame_sampling_design.md 或后续报告附录。")
    save(img, "frame_sampling_strategy.png")


if __name__ == "__main__":
    system_architecture()
    evidence_graph()
    vlm_gate_flow()
    frame_sampling_strategy()
    print("redrawn chinese report figures")
