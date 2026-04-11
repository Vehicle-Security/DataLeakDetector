# 01-FrameAnalyzer

> 类型：Module Spec  
> 版本：v2
> 本文档描述的是对现有 FrameAnalyzer 模块的定向重构，而不是全新推倒重写。请先审查旧项目中可复用的实现，包括抽帧、OCR、时间窗裁剪、VLM 调用与结果缓存；在此基础上重点修改以下部分：1）prompt，从泄露导向改为动作事实抽取；2）结构化输出字段，收敛为 segment 级结果；3）OCR 匹配对象，从仅文件名扩展为文件名/文本片段/应用或网站名/场景词；4）命中信号聚合，避免按单帧重复扩展片段。除非旧逻辑明显阻碍新模块职责，否则不要主动推翻现有流程骨架。

---

## 1. 模块概述

FrameAnalyzer 是系统中确定性的视频行为分析模块，负责从给定视频时间窗中识别界面操作片段，并输出结构化行为结果。

不负责：

* 视频采集与录制
* 下游风险判定与告警

---

## 2. 模块目标

- 自动化分析视频片段，定位与查询目标相关的操作行为
- 结构化输出行为事件，便于后续关联、追踪和推理。
- 相同输入应产生一致输出，保证结果可复现、可调试。

---

## 3. 核心职责

负责：

- 视频帧特征提取与去冗余
- OCR 检索与模糊匹配
- 上下文关键帧补充
- 调用大模型进行界面动作与可观察事实识别
- 结构化输出结果。

边界：

* 不负责视频采集、存储与播放
* 不负责最终风险判定与外部告警

---

## 4. 输入与输出

### 输入

```python
{
    "video_path": "str",
    "recording_start_time": "YYYY-MM-DD HH:MM:SS", # 录屏开始时间

    "time_window": {
        "start": "YYYY-MM-DD HH:MM:SS",# 检索起始时间
        "end": "YYYY-MM-DD HH:MM:SS"# 检索结束时间
    },

    "query": {
        "file_names": ["str", ...],        # 可选
        "text_snippets": ["str", ...],     # 可选
        "app_names": ["str", ...],         # 可选
        "scene_keywords": ["str", ...]     # 可选（上传/会议/聊天等）
    },

    "analysis_config": {
        "enable_fallback": False, #OCR 没命中时，是否还要用 VLM 再扫一遍
        "mode": "ocr_first"   # or "vlm_direct"
    }
}
```

### 输出

```python
{
    "time_window": {
        "start": "YYYY-MM-DD HH:MM:SS",
        "end": "YYYY-MM-DD HH:MM:SS"
    },

    "ocr_hit": True, # 表示是否存在 OCR 命中片段（用于区分 no_match 与 fallback 触发情况）

    "segments": [
        {
            "time_range": "str",
            "app_name": "str",
            "operation_type": "str",

            "primary_resource": "str",        # 若存在明确主资源则填写，否则为空
            "related_resources": ["str"],     # 可选

            "action_description": "str",      # 描述界面中实际发生的操作（不包含推理）

            "visible_evidence": ["str"],

            "supporting_timestamps": ["str"],
            "confidence": 0.0
        }
    ],

    "summary": {
        "apps": ["str"],
        "operations": ["str"],
        "resources": ["str"]
    },

    "status": "success" | "no_match" | "failed"
}
```

---

## 5. 核心流程

**1. 视觉预处理**：对时间窗内视频进行抽帧，去除冗余静止帧，保留界面变化附近的候选帧。

**2. OCR 过滤与命中提取**：对候选帧进行 OCR，匹配收到的文件名、文本片段、应用/网站名称和场景词，提取命中信号。

**3. 命中聚合与片段生成**：将时间上接近或重叠的命中信号进行聚合，并补充前后上下文关键帧，生成尽量少且语义相对独立的候选片段。

**4. VLM 行为分析**：对每个候选片段调用 VLM，识别界面中的操作类型与可观察事实，输出结构化 segment（不做风险判断与推理）。

**5. 结构化结果输出**：返回 segment 级结果与整体摘要；若无命中则返回 `no_match`，并按配置决定是否进行轻量回退检查当前时间窗。

---

## 6. 依赖与接口

上游：

* 视频文件（外部采集/存储模块）
* 查询配置（外部输入，可选文件名/文本/应用/场景词）

下游：

* 行为关联与推理模块（EventCorrelator / LeakReasoner）
* 统计与可视化模块

---

## 7. 异常处理

- 视频文件不存在或无法读取：抛出异常并返回 status: "failed"
- OCR/模型推理失败：捕获异常，报告中返回错误信息
- 某片段的VLM 返回空结果 / 无法解析 ，不中断流程，继续处理其他片段

- 未命中有效片段 → `status: no_match`

---

## 8. 规划

* 当前：定向重构旧实现，支持基于时间窗的视频行为检索与片段级分析；结合 OCR 与多模态模型进行分层处理，输出结构化行为证据，接口清晰，便于下游模块进行关联与推理。
