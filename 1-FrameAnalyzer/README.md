# 风险操作分析模块

风险操作分析模块。从海量录屏帧中精准定位并分析特定行为的功能。实现了2个Tool API给Agent调用。
Tool-1: 检索前后相关帧，并描述，分析其行为。
Tool-2: 将某些帧分析得更细致一些。利用 MLLM+OCR，分析某一帧信息+行为。


## 🚀 核心功能

1.  **检索与keyword相关的帧并描述行为 (`relavance_frame.py`)**：keyword可能是文件名、应用名、一大段文字等。
    
    -   **三阶段过滤策略**：画面特征去重（ResNet50）-> 关键词文本过滤（OCR）-> 行为意图分析（Qwen2.5-VL）。
        
    -   **高效处理**：通过采样和特征比对大幅降低 VLM 的调用成本。
        
    -   **结构化报告**：自动生成包含时间轴和事件描述的 JSON 报告。
        
2.  **精细化场景溯源 (`analyze_scene_deep_dive`)**
    
    -   **深度推理**：针对特定关键帧序列进行文分析。
        
    -   **语义增强**：在发送给大模型前，先通过 OCR 提取环境文本作为辅助信息。
        
    -   **高精度意图识别**：利用 Qwen2.5-VL 的强大推理能力，还原操作细节。


## 🛠️ 环境准备

### 1. 依赖安装

确保您的环境已安装 Python 3.9+ 及 CUDA 驱动（推荐）。

Bash

```
pip install opencv-python torch torchvision pillow tqdm easyocr langchain-openai langchain-core

```

### 2. 模型与 API 配置

-   **Local Models**: 脚本会自动下载 ResNet50 权重。
    
-   **VLM API**: 本工具使用通义千问 `qwen2.5-vl-72b-instruct`。请确保拥有阿里云 DashScope 的 API Key。


## 💻 使用示例

### 场景 A：从视频中检索与文件名‘AAA公司员工守则’有关的帧并描述

Python

```
from your_script_name import analyze_video_behavior

analyze_video_behavior(
    target_keywords=['AAA公司员工守则'],
    video_path="office_surveillance.mp4",
    output_dir="./analysis_result/",
    similarity_threshold=0.98,
    sample_interval=1.0  # 每秒采样一帧
)

```

### 场景 B：对某些帧进行精细描述

Python

```
test_data = [
        {'frame': './output/frame_002773.jpg', 'frame_index': 2773, 'timestamp': 46.2},
        {'frame': './output/frame_002832.jpg', 'frame_index': 2832, 'timestamp': 47.2},
        {'frame': './output/frame_002891.jpg', 'frame_index': 2891, 'timestamp': 48.2},
        {'frame': './output/frame_003422.jpg', 'frame_index': 3422, 'timestamp': 57.0},
    ]
    
final_report = analyze_scene_deep_dive(
    filtered_frames=test_data,
    output_path="./output/security_deep_dive.json"
)
```


## 📊 输出结果说明

系统将生成一个 JSON 格式的报告：

### 1. 行为检索结果 (Scenario 1: `relavance_frame.py` Output)

该结果由第一个函数生成，主要用于在长视频中快速检索与文件名或应用名称或文本有关的事件片段，并进行操作描述。

JSON

```
{
    "total_events": 2,
    "events": [
        {
            "time_range": "46.2 - 57.0 (秒)",
            "involved_timestamps": [46.2, 47.2, 48.2, 57.0],
            "app_name": "QQ",
            "operation_type": "邮件附件外发",
            "description": "用户在QQ中选中了 'AAA公司员工守则.docx' 文件，并通过鼠标右键菜单选择了发送给联系人 '青山撞入怀'。从弹出的发送窗口到文件成功发送，整个过程在这些帧中被记录。"
        },
        {
            "time_range": "112.1 - 128.8 (秒)",
            "involved_timestamps": [112.1, 121.0, 122.0, 128.8],
            "app_name": "Kimi",
            "operation_type": "文档上传与查看",
            "description": "用户在Kimi应用中上传了名为'AAA公司员工守则'的Word文档，并在AI对话框中查看了该文档的内容。从点击上传到查看文档内容，整个过程覆盖了多个连续帧。"
        }
    ]
}

```

----------

### 2. 深度场景分析 (Scenario 2: `analyze_scene_deep_dive` Output)
对于某些帧再次精细化分析。

JSON

```
{
    "overall_summary": "用户在QQ聊天窗口中发送了一份名为'AAA公司员工守则.docx'的文件。",
    "environment": "QQ即时通讯软件，具体页面为与联系人'青山撞入怀'的聊天窗口",
    "action_chain": [
        {
            "timestamp": 46.2,
            "description": "用户在QQ聊天窗口中选择并上传了名为'AAA公司员工守则.docx'的文件。"
        },
        {
            "timestamp": 47.2,
            "description": "用户确认发送文件，点击了发送按钮。"
        },
        {
            "timestamp": 48.2,
            "description": "文件成功发送至联系人'青山撞入怀'。"
        },
        {
            "timestamp": 57.0,
            "description": "用户在聊天窗口中输入了一些文本内容。"
        }
    ],
    "risk_assessment": {
        "level": "高",
        "reasoning": "1. 关键OCR文字：'AAA公司员工守则.docx'，这可能是一份敏感的内部文件；2. 关键视觉动作：用户将文件拖拽至QQ聊天窗口并点击发送按钮；3. 意图推导逻辑：用户有明确的数据外发行为，将可能包含敏感信息的文件发送给外部联系人。",
        "hit_criteria": [
            "存在数据外发行为"
        ]
    },
    "final_intent": "用户意图将一份可能包含敏感信息的公司内部文件通过QQ发送给外部联系人'青山撞入怀'。"
}

```

----------

### 💡 字段说明

-   **time_range**: 事件在原始视频中发生的时间区间。
    
-   **action_chain**: 深度溯源模式下，模型对每一帧画面所代表的具体原子动作的推导。
    
-   **risk_assessment**: 结合 OCR 文本和视觉动作，由大模型生成的风险判定标准，可直接用于审计告警。

