# EvidenceTracer - 敏感资源跨过程追踪模块

## 📌 模块简介

**EvidenceTracer** 是 ScreenGuard 数据泄露检测系统的第二个核心模块，负责跨过程追踪敏感资源的流转和变化。

### 模块定位
- **输入**：RiskSieve 输出的敏感操作片段（操作类型、对象、起止时间、关键帧）
- **输出**：以资源为中心的操作链路与证据
- **核心能力**：追踪文件重命名、压缩、加密、截图、格式转换等隐蔽操作，构建完整的数据流转证据链

### 与其他模块的关系
```
RiskSieve (录屏分析) → EvidenceTracer (资源追踪) → ThreatHunter (威胁判断)
```

**注意**：EvidenceTracer 不直接与系统日志或视频交互，只处理 RiskSieve 的结构化输出。

---

## 🏗️ 架构设计

本模块基于 **LangGraph** 构建，采用 ReAct (Reasoning-Acting-Observing) 模式，通过 LLM + Tools 实现智能资源追踪。

### 核心组件

## 🚀 使用方法

### 安装依赖

```bash
pip install langgraph langchain-openai python-dotenv
```

### 环境配置

创建 `.env` 文件：

```env
MODEL_NAME=qwen2-vl-72b-instruct
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your_api_key_here
TEMPERATURE=0.01
```

### 基本使用

```python
from graph import run_evidence_tracer

# 准备输入：来自 RiskSieve 的操作片段
input_operations = [
    {
        "operation_id": "op_001",
        "operation_type": "file_access",
        "resource_name": "机密报告.pdf",
        "app_name": "Adobe Reader",
        "start_time": "10:23:15",
        "end_time": "10:23:45",
        "keyframes": ["/path/to/frame1.jpg"],
        "raw_description": "用户打开了机密报告.pdf并浏览内容"
    },
    {
        "operation_id": "op_002",
        "operation_type": "file_compress",
        "resource_name": "report.zip",
        "app_name": "7-Zip",
        "start_time": "10:25:10",
        "end_time": "10:25:20",
        "keyframes": ["/path/to/frame2.jpg"],
        "raw_description": "用户将机密报告.pdf压缩为report.zip并设置密码"
    },
    {
        "operation_id": "op_003",
        "operation_type": "file_upload",
        "resource_name": "report.zip",
        "app_name": "Chrome",
        "start_time": "10:26:00",
        "end_time": "10:26:30",
        "keyframes": ["/path/to/frame3.jpg"],
        "raw_description": "用户在Chrome中上传report.zip到云存储"
    }
]

# 运行分析
result = run_evidence_tracer(input_operations, max_iterations=10)

# 输出结果
print(result)
```

### 输出格式

```json
{
    "tracked_resources": [
        {
            "resource_id": "res_001",
            "resource_name": "机密报告.pdf",
            "resource_type": "document",
            "first_seen": "op_001",
            "last_seen": "op_002",
            "derived_resources": ["report.zip"]
        },
        {
            "resource_id": "res_002",
            "resource_name": "report.zip",
            "resource_type": "archive",
            "first_seen": "op_002",
            "last_seen": "op_003",
            "derived_resources": []
        }
    ],
    "evidence_chains": [
        {
            "chain_id": "chain_001",
            "root_resource": "机密报告.pdf",
            "operations": ["op_001", "op_002", "op_003"],
            "risk_indicators": [
                "file_obfuscation",
                "encryption_before_transfer",
                "cross_application_transfer"
            ]
        }
    ],
    "summary": "追踪到敏感文件从访问、压缩加密到上传的完整链路"
}
```

---

## 🔍 关键特性

### 1. 高效的资源识别
- ✅ **直接使用结构化字段**：优先读取 `resource_name`，避免重复识别
- ✅ **补充发现额外资源**：从描述中识别源文件、派生文件（如"将A压缩为B"中的A）
- ✅ **处理边缘情况**：当 RiskSieve 未准确识别时，从描述中提取

### 2. 操作类型深度分析
自动识别以下操作类型：
- ✅ 文件访问、复制、重命名
- ✅ 文件压缩、加密
- ✅ 截图、文本复制
- ✅ 格式转换、导出
- ✅ 文件上传、分享

### 3. 风险指标检测
- `file_obfuscation`: 文件脱敏（重命名、压缩）
- `encryption_before_transfer`: 传输前加密
- `format_change`: 格式转换
- `content_extraction`: 内容提取（截图、复制）
- `cross_application_transfer`: 跨应用传输

### 4. 证据链构建
- 广度优先搜索追踪资源流转
- 自动识别派生资源
- 时间序列重建

---

## 📊 典型场景

### 场景 1：文件重命名后上传
```
机密文件.pdf → 工作文档.pdf → 上传到个人邮箱
```
**检测能力**：通过文件基础名称匹配识别重命名关系

### 场景 2：压缩加密后外发
```
敏感报告.docx → 压缩为 report.zip (加密) → 上传到云盘
```
**检测能力**：识别 `file_obfuscation` + `encryption_before_transfer` 风险指标

### 场景 3：截图+文本复制
```
查看机密数据库 → 截图保存 → 复制文本 → 粘贴到外部应用
```
**检测能力**：追踪内容提取操作，建立源数据与派生内容的关联

### 场景 4：格式转换链
```
源代码.py → 导出为 .txt → 重命名为 config.log → 上传
```
**检测能力**：识别多步转换，恢复完整流转路径

---


## 🧪 测试

运行内置测试用例：

```bash
python graph.py
```

这将执行一个包含文件访问→压缩加密→上传的完整测试场景。

---

## 🔗 与其他模块集成

### 从 RiskSieve 接收输入
```python
from risksieve import RiskSieveAnalyzer
from graph import run_evidence_tracer

# 第一步：RiskSieve 分析录屏
risksieve_result = RiskSieveAnalyzer.analyze(video_path, system_logs)
sensitive_operations = risksieve_result['operations']

# 第二步：EvidenceTracer 追踪资源
evidence_result = run_evidence_tracer(sensitive_operations)
```

### 输出给 ThreatHunter
```python
from threathunter import ThreatHunter

# 第三步：ThreatHunter 风险判断
threat_result = ThreatHunter.evaluate(evidence_result)
```

---

## 📝 注意事项

1. **不要直接处理原始日志和视频**  
   EvidenceTracer 的设计哲学是只处理结构化的操作描述，保持模块职责单一。

2. **关注资源流转，而非风险判断**  
   风险评估由 ThreatHunter 完成，本模块只负责客观记录和追踪证据。

3. **LLM 能力要求**  
   需要支持 Function Calling 的模型（如 GPT-4、Claude 3+）。

4. **迭代次数控制**  
   对于复杂的多步操作链，可能需要调整 `max_iterations` 参数。

---

## 🛠️ 开发路线图

- [ ] 支持更多资源类型（音频、视频）
- [ ] 优化文件相似度匹配算法
- [ ] 添加可视化证据链展示
- [ ] 支持增量分析（在线追踪）
- [ ] 与数据库集成，持久化追踪结果
