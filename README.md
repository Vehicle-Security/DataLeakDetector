# DataLeakDetector

DataLeakDetector 是一个面向 `spec/data` 真实样本目录的数据泄漏证据分析流水线。
当前项目已经收敛为单包结构，日志、视频帧、OCR/VLM 视觉证据、事件关联、
符号推理和可选 Neo4j 图谱写入都从同一套模型与配置进入。

```text
spec/data 样本目录
  -> logs/logs.json 或 logs/keyevents.json
  -> video/*.mp4
  -> groundtruth.json 中声明的初始敏感源文件
  -> FrameAnalyzer：日志时间窗 + 非均匀关键帧 + OCR 预筛 + VLM
  -> EventCorrelator：文件血缘 + 前端应用识别 + 泄漏出口候选
  -> LeakReasoner：符号化污点传播
  -> 可选 Neo4j 证据图谱
  -> JSON 报告
```

初始敏感文件只来自 `groundtruth.json` 或显式传入的 `--sensitive-file`。
敏感文件被打开、转换、复制、上传后产生的文件都视为衍生文件，由血缘分析和推理
得出，不写入初始敏感文件表。

当前 `conclusion` 的评估口径优先以 `groundtruth.json` 为准：只要标注中存在
符合 `spec/config/groundtruth_policy.json` 的泄密操作，报告结论就是
`data_leak_risk_detected`；没有标注文件时才回退到推理器结果。推理器自身判断会
保留在 `leak_reasoner.detector_conclusion`，方便后续评估召回和误报。
`groundtruth` 不参与证据生成，它只解释当前数据集的标签口径。

项目区别于传统日志规则检测的核心链路是：

1. 从日志定位可疑时间窗，而不是全视频均匀抽帧。
2. 在可疑时间窗内按画面变化选择关键帧，捕捉应用切换、上传弹窗、粘贴内容等状态变化。
3. 用 OCR 做低成本预筛，只把低置信或高风险帧交给 VLM。
4. 用 VLM 补全日志缺失的前端应用、屏幕内容、文件名和外发动作事实。
5. 将日志事实和视觉事实统一成 Datalog 事实，做可解释的污点传播推理。

报告中的 `detection_core` 会单独展示这条主链路的抽帧、VLM 补全和 Datalog 推理统计。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

如果要启用本地视频抽帧和 OCR，可安装视觉相关依赖：

```powershell
python -m pip install -e ".[dev,vision]"
```

## 运行真实样本

直接运行 `spec/data` 下的样本目录：

```powershell
python main/run_e2e.py --case spec\data\nas_samples\stage1\0-normal-ai-chatgpt-1
```

启用视觉分析：

```powershell
python main/run_e2e.py --case spec\data\nas_samples\stage2\2-filestruct-pdfconvert-2 --vision --max-vlm-frames 8
```

使用 `--case` 时，程序会自动发现：

- 优先使用 `logs/logs.json`，不存在时回退到 `logs/keyevents.json`
- `video/*.mp4`
- `groundtruth.json`
- `groundtruth.json` 中声明的初始敏感源文件

## OCR/VLM 流程

视觉分析默认关闭，用来避免不必要的模型调用成本。启用后流程如下：

1. 根据日志中的敏感文件、传输行为和泄漏出口行为定位可疑时间窗。
2. `frames.py` 使用画面变化抽取关键帧，不做简单均匀抽帧。
3. `ocr.py` 对关键帧做 OCR。
4. 高置信 OCR 结果直接作为本地证据。
5. 低置信或可疑 OCR 帧再送入 VLM，并受 `DLD_MAX_VLM_FRAMES` 限制。
6. `parser.py` 统一解析 Qwen/OpenAI 兼容接口返回的 JSON。
7. 视觉观察结果进入 `EventCorrelator`，即使日志里没有泄漏文件路径，也能补充 Datalog 事实。

## 配置

密钥只放在本地 `.env`，不要提交到仓库。

```text
DLD_VISION_ENABLED=1
DLD_OCR_PROVIDER=tesseract
DLD_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DLD_VLM_MODEL=qwen-vl-max-latest
DLD_VLM_API_KEY=...
DLD_MAX_VLM_FRAMES=8
```

不同数据集的策略词可以通过环境变量追加，不需要改代码：

```text
DLD_POLICY_CONFIG=spec/config/policy.json
DLD_GROUNDTRUTH_POLICY_CONFIG=spec/config/groundtruth_policy.json
DLD_SENSITIVE_TOKENS=prototype,pricing
DLD_TRANSFER_TOKENS=watermark,print
DLD_SINK_TOKENS=slack,github issue
```

`spec/config/policy.json` 是主要策略入口，包含敏感语义词、文件传输动作、外部汇聚点、
风险等级、前端应用识别和语义别名。代码里的策略只作为配置缺失时的最小兜底，
不要为了适配新数据集去堆 Python 关键词。

`spec/config/groundtruth_policy.json` 是当前样本标注的结论口径，负责解释
`groundtruth.json` 里的操作文本。以后如果数据集把“泄密”“正常”“未知风险”的
标注方式换掉，优先改这个文件。

初始敏感源文件的提取也可以配置。这里应该只指向标注文件里的“源文件”字段，
不要把转换后、复制后、上传前临时文件等衍生字段加进去。空值会使用内置默认值，
所以换数据集时通常只需要调整字段名或 JSON 路径。

```text
DLD_SENSITIVE_SOURCE_FIELDS=sensitive_file_path,sensitive_file,sensitive_path,source_file
DLD_SENSITIVE_SOURCE_JSON_PATHS=operations.*.sensitive_file_path
DLD_SENSITIVE_SOURCE_REGEXES=
```

## Neo4j

```powershell
tools\start_neo4j.ps1
python main/run_e2e.py --case spec\data\nas_samples\stage1\0-normal-ai-chatgpt-1 --neo4j --neo4j-strict
tools\stop_neo4j.ps1
```

## 关键文件

| 路径 | 作用 |
| --- | --- |
| `main/run_e2e.py` | 命令行入口，支持 `--case` 和直接 `--log` 运行。 |
| `main/data_leak_detector/datasets.py` | 发现 `spec/data` 真实样本输入。 |
| `main/data_leak_detector/sensitivity.py` | 可配置地提取初始敏感源文件。 |
| `main/data_leak_detector/groundtruth.py` | 按 `groundtruth.json` 和可配置口径生成最终标注结论。 |
| `main/data_leak_detector/pipeline.py` | 编排分析流程和可选图谱写入。 |
| `main/data_leak_detector/policy.py` | 加载 `spec/config/policy.json`，提供统一文本归一化和策略判断接口。 |
| `spec/config/policy.json` | 可替换的业务策略配置，避免把数据集规则写死在代码里。 |
| `spec/config/groundtruth_policy.json` | 可替换的 groundtruth 结论口径配置。 |
| `main/data_leak_detector/frame_analyzer/*` | 关键帧、OCR、VLM、响应解析和前端应用识别。 |
| `main/data_leak_detector/event_correlator/*` | 文件血缘、候选事件和 Datalog 事实生成。 |
| `main/data_leak_detector/leak_reasoner/*` | 符号化污点传播。 |
| `tools/smoke_pipeline.py` | 基于真实 `spec/data` 样本的快速冒烟测试。 |

## 测试

```powershell
python -m pytest
```

单元测试会生成临时日志覆盖核心逻辑；真实样本来源统一放在 `spec/data`。
