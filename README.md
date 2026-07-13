# DataLeakDetector 使用说明

DataLeakDetector 根据终端日志和录屏视频定位数据外发、内容复制、文件转换、屏幕共享等风险行为，并输出可追溯的 JSON 报告。

本文只说明如何准备环境、组织数据并把项目跑通。默认使用内存日志挖掘，不需要安装 Neo4j；当前视觉流程是“日志定位时间窗口 -> 非均匀关键帧 -> VLM -> 事件关联与推理”，不包含 OCR、ROI 或其他视觉模式。

## 1. 运行前准备

推荐环境：

- Windows 10/11
- PowerShell 5.1 或更高版本
- Python 3.10 或更高版本，推荐 Python 3.12
- 能访问所配置 VLM 接口的网络
- 足够的磁盘空间；全量运行会在 `artifacts/` 下保存关键帧和报告

在 PowerShell 中进入仓库根目录：

```powershell
Set-Location "D:\Projects\Job\DataLeakDetector"
```

确认 Python 可用：

```powershell
python --version
```

如果输出低于 `3.10`，先安装新版 Python，并确保安装时勾选 `Add Python to PATH`。

## 2. 创建虚拟环境并安装依赖

首次使用时执行：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vision]"
```

以后重新打开 PowerShell，只需要进入仓库并激活环境：

```powershell
Set-Location "D:\Projects\Job\DataLeakDetector"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

验证安装：

```powershell
python -m compileall -q main tests tools
python -m pytest tests\test_pipeline.py -q
```

两条命令都应以退出码 `0` 结束。

## 3. 配置 VLM

复制配置模板：

```powershell
Copy-Item .env.example .env
```

用编辑器打开 `.env`，至少填写当前使用的 VLM 密钥：

```text
DLD_VLM_MODEL=qwen3.7-plus
DLD_VLM_CODING_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
DLD_VLM_CODING_API_KEY=在这里填写真实密钥
DLD_VLM_USE_CODING_PLAN=1
DLD_VLM_WORKERS=2
DLD_VLM_DRY_RUN=0
DLD_VLM_MAX_IMAGE_SIDE=1280
```

注意：

- `.env` 保存真实密钥，不要提交到 Git。
- 初次运行建议先用 `DLD_VLM_WORKERS=2`，确认配额稳定后再调整并发。
- `DLD_VLM_DRY_RUN=1` 表示只生成模型请求和关键帧，不真正调用模型。
- 命令行的 `--vlm-workers`、`--vlm-grid-layout` 等参数会覆盖对应运行配置。

执行一次最小真实请求，检查密钥、地址、模型和配额：

```powershell
python tools\vlm_preflight.py
```

成功时会看到类似输出：

```text
VLM preflight passed: model=qwen3.7-plus endpoint=https://coding.dashscope.aliyuncs.com/v1 workers=2
```

如果这里出现 `401`、`403` 或 `429`，不要启动全量任务：

- `401/403`：检查密钥、接口地址和套餐类型。
- `429 allocated quota exceeded`：当前额度不可用，等待额度恢复或更换可用套餐。
- 超时：先检查网络，再适当增加 `.env` 中的 `DLD_VLM_TIMEOUT_SECONDS`。

## 4. 数据存放

默认数据根目录是：

```text
spec/data/nas_samples/
```

可以继续使用这个目录，也可以创建自己的 case 根目录。批量扫描时，一个可识别的 case 必须直接包含 `logs/` 和 `video/` 两个目录。

推荐结构：

```text
spec/data/nas_samples/
  stage1/
    my-case-001/
      logs/
        logs.json
        keyevents.json          # 可选
      video/
        recording_20260713_120000.mp4
      INDEX.md                  # 推荐
      groundtruth.json          # 可选，只用于评测
```

可以有多级目录，例如 `stage1/my-case/session_001/`。只要某一级目录直接包含 `logs/` 和 `video/`，批量模式就会把它识别成独立 case。

### 4.1 日志文件

日志查找顺序如下：

1. `logs/logs.json`
2. `logs/keyevents.json`
3. case 根目录下的 `keyevents.json`
4. case 根目录下的 `logs.json`

建议统一使用 `logs/logs.json`，内容使用 UTF-8 JSON 数组。最小示例：

```json
[
  {
    "timestamp": "2026-07-13T12:00:10.000",
    "event_type": "opened",
    "file_path": "C:\\Users\\alice\\Desktop\\customer_contract.docx",
    "process_info": {
      "process_name": "WINWORD.EXE"
    },
    "window_info": {
      "window_title": "customer_contract.docx - Word"
    },
    "extra": {
      "source": "file_monitor",
      "relative_timestamp": 10.0
    }
  },
  {
    "timestamp": "2026-07-13T12:00:35.000",
    "event_type": "file_upload",
    "file_path": "C:\\Users\\alice\\Desktop\\customer_contract.docx",
    "process_info": {
      "process_name": "chrome.exe"
    },
    "window_info": {
      "window_title": "Upload - Chrome"
    },
    "extra": {
      "source": "file_dialog_monitor",
      "raw_operation": "file_upload",
      "relative_timestamp": 35.0
    }
  }
]
```

关键字段：

- `timestamp`：事件的绝对时间，必须能被解析。
- `event_type`：如 `opened`、`clipboard_text`、`file_selected`、`file_upload`、`send`、`copy`、`print`。
- `file_path`：事件涉及的文件路径；没有文件时可以为空字符串。
- `process_info.process_name`：前台或执行进程。
- `window_info.window_title`：窗口标题，建议保留。
- `extra.relative_timestamp`：相对录屏开始的秒数，强烈推荐写入。

日志可以包含更多采集字段，程序会保留原始记录并提取所需信息。不要为了“让检测通过”手工伪造上传或敏感标签。

### 4.2 录屏视频

把 MP4 放在 case 的 `video/` 目录下：

```text
video/recording_20260713_120000.mp4
```

注意：

- 启用 `--vision` 时必须有可读取的 MP4。
- 同一 case 最好只放一个 MP4。
- 如果有多个 MP4，请在 `INDEX.md` 中明确指定。
- 日志时间和视频时间必须对应，否则会抽到错误画面。

推荐的 `INDEX.md`：

```markdown
# my-case-001

**Recording Time**: 2026-07-13 12:00:00
**Session ID**: 20260713_120000

Video: `video/recording_20260713_120000.mp4`
```

如果日志里已经有可靠的 `extra.relative_timestamp`，程序会优先使用相对时间；否则会用 `INDEX.md` 的 `Recording Time` 将绝对日志时间换算为视频时间。

### 4.3 groundtruth.json

`groundtruth.json` 是可选文件，只用于检测完成后的准确率评测，不参与敏感文件发现、日志窗口生成、VLM 请求或最终推理。

示例：

```json
{
  "record_id": "my-case-001",
  "recording_start_time": "2026-07-13 12:00:00",
  "total_operations": 1,
  "operations": [
    {
      "operation_time": "2026-07-13 12:00:35",
      "sensitive_file_path": "C:\\Users\\alice\\Desktop\\customer_contract.docx",
      "operation": "direct leak - mail attachment"
    }
  ]
}
```

没有 groundtruth 也可以检测，只是该 case 会显示为未评分。子 session 可以在 release 模式下继承最近祖先目录的 groundtruth，但仍然仅用于评测。

## 5. 配置原始敏感文件

检测器唯一的初始敏感源来自：

```text
spec/config/sensitive_files..json
```

注意文件名中确实有两个点：`sensitive_files..json`。

格式：

```json
{
  "sensitive_files": [
    "C:\\Users\\alice\\Desktop\\customer_contract.docx",
    "D:\\finance\\quarterly_report.xlsx"
  ]
}
```

只添加经过日志确认的原始敏感文件：

- 应添加：用户最初打开、读取或操作的原始合同、报表、源码等。
- 不应添加：复制件、重命名文件、压缩包、转换后的 PDF、截图、拆分文件、上传缓存等派生文件。
- 不要从 `groundtruth.json` 自动复制敏感路径到这里。
- 不要仅因为文件名含有 `secret`、`合同`、`机密` 就直接认定为敏感源。

使用自定义敏感源文件时：

```powershell
python main\run_e2e.py `
  --case "spec\data\nas_samples\stage1\my-case-001" `
  --sensitive-files-config "D:\my-config\sensitive_files.json" `
  --vision
```

## 6. 第一次运行：先干跑一个 case

先选择一个日志和视频都完整的小 case。干跑会执行日志挖掘和抽帧，但不会调用真实 VLM：

```powershell
python main\run_e2e.py `
  --case "spec\data\nas_samples\stage1\1-email-fastmail-1" `
  --output-dir "artifacts\smoke_fastmail" `
  --vision `
  --vlm-dry-run `
  --max-vlm-frames 4 `
  --vlm-workers 1 `
  --no-neo4j-log-miner
```

运行结束后检查：

```powershell
Get-ChildItem -Recurse "artifacts\smoke_fastmail"
```

至少应看到主报告和下列部分产物：

```text
artifacts/smoke_fastmail/
  <report_id>.json
  <report_id>/
    artifact_manifest.json
    vision_precompute.json
    keyframes_raw_all/
    keyframes_raw/
    keyframes_vlm_input/       # 经过缩放时存在
    keyframes_vlm_grid/        # 使用网格时存在
    frame_observations.json
    event_correlator_details.json
    leak_paths.json
```

重点查看主报告中的：

- `frame_analyzer.statistics.vision.analysis_windows`
- `frame_analyzer.statistics.vision.keyframes_raw_all`
- `frame_analyzer.statistics.vision.keyframes`
- `frame_analyzer.warnings` 和 `frame_analyzer.errors`

干跑的 VLM 事件为空是正常现象，它只验证数据发现、日志时间、视频读取和关键帧链路。

## 7. 真实运行一个 case

确保 `python tools\vlm_preflight.py` 通过，然后执行：

```powershell
python main\run_e2e.py `
  --case "spec\data\nas_samples\stage1\1-email-fastmail-1" `
  --output-dir "artifacts\real_fastmail" `
  --vision `
  --max-vlm-frames -1 `
  --vlm-grid-layout "4x1" `
  --vlm-workers 2 `
  --vlm-fast-dispatch `
  --no-neo4j-log-miner
```

说明：

- `--max-vlm-frames -1`：不额外限制已选关键帧数量。
- `--vlm-grid-layout 4x1`：每个请求图按 4 行 1 列拼接源帧。
- `--vlm-workers 2`：最多 2 路 VLM 并发，首次运行建议保持较低值。
- `--no-neo4j-log-miner`：明确使用默认内存日志挖掘。

单 case 模式会保留详细 VLM 请求、响应和解析产物，适合排查问题。

## 8. 只分析一个日志文件

没有标准 case 目录、也不需要视频分析时，可以直接传日志：

```powershell
python main\run_e2e.py `
  --log "D:\samples\logs.json" `
  --output-dir "artifacts\log_only"
```

如果同时需要视频：

```powershell
python main\run_e2e.py `
  --log "D:\samples\logs.json" `
  --video "D:\samples\recording.mp4" `
  --output-dir "artifacts\log_and_video" `
  --vision
```

直接传 `--log` 时没有 `INDEX.md` 帮助换算时间，因此日志最好包含 `extra.relative_timestamp`。

## 9. 全量运行

推荐使用 `tools/run_all.ps1`。它会按顺序执行：

1. 发起一个最小真实 VLM 请求，检查密钥和配额。
2. 为全部 case 生成可复用的视觉预计算缓存。
3. 使用缓存运行真实 VLM 和推理。
4. 写出全量报告和评测对比。

建议从保守并发开始：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = "artifacts\full_release_grid4x1_$stamp"

powershell -ExecutionPolicy Bypass -File tools\run_all.ps1 `
  -CaseRoot "spec\data\nas_samples" `
  -RunDir $runDir `
  -PrecomputeWorkers 4 `
  -VlmCaseWorkers 2 `
  -VlmWorkers 2 `
  -VlmGridLayout "4x1"
```

运行时间可能较长。不要关闭执行脚本的 PowerShell 窗口，也不要在同一个输出目录同时启动第二个任务。

实时查看预计算进度：

```powershell
Get-Content "$runDir\release_precompute_progress.json" -Wait
```

进入 VLM 阶段后查看：

```powershell
Get-Content "$runDir\release_progress.json" -Wait
```

查看日志尾部：

```powershell
Get-Content "$runDir\precompute.log" -Tail 30
Get-Content "$runDir\vlm.stdout.log" -Tail 30
```

VLM 阶段运行时持续写入 `vlm.stdout.log` 和 `vlm.stderr.log`；阶段结束后脚本会把两者合并为 `vlm.log`。

正常完成后主要文件包括：

```text
<runDir>/
  vlm_preflight.log
  precompute.log
  vlm.log
  release_precompute_progress.json
  release_progress.json
  release_report.json
  release_comparison.json
  vision_precompute/
  case_debug/                 # 启用 release debug artifacts 时存在
```

## 10. 只运行指定 case

创建 UTF-8 文本文件，每行写一个相对 `--case-root` 的 case ID：

```text
stage1/1-email-fastmail-1
stage2/2-filestruct-pdfconvert-2
stage4/e2e-1/session_20260505_213253
```

例如保存为 `artifacts/my_case_list.txt`，然后执行：

```powershell
python main\run_e2e.py `
  --case-root "spec\data\nas_samples" `
  --case-list "artifacts\my_case_list.txt" `
  --case-workers 2 `
  --release `
  --release-debug-artifacts `
  --output-dir "artifacts\selected_cases" `
  --vlm-grid-layout "4x1" `
  --vlm-workers 2 `
  --vlm-fast-dispatch
```

## 11. VLM 失败后的续跑

Release 模式下，只要某个 VLM 批次最终失败，该 case 会被标记为失败，任务会尽快停止，而不是把空响应当成 `no_confirmed_data_leak`。

失败目录中会生成：

```text
release_retry_cases.txt
```

它包含失败 case 和尚未完成的 case。恢复配额后，先重新预检：

```powershell
python tools\vlm_preflight.py
```

预检通过后，使用原预计算缓存续跑到一个新目录：

```powershell
$failedRun = "artifacts\full_release_grid4x1_失败任务目录"
$retryRun = "artifacts\full_release_grid4x1_retry_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

powershell -ExecutionPolicy Bypass -File tools\run_all.ps1 `
  -CaseRoot "spec\data\nas_samples" `
  -RunDir $retryRun `
  -CaseList "$failedRun\release_retry_cases.txt" `
  -VisionPrecomputeRoot "$failedRun\vision_precompute" `
  -SkipPrecompute `
  -VlmCaseWorkers 2 `
  -VlmWorkers 2 `
  -VlmGridLayout "4x1"
```

不要在原失败目录上直接覆盖运行。使用新目录可以保留失败现场，并避免旧的空响应或半成品影响新报告。

## 12. 如何看结果

最终结论只有三种：

- `data_leak_risk_detected`：存在已确认的数据外发风险路径。
- `suspicious_behavior_detected`：存在隐藏转换、复制、截图等可疑行为，但证据不足以确认直接外发。
- `no_confirmed_data_leak`：当前成功完成的证据链没有确认泄漏。

重点文件：

- `release_report.json`：全部完成 case 的详细结果。
- `release_comparison.json`：检测结论与 groundtruth 的对比。
- `release_progress.json`：运行中状态、错误和 VLM 调度快照。
- `case_debug/<case>/.../vlm_parse_result.json`：模型事件、丢弃事件和批次错误。
- `case_debug/<case>/.../leak_paths.json`：推理出的泄漏路径。

判断结果前先检查：

```powershell
$progress = Get-Content "$runDir\release_progress.json" -Raw | ConvertFrom-Json
$progress | Select-Object state, completed_cases, failed_cases, aborted, abort_reason
```

只有 `state=completed` 且 `failed_cases=0` 的 release 才能作为有效全量结果。`state=failed` 或存在 VLM 错误时，不要把缺失事件解释成阴性结论。

## 13. 常见问题

### 找不到 case

错误示例：

```text
no case directories found
```

检查目标目录是否直接包含：

```text
logs/
video/
```

并确认 `logs/logs.json` 或 `logs/keyevents.json` 不是空文件。

### 找不到日志

错误示例：

```text
no logs.json or keyevents.json found
```

优先把日志放到 `<case>/logs/logs.json`，确保文件大于空数组 `[]`，并使用 UTF-8 编码。

### 没有选出关键帧

先看 `vision_precompute.json` 中的：

- `windows`
- `raw_keyframe_count`
- `keyframes`
- `warnings`

常见原因：

- 日志绝对时间与录屏开始时间不匹配。
- 缺少 `extra.relative_timestamp`，同时 `INDEX.md` 又没有正确的 `Recording Time`。
- 视频损坏或 OpenCV 无法读取。
- 日志只记录了普通阅读，没有外部应用或可报告动作。

### PowerShell 不允许执行脚本

只为当前窗口临时放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 中文路径或日志乱码

- 新增的 JSON、Markdown 和配置文件统一保存为 UTF-8。
- 不要用 ANSI/GBK 覆盖原文件。
- 日志原始字段已经乱码时，应优先修复采集或导出过程，而不是在 groundtruth 中补标签。

### VLM 返回 429

停止当前任务并执行：

```powershell
python tools\vlm_preflight.py
```

只有预检恢复成功后再使用 `release_retry_cases.txt` 续跑。降低并发只能缓解速率限制，不能解决套餐总额度耗尽。

## 14. Neo4j（可选）

默认内存日志挖掘已经可以完成正常运行。只有明确需要 Neo4j 日志查询时才启用：

```powershell
tools\start_neo4j.ps1

python main\run_e2e.py `
  --case "spec\data\nas_samples\stage1\1-email-fastmail-1" `
  --vision `
  --neo4j-log-miner

tools\stop_neo4j.ps1
```

Neo4j 不可用且未启用 strict 模式时，程序会回退到内存日志挖掘。首次跑通项目时不要开启 Neo4j，以减少额外变量。

## 15. 最短跑通清单

按顺序完成以下操作：

1. `python -m venv .venv`
2. 激活 `.venv`
3. `python -m pip install -e ".[dev,vision]"`
4. `Copy-Item .env.example .env`
5. 在 `.env` 填写真实 VLM 密钥
6. 把 case 放成 `<case>/logs/logs.json + <case>/video/*.mp4`
7. 把已确认的原始敏感文件写入 `spec/config/sensitive_files..json`
8. `python tools\vlm_preflight.py`
9. 先用 `--vlm-dry-run` 跑一个 case
10. 再真实运行一个 case
11. 最后使用 `tools/run_all.ps1` 跑全量

任何一步失败，都先解决当前错误，不要直接跳到全量运行。
