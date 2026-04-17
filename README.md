# DataLeakDetector

面向系统验收与研究验证的新三模块数据泄露检测原型。

当前主链路：
- `01-FrameAnalyzer`
- `02-EventCorrelator`
- `03-LeakReasoner`
- 主入口：`main/main_v2.py`

## 当前状态

- `main/main.py` 已转发到 `main/main_v2.py`
- `--mode full` 运行真实的模块 1 视频/OCR/VLM 分析链路
- `--mode demo` 仅用于基于样例元数据生成演示片段
- 当前验收样例为桌面上的 `10-2` 和 `5-2`

## 运行方式

请使用仓库内虚拟环境：

```powershell
venv\Scripts\python.exe main\main_v2.py --samples 10-2 5-2 --mode full
```

如需强制模块 1 绕过缓存重跑：

```powershell
venv\Scripts\python.exe main\main_v2.py --samples 10-2 5-2 --mode full --fresh-run
```

默认输出：

- `output/e2e_v2_summary.json`

FrameAnalyzer 缓存目录：

- `output/frame_cache`
- debug 帧导出默认关闭
- 仅在需要人工排查时启用：`FRAME_ANALYZER_SAVE_DEBUG_FRAMES=true`
- 可选导出目录覆盖：`FRAME_ANALYZER_DEBUG_FRAME_DIR=...`

## 输出约定

最终 summary 中的 `frame_analysis` 当前稳定包含：

- `mode`
- `status`
- `metadata`
- `segments`
- `summary`

`metadata` 当前包含：

- `analysis_backend`
- `analysis_backend_version`
- `prompt_signature`
- `cache_hit`
- `cache_schema_version`
- `request_signature`
- `cache_path`
- `fresh_run_requested`

## 当前限制

模块 1 当前仍是在新数据契约外壳下复用迁移后的视觉后端。
因此它还不能被描述为“从零重写完成”的版本。
