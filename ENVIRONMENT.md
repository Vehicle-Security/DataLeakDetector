# 环境说明

当前工作区的真实运行口径：

- 本仓库当前使用 `venv\Scripts\python.exe`
- 当前仓库下不存在 repo-local `env\Scripts\python.exe`

推荐命令：

```powershell
venv\Scripts\python.exe -m unittest tests.test_frame_analyzer_adapter tests.test_event_correlator tests.test_leak_reasoner tests.test_v2_acceptance tests.test_e2e_v2
```

```powershell
venv\Scripts\python.exe main\main_v2.py --samples 10-2 5-2 --mode full
```

- debug 帧导出默认关闭；仅在定向排查时启用：
  `FRAME_ANALYZER_SAVE_DEBUG_FRAMES=true`
- 可选 debug 帧导出目录覆盖：
  `FRAME_ANALYZER_DEBUG_FRAME_DIR=output/vlm_debug_frames`

如果后续某个验收脚本仍坚持使用 `env\Scripts\python.exe`，应优先判断为环境命名差异，而不是主链路回退。
