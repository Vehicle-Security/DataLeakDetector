# 2-EventCorrelator

该目录承接 `docs/introduce.md` 中“EventCorrelator / 日志挖掘和敏感文件发现”职责。
当前仓库仍保留历史模块目录名，后续大目录重命名应单独做迁移，避免影响现有运行脚本。

## 当前职责

`EventCorrelator` 当前负责：

- 前台网页/应用分类，将浏览器窗口进一步解析为 `email`、`ai_service`、`cloud_storage`、`meeting` 等 `FrontendApp`
- 敏感实体窗口构建，输出可供 FrameAnalyzer 消费的 `analysis_windows`
- 时间线归一化
- 敏感文件 lineage 构建，支持沿已知 artifact 多跳推断派生链
- 日志与片段的匹配关联
- 结构化 `CorrelatedEvent` 生成
- `UploadCandidate` 生成
- 去重与证据合并
- 为下游推理补充 `object_binding` 元数据
- 导出 `artifact_instances`，用 `path + nearest_evidence_time` 区分同名同路径的不同派生实体
- 输出稳定的 `CorrelationBundle`

## 当前行为

该模块已经处于 `main_v2.py --mode full` 的真实主链路中。

当前已经支持：

- 从 `window_info.window_title` 解析高风险前台应用类别，浏览器不再只保留进程名
- 按敏感文件 anchor、非白名单共现应用、后续外部前台应用构建分析窗口，并附带 `post_buffer_seconds`
- 将派生文件的多次上传收敛为单个上传候选
- 对派生文件继续派生的场景进行多跳 lineage 推断，减少只追一轮的断链
- 在 `CorrelatedEvent` 和 `UploadCandidate` 上显式输出 `object_binding`
- 在屏幕共享类场景下给出 segment-only 候选及其绑定依据

## 设计说明

该模块已经不再只是一个冻结边界的占位层。
它现在已经是以下三类信息之间的真实集成点：

- `FrameAnalyzer` 输出的结构化观察片段
- 归一化后的系统日志
- 下游 `LeakReasoner`

当前仍存在一定启发式规则，尤其是屏幕共享绑定部分，但它已经属于工作中的主链路模块，而不是纯占位迁移层。
