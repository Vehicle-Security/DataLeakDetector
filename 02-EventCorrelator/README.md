# 02-EventCorrelator

该目录对应新三模块架构中的模块 2。

## 当前职责

`EventCorrelator` 当前负责：

- 时间线归一化
- 敏感文件 lineage 构建
- 日志与片段的匹配关联
- 结构化 `CorrelatedEvent` 生成
- `UploadCandidate` 生成
- 去重与证据合并
- 为下游推理补充 `object_binding` 元数据
- 输出稳定的 `CorrelationBundle`

## 当前行为

该模块已经处于 `main_v2.py --mode full` 的真实主链路中。

当前已经支持：

- 将派生文件的多次上传收敛为单个上传候选
- 在 `CorrelatedEvent` 和 `UploadCandidate` 上显式输出 `object_binding`
- 在屏幕共享类场景下给出 segment-only 候选及其绑定依据

## 设计说明

该模块已经不再只是一个冻结边界的占位层。
它现在已经是以下三类信息之间的真实集成点：

- `FrameAnalyzer` 输出的结构化观察片段
- 归一化后的系统日志
- 下游 `LeakReasoner`

当前仍存在一定启发式规则，尤其是屏幕共享绑定部分，但它已经属于工作中的主链路模块，而不是纯占位迁移层。
