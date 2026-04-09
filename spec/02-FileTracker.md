# Module 2: FileTracker

当前文件名原先采用 `02-FileTracer.md`，但模块正式名称统一为 `FileTracker`。后续代码、文档、口头沟通和汇报默认都使用 `FileTracker` 这一名称。

## Summary

模块 2 的职责是围绕单个敏感文件相关事件执行“事件分析 + 派生文件追踪”。它负责调用模块 1 获取视频行为结果，从模块 1 输出中抽取隐藏操作，生成派生敏感事件，并把这些结果交回模块 3 使用。

本模块不承担最终上传判定，也不直接决定是否触发告警。

## Responsibilities

模块 2 需要完成以下工作：

1. 接收一个敏感文件相关事件作为分析入口。
2. 调用模块 1，在对应时间窗内分析视频行为。
3. 从模块 1 结果中抽取隐藏操作。
4. 为隐藏操作生成派生敏感事件。
5. 更新 worklist 所需的敏感文件集合和文件映射信息。
6. 向模块 3 返回结构稳定的分析结果。

## Input Contract

模块 2 分析单个事件时，最小输入契约如下：

- `current_event: SensitiveFileEvent`
- `index_path`
- `video_path`
- `log_events`
- `search_duration`

### Input Semantics

- `current_event` 表示当前待分析的敏感文件相关事件。
- `index_path` 用于读取录屏起始时间。
- `video_path` 用于向模块 1 提供视频分析目标。
- `log_events` 用于在路径解析阶段辅助恢复完整文件路径。
- `search_duration` 决定视频搜索时间窗长度。

## Core Output Contract

重构后，本模块对外必须继续提供以下稳定字段：

- `frame_analysis_result`
- `hidden_operations`
- `file_mappings`
- `new_events`
- `has_hidden_behavior`
- `analysis_complete`
- `error_message`

### Output Semantics

- `frame_analysis_result` 保存模块 1 的原始事件级分析结果。
- `hidden_operations` 保存从模块 1 结果中抽出的隐藏操作。
- `file_mappings` 保存隐藏操作中抽出的文件映射语义。
- `new_events` 保存根据隐藏操作生成的派生敏感事件。
- `has_hidden_behavior` 表示当前事件是否检测到隐藏行为。
- `analysis_complete` 表示当前单次分析流程是否完成。
- `error_message` 表示模块 2 自身的失败信息，供模块 3 决定如何兜底。

## Core Objects And Invariants

### SensitiveFileEvent

`SensitiveFileEvent` 是模块 2 和模块 3 共享的核心事件对象，必须满足以下约束：

- 必须明确区分 `original_file` 和 `current_file`。
- `original_file` 表示追溯后的源敏感文件。
- `current_file` 表示当前正在被处理的文件路径，可能已经经过重命名、复制或格式转换。

### File Mapping Invariants

文件映射必须满足以下不变量：

- `new_events` 中的派生事件必须可追溯到原始敏感文件。
- `file_mappings` 必须表达“派生文件 <- 直接父文件”的关系，并能导出完整映射链。
- 已经是已知敏感文件的派生结果不应重复入队。
- 文件路径必须统一做规范化处理。
- 一条隐藏操作允许拆出多个输出文件名。
- 映射链不能因为循环关系无限扩张。

## Current Flow

当前模块 2 的主流程可以概括为：

1. 初始化单次行为分析状态。
2. 调用模块 1 分析指定时间窗的视频帧。
3. 从模块 1 的 `events` 中抽取隐藏操作。
4. 结合日志和当前文件目录解析派生文件完整路径。
5. 为每个派生文件生成新的 `SensitiveFileEvent`。
6. 更新 `WorklistManager` 中的敏感文件集合和文件映射。
7. 将分析结果返回给模块 3。

## Refactor Directions

本轮建议按照以下方向重构模块 2：

- 将模块 2 聚焦为“事件分析 + 派生文件追踪”，不承担上传判定。
- 将路径解析、操作抽取、去重规则继续收口在工具层。
- 将 `BehaviorAnalysisGraph` 保持为薄编排层，节点只负责状态转移。
- 明确 `update_worklist_node` 只负责入队、标记敏感、建立映射，不夹带上传语义。
- 保持 `SensitiveFileEvent` 和 `WorklistManager` 的共享语义稳定，减少模块间重复解释。
- （可考虑把 `resolve_full_path` 的日志推断策略拆成单独策略函数，便于测试。）
- （可考虑把 `file_mappings` 输出结构从 `dict` 扩成更明确的数据类，但前提是不破坏下游。）

## Failure Handling

模块 2 需要显式定义失败处理策略，避免静默吞错：

- 模块 1 失败时，模块 2 返回空事件结果，不直接使全链路崩溃。
- 路径找不到时，允许退回同目录推断。
- 失败信息必须落入 `error_message`，不能静默吞掉。
- 如果没有检测到相关事件，应返回空结构，而不是返回不完整对象。

## Acceptance Criteria

本模块在本轮重构中的最小验收标准如下：

- 能从模块 1 结果中正确拆出多文件隐藏操作。
- 不会把已知派生敏感文件重复入队。
- 能导出直接映射和完整映射链。
- 不破坏现有相关回归测试。

## Related Regression Coverage

当前已有回归测试重点保护以下模块 2 行为：

- 模型返回分号分隔的多个输出文件名时，能够正确拆分。
- 已知派生敏感文件不会被重复加入 worklist。
- 文件映射相关逻辑不会诱发下游循环扩张。

## Assumptions

- 模块 2 当前对模块 1 的调用接口保持不变。
- 模块 2 输出仍需被模块 3 直接消费，因此不在本轮大幅改变字段名。
- （如果后续决定把路径解析规则拆得更细，则优先通过新增测试保护行为，再做内部拆分。）
