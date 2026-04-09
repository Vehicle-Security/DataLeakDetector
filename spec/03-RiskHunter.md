# Module 3: RiskHunter

## Summary

模块 3 `RiskHunter` 是当前端到端流水线的总控模块。它负责初始化 `WorklistManager`、从日志构建初始敏感事件 worklist、调度模块 2 分析单个事件、生成敏感操作记录、判定上传和告警，并将结果汇总给模块 4 使用。

本轮重构默认优先处理模块 3，因为它是模块 2 与模块 4 之间的关键连接层，也是当前重复告警、状态耦合和输出不稳定问题的主要承载位置。

## Responsibilities

模块 3 需要完成以下职责：

1. 初始化 `WorklistManager`。
2. 从日志中构建初始敏感事件 worklist。
3. 循环取出待处理事件并调用模块 2。
4. 基于模块 2 和模块 1 输出生成敏感操作记录。
5. 判断是否属于上传或外发行为。
6. 依据黑白名单生成 `alert_events` 或 `info_events`。
7. 汇总统计信息和文件映射并提供给模块 4。

## State Contract

重构后，模块 3 的状态契约需要继续保留以下稳定字段。

### Input Configuration

- `record_id`
- `base_path`
- `log_file`
- `video_path`
- `index_path`

### Config Fields

- `sensitive_files`
- `blacklist_apps`
- `whitelist_apps`
- `search_duration`

### Runtime Fields

- `worklist_size`
- `processed_count`
- `current_event`
- `module1_result`

### Output Fields

- `upload_events`
- `operation_records`
- `alert_events`
- `info_events`
- `statistics`

### Internal Fields

- `_worklist_manager`
- `_log_events`
- `_operation_record_keys`
- `_hidden_transformed_paths`

## Output Semantics

模块 3 输出字段的语义需要保持清晰且稳定：

- `upload_events` 是所有识别出的上传类事件全集。
- `alert_events` 是需要告警的上传事件子集。
- `info_events` 是仅记录不告警的上传事件。
- `operation_records` 是敏感文件相关操作轨迹，用于评估、排错和证据补充。
- `file_mappings` 必须来自 `WorklistManager.export_file_mappings()`。
- `statistics.total_events_processed` 必须与 `processed_count` 同步。

## Detection Rules

模块 3 在本轮重构中需要显式保持以下判定规则：

1. 上传判定来自 `behavior_category` 与 `operation_type` 的组合。
2. 黑名单应用中的外发行为默认记为 `critical`。
3. 白名单应用的上传默认记为 `info`。
4. 未知应用的外发默认先记录不告警。
5. 相同时间、相同敏感文件、相同操作的 `operation_records` 需要去重。
6. 同一上传事实不应在结果层重复扩张成多条完全等价告警。
7. `statistics` 的更新必须显式、可测试，不能依赖隐式 side effect。
8. （可进一步把“上传判定关键词”和“告警级别规则”从类内常量迁到外部配置文件。）

## Current Flow

当前模块 3 的运行流程可以概括为：

1. `initialize_node` 加载日志，初始化 `WorklistManager`，扫描并构建 worklist。
2. `process_event_node` 取出下一个敏感事件并调用模块 2。
3. `analyze_upload_node` 基于模块 2 和模块 1 结果生成操作记录与上传事件，并应用黑白名单规则。
4. `finalize_node` 汇总统计、导出文件映射并结束流水线。

这一设计方向本身是合理的，但当前实现仍存在职责交叉和重复逻辑，需要借助本轮重构进一步收口。

## Refactor Directions

本轮建议按以下方向收敛模块 3：

- 模块 3 保持为总控编排层，不重复实现模块 2 已有的路径解析和记录构建逻辑。
- `initialize/process/analyze/finalize` 四个节点保持职责单一。
- `upload_detector_tools.py` 继续承担跨节点共享的小型纯函数。
- `upload_detection_config.py` 当前先保留兼容接口，避免影响可跑通基线。
- 报警去重、统计同步、映射输出要成为显式规则，而不是隐式副作用。
- 模块 3 对外输出应优先关注“结构稳定”和“下游兼容”，而不是追求一次性重做全部数据规则。
- （可考虑后续把样例敏感文件列表与黑白名单迁出 Python 代码。）
- （可考虑给模块 3 增加单独调试入口，但不作为本轮交付前置。）

## Interface Constraints With Module 4

模块 3 与模块 4 的接口必须保持兼容，至少满足以下约束：

- 模块 3 必须继续输出 `alert_events`、`info_events`、`operation_records`、`file_mappings`。
- `file_mappings.direct_file_mappings` 和 `file_mappings.full_file_mapping_chains` 的结构保持兼容。
- 不能因为模块 2/3 重构而破坏 `run_e2e.py` 中 `_inject_connected_facts_from_module3()` 的事实注入逻辑。
- 不能改变模块 4 当前依赖的报告主结构。

## Acceptance Criteria

模块 3 在本轮重构中的最小验收标准如下：

- 能在 10-2 样例上完成初始化、循环处理、告警生成和结果汇总。
- 不出现“模块 3 未加载，跳过”的假成功。
- 能给模块 4 提供可用的映射和事件结果。
- 不破坏现有统计和回归测试语义。
- 报告仍能在根目录 `output/` 下正确生成。

## Related Regression Coverage

当前已有回归测试重点保护以下模块 3 行为：

- `statistics.total_events_processed` 与 `processed_count` 同步。
- 模块 3 输出可以为模块 4 注入补充事实，构造可连接的泄露路径。
- 文件映射循环不会在模块 4 中被放大为无限推理链。

## Assumptions

- 模块 3 继续作为总控编排层存在，不在本轮拆成多个独立顶层服务。
- 模块 3 当前对模块 2 和模块 4 的字段依赖关系默认保持不变。
- 模块 3 的重构优先级高于配置彻底产品化。
- （如果后续导师明确要求“配置去硬编码”必须本轮完成，则把 `upload_detection_config.py` 外部化提升为正式交付项。）
