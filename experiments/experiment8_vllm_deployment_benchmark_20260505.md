# Experiment 8: vLLM Server Deployment Inference Performance

## Experiment Setup

- Experiment name: Experiment 8, server deployment inference performance evaluation
- Model service: vLLM
- Model: `qwen2.5-vl-72b`
- Dataset: `/home/wh/datasets/log_driven_e2e_selected_stage1245`
- Number of samples: 74
- Concurrency levels: 1, 2, 3, 4
- Module4 LLM reasoning: disabled, `DLD_THREAT_USE_LLM=false`
- Frame sampling: `sample_fps=1.0`
- Run directory: `/home/wh/logs/log_driven_e2e_bench_selected_stage1245/run_20260505_143705`

This experiment evaluates deployment inference cost only. It does not report detection accuracy.

## Performance Results

| Concurrency | Samples | Completed | Failed | Timeout | CUDA OOM | video_sec/sec | sampled_frames/sec | mean wall(s) | P95 wall(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 74 | 0 | 0 | 0 | 3.749 | 3.749 | 23.718 | 94.250 |
| 2 | 74 | 74 | 0 | 0 | 0 | 10.121 | 10.121 | 17.449 | 72.772 |
| 3 | 74 | 74 | 0 | 0 | 0 | 15.758 | 15.758 | 15.413 | 32.940 |
| 4 | 74 | 74 | 0 | 0 | 0 | 6.748 | 6.748 | 40.687 | 105.372 |

Because `sample_fps=1.0`, `sampled_frames/sec` is numerically equal to `video_sec/sec`.

## Conclusion

- Concurrency 3 achieves the best normalized throughput: `15.758 video seconds/s`.
- Concurrency 3 also has the lowest P95 single-sample wall time: `32.940 s`.
- Increasing concurrency from 3 to 4 reduces throughput to `6.748 video seconds/s` and increases P95 wall time to `105.372 s`.
- All concurrency levels finished with zero failure, zero timeout, and zero CUDA OOM.
- Recommended deployment concurrency for this server configuration: `concurrency=3`.

## Suggested Document Text

我们使用 74 个筛选后的 log-driven E2E 样本评估服务器端 vLLM 部署推理性能。由于不同样本的视频长度不同，本文采用归一化吞吐量 `video seconds/s` 作为主要性能指标，而不是仅统计每分钟处理样本数。实验结果表明，当并发数为 3 时，系统达到最高归一化吞吐量，可处理 15.758 秒视频/秒，同时 P95 单样本耗时为 32.940 秒。当并发数进一步增加到 4 时，吞吐量下降至 6.748 秒视频/秒，P95 耗时上升至 105.372 秒，说明系统已超过当前硬件配置下的高效运行区间，出现资源竞争或 vLLM 请求排队。因此，并发 3 是当前服务器配置下推荐的部署并发度。

## Drawing Prompt

```text
请生成一张学术论文风格的实验图，不要添加主标题，白色背景，简洁配色，清晰坐标轴和图例。

图类型：双轴组合图，柱状图 + 折线图。

横轴：Concurrency，包含 1、2、3、4。

左纵轴：Normalized throughput (video seconds/s)，用柱状图表示。
右纵轴：P95 wall time (s)，用折线图表示。

数据：
Concurrency 1: throughput 3.749 video seconds/s, P95 wall time 94.250 s。
Concurrency 2: throughput 10.121 video seconds/s, P95 wall time 72.772 s。
Concurrency 3: throughput 15.758 video seconds/s, P95 wall time 32.940 s。
Concurrency 4: throughput 6.748 video seconds/s, P95 wall time 105.372 s。

要求：
1. 高亮 concurrency=3。
2. 在 concurrency=3 附近标注 “Best deployment point”。
3. 柱状图表示吞吐量，折线图表示 P95 延迟。
4. 左轴标签为 “Normalized throughput (video seconds/s)”。
5. 右轴标签为 “P95 wall time (s)”。
6. 横轴标签为 “Concurrency”。
7. 图例清晰，颜色不要太花，适合论文或技术报告。
8. 不要图标题。
```

## Notes

- `samples/min` is not used as the main metric because sample video lengths differ.
- `video_sec/sec` is the primary normalized throughput metric.
- This record only covers Experiment 8 and does not include Experiment 7.
