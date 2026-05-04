# Module1/E2E Benchmark Notes - 2026-05-04

This note records the server-side module1 benchmark results and what each number means. The code changes in this branch target the next step: full E2E detection from video plus logs, without using `groundtruth.json`.

## Context

- Server workspace used for the benchmark: `/home/wh/workspace/DataLeakDetector-main-test`
- Source baseline: `459753f`
- Dataset root: `/home/wh/datasets/stage1/stage1`
- Model service: `qwen2.5-vl-72b` through vLLM on port `8000`
- Best vLLM setting found: `--gpu-memory-utilization 0.78`
- Module1 test limits: `FRAME_ANALYZER_MAX_VLM_IMAGES=4`, `FRAME_ANALYZER_VLM_MAX_SIDE=640`

## Run Meanings

### `run_20260504_000501`

Early baseline after image cap and resize experiments. This run is useful as a failure diagnosis, but should not be used as the final throughput result because many failures were CUDA OOM or malformed VLM JSON.

| concurrency | cases | samples/min | success | leak_detected | video_sec/sec | raw_frames/sec | p95_wall_sec |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82 | 1.850 | 35 | 30 | 1.218 | 12.082 | 74.350 |
| 2 | 82 | 3.344 | 32 | 29 | 2.202 | 21.836 | 88.744 |
| 4 | 82 | 8.489 | 18 | 17 | 5.590 | 55.434 | 93.310 |

### `run_20260504_125450`

vLLM was stable at `0.78`, but OCR was pinned to GPU0 only. `concurrency=1` improved, while `concurrency=2` became invalid because both OCR workers competed for GPU0 and caused many OOM failures.

| concurrency | cases | status summary | leak_detected | note |
|---:|---:|---|---:|---|
| 1 | 82 | success 54, bad groundtruth 10, no_hits 10, failed 8 | 45 | JSON parse and no_hits remained. |
| 2 | 82 | failed 59, success 11, bad groundtruth 10, no_hits 2 | 11 | Invalid throughput because most failures were OCR CUDA OOM. |

### `run_20260504_151354`

Best current module1-only result. OCR workers were assigned to fixed GPU slots 0 and 6, so OOM disappeared. VLM JSON output was repaired with `json-repair`. Remaining non-success rows are bad dataset labels or OCR gate misses.

| metric | value |
|---|---:|
| total rows | 82 |
| bad `groundtruth.json` rows | 10 |
| valid module1 cases | 72 |
| valid successes | 62 |
| valid `no_hits_found` | 10 |
| valid success rate | 86.11% |
| total success rate | 75.61% |
| valid leak detected | 51 |
| total samples/min | 2.185 |
| valid cases/min | 1.918 |
| success/min | 1.652 |
| valid video_sec/sec | 1.439 |
| valid raw_frames/sec | 14.267 |

Bad `groundtruth.json` cases are data-label issues, not module1 failures:

- `1-email-gmail-1`
- `2-ai-Gemini-3`
- `3-Messaging-dingding-1`
- `4-Drive-OneDrive-2`
- `4-Drive-baidu-2`
- `5-meeting-Zoom-1`
- `6-workplace-youdao-1`
- `7-Tech community-stackoverflow-2`
- `8-git-GitLab-2`
- `8-git-Gitee-2`

## Term Glossary

- `success`: Module1 returned VLM events successfully. It may or may not contain a leak event.
- `leak_detected`: The benchmark counted at least one returned event as leak/sensitive behavior.
- `no_hits_found`: Module1 did not send frames to VLM because OCR keyword filtering found no matching frame.
- `failed`: Module1 returned an error, such as CUDA OOM or malformed VLM JSON.
- `bad_groundtruth_json`: The sample label file is invalid JSON. This is a dataset issue, not an OCR/VLM issue.

## Code Diagnosis

The observed `no_hits_found` rows are mainly caused by the pre-VLM OCR gate:

- Module1 sampled roughly one frame per second, then removed visually similar frames before OCR.
- OCR used exact substring matching for ordinary keywords.
- Module2 passed only the file stem as the target keyword.
- Module2 searched from event time to event time plus 30 seconds, with no pre-buffer.
- Without groundtruth, app/sink words like `发送`, `附件`, `上传`, `邮箱`, `网盘`, and `共享屏幕` were missing from the target context.

This branch changes those points so full E2E can work from logs plus video.
