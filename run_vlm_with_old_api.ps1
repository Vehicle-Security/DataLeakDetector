# Use old pay-as-you-go API configuration
$env:OPENAI_API_KEY="sk-1102995c430c46e69dde0bc8ef628c66"
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:DASHSCOPE_API_KEY="sk-1102995c430c46e69dde0bc8ef628c66"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_API_KEY="sk-1102995c430c46e69dde0bc8ef628c66"
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VL_API_KEY="sk-1102995c430c46e69dde0bc8ef628c66"
$env:VL_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VL_MODEL_NAME="qwen3.6-plus"

# Other settings
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
$env:DLD_VLM_ENABLE_OCR_PREFILTER="1"
$env:DLD_VLM_OCR_ENGINE="easyocr"
$env:DLD_VLM_REVIEW_CACHE="0"
$env:DLD_FRAME_DEBUG="1"
$env:DLD_SAVE_FRAME_DEBUG="1"
$env:DLD_VLM_REQUEST_DELAY="1.0"

Write-Host "Using old pay-as-you-go API (qwen-vl-plus)" -ForegroundColor Cyan
Write-Host "Starting VLM test with 12 workers..." -ForegroundColor Cyan

python -u tools\run_benchmark_guarded.py `
  --run-name nas_vlm_old_api_test `
  --use-vlm `
  --vlm-gate-mode adaptive `
  --vlm-workers 12 `
  --max-vlm-cases 0 `
  --max-vlm-frames 24 `
  --json
