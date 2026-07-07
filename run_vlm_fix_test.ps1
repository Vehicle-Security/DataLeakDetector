# Use old pay-as-you-go API configuration
$env:OPENAI_API_KEY="sk-1102995c430c46e69dde0bc8ef628c66"
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:DASHSCOPE_API_KEY="sk-1102995c430c46e69dde0bc8ef628c66"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VL_MODEL_NAME="qwen-vl-plus"

# Other settings
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
$env:DLD_VLM_ENABLE_OCR_PREFILTER="1"
$env:DLD_VLM_OCR_ENGINE="easyocr"
$env:DLD_VLM_REVIEW_CACHE="0"
$env:DLD_FRAME_DEBUG="1"
$env:DLD_SAVE_FRAME_DEBUG="1"
$env:DLD_VLM_REQUEST_DELAY="1.0"

Write-Host "Running SMALL TEST (10 cases) to validate fix..." -ForegroundColor Cyan
Write-Host "Fix: Allow selected_or_attached and in_progress to generate LeakFile facts" -ForegroundColor Yellow

python -u tools\run_benchmark_guarded.py `
  --run-name nas_vlm_fix_test `
  --use-vlm `
  --vlm-gate-mode adaptive `
  --vlm-workers 4 `
  --max-vlm-cases 0 `
  --max-vlm-frames 24 `
  --json
