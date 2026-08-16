#!/usr/bin/env bash

# Sensitivity study for global visual near-duplicate suppression.  Every
# setting regenerates the precompute cache because the selected keyframes are
# themselves the independent variable.  The VLM model, prompt, grid layout,
# sample set, and pipeline stay fixed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="artifacts/frame_dedup_sensitivity_$(date +%Y%m%d_%H%M%S)"
CASE_ROOT="spec/data/nas_samples"
PRECOMPUTE_WORKERS=1
VLM_CASE_WORKERS=15
VLM_WORKERS=15
SETTINGS="conservative,default,aggressive"

usage() {
  cat <<'EOF'
Usage: tools/run_frame_dedup_sensitivity.sh [options]

Options:
  --run-root PATH            Root directory for selected runs
  --case-root PATH           Case root (default: spec/data/nas_samples)
  --precompute-workers N     Precompute workers (default: 1)
  --vlm-case-workers N       Case workers (default: 15)
  --vlm-workers N            Shared VLM workers (default: 15)
  --settings LIST            Comma-separated settings: conservative,default,aggressive,extreme
EOF
}

require_value() {
  if (($# < 2)) || [[ -z ${2:-} ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --run-root) require_value "$@"; RUN_ROOT=$2; shift 2 ;;
    --case-root) require_value "$@"; CASE_ROOT=$2; shift 2 ;;
    --precompute-workers) require_value "$@"; PRECOMPUTE_WORKERS=$2; shift 2 ;;
    --vlm-case-workers) require_value "$@"; VLM_CASE_WORKERS=$2; shift 2 ;;
    --vlm-workers) require_value "$@"; VLM_WORKERS=$2; shift 2 ;;
    --settings) require_value "$@"; SETTINGS=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
SUMMARY="$RUN_ROOT/experiment_plan.md"
cat >"$SUMMARY" <<EOF
# Global visual deduplication threshold sensitivity

Fixed controls: all 326 cases; Qwen3.7-plus; token-plan endpoint; 4x1 grid;
full ScreenGuard pipeline; 15 case workers and 15 VLM request workers.

| Setting | Near-pixel threshold | pHash distance | Entropy threshold |
| --- | ---: | ---: | ---: |
| conservative | 0.003 | 4 | 0.005 |
| default | 0.010 | 12 | 0.015 |
| aggressive | 0.040 | 28 | 0.060 |
| extreme | 0.080 | 40 | 0.100 |
EOF

run_setting() {
  local name=$1 near_pixel=$2 hash_distance=$3 entropy=$4 run_dir
  run_dir="$RUN_ROOT/$name"
  echo "[$(date -Is)] Starting $name: near=$near_pixel hash=$hash_distance entropy=$entropy"
  DLD_FRAME_NEAR_DUPLICATE_THRESHOLD="$near_pixel" \
  DLD_FRAME_HASH_DISTANCE_THRESHOLD="$hash_distance" \
  DLD_FRAME_ENTROPY_DUPLICATE_THRESHOLD="$entropy" \
  tools/run_all.sh \
    --case-root "$CASE_ROOT" \
    --run-dir "$run_dir" \
    --precompute-workers "$PRECOMPUTE_WORKERS" \
    --vlm-case-workers "$VLM_CASE_WORKERS" \
    --vlm-workers "$VLM_WORKERS" \
    --vlm-grid-layout 4x1
  echo "[$(date -Is)] Finished $name"
}

IFS=',' read -r -a selected_settings <<<"$SETTINGS"
for setting in "${selected_settings[@]}"; do
  case "$setting" in
    conservative) run_setting conservative 0.003 4 0.005 ;;
    default) run_setting default 0.010 12 0.015 ;;
    aggressive) run_setting aggressive 0.040 28 0.060 ;;
    extreme) run_setting extreme 0.080 40 0.100 ;;
    *) echo "Unknown sensitivity setting: $setting" >&2; exit 2 ;;
  esac
done

echo "All sensitivity settings finished: $RUN_ROOT"
