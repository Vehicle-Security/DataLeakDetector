#!/usr/bin/env bash

# Continue the keyframe-selection sensitivity study after the three global
# deduplication settings have finished.  One factor changes at a time relative
# to the shared `default` run: sampling density, window budget, or VLM grid
# layout.  Fresh selection settings must regenerate precompute; grid-only
# settings reuse the default selection cache by design.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="artifacts/frame_dedup_sensitivity_20260813"
CASE_ROOT="spec/data/nas_samples"
PRECOMPUTE_WORKERS=1
VLM_CASE_WORKERS=15
VLM_WORKERS=15

usage() {
  cat <<'EOF'
Usage: tools/run_frame_selection_sensitivity.sh [options]

Options:
  --run-root PATH            Root shared with the deduplication study
  --case-root PATH           Case root (default: spec/data/nas_samples)
  --precompute-workers N     Precompute workers (default: 1)
  --vlm-case-workers N       Case workers (default: 15)
  --vlm-workers N            Shared VLM workers (default: 15)
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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run_fresh_selection() {
  local name=$1 frame_step=$2 strong_step=$3 weak_step=$4 budget=$5 medium_budget=$6 weak_budget=$7
  echo "[$(date -Is)] Starting $name: steps=${frame_step}/${strong_step}/${weak_step}ms budgets=${budget}/${medium_budget}/${weak_budget}"
  DLD_FRAME_STEP_MS="$frame_step" \
  DLD_STRONG_FRAME_STEP_MS="$strong_step" \
  DLD_WEAK_FRAME_STEP_MS="$weak_step" \
  DLD_MAX_KEYFRAMES_PER_WINDOW="$budget" \
  DLD_MAX_KEYFRAMES_PER_STRONG_WINDOW="$budget" \
  DLD_MAX_KEYFRAMES_PER_MEDIUM_WINDOW="$medium_budget" \
  DLD_MAX_KEYFRAMES_PER_WEAK_WINDOW="$weak_budget" \
  tools/run_all.sh \
    --case-root "$CASE_ROOT" \
    --run-dir "$RUN_ROOT/$name" \
    --precompute-workers "$PRECOMPUTE_WORKERS" \
    --vlm-case-workers "$VLM_CASE_WORKERS" \
    --vlm-workers "$VLM_WORKERS" \
    --vlm-grid-layout 4x1
  echo "[$(date -Is)] Finished $name"
}

run_grid_only() {
  local name=$1 layout=$2 precompute_root="$RUN_ROOT/default/vision_precompute"
  if [[ ! -d $precompute_root ]]; then
    echo "Missing default precompute cache: $precompute_root" >&2
    exit 1
  fi
  echo "[$(date -Is)] Starting $name: grid=$layout, reused cache=$precompute_root"
  tools/run_all.sh \
    --case-root "$CASE_ROOT" \
    --run-dir "$RUN_ROOT/$name" \
    --precompute-workers "$PRECOMPUTE_WORKERS" \
    --vlm-case-workers "$VLM_CASE_WORKERS" \
    --vlm-workers "$VLM_WORKERS" \
    --vlm-grid-layout "$layout" \
    --vision-precompute-root "$precompute_root" \
    --skip-precompute
  echo "[$(date -Is)] Finished $name"
}

# Only one parameter family differs from the default values per run.
run_fresh_selection sparse_sampling 2000 500 4000 18 2 2
run_fresh_selection dense_sampling 500 125 1000 18 2 2
run_fresh_selection low_budget 1000 250 2000 10 1 1
run_fresh_selection high_budget 1000 250 2000 26 3 3
run_grid_only grid_2x1 2x1
run_grid_only grid_4x2 4x2

echo "All additional keyframe-selection sensitivity settings finished: $RUN_ROOT"
