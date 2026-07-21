#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CASE_ROOT="spec/data/nas_samples"
RUN_DIR=""
PRECOMPUTE_WORKERS=2
VLM_CASE_WORKERS=20
VLM_WORKERS=20
VLM_GRID_LAYOUT="4x1"
VLM_TIMEOUT_SECONDS=300
VLM_RETRY_ATTEMPTS=3
VLM_RETRY_BACKOFF_SECONDS=2
CASE_LIST=""
VISION_PRECOMPUTE_ROOT=""
SKIP_PRECOMPUTE=0
VLM_TOKEN_BASE_URL="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
USE_NEO4J_LOG_MINER_FOR_PRECOMPUTE=0

usage() {
  cat <<'EOF'
Usage: tools/run_all.sh [options]

Options:
  --case-root PATH                         Case root (default: spec/data/nas_samples)
  --run-dir PATH                           Output directory (default: timestamped artifacts directory)
  --precompute-workers N                   Precompute case workers (default: 2)
  --vlm-case-workers N                     VLM case workers (default: 20)
  --vlm-workers N                          Shared VLM requests per API key (default: 20)
  --vlm-grid-layout ROWSxCOLUMNS           VLM grid layout (default: 4x1)
  --vlm-timeout-seconds SECONDS            VLM response timeout (default: 300)
  --vlm-retry-attempts N                   VLM retry attempts (default: 3)
  --vlm-retry-backoff-seconds SECONDS      Initial retry backoff (default: 2)
  --case-list PATH                         Optional UTF-8 case ID list
  --vision-precompute-root PATH            Existing precompute cache root
  --skip-precompute                        Reuse --vision-precompute-root
  --vlm-token-base-url URL                 Token-plan endpoint
  --use-neo4j-log-miner-for-precompute     Enable Neo4j only during precompute
  -h, --help                               Show this help
EOF
}

require_value() {
  if (($# < 2)) || [[ -z ${2:-} ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --case-root) require_value "$@"; CASE_ROOT=$2; shift 2 ;;
    --run-dir) require_value "$@"; RUN_DIR=$2; shift 2 ;;
    --precompute-workers) require_value "$@"; PRECOMPUTE_WORKERS=$2; shift 2 ;;
    --vlm-case-workers) require_value "$@"; VLM_CASE_WORKERS=$2; shift 2 ;;
    --vlm-workers) require_value "$@"; VLM_WORKERS=$2; shift 2 ;;
    --vlm-grid-layout) require_value "$@"; VLM_GRID_LAYOUT=$2; shift 2 ;;
    --vlm-timeout-seconds) require_value "$@"; VLM_TIMEOUT_SECONDS=$2; shift 2 ;;
    --vlm-retry-attempts) require_value "$@"; VLM_RETRY_ATTEMPTS=$2; shift 2 ;;
    --vlm-retry-backoff-seconds) require_value "$@"; VLM_RETRY_BACKOFF_SECONDS=$2; shift 2 ;;
    --case-list) require_value "$@"; CASE_LIST=$2; shift 2 ;;
    --vision-precompute-root) require_value "$@"; VISION_PRECOMPUTE_ROOT=$2; shift 2 ;;
    --skip-precompute) SKIP_PRECOMPUTE=1; shift ;;
    --vlm-token-base-url) require_value "$@"; VLM_TOKEN_BASE_URL=$2; shift 2 ;;
    --use-neo4j-log-miner-for-precompute) USE_NEO4J_LOG_MINER_FOR_PRECOMPUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x $PYTHON ]]; then
  echo "Virtual environment not found at $REPO_ROOT/.venv" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev,vision]'" >&2
  exit 1
fi
if [[ ! $VLM_GRID_LAYOUT =~ ^[0-9]+x[0-9]+$ ]]; then
  echo "VLM grid layout must use ROWSxCOLUMNS, for example 4x1." >&2
  exit 2
fi
if ((SKIP_PRECOMPUTE)); then
  if [[ -z $VISION_PRECOMPUTE_ROOT ]]; then
    echo "--skip-precompute requires --vision-precompute-root." >&2
    exit 2
  fi
  if [[ ! -e $VISION_PRECOMPUTE_ROOT ]]; then
    echo "Vision precompute root does not exist: $VISION_PRECOMPUTE_ROOT" >&2
    exit 2
  fi
fi

if [[ -z $RUN_DIR ]]; then
  RUN_DIR="artifacts/full_release_grid4x1_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_DIR"

export DLD_VLM_RETRY_ATTEMPTS="$VLM_RETRY_ATTEMPTS"
export DLD_VLM_RETRY_BACKOFF_SECONDS="$VLM_RETRY_BACKOFF_SECONDS"
export DLD_VLM_TIMEOUT_SECONDS="$VLM_TIMEOUT_SECONDS"
export DLD_VLM_WORKERS="$VLM_WORKERS"
export DLD_VLM_GRID_LAYOUT="$VLM_GRID_LAYOUT"
export DLD_VLM_TOKEN_BASE_URL="$VLM_TOKEN_BASE_URL"
export DLD_VLM_USE_CODING_PLAN=0
export PYTHONUNBUFFERED=1

PREFLIGHT_LOG="$RUN_DIR/vlm_preflight.log"
PRECOMPUTE_LOG="$RUN_DIR/precompute.log"
VLM_STDOUT_LOG="$RUN_DIR/vlm.stdout.log"
VLM_STDERR_LOG="$RUN_DIR/vlm.stderr.log"
VLM_LOG="$RUN_DIR/vlm.log"

echo "Run directory: $RUN_DIR"
echo "Case root: $CASE_ROOT"
echo "VLM grid layout: $VLM_GRID_LAYOUT"
echo "VLM timeout: ${VLM_TIMEOUT_SECONDS}s"
echo "VLM case workers: $VLM_CASE_WORKERS"
echo "VLM request workers per API key: $VLM_WORKERS"
echo "Case list: ${CASE_LIST:-all discovered cases}"
echo "VLM endpoint plan: token-plan ($VLM_TOKEN_BASE_URL)"

echo "Checking VLM endpoint and quota with a minimal real request..."
if ! "$PYTHON" tools/vlm_preflight.py >"$PREFLIGHT_LOG" 2>&1; then
  echo "VLM preflight failed. See $PREFLIGHT_LOG" >&2
  exit 1
fi

precompute_args=(
  main/run_e2e.py
  --case-root "$CASE_ROOT"
  --case-workers "$PRECOMPUTE_WORKERS"
  --release
  --release-precompute-only
  --output-dir "$RUN_DIR"
)
[[ -n $CASE_LIST ]] && precompute_args+=(--case-list "$CASE_LIST")
((USE_NEO4J_LOG_MINER_FOR_PRECOMPUTE)) && precompute_args+=(--release-precompute-neo4j-log-miner)

if ((SKIP_PRECOMPUTE)); then
  echo "Skipping precompute; cache root: $VISION_PRECOMPUTE_ROOT"
else
  echo "Starting full release precompute..."
  set +e
  "$PYTHON" "${precompute_args[@]}" >"$PRECOMPUTE_LOG" 2>&1
  precompute_status=$?
  set -e
  if ((precompute_status != 0)); then
    if ((precompute_status == 137)); then
      echo "Release precompute was killed by the OS, usually due to exhausted memory." >&2
      echo "Resume this run directory with fewer workers: $0 --run-dir '$RUN_DIR' --precompute-workers 1" >&2
    fi
    echo "Release precompute failed. See $PRECOMPUTE_LOG" >&2
    exit "$precompute_status"
  fi
fi

vlm_args=(
  main/run_e2e.py
  --case-root "$CASE_ROOT"
  --case-workers "$VLM_CASE_WORKERS"
  --release
  --release-debug-artifacts
  --output-dir "$RUN_DIR"
  --vlm-grid-layout "$VLM_GRID_LAYOUT"
  --vlm-workers "$VLM_WORKERS"
  --vlm-fast-dispatch
)
[[ -n $CASE_LIST ]] && vlm_args+=(--case-list "$CASE_LIST")
[[ -n $VISION_PRECOMPUTE_ROOT ]] && vlm_args+=(--vision-precompute-root "$VISION_PRECOMPUTE_ROOT")

vlm_pid=""
cleanup() {
  if [[ -n $vlm_pid ]] && kill -0 "$vlm_pid" 2>/dev/null; then
    kill "$vlm_pid" 2>/dev/null || true
    wait "$vlm_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting full release VLM..."
"$PYTHON" "${vlm_args[@]}" >"$VLM_STDOUT_LOG" 2>"$VLM_STDERR_LOG" &
vlm_pid=$!
set +e
wait "$vlm_pid"
vlm_status=$?
set -e
vlm_pid=""
{
  [[ ! -s $VLM_STDOUT_LOG ]] || cat "$VLM_STDOUT_LOG"
  [[ ! -s $VLM_STDERR_LOG ]] || cat "$VLM_STDERR_LOG"
} >"$VLM_LOG"
if ((vlm_status != 0)); then
  echo "Release VLM failed. See $VLM_LOG" >&2
  exit "$vlm_status"
fi

trap - EXIT INT TERM
echo "Done."
echo "Release report: $RUN_DIR/release_report.json"
echo "Comparison: $RUN_DIR/release_comparison.json"
