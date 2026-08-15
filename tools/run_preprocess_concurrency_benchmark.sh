#!/usr/bin/env bash

# Benchmark the server-side, VLM-free portion of ScreenGuard. Every measured
# repeat gets an empty output directory so no visual precompute is reused.
# The second pass uses VLM dry-run mode to construct the actual 4x1 grids and
# prompts without issuing an API request.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CASE_ROOT="spec/data/nas_samples"
CASE_LIST="spec/config/concurrency_benchmark_32_cases.txt"
RUN_DIR=""
WORKERS=(1 2 4 8 12 16 20 24 32)
REPEATS=3
WARMUP_CASE="stage0/0-normal-git-github-1"

usage() {
  cat <<'EOF'
Usage: tools/run_preprocess_concurrency_benchmark.sh [options]

Options:
  --case-root PATH     Case root (default: spec/data/nas_samples)
  --case-list PATH     Fixed case ID list (default: concurrency 32-case list)
  --run-dir PATH       Output directory (default: timestamped artifacts directory)
  --workers CSV        Case concurrency, e.g. 1,2,4,8 (default: 1,2,4,8,12,16,20,24,32)
  --repeats N          Measured repeats per concurrency (default: 3)
  --warmup-case ID     One unmeasured case used to warm the runtime
  -h, --help           Show this help
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
    --case-list) require_value "$@"; CASE_LIST=$2; shift 2 ;;
    --run-dir) require_value "$@"; RUN_DIR=$2; shift 2 ;;
    --workers) require_value "$@"; IFS=',' read -r -a WORKERS <<<"$2"; shift 2 ;;
    --repeats) require_value "$@"; REPEATS=$2; shift 2 ;;
    --warmup-case) require_value "$@"; WARMUP_CASE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="$REPO_ROOT/.venv/bin/python"
[[ -x $PYTHON ]] || { echo "Virtual environment not found: $PYTHON" >&2; exit 1; }
[[ -f $CASE_LIST ]] || { echo "Case list not found: $CASE_LIST" >&2; exit 1; }
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be a positive integer" >&2; exit 2; }
for workers in "${WORKERS[@]}"; do
  [[ $workers =~ ^[1-9][0-9]*$ ]] || { echo "Invalid worker count: $workers" >&2; exit 2; }
done

if [[ -z $RUN_DIR ]]; then
  RUN_DIR="artifacts/preprocess_concurrency_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_DIR"

case_count=$(awk 'NF { count += 1 } END { print count + 0 }' "$CASE_LIST")
warmup_list="$RUN_DIR/warmup_case.txt"
printf '%s\n' "$WARMUP_CASE" >"$warmup_list"
summary="$RUN_DIR/runs.jsonl"
: >"$summary"

run_one() {
  local workers=$1 repeat=$2 output="$RUN_DIR/workers_${workers}/repeat_${repeat}"
  local started ended elapsed status progress
  mkdir -p "$output"
  started=$(date +%s.%N)
  echo "[$(date -Is)] Starting measured run workers=$workers repeat=$repeat cases=$case_count"
  set +e
  "$PYTHON" main/run_e2e.py \
    --case-root "$CASE_ROOT" \
    --case-list "$CASE_LIST" \
    --case-workers "$workers" \
    --release \
    --release-precompute-only \
    --output-dir "$output" \
    >"$output/stdout.log" 2>"$output/stderr.log"
  status=$?
  if ((status == 0)); then
    "$PYTHON" main/run_e2e.py \
      --case-root "$CASE_ROOT" \
      --case-list "$CASE_LIST" \
      --case-workers "$workers" \
      --release \
      --release-debug-artifacts \
      --vlm-dry-run \
      --vlm-grid-layout 4x1 \
      --vlm-workers "$workers" \
      --vision-precompute-root "$output/vision_precompute" \
      --output-dir "$output" \
      >>"$output/stdout.log" 2>>"$output/stderr.log"
    status=$?
  fi
  set -e
  ended=$(date +%s.%N)
  elapsed=$(awk -v a="$started" -v b="$ended" 'BEGIN { printf "%.3f", b-a }')
  progress="$output/release_precompute_progress.json"
  "$PYTHON" - "$summary" "$workers" "$repeat" "$case_count" "$status" "$elapsed" "$progress" <<'PY'
import json
import sys
from pathlib import Path

target, workers, repeat, cases, status, elapsed, progress = sys.argv[1:]
payload = {
    "workers": int(workers), "repeat": int(repeat), "case_count": int(cases),
    "exit_code": int(status), "wall_seconds": float(elapsed),
}
source = Path(progress)
if source.is_file():
    state = json.loads(source.read_text(encoding="utf-8"))
    payload.update({key: state.get(key) for key in ("state", "completed_cases", "failed_cases", "elapsed_seconds")})
with Path(target).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
  if ((status != 0)); then
    echo "[$(date -Is)] Failed workers=$workers repeat=$repeat; see $output/stderr.log" >&2
    return "$status"
  fi
  echo "[$(date -Is)] Completed workers=$workers repeat=$repeat wall=${elapsed}s"
}

echo "Run directory: $RUN_DIR"
echo "Stage: server preprocessing only (no VLM request)"
echo "Case root: $CASE_ROOT"
echo "Case list: $CASE_LIST ($case_count cases)"
echo "Workers: ${WORKERS[*]}"
echo "Measured repeats: $REPEATS"
echo "Warmup case: $WARMUP_CASE"

echo "[$(date -Is)] Starting unmeasured runtime warmup..."
"$PYTHON" main/run_e2e.py \
  --case-root "$CASE_ROOT" \
  --case-list "$warmup_list" \
  --case-workers 1 \
  --release \
  --release-precompute-only \
  --output-dir "$RUN_DIR/warmup" \
  >"$RUN_DIR/warmup.stdout.log" 2>"$RUN_DIR/warmup.stderr.log"
"$PYTHON" main/run_e2e.py \
  --case-root "$CASE_ROOT" \
  --case-list "$warmup_list" \
  --case-workers 1 \
  --release \
  --release-debug-artifacts \
  --vlm-dry-run \
  --vlm-grid-layout 4x1 \
  --vlm-workers 1 \
  --vision-precompute-root "$RUN_DIR/warmup/vision_precompute" \
  --output-dir "$RUN_DIR/warmup" \
  >>"$RUN_DIR/warmup.stdout.log" 2>>"$RUN_DIR/warmup.stderr.log"
echo "[$(date -Is)] Warmup complete."

for workers in "${WORKERS[@]}"; do
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    run_one "$workers" "$repeat"
  done
done

"$PYTHON" - "$summary" "$RUN_DIR/summary.json" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

records = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
groups = defaultdict(list)
for record in records:
    groups[record["workers"]].append(record)
rows = []
for workers, items in sorted(groups.items()):
    successful = [item for item in items if item["exit_code"] == 0 and item.get("completed_cases") == item["case_count"]]
    durations = sorted(item["wall_seconds"] for item in successful)
    def percentile(p):
        if not durations:
            return None
        position = (len(durations) - 1) * p
        lower, upper = math.floor(position), math.ceil(position)
        return durations[lower] if lower == upper else durations[lower] + (durations[upper] - durations[lower]) * (position - lower)
    rows.append({
        "workers": workers,
        "measured_repeats": len(items),
        "successful_repeats": len(successful),
        "successful_cases_per_repeat": successful[0]["case_count"] if successful else 0,
        "throughput_case_per_min": round(statistics.mean(item["case_count"] / item["wall_seconds"] * 60 for item in successful), 3) if successful else None,
        "batch_seconds_p50": round(percentile(0.5), 3) if durations else None,
        "batch_seconds_p95": round(percentile(0.95), 3) if durations else None,
    })
Path(sys.argv[2]).write_text(json.dumps({"rows": rows, "runs": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
PY

echo "Done. Summary: $RUN_DIR/summary.json"
