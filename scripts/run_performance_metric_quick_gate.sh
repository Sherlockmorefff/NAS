#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/performance_metric_quick_gate_v2.yaml"
PROGRAM="$REPO_ROOT/analyse/performance_metric_quick_gate.py"
LAUNCHER="$REPO_ROOT/scripts/run_performance_metric_quick_gate.sh"
OUTPUT="$REPO_ROOT/results/posthoc/performance-metric-quick-gate-global4-v2"
ARCHIVE="$REPO_ROOT/results/posthoc/performance-metric-quick-gate-global4-v2.tar.gz"
ARCHIVE_SHA256="$ARCHIVE.sha256"

TASK_PYTHON="${PERFORMANCE_METRIC_PYTHON:-python}"
if ! "$TASK_PYTHON" -c 'import gpytorch,numpy,scipy,sklearn,torch,yaml' >/dev/null 2>&1; then
  repository_parent="$(dirname "$REPO_ROOT")"
  conda_candidate="$repository_parent/.conda/envs/nas/bin/python"
  if [[ -x "$conda_candidate" ]] && "$conda_candidate" -c 'import gpytorch,numpy,scipy,sklearn,torch,yaml' >/dev/null 2>&1; then
    TASK_PYTHON="$conda_candidate"
  else
    echo "no Python environment provides gpytorch, numpy, scipy, sklearn, torch, and yaml" >&2
    exit 4
  fi
fi

resume=false
dry_run=false
restart_after_source_fix=false
for argument in "$@"; do
  case "$argument" in
    --resume) resume=true ;;
    --dry-run) dry_run=true ;;
    --restart-after-source-fix) restart_after_source_fix=true ;;
    *) echo "unsupported argument: $argument" >&2; exit 2 ;;
  esac
done

if [[ "$resume" == true && "$restart_after_source_fix" == true ]] || \
   [[ "$dry_run" == true && "$restart_after_source_fix" == true ]]; then
  echo "--restart-after-source-fix cannot be combined with --resume or --dry-run" >&2
  exit 2
fi

CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
SOURCE_SHA256="$(sha256sum "$PROGRAM" "$LAUNCHER" | sha256sum | awk '{print $1}')"

common_args=(
  --config "$CONFIG"
  --output "$OUTPUT"
  --config-sha256 "$CONFIG_SHA256"
  --source-sha256 "$SOURCE_SHA256"
)

if [[ "$dry_run" == true ]]; then
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --dry-run
  exit 0
fi

if [[ "$restart_after_source_fix" == true ]]; then
  if [[ ! -f "$OUTPUT/state.json" ]] || [[ -f "$OUTPUT/COMPLETE" ]]; then
    echo "source-fix restart requires a failed, incomplete state.json" >&2
    exit 3
  fi
  state_fields="$("$TASK_PYTHON" - "$OUTPUT/state.json" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
print(state.get("status", ""))
print(state.get("config_sha256", ""))
print(state.get("source_sha256", ""))
print("null" if state.get("split_sha256") is None else state.get("split_sha256"))
PY
)"
  mapfile -t state_values <<< "$state_fields"
  old_status="${state_values[0]:-}"
  old_config_sha256="${state_values[1]:-}"
  old_source_sha256="${state_values[2]:-}"
  old_split_sha256="${state_values[3]:-}"
  if [[ "$old_status" != "failed" ]] || [[ "$old_config_sha256" != "$CONFIG_SHA256" ]] || \
     [[ "$old_source_sha256" == "$SOURCE_SHA256" ]] || [[ "$old_split_sha256" != "null" ]]; then
    echo "source-fix restart admission failed: expected failed pre-split state, same config, changed source" >&2
    exit 3
  fi
  preservation_suffix="failed-attempt-${old_source_sha256:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  preserved_output="$OUTPUT.$preservation_suffix"
  preserved_archive="${ARCHIVE%.tar.gz}.$preservation_suffix.tar.gz"
  preserved_archive_sha256="$preserved_archive.sha256"
  if [[ -e "$preserved_output" ]] || [[ -e "$preserved_archive" ]] || [[ -e "$preserved_archive_sha256" ]]; then
    echo "preservation target already exists; refusing to overwrite" >&2
    exit 3
  fi
  mv "$OUTPUT" "$preserved_output"
  if [[ -e "$ARCHIVE" ]]; then
    mv "$ARCHIVE" "$preserved_archive"
  fi
  if [[ -e "$ARCHIVE_SHA256" ]]; then
    mv "$ARCHIVE_SHA256" "$preserved_archive_sha256"
  fi
  echo "preserved failed result: $preserved_output"
  if [[ -e "$preserved_archive" ]]; then
    echo "preserved failed archive: $preserved_archive"
  fi
fi

if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ "$resume" != true ]]; then
  echo "formal result directory exists and is non-empty: $OUTPUT" >&2
  exit 3
fi
if [[ "$resume" == true ]] && [[ -f "$OUTPUT/COMPLETE" ]]; then
  echo "completed formal result cannot be resumed: $OUTPUT" >&2
  exit 3
fi
if [[ "$resume" != true ]] && { [[ -e "$ARCHIVE" ]] || [[ -e "$ARCHIVE_SHA256" ]]; }; then
  echo "formal archive path already exists; refusing to overwrite" >&2
  exit 3
fi

mkdir -p "$OUTPUT/logs"
LOG="$OUTPUT/logs/run.log"

package_result() {
  local previous_suffix temporary
  previous_suffix="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -e "$ARCHIVE" ]]; then
    mv "$ARCHIVE" "$ARCHIVE.previous-$previous_suffix"
  fi
  if [[ -e "$ARCHIVE_SHA256" ]]; then
    mv "$ARCHIVE_SHA256" "$ARCHIVE_SHA256.previous-$previous_suffix"
  fi
  temporary="$ARCHIVE.tmp-$$"
  tar \
    --exclude="$(basename "$OUTPUT")/cache" \
    --exclude="$(basename "$OUTPUT")/cache/*" \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    -czf "$temporary" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
  mv "$temporary" "$ARCHIVE"
  sha256sum "$ARCHIVE" > "$ARCHIVE_SHA256.tmp-$$"
  mv "$ARCHIVE_SHA256.tmp-$$" "$ARCHIVE_SHA256"
}

mark_failed() {
  local message="$1"
  set +e
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --mark-failed "$message" >/dev/null 2>&1
  if [[ -f "$OUTPUT/COMPLETE" ]]; then
    mv "$OUTPUT/COMPLETE" "$OUTPUT/COMPLETE.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  touch "$OUTPUT/FAILED"
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --refresh-manifest >/dev/null 2>&1
  package_result
}

trap 'mark_failed "launcher interrupted"; exit 130' INT TERM

run_args=("${common_args[@]}")
if [[ "$resume" == true ]]; then
  run_args+=(--resume)
fi

set +e
if [[ "$resume" == true ]]; then
  "$TASK_PYTHON" "$PROGRAM" "${run_args[@]}" 2>&1 | tee -a "$LOG"
else
  "$TASK_PYTHON" "$PROGRAM" "${run_args[@]}" 2>&1 | tee "$LOG"
fi
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 ]]; then
  mark_failed "formal program exited with status $status"
  exit "$status"
fi

if [[ -f "$OUTPUT/FAILED" ]]; then
  mv "$OUTPUT/FAILED" "$OUTPUT/FAILED.previous-$(date -u +%Y%m%dT%H%M%SZ)"
fi
touch "$OUTPUT/COMPLETE"
"$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --refresh-manifest >/dev/null
package_result
