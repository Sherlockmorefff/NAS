#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/joint_configuration_metric_gate_v1.yaml"
PROGRAM="$REPO_ROOT/analyse/joint_configuration_metric_gate.py"
LAUNCHER="$REPO_ROOT/scripts/run_joint_configuration_metric_gate.sh"
FOCUSED_TEST="$REPO_ROOT/tests/test_joint_configuration_metric_gate.py"
REUSED_V2="$REPO_ROOT/analyse/performance_metric_quick_gate.py"
OUTPUT="$REPO_ROOT/results/posthoc/joint-configuration-metric-gate-global4-v1"
ARCHIVE="$REPO_ROOT/results/posthoc/joint-configuration-metric-gate-global4-v1.tar.gz"
ARCHIVE_SHA256="$ARCHIVE.sha256"

dry_run=false
restart_after_source_fix=false
if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run) dry_run=true ;;
    --restart-after-source-fix) restart_after_source_fix=true ;;
    *) echo "usage: $0 [--dry-run|--restart-after-source-fix]" >&2; exit 2 ;;
  esac
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run|--restart-after-source-fix]" >&2
  exit 2
fi

for required_file in "$CONFIG" "$PROGRAM" "$LAUNCHER" "$FOCUSED_TEST" "$REUSED_V2"; do
  if [[ ! -f "$required_file" ]]; then
    echo "required experiment file is missing: $required_file" >&2
    exit 4
  fi
done

HASH_PYTHON="${JOINT_CONFIGURATION_METRIC_PYTHON:-python}"
if ! "$HASH_PYTHON" -c 'import hashlib,pathlib' >/dev/null 2>&1; then
  HASH_PYTHON=python
fi
CONFIG_SHA256="$("$HASH_PYTHON" - "$CONFIG" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
SOURCE_SHA256="$("$HASH_PYTHON" - "$PROGRAM" "$LAUNCHER" "$FOCUSED_TEST" "$REUSED_V2" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

digest = sha256()
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"

common_args=(
  --config "$CONFIG"
  --output "$OUTPUT"
  --config-sha256 "$CONFIG_SHA256"
  --source-sha256 "$SOURCE_SHA256"
)

TASK_PYTHON="${JOINT_CONFIGURATION_METRIC_PYTHON:-python}"
required_imports='import gpytorch,numpy,scipy,sklearn,torch,yaml'
if ! "$TASK_PYTHON" -c "$required_imports" >/dev/null 2>&1; then
  repository_parent="$(dirname "$REPO_ROOT")"
  conda_candidate="$repository_parent/.conda/envs/nas/bin/python"
  if [[ -x "$conda_candidate" ]] && "$conda_candidate" -c "$required_imports" >/dev/null 2>&1; then
    TASK_PYTHON="$conda_candidate"
  else
    if [[ "$dry_run" == true ]]; then
      echo "no Python environment provides gpytorch, numpy, scipy, sklearn, torch, and yaml" >&2
      exit 4
    fi
    if [[ -e "$OUTPUT" && ! -d "$OUTPUT" ]]; then
      echo "formal result path exists and is not a directory: $OUTPUT" >&2
      exit 3
    fi
    if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
      echo "formal result directory exists and is non-empty: $OUTPUT" >&2
      exit 3
    fi
    if [[ -e "$ARCHIVE" || -e "$ARCHIVE_SHA256" ]]; then
      echo "formal archive or SHA-256 sidecar already exists; refusing to overwrite" >&2
      exit 3
    fi
    mkdir -p "$OUTPUT/logs"
    dependency_message="no Python environment provides gpytorch, numpy, scipy, sklearn, torch, and yaml"
    : >"$OUTPUT/logs/py_compile.log"
    : >"$OUTPUT/logs/pytest_focused.log"
    : >"$OUTPUT/logs/synthetic_smoke.log"
    printf '%s\n' "$dependency_message" >"$OUTPUT/logs/run.log"
    "$HASH_PYTHON" "$PROGRAM" "${common_args[@]}" --write-failure-bundle "$dependency_message" \
      >>"$OUTPUT/logs/run.log" 2>&1
    touch "$OUTPUT/FAILED"
    "$HASH_PYTHON" "$PROGRAM" "${common_args[@]}" --refresh-manifest \
      >>"$OUTPUT/logs/run.log" 2>&1
    temporary_archive="$ARCHIVE.tmp-$$"
    temporary_sidecar="$ARCHIVE_SHA256.tmp-$$"
    tar -czf "$temporary_archive" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
    "$HASH_PYTHON" - "$temporary_archive" "$temporary_sidecar" "$(basename "$ARCHIVE")" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

archive = Path(sys.argv[1])
Path(sys.argv[2]).write_text(f"{sha256(archive.read_bytes()).hexdigest()}  {sys.argv[3]}\n", encoding="utf-8")
PY
    mv "$temporary_archive" "$ARCHIVE"
    mv "$temporary_sidecar" "$ARCHIVE_SHA256"
    echo "$dependency_message" >&2
    exit 4
  fi
fi

if [[ "$dry_run" == true ]]; then
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --dry-run
  exit 0
fi

if [[ "$restart_after_source_fix" == true ]]; then
  if [[ ! -f "$OUTPUT/state.json" ]] || [[ ! -f "$OUTPUT/FAILED" ]] || [[ -f "$OUTPUT/COMPLETE" ]]; then
    echo "source-fix restart requires a failed, incomplete formal result" >&2
    exit 3
  fi
  state_fields="$("$HASH_PYTHON" - "$OUTPUT/state.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
print(state.get("status", ""))
print(state.get("config_sha256", ""))
print(state.get("source_sha256", ""))
print("null" if state.get("split_sha256") is None else state.get("split_sha256"))
PY
)"
  mapfile -t state_values <<<"$state_fields"
  old_status="${state_values[0]:-}"
  old_config_sha256="${state_values[1]:-}"
  old_source_sha256="${state_values[2]:-}"
  old_split_sha256="${state_values[3]:-}"
  if [[ "$old_status" != "failed" ]] || [[ "$old_config_sha256" != "$CONFIG_SHA256" ]] || \
     [[ -z "$old_source_sha256" ]] || [[ "$old_source_sha256" == "$SOURCE_SHA256" ]] || \
     [[ "$old_split_sha256" != "null" ]]; then
    echo "source-fix restart admission failed: require failed pre-split state, same config, changed source" >&2
    exit 3
  fi
  preservation_suffix="failed-attempt-${old_source_sha256:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  preserved_output="$OUTPUT.$preservation_suffix"
  preserved_archive="${ARCHIVE%.tar.gz}.$preservation_suffix.tar.gz"
  preserved_archive_sha256="$preserved_archive.sha256"
  if [[ -e "$preserved_output" ]] || [[ -e "$preserved_archive" ]] || [[ -e "$preserved_archive_sha256" ]]; then
    echo "source-fix preservation target already exists; refusing to overwrite" >&2
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

if [[ -e "$OUTPUT" && ! -d "$OUTPUT" ]]; then
  echo "formal result path exists and is not a directory: $OUTPUT" >&2
  exit 3
fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
  echo "formal result directory exists and is non-empty: $OUTPUT" >&2
  exit 3
fi
if [[ -e "$ARCHIVE" || -e "$ARCHIVE_SHA256" ]]; then
  echo "formal archive or SHA-256 sidecar already exists; refusing to overwrite" >&2
  exit 3
fi

mkdir -p "$OUTPUT/logs"
PY_COMPILE_LOG="$OUTPUT/logs/py_compile.log"
PYTEST_LOG="$OUTPUT/logs/pytest_focused.log"
SYNTHETIC_LOG="$OUTPUT/logs/synthetic_smoke.log"
RUN_LOG="$OUTPUT/logs/run.log"
: >"$PY_COMPILE_LOG"
: >"$PYTEST_LOG"
: >"$SYNTHETIC_LOG"
: >"$RUN_LOG"

run_logged() {
  local log_path="$1"
  shift
  local command_status tee_status
  local -a pipeline_status
  set +e
  "$@" 2>&1 | tee "$log_path"
  pipeline_status=("${PIPESTATUS[@]}")
  command_status=${pipeline_status[0]}
  tee_status=${pipeline_status[1]}
  set -e
  if [[ $command_status -ne 0 ]]; then
    return "$command_status"
  fi
  return "$tee_status"
}

package_result() {
  local temporary_archive temporary_sidecar
  if [[ -e "$ARCHIVE" || -e "$ARCHIVE_SHA256" ]]; then
    echo "formal archive or SHA-256 sidecar appeared during execution; refusing to overwrite" >&2
    return 3
  fi
  temporary_archive="$ARCHIVE.tmp-$$"
  temporary_sidecar="$ARCHIVE_SHA256.tmp-$$"
  if [[ -e "$temporary_archive" || -e "$temporary_sidecar" ]]; then
    echo "temporary archive target already exists; refusing to overwrite" >&2
    return 3
  fi
  tar \
    --exclude="$(basename "$OUTPUT")/cache" \
    --exclude="$(basename "$OUTPUT")/cache/*" \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    -czf "$temporary_archive" \
    -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")" || return $?
  "$TASK_PYTHON" - "$temporary_archive" "$temporary_sidecar" "$(basename "$ARCHIVE")" <<'PY' || return $?
from hashlib import sha256
from pathlib import Path
import sys

archive = Path(sys.argv[1])
sidecar = Path(sys.argv[2])
sidecar.write_text(f"{sha256(archive.read_bytes()).hexdigest()}  {sys.argv[3]}\n", encoding="utf-8")
PY
  mv "$temporary_archive" "$ARCHIVE" || return $?
  mv "$temporary_sidecar" "$ARCHIVE_SHA256" || return $?
}

fallback_failure_bundle() {
  local message="$1"
  "$HASH_PYTHON" - "$OUTPUT" "$message" "$CONFIG_SHA256" "$SOURCE_SHA256" "$REPO_ROOT" <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

output = Path(sys.argv[1])
message = sys.argv[2]
config_hash = sys.argv[3]
source_hash = sys.argv[4]
repo = Path(sys.argv[5])
output.mkdir(parents=True, exist_ok=True)
(output / "logs").mkdir(parents=True, exist_ok=True)
for name in ("py_compile.log", "pytest_focused.log", "synthetic_smoke.log", "run.log"):
    (output / "logs" / name).touch(exist_ok=True)

def write_json(relative, value, overwrite=False):
    path = output / relative
    if overwrite or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

now = datetime.now(timezone.utc).isoformat()
write_json("state.json", {
    "format_version": "joint_configuration_metric_gate_global4_v1",
    "status": "failed", "updated_at_utc": now, "config_sha256": config_hash,
    "source_sha256": source_hash, "decision": "RUN_FAILED", "error": message,
    "formal_gnn_training_started": False, "test_data_accessed": False,
}, overwrite=True)
write_json("frozen_protocol.json", {
    "format_version": "joint_configuration_metric_gate_global4_v1",
    "config_sha256": config_hash, "source_sha256": source_hash,
    "status": "fallback_failure_bundle",
})
write_json("data_inventory.json", {"status": "failed_before_or_during_inventory", "error": message})
write_json("data_provenance.json", {"status": "failed_before_or_during_provenance", "error": message})
for relative in ("split_manifest.json", "pair_inventory.json", "metric_training_audit.json", "geometry_metrics.json"):
    write_json(relative, {"format_version": "joint_configuration_metric_gate_global4_v1", "folds": [], "reason": message})
write_json("best_of_4_batches.json", {"format_version": "joint_configuration_metric_gate_global4_v1", "folds": [], "reason": message})
write_json("best_of_4_results.json", {"format_version": "joint_configuration_metric_gate_global4_v1", "models": {}, "reason": message})
write_json("branch_ablation.json", {"format_version": "joint_configuration_metric_gate_global4_v1", "reason": message})
write_json("shuffle_controls.json", {"format_version": "joint_configuration_metric_gate_global4_v1", "reason": message})
for relative, fields in (
    ("metrics_per_fold.csv", ("fold", "model", "mae")),
    ("oof_predictions.csv", ("fold", "joint_fingerprint", "raw_validation_mean")),
):
    path = output / relative
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(fields)
decision = {
    "classification": "RUN_FAILED", "reason": message,
    "failure_code": "LAUNCHER_FALLBACK", "bo_authorized": False,
    "uncertainty_calibration_claimed": False,
}
write_json("decision.json", decision, overwrite=True)
(output / "final_report.md").write_text(
    "# Joint configuration metric gate v1\n\n"
    "- Frozen classification: `RUN_FAILED`\n"
    f"- Reason: `{message}`\n"
    "- No BO, uncertainty, or acquisition claim is authorized.\n",
    encoding="utf-8",
)
try:
    status = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout.strip()
except Exception as error:
    status = f"unavailable: {error}\n"
    branch = "unavailable"
    head = "unavailable"
if not (output / "git_status.txt").exists():
    (output / "git_status.txt").write_text(f"branch={branch}\nHEAD={head}\n{status}", encoding="utf-8")
if not (output / "git_diff.patch").exists():
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        ).stdout
    except Exception:
        diff = ""
    (output / "git_diff.patch").write_text(diff, encoding="utf-8")
write_json("environment.json", {
    "timestamp_utc": now, "python": sys.version, "platform": platform.platform(),
    "branch": branch, "head": head, "fallback_failure_bundle": True,
})
complete = output / "COMPLETE"
if complete.exists():
    complete.rename(output / f"COMPLETE.interrupted-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
(output / "FAILED").touch()
files = []
for path in sorted(output.rglob("*")):
    if not path.is_file() or path.name == "manifest.json":
        continue
    relative = path.relative_to(output).as_posix()
    files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()})
write_json("manifest.json", {
    "format_version": "joint_configuration_metric_gate_global4_v1",
    "generated_at_utc": now, "files": files, "file_count": len(files),
}, overwrite=True)
PY
}

quarantine_interrupted_package_files() {
  local suffix path
  suffix="interrupted-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  for path in \
    "$ARCHIVE" "$ARCHIVE_SHA256" \
    "$ARCHIVE.tmp-$$" "$ARCHIVE_SHA256.tmp-$$"; do
    if [[ -e "$path" ]]; then
      mv "$path" "$path.$suffix" || return $?
    fi
  done
}

failure_active=false
write_failure_and_exit() {
  local message="$1"
  local exit_status="$2"
  local bundle_status refresh_status package_status
  if [[ "$failure_active" == true ]]; then
    exit "$exit_status"
  fi
  failure_active=true
  trap - ERR INT TERM
  set +e
  touch "$PY_COMPILE_LOG" "$PYTEST_LOG" "$SYNTHETIC_LOG" "$RUN_LOG"
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" \
    --write-failure-bundle "$message" >>"$RUN_LOG" 2>&1
  bundle_status=$?
  if [[ $bundle_status -ne 0 ]]; then
    fallback_failure_bundle "$message; program failure-bundle status $bundle_status" >>"$RUN_LOG" 2>&1
    bundle_status=$?
    if [[ $bundle_status -ne 0 ]]; then
      echo "both program and fallback failure-bundle writers failed" >&2
      exit 125
    fi
  fi
  if [[ -f "$OUTPUT/COMPLETE" ]]; then
    mv "$OUTPUT/COMPLETE" "$OUTPUT/COMPLETE.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  touch "$OUTPUT/FAILED"
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --refresh-manifest >>"$RUN_LOG" 2>&1
  refresh_status=$?
  if [[ $refresh_status -ne 0 ]]; then
    fallback_failure_bundle "$message; manifest refresh status $refresh_status" >>"$RUN_LOG" 2>&1
    refresh_status=$?
    if [[ $refresh_status -ne 0 ]]; then
      echo "failure manifest refresh and fallback both failed" >&2
      exit 125
    fi
  fi
  quarantine_interrupted_package_files
  package_status=$?
  if [[ $package_status -ne 0 ]]; then
    echo "could not preserve interrupted package files" >&2
    exit 125
  fi
  package_result >>"$RUN_LOG" 2>&1
  package_status=$?
  if [[ $package_status -ne 0 ]]; then
    echo "failure bundle exists in $OUTPUT but archive packaging failed with status $package_status" >&2
    exit 125
  fi
  exit "$exit_status"
}

unexpected_error() {
  local exit_status="$1"
  local line_number="$2"
  write_failure_and_exit "launcher failed at line $line_number with status $exit_status" "$exit_status"
}

trap 'unexpected_error "$?" "$LINENO"' ERR
trap 'write_failure_and_exit "launcher interrupted by INT" 130' INT
trap 'write_failure_and_exit "launcher interrupted by TERM" 143' TERM

if run_logged "$PY_COMPILE_LOG" \
  "$TASK_PYTHON" -m py_compile "$PROGRAM" "$FOCUSED_TEST"; then
  :
else
  status=$?
  write_failure_and_exit "py_compile exited with status $status" "$status"
fi

if run_logged "$PYTEST_LOG" \
  "$TASK_PYTHON" -m pytest -q "$FOCUSED_TEST"; then
  :
else
  status=$?
  write_failure_and_exit "focused pytest exited with status $status" "$status"
fi

if run_logged "$SYNTHETIC_LOG" \
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --synthetic-smoke; then
  :
else
  status=$?
  write_failure_and_exit "synthetic smoke test exited with status $status" "$status"
fi

if run_logged "$RUN_LOG" \
  "$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --preflight-ready; then
  :
else
  status=$?
  write_failure_and_exit "formal program exited with status $status" "$status"
fi

if [[ -f "$OUTPUT/FAILED" ]]; then
  write_failure_and_exit "formal program returned success after creating FAILED" 1
fi
touch "$OUTPUT/COMPLETE"
"$TASK_PYTHON" "$PROGRAM" "${common_args[@]}" --refresh-manifest >>"$RUN_LOG" 2>&1
package_result
