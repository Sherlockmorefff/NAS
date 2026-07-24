#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 --final-base-seed N [--run-dir DIR] [--top-k N] [--n-replicates N]"
  echo "          [--eval-epochs N] [--patience N] [--train] [--resume]"
  echo
  echo "Without --train, all 15 histories are preflighted and no Cora training starts."
}

FINAL_BASE_SEED=""
RUN_DIR=""
TOP_K=10
N_REPLICATES=10
EVAL_EPOCHS=300
PATIENCE=80
START_TRAINING=0
RESUME=0

while (($#)); do
  case "$1" in
    --final-base-seed)
      FINAL_BASE_SEED="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --n-replicates)
      N_REPLICATES="$2"
      shift 2
      ;;
    --eval-epochs)
      EVAL_EPOCHS="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --train)
      START_TRAINING=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$FINAL_BASE_SEED" ]]; then
  echo "--final-base-seed is required" >&2
  exit 2
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for python_candidate in .venv310/bin/python .venv/bin/python python3 python; do
    if command -v "$python_candidate" >/dev/null 2>&1 &&
      "$python_candidate" -c 'import numpy, torch, torch_geometric' >/dev/null 2>&1; then
      PYTHON_BIN="$python_candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter with numpy, torch, and torch_geometric was found." >&2
  exit 2
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="results/final_eval_topk_seedfair_$(date +%Y%m%d_%H%M%S)"
elif [[ -d "$RUN_DIR" ]]; then
  RESUME=1
fi

RUN_NAME="$(basename "$RUN_DIR")"
LOG_DIR="${LOG_DIR:-logs/final_eval_topk_seedfair/$RUN_NAME}"
STATUS_FILE="$RUN_DIR/subtask_exit_codes.tsv"
mkdir -p "$RUN_DIR" "$LOG_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf "timestamp\tstage\tmethod_label\tsearch_seed\texit_code\toutput\n" > "$STATUS_FILE"
fi

history_path() {
  local method_label="$1"
  local search_seed="$2"
  case "$method_label" in
    schur)
      printf "results/formal_seedfair_full300_schur_global4_seed%s_pool768/history_final.json" "$search_seed"
      ;;
    gmm_exp100)
      printf "results/formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp100_global4_seed%s_pool768/history_final.json" "$search_seed"
      ;;
    gmm_exp150)
      printf "results/formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp150_global4_seed%s_pool768/history_final.json" "$search_seed"
      ;;
    *)
      return 2
      ;;
  esac
}

record_status() {
  local stage="$1"
  local method_label="$2"
  local search_seed="$3"
  local exit_code="$4"
  local output="$5"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Iseconds)" "$stage" "$method_label" "$search_seed" "$exit_code" "$output" \
    >> "$STATUS_FILE"
}

run_task() {
  local stage="$1"
  local method_label="$2"
  local search_seed="$3"
  local output="$4"
  local history
  history="$(history_path "$method_label" "$search_seed")"
  local command=(
    "$PYTHON_BIN" final_eval.py
    --history_path "$history"
    --method_label "$method_label"
    --search_seed "$search_seed"
    --final_base_seed "$FINAL_BASE_SEED"
    --top_k "$TOP_K"
    --n_replicates "$N_REPLICATES"
    --eval_epochs "$EVAL_EPOCHS"
    --patience "$PATIENCE"
    --hp_mode global4
    --output "$output"
    --log_dir "$LOG_DIR"
    --version "${stage}_${method_label}_seed${search_seed}"
  )
  if [[ "$stage" == "preflight" ]]; then
    command+=(--dry_run)
  fi
  if [[ "$RESUME" -eq 1 && -f "$output/final_eval_config.json" ]]; then
    command+=(--resume)
  fi

  echo "[$stage] method=$method_label search_seed=$search_seed output=$output"
  "${command[@]}"
  local exit_code=$?
  record_status "$stage" "$method_label" "$search_seed" "$exit_code" "$output"
  return "$exit_code"
}

METHODS=(schur gmm_exp100 gmm_exp150)
PREFLIGHT_FAILED=0
for method_label in "${METHODS[@]}"; do
  for search_seed in 0 1 2 3 4; do
    preflight_output="$RUN_DIR/preflight/${method_label}_seed${search_seed}"
    run_task preflight "$method_label" "$search_seed" "$preflight_output" || PREFLIGHT_FAILED=1
  done
done

if [[ "$PREFLIGHT_FAILED" -ne 0 ]]; then
  echo "At least one preflight failed. No training task was started." >&2
  exit 1
fi
echo "All 15 history preflights passed."

if [[ "$START_TRAINING" -ne 1 ]]; then
  echo "Preflight-only run complete. Re-run with the same --run-dir and --train on the GPU server."
  exit 0
fi

"$PYTHON_BIN" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
CUDA_EXIT=$?
record_status gpu_check all - "$CUDA_EXIT" "$RUN_DIR"
if [[ "$CUDA_EXIT" -ne 0 ]]; then
  echo "CUDA is unavailable. Refusing to start full final evaluation." >&2
  exit 1
fi

TRAIN_FAILED=0
for method_label in "${METHODS[@]}"; do
  for search_seed in 0 1 2 3 4; do
    task_output="$RUN_DIR/runs/${method_label}_seed${search_seed}"
    run_task train "$method_label" "$search_seed" "$task_output" || TRAIN_FAILED=1
  done
done

if [[ "$TRAIN_FAILED" -ne 0 ]]; then
  echo "One or more training subtasks failed; inspect $STATUS_FILE and resume with the same --run-dir." >&2
  exit 1
fi
echo "All 15 final-evaluation subtasks completed."
