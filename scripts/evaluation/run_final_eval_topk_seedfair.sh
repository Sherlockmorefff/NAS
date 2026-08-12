#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  echo "Usage: $0 --final-base-seed N --protocol-id ID [--run-dir DIR] [--top-k N]"
  echo "          [--dataset NAME] [--data-root DIR] [--search-run-tag TAG]"
  echo "          [--search-protocol-id ID] [--legacy-root DIR]"
  echo "          [--eval-epochs N] [--patience N] [--python PATH] [--train] [--resume]"
  echo
  echo "Without --train, all 15 histories are preflighted and no GNN training starts."
  echo "Set PYTHON_BIN=/path/to/python as an alternative to --python."
  echo "The historical root command run_final_eval_topk_seedfair.sh remains supported."
}

FINAL_BASE_SEED=""
RUN_DIR=""
TOP_K=10
N_REPLICATES=10
EVAL_EPOCHS=300
PATIENCE=80
START_TRAINING=0
RESUME=0
DATASET="cora"
DATA_ROOT=""
SEARCH_RUN_TAG=""
PROTOCOL_ID=""
SEARCH_PROTOCOL_ID=""
LEGACY_ROOT="legacy_artifacts/pre_20260813"
PYTHON_OVERRIDE=""

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
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --search-run-tag)
      SEARCH_RUN_TAG="$2"
      shift 2
      ;;
    --protocol-id)
      PROTOCOL_ID="$2"
      shift 2
      ;;
    --search-protocol-id)
      SEARCH_PROTOCOL_ID="$2"
      shift 2
      ;;
    --legacy-root)
      LEGACY_ROOT="$2"
      shift 2
      ;;
    --python)
      if [[ -z "${2:-}" ]]; then
        echo "--python requires a path or command name" >&2
        exit 2
      fi
      PYTHON_OVERRIDE="$2"
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
    --print-repo-root)
      printf '%s\n' "$REPO_ROOT"
      exit 0
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

case "${DATASET,,}" in
  cora) DATASET="cora" ;;
  citeseer|cite-seer|cite_seer) DATASET="citeseer" ;;
  pubmed|pub-med|pub_med) DATASET="pubmed" ;;
  dblp|citationfull-dblp|citationfull_dblp) DATASET="dblp" ;;
  flickr) DATASET="flickr" ;;
  ogbn-arxiv|ogbn_arxiv) DATASET="ogbn-arxiv" ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac
if [[ -z "$DATA_ROOT" ]]; then
  if [[ "$DATASET" == "cora" ]]; then
    DATA_ROOT="/tmp/Cora"
  else
    DATA_ROOT="/tmp/gnn_datasets"
  fi
fi
if [[ "$DATASET" != "cora" && -z "$SEARCH_RUN_TAG" && -z "$SEARCH_PROTOCOL_ID" ]]; then
  echo "--search-protocol-id or legacy --search-run-tag is required for non-Cora histories" >&2
  exit 2
fi

if [[ -z "$FINAL_BASE_SEED" ]]; then
  echo "--final-base-seed is required" >&2
  exit 2
fi

python_command_exists() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]]
  else
    command -v "$candidate" >/dev/null 2>&1
  fi
}

python_has_dependencies() {
  local candidate="$1"
  python_command_exists "$candidate" &&
    "$candidate" -c 'import numpy, torch, torch_geometric' >/dev/null 2>&1
}

if [[ -n "$PYTHON_OVERRIDE" ]]; then
  PYTHON_BIN="$PYTHON_OVERRIDE"
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! python_has_dependencies "$PYTHON_BIN"; then
    echo "Configured Python cannot import numpy, torch, and torch_geometric: $PYTHON_BIN" >&2
    exit 2
  fi
else
  for python_candidate in "${PYTHON:-python}" .venv310/bin/python .venv/bin/python python3; do
    if python_has_dependencies "$python_candidate"; then
      PYTHON_BIN="$python_candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter with numpy, torch, and torch_geometric was found." >&2
  echo "Activate the intended environment, pass --python PATH, or set PYTHON_BIN." >&2
  exit 2
fi

if [[ -z "$RUN_DIR" ]]; then
  if [[ -z "$PROTOCOL_ID" ]]; then
    echo "--protocol-id is required unless the legacy --run-dir override is explicit" >&2
    exit 2
  fi
  RUN_DIR="results/final_eval/$PROTOCOL_ID/$DATASET"
elif [[ -d "$RUN_DIR" ]]; then
  RESUME=1
fi

RUN_NAME="$(basename "$RUN_DIR")"
if [[ -n "$PROTOCOL_ID" ]]; then
  LOG_DIR="${LOG_DIR:-logs/$PROTOCOL_ID/final_eval/$DATASET}"
else
  LOG_DIR="${LOG_DIR:-logs/final_eval_topk_seedfair/$RUN_NAME}"
fi
STATUS_FILE="$RUN_DIR/subtask_exit_codes.tsv"
mkdir -p "$RUN_DIR" "$LOG_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf "timestamp\tstage\tmethod_label\tsearch_seed\texit_code\toutput\n" > "$STATUS_FILE"
fi

history_path() {
  local method_label="$1"
  local search_seed="$2"
  local method_key=""
  case "$method_label" in
    global_schur) method_key="s0" ;;
    gmm_exp100) method_key="g100" ;;
    gmm_exp150) method_key="g150" ;;
    *) return 2 ;;
  esac
  if [[ -n "$SEARCH_PROTOCOL_ID" ]]; then
    printf "results/search/%s/%s/%s/search_seed%s/history_final.json" \
      "$SEARCH_PROTOCOL_ID" "$DATASET" "$method_key" "$search_seed"
    return
  fi
  if [[ -z "$SEARCH_RUN_TAG" && "$DATASET" == "cora" ]]; then
    case "$method_label" in
      global_schur)
        printf "%s/results/formal_seedfair_full300_schur_global4_seed%s_pool768/history_final.json" "$LEGACY_ROOT" "$search_seed"
        ;;
      gmm_exp100)
        printf "%s/results/formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp100_global4_seed%s_pool768/history_final.json" "$LEGACY_ROOT" "$search_seed"
        ;;
      gmm_exp150)
        printf "%s/results/formal_seedfair_full300_gmm_ted_lowfid_gmm_fit_pool_exp150_global4_seed%s_pool768/history_final.json" "$LEGACY_ROOT" "$search_seed"
        ;;
      *)
        return 2
        ;;
    esac
    return
  fi
  printf "%s/results/%s/%s/%s/search_seed%s/history_final.json" \
    "$LEGACY_ROOT" "$SEARCH_RUN_TAG" "$DATASET" "$method_label" "$search_seed"
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
    "$PYTHON_BIN" "$REPO_ROOT/final_eval.py"
    --history_path "$history"
    --method_label "$method_label"
    --search_seed "$search_seed"
    --final_base_seed "$FINAL_BASE_SEED"
    --top_k "$TOP_K"
    --n_replicates "$N_REPLICATES"
    --eval_epochs "$EVAL_EPOCHS"
    --patience "$PATIENCE"
    --hp_mode global4
    --dataset "$DATASET"
    --require_cuda
    --output "$output"
    --log_dir "$LOG_DIR"
    --version "${stage}_${method_label}_seed${search_seed}"
  )
  if [[ -n "$PROTOCOL_ID" ]]; then
    command+=(--protocol-id "$PROTOCOL_ID")
  fi
  if [[ -z "$SEARCH_RUN_TAG" && -z "$SEARCH_PROTOCOL_ID" && "$DATASET" == "cora" ]]; then
    command+=(--cora_root "$DATA_ROOT")
  else
    command+=(--data_root "$DATA_ROOT")
  fi
  if [[ "$DATASET" == "dblp" ]]; then
    command+=(--split_seed 0)
  fi
  if [[ "$DATASET" == "ogbn-arxiv" ]]; then
    command+=(--ogbn_arxiv_edge_mode undirected)
  fi
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

METHODS=(global_schur gmm_exp100 gmm_exp150)
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
    case "$method_label" in
      global_schur) method_dir="s0" ;;
      gmm_exp100) method_dir="g100" ;;
      gmm_exp150) method_dir="g150" ;;
    esac
    task_output="$RUN_DIR/${method_dir}/search_seed${search_seed}"
    run_task train "$method_label" "$search_seed" "$task_output" || TRAIN_FAILED=1
  done
done

if [[ "$TRAIN_FAILED" -ne 0 ]]; then
  echo "One or more training subtasks failed; inspect $STATUS_FILE and resume with the same --run-dir." >&2
  exit 1
fi
echo "All 15 final-evaluation subtasks completed."
