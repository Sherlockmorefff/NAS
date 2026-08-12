#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYSE_DIR="$REPO_ROOT/analyse"
cd "$REPO_ROOT"

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--python PATH] [--print-repo-root]

Run the repository analysis and diagnostics suite from any working directory.
Set PYTHON=/path/to/python to override the default current-environment python.
--dry-run prints commands without creating directories or executing Python.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
PY="${PYTHON:-python}"
while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --python)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --python requires a path or command name" >&2
        exit 2
      fi
      PY="$2"
      shift 2
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
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$PY" == */* ]]; then
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python interpreter is not executable: $PY" >&2
    exit 2
  fi
elif ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: $PY" >&2
  echo "Set PYTHON=/path/to/python or pass --python PATH." >&2
  exit 2
fi

BUNDLE_DIR="${BUNDLE_DIR:-results_analysis_bundle_after_diagnostics}"
DIAG_DIR="${DIAG_DIR:-results/diagnostics}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
CORA_ROOT="${CORA_ROOT:-}"
RUN_FULL_SANITY="${RUN_FULL_SANITY:-0}"
RUN_OFFICIAL_BASELINE="${RUN_OFFICIAL_BASELINE:-0}"
RUN_CANONICAL_DIAGNOSTICS="${RUN_CANONICAL_DIAGNOSTICS:-1}"

result_root_has_artifacts() {
  [ -d "$1" ] || return 1
  find "$1" \( -name history_final.json -o -name 'final_results*.json' \) -print -quit 2>/dev/null | grep -q .
}

detect_results_root() {
  if result_root_has_artifacts "results"; then
    echo "results"
    return
  fi
  if result_root_has_artifacts "results-wgmm1"; then
    echo "results-wgmm1"
    return
  fi
  for candidate in results-*; do
    if result_root_has_artifacts "$candidate"; then
      echo "$candidate"
      return
    fi
  done
  echo "results"
}

if [ -z "$RESULTS_ROOT" ]; then
  RESULTS_ROOT="$(detect_results_root)"
fi

if [ "$BUNDLE_DIR" = "results_analysis_bundle_after_diagnostics" ] && [ "$RESULTS_ROOT" != "results" ]; then
  safe_results_name="${RESULTS_ROOT//\//_}"
  BUNDLE_DIR="results_analysis_bundle_after_diagnostics_${safe_results_name}"
fi

if [ -z "$CORA_ROOT" ]; then
  if [ -d "Cora" ]; then
    CORA_ROOT="Cora"
  elif [ -d "data/Cora" ]; then
    CORA_ROOT="data/Cora"
  else
    CORA_ROOT="/tmp/Cora"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$BUNDLE_DIR" "$DIAG_DIR"
fi

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run_or_warn() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if ! "$@"; then
    echo "WARNING: command failed but diagnostics will continue: $*"
  fi
}

existing_files=()
add_if_exists() {
  if [ -f "$1" ]; then
    existing_files+=("$1")
  else
    echo "skip missing $1"
  fi
}

section "1. File Check"
echo "REPO_ROOT  : $REPO_ROOT"
echo "Python      : $PY"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "BUNDLE_DIR  : $BUNDLE_DIR"
echo "DIAG_DIR    : $DIAG_DIR"
echo "CORA_ROOT   : $CORA_ROOT"
if ! result_root_has_artifacts "$RESULTS_ROOT"; then
  echo "WARNING: RESULTS_ROOT has no history_final.json or final_results*.json artifacts: $RESULTS_ROOT"
fi

for f in \
  "$REPO_ROOT/final_eval.py" \
  "$ANALYSE_DIR/collect_experiment_results.py" \
  "$ANALYSE_DIR/analyze_nas_hpo_contributions.py" \
  "$ANALYSE_DIR/nas_pipeline_audit_report.py" \
  "$ANALYSE_DIR/sanity_check_2gcn_modes.py" \
  "$ANALYSE_DIR/canonicalize_search_space.py" \
  "$ANALYSE_DIR/diagnose_gmm_density.py" \
  "$ANALYSE_DIR/diagnose_bo_gp_surrogate.py" \
  "$REPO_ROOT/weighted_diag_gmm_init.py" \
  "$REPO_ROOT/bo_phase4.py" \
  "$REPO_ROOT/bo_phase4_tpe.py"
do
  if [ -f "$f" ]; then
    echo "OK      $f"
  else
    echo "MISSING $f"
  fi
done

section "2. Static py_compile"
for f in \
  "$REPO_ROOT/final_eval.py" \
  "$ANALYSE_DIR/collect_experiment_results.py" \
  "$ANALYSE_DIR/analyze_nas_hpo_contributions.py" \
  "$ANALYSE_DIR/nas_pipeline_audit_report.py" \
  "$ANALYSE_DIR/sanity_check_2gcn_modes.py" \
  "$ANALYSE_DIR/canonicalize_search_space.py" \
  "$ANALYSE_DIR/diagnose_gmm_density.py" \
  "$ANALYSE_DIR/diagnose_bo_gp_surrogate.py"
do
  add_if_exists "$f"
done

if [ "${#existing_files[@]}" -gt 0 ]; then
  run_or_warn "$PY" -m py_compile "${existing_files[@]}"
else
  echo "WARNING: no Python files available for py_compile"
fi

section "3. Collect Results Bundle"
if [ -f "$ANALYSE_DIR/collect_experiment_results.py" ]; then
  run_or_warn "$PY" "$ANALYSE_DIR/collect_experiment_results.py" \
    --results_root "$RESULTS_ROOT" \
    --output "$BUNDLE_DIR"
else
  echo "WARNING: collect_experiment_results.py missing; skipped"
fi

section "4. NAS/HPO Contribution Summary"
if [ -f "$ANALYSE_DIR/analyze_nas_hpo_contributions.py" ]; then
  run_or_warn "$PY" "$ANALYSE_DIR/analyze_nas_hpo_contributions.py" \
    --final_eval_summary "$BUNDLE_DIR/final_eval_summary.csv" \
    --search_summary "$BUNDLE_DIR/search_runs_summary.csv" \
    --output_csv "$BUNDLE_DIR/nas_hpo_contribution_summary.csv" \
    --output_md "$BUNDLE_DIR/nas_hpo_contribution_summary.md"
else
  echo "WARNING: analyze_nas_hpo_contributions.py missing; skipped"
fi

section "5. Pipeline Audit Report"
if [ -f "$ANALYSE_DIR/nas_pipeline_audit_report.py" ]; then
  run_or_warn "$PY" "$ANALYSE_DIR/nas_pipeline_audit_report.py" \
    --results_bundle "$BUNDLE_DIR" \
    --results_root "$RESULTS_ROOT" \
    --output "$BUNDLE_DIR/nas_pipeline_audit_report.md"
else
  echo "WARNING: nas_pipeline_audit_report.py missing; skipped"
fi

section "6. Quick 2xGCN hp_mode Sanity Check"
if [ -f "$ANALYSE_DIR/sanity_check_2gcn_modes.py" ]; then
  run_or_warn "$PY" "$ANALYSE_DIR/sanity_check_2gcn_modes.py" \
    --cora_root "$CORA_ROOT" \
    --epochs 1 \
    --patience 1 \
    --seeds 0 \
    --output "$DIAG_DIR/2gcn_mode_sanity_quick.json" \
    --output_md "$DIAG_DIR/2gcn_mode_sanity_quick.md"
else
  echo "WARNING: sanity_check_2gcn_modes.py missing; skipped"
fi

if [ "$RUN_FULL_SANITY" = "1" ]; then
  section "6b. Full 2xGCN hp_mode Sanity Check"
  run_or_warn "$PY" "$ANALYSE_DIR/sanity_check_2gcn_modes.py" \
    --cora_root "$CORA_ROOT" \
    --run_full_check \
    --seeds 0 1 2 3 4 \
    --output "$DIAG_DIR/2gcn_mode_sanity_full.json" \
    --output_md "$DIAG_DIR/2gcn_mode_sanity_full.md"
else
  echo
  echo "Full sanity check skipped by default. To run it: RUN_FULL_SANITY=1 scripts/analysis/run_diagnostics_suite.sh"
fi

section "7. GMM Density Diagnosis"
if [ -f "$ANALYSE_DIR/diagnose_gmm_density.py" ]; then
  for mode in global4 hybrid_cond7 layer_cond19; do
    mapfile -t histories < <(find "$RESULTS_ROOT" -name history_final.json -path "*${mode}*" -print 2>/dev/null || true)
    if [ "${#histories[@]}" -eq 0 ]; then
      echo "skip GMM density for $mode: no history_final.json found"
      continue
    fi

    run_or_warn "$PY" "$ANALYSE_DIR/diagnose_gmm_density.py" \
      --history_paths "${histories[@]}" \
      --hp_mode "$mode" \
      --output_dir "$DIAG_DIR/gmm_density_${mode}" \
      --top_frac 0.3 \
      --n_components 4 \
      --weight_temp 8.0

    if [ "$RUN_CANONICAL_DIAGNOSTICS" = "1" ]; then
      run_or_warn "$PY" "$ANALYSE_DIR/diagnose_gmm_density.py" \
        --history_paths "${histories[@]}" \
        --hp_mode "$mode" \
        --output_dir "$DIAG_DIR/gmm_density_${mode}_canonical" \
        --top_frac 0.3 \
        --n_components 4 \
        --weight_temp 8.0 \
        --use_canonical_z \
        --inactive_value 0.5
    fi
  done
else
  echo "WARNING: diagnose_gmm_density.py missing; skipped"
fi

section "8. BO-GP Surrogate Diagnosis"
if [ -f "$ANALYSE_DIR/diagnose_bo_gp_surrogate.py" ]; then
  mapfile -t phase4_histories < <(find "$RESULTS_ROOT" -path "*phase4*" -name history_final.json -print 2>/dev/null || true)
  if [ "${#phase4_histories[@]}" -eq 0 ]; then
    echo "skip BO-GP surrogate diagnosis: no Phase4 history_final.json found"
  else
    for hist in "${phase4_histories[@]}"; do
      case "$hist" in
        *global4*) mode="global4" ;;
        *hybrid_cond7*) mode="hybrid_cond7" ;;
        *layer_cond19*) mode="layer_cond19" ;;
        *) echo "skip unknown hp_mode: $hist"; continue ;;
      esac
      run_id="$(basename "$(dirname "$hist")")"

      run_or_warn "$PY" "$ANALYSE_DIR/diagnose_bo_gp_surrogate.py" \
        --history_path "$hist" \
        --hp_mode "$mode" \
        --output_dir "$DIAG_DIR/gp_surrogate_${run_id}" \
        --holdout_frac 0.2 \
        --seed 42

      if [ "$RUN_CANONICAL_DIAGNOSTICS" = "1" ]; then
        run_or_warn "$PY" "$ANALYSE_DIR/diagnose_bo_gp_surrogate.py" \
          --history_path "$hist" \
          --hp_mode "$mode" \
          --output_dir "$DIAG_DIR/gp_surrogate_${run_id}_canonical" \
          --holdout_frac 0.2 \
          --seed 42 \
          --use_canonical_z \
          --inactive_value 0.5
      fi
    done
  fi
else
  echo "WARNING: diagnose_bo_gp_surrogate.py missing; skipped"
fi

if [ "$RUN_OFFICIAL_BASELINE" = "1" ]; then
  section "9. Optional Official 2xGCN final_eval Baselines"
  if [ -f "$REPO_ROOT/final_eval.py" ]; then
    for mode in global4 hybrid_cond7 layer_cond19; do
      run_or_warn "$PY" "$REPO_ROOT/final_eval.py" \
        --cora_root "$CORA_ROOT" \
        --hp_mode "$mode" \
        --include_official_2gcn_baseline \
        --disable_auto_best \
        --n_seeds 10 \
        --eval_epochs 200 \
        --patience 30 \
        --output "$RESULTS_ROOT/final_eval_official_baseline_${mode}" \
        --version "official_baseline_${mode}"
    done
    echo "Official baseline runs finished or skipped. Re-run this script to refresh bundle reports."
  else
    echo "WARNING: final_eval.py missing; official baseline skipped"
  fi
else
  echo
  echo "Official final_eval baselines skipped by default. To run them: RUN_OFFICIAL_BASELINE=1 scripts/analysis/run_diagnostics_suite.sh"
fi

section "10. Expected Output Files"
cat <<EOF
Bundle:
  $BUNDLE_DIR/search_runs_summary.csv
  $BUNDLE_DIR/final_eval_summary.csv
  $BUNDLE_DIR/final_eval_per_seed.csv
  $BUNDLE_DIR/search_to_final_summary.csv
  $BUNDLE_DIR/comparison_report.md
  $BUNDLE_DIR/nas_hpo_contribution_summary.csv
  $BUNDLE_DIR/nas_hpo_contribution_summary.md
  $BUNDLE_DIR/nas_pipeline_audit_report.md

Diagnostics:
  $DIAG_DIR/2gcn_mode_sanity_quick.json
  $DIAG_DIR/2gcn_mode_sanity_quick.md
  $DIAG_DIR/gmm_density_<hp_mode>/gmm_density_report.md
  $DIAG_DIR/gp_surrogate_<run_id>/gp_surrogate_report.md

Skipped diagnostics are expected when Cora, torch_geometric, botorch/gpytorch, or history_final.json files are unavailable.
GMM density is not predicted accuracy. GP surrogate predicts search-time validation accuracy, not final test accuracy.

To scan a non-default results directory:
  RESULTS_ROOT=results-wgmm1 BUNDLE_DIR=results_analysis_bundle_after_diagnostics_wgmm1 scripts/analysis/run_diagnostics_suite.sh

To force a Cora location:
  CORA_ROOT=Cora scripts/analysis/run_diagnostics_suite.sh

Historical compatibility command:
  analyse/run_diagnostics_suite.sh
EOF

section "Done"
