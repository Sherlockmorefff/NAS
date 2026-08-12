# Repository layout and artifact policy

This document describes the current repository layout. It is a compatibility
guide, not a declaration that every root-level Python file should be moved into
a package. The stable Python entry points remain at the repository root during
the first organization phase.

## Stable root entry points

The following formal entry points remain in place so existing CLI commands,
subprocess launch records, source freezes, and historical run instructions keep
working:

- `bo_phase4.py`: Phase4 NAS search, initialization, surrogate fitting, online
  acquisition, history, and resume state.
- `cross_dataset_runner.py`: isolated S0/G100/G150 cross-dataset command
  construction and execution.
- `final_eval.py`: seed-fair final evaluation from persisted search histories.
- `deterministic_three_strategy_pipeline.py` and `determinism_smoke.py`:
  deterministic formal orchestration and acceptance checks.
- `formal_matrix.py`, `source_freeze.py`, and `source_verification.py`: formal
  task matrices, source snapshots, and launch-gate verification.
- `first_formal_top10_test30.py`: historical first-formal Top-10/test30
  evaluation workflow.
- `validate_dataset_loading.py`, `gpu_dataset_smoke.py`,
  `resource_preflight.py`, `method_path_smoke.py`, and the Flickr preflight
  programs: explicit validation entry points.

Moving these implementations requires a separate compatibility change with
root-level wrappers and updates to imports, tests, documentation, source-gate
discovery, and recorded subprocess paths.

## Core library modules

Core library modules currently remain at the root. `dataset_utils.py` owns
dataset and split identity; `nas_space.py` and `hp_modes.py` own the search
representation; `eval_utils.py` owns candidate decoding and evaluation;
`deterministic_runtime.py`, `evaluation_errors.py`, `run_provenance.py`, and
`source_verification.py` provide deterministic, error, and provenance
boundaries. `models.py` and `util.py` retain D-VAE checkpoint compatibility.

## Initialization

Initialization implementations are `initialization_gmm_schur.py`,
`initialization_wgmm_ted.py`, and `weighted_diag_gmm_init.py`. They remain at
the root because Phase4, tests, documentation, and frozen source manifests
refer to these paths.

## Surrogate

`surrogate/` is the maintained package for Exact GP and DKL predictors,
checkpoint I/O, history datasets, metrics, and standalone predictor tools.

## Orchestration and preflight

Cross-dataset orchestration is split across the stable root entry points listed
above. Formal matrix construction, source freeze/verification, deterministic
queue execution, dataset loading checks, GPU smokes, and resource preflights
remain separate so validation can stop before a costly search or evaluation.

## Analysis

`analyse/` contains Python posthoc and diagnostic programs. Its two historical
shell paths are now compatibility wrappers; maintained launchers live under
`scripts/analysis/`.

## Launch scripts

Maintained shell entry points are grouped by purpose:

```text
scripts/
├── analysis/
│   ├── run_collect_results.sh
│   └── run_diagnostics_suite.sh
└── evaluation/
    └── run_final_eval_topk_seedfair.sh
```

Each launcher resolves the repository from its own file location and can be
called from a different working directory. The analysis launchers use the
current environment's `python` by default; set `PYTHON=/path/to/python`, or pass
`--python PATH` to the diagnostics suite. The final-evaluation launcher accepts
`--python PATH` or `PYTHON_BIN=/path/to/python` and checks its required imports.

These historical commands remain supported through thin argument-forwarding
wrappers:

```bash
analyse/run_collect_results.sh --help
analyse/run_diagnostics_suite.sh --help
run_final_eval_topk_seedfair.sh --help
```

The canonical commands for new instructions are:

```bash
scripts/analysis/run_collect_results.sh --help
scripts/analysis/run_diagnostics_suite.sh --help
scripts/evaluation/run_final_eval_topk_seedfair.sh --help
```

## Configuration, tests, and documentation

- `configs/` contains machine-readable experiment definitions. In particular,
  `configs/cross_dataset_methods.json` is the formal S0/G100/G150 method source.
- `tests/` contains CPU-safe unit, mock, provenance, launch-gate, initialization,
  surrogate, final-evaluation, and launcher tests. Tests must not rely on a GPU
  or download Cora during local validation.
- `docs/` contains experiment protocols, initialization specifications,
  reproducibility notes, and this layout policy.

## Legacy and historical code

The original D-VAE/ENAS workflow and staged Cora experiments remain in their
historical locations. This includes `train.py`, `train_joint.py`,
`generate_mini_data.py`, `bo_phase2.py`, `bo_phase3.py`, `bo_phase4_tpe.py`,
`eval_joint.py`, `bayesian_optimization/`, and `software/enas/`. Experimental
geometric-acquisition files also remain at the root for now.

Lack of a repository-internal import is not sufficient evidence that a file is
unused: it may be a direct CLI, subprocess target, frozen-source member, or
historical reproduction dependency. Historical code should be assessed for
reproduction value before it is deleted or moved into a future `legacy/`
directory.

## Results, logs, data, and checkpoints

Existing `results/`, `results-1-LHS/`, `results-wgmm1/`, analysis bundles, and
all `logs*` directories are historical experiment products and are not being
migrated. `data/`, `Cora/`, and `raw/` retain their current dataset-root
semantics. Checkpoints currently live beside the training runs that produced
them, including the joint-search directories under historical result roots.

The following files are experiment state, not ordinary disposable caches:

- `history_final.json`, intermediate histories, and history metadata;
- dataset and source manifests, run provenance, fingerprints, and exact argv;
- GP checkpoints, online GP state, RNG state, and resume state;
- SQLite/Optuna databases;
- D-VAE and GNN checkpoints;
- final-evaluation records and per-seed results.

Ignored files are not automatically safe to remove. Git ignore rules only
control discovery and staging; they do not classify scientific value or
reproducibility requirements. Python and pytest caches may be regenerable, but
must still be handled explicitly rather than inferred from broad ignore rules.

## Suggested layout for future runs

No current CLI default is changed by this policy. After a separate CLI and
compatibility change, new runs may use:

```text
results/
├── search/<run-tag>/
├── final_eval/<run-tag>/
└── posthoc/<run-tag>/

logs/
└── <run-tag>/

artifacts/
├── archives/
└── indexes/
```

Manifest, provenance, history, and resume state should continue to coexist with
the run that owns them. `artifacts/` is only for additional archives and
cross-run indexes; it must not become a reason to remove original run metadata.
The proposed layout becomes active only after a later change updates CLI output
options, tests, and documentation. Existing result paths must remain supported
for final evaluation, resume, audit, and historical reproduction.
