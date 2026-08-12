# Repository layout and artifact policy

This document describes the current repository layout. It is a compatibility
guide, not a declaration that every root-level Python file should be moved into
a package. Stable formal entry points and compatibility modules remain at the
repository root.

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
Validation commands retain stable root compatibility modules, but their
implementations now live under `scripts/validation/`; they are not formal
search entry points.

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
above. Formal matrix construction, source freeze/verification, and deterministic
queue execution remain at the root. Dataset loading checks, GPU smokes, cost
estimates, method-path smokes, and resource preflights live under
`scripts/validation/` so validation can stop before a costly search or
evaluation. Their historical root paths remain import- and CLI-compatible.

## Analysis

`analyse/` contains Python posthoc and diagnostic programs. Its two historical
shell paths are now compatibility wrappers; maintained launchers live under
`scripts/analysis/`.

## Launch scripts

Maintained shell entry points are grouped by purpose:

```text
scripts/
├── __init__.py
├── analysis/
│   ├── run_collect_results.sh
│   └── run_diagnostics_suite.sh
├── evaluation/
│   └── run_final_eval_topk_seedfair.sh
├── maintenance/
│   └── inventory_repository.py
└── validation/
    ├── cross_dataset_cost_estimate.py
    ├── flickr_continuous_extreme_preflight.py
    ├── flickr_failed_candidate_resource_smoke.py
    ├── gpu_dataset_smoke.py
    ├── method_path_smoke.py
    ├── resource_preflight.py
    └── validate_dataset_loading.py
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

New validation instructions should likewise use canonical paths, for example:

```bash
python scripts/validation/validate_dataset_loading.py --help
python scripts/validation/resource_preflight.py --help
python scripts/validation/cross_dataset_cost_estimate.py --help
```

The seven former root commands remain compatibility modules. They forward the
same arguments and preserve imports, including historically imported helper
functions, so source manifests and exact argv records do not need rewriting.
GPU/data validation is never run implicitly by these wrappers.

## Configuration, tests, and documentation

- `configs/` contains machine-readable experiment definitions. In particular,
  `configs/cross_dataset_methods.json` is the formal S0/G100/G150 method source.
- `tests/` contains the currently maintained CPU-safe unit, mock, provenance,
  launch-gate, initialization, surrogate, final-evaluation, and launcher tests.
  `pytest.ini` makes this the default collection root. Tests must not rely on a
  GPU or download Cora during local validation.
- `docs/` contains experiment protocols, initialization specifications,
  reproducibility notes, and this layout policy.

Default pytest collection deliberately excludes `legacy/`, vendored Theano
tests, and `software/enas/` historical tests. This defines the maintained test
boundary; it is not a claim that excluded tests pass under the current
environment.

## Legacy and historical code

`legacy/geometric_acquisition/` contains the archived differentiable-decoder,
Jacobian, and geometric acquisition prototype. It has no formal dependency,
has no root wrapper, and is excluded from the current source freeze and tests.
Its README records the incompatible historical API and limited import-only
support status.

`legacy/phased_cora_pipeline/` contains the historical mini-data, joint
training/evaluation, Phase2, Phase3, and TPE implementations. The corresponding
root filenames are lightweight wrappers, so commands such as
`python bo_phase2.py --help` and checkpoint module paths remain stable. These
implementations are included in the current source freeze because those root
wrappers call them.

The original D-VAE/BN and vendored ENAS stack remains in `train.py`,
`bayesian_optimization/`, and `software/enas/`. The compatibility-sensitive
`models.py`, `util.py`, and `nas_space.py` modules remain at the root.

Lack of a repository-internal import is not sufficient evidence that a file is
unused: it may be a direct CLI, subprocess target, frozen-source member, or
historical reproduction dependency. Historical code should be assessed for
reproduction value before it is deleted or moved into `legacy/`.

## Formal source-freeze boundary

The current source gate discovers root Python entry points, `analyse/`,
`configs/`, `surrogate/`, and all maintained `scripts/` Python/shell files. It
also includes `legacy/phased_cora_pipeline/` because root compatibility wrappers
execute those implementations. `legacy/geometric_acquisition/` remains outside
the formal gate. This new layout produces a new source ID; historical manifests
and source IDs are not edited and continue to verify their historical paths.

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

`scripts/maintenance/inventory_repository.py` creates a metadata-only JSON,
CSV, and Markdown inventory under an explicit output directory. It does not
open artifact contents or follow symlinks. Machine-specific snapshots under
`artifacts/indexes/repository_inventory_*/` stay local and ignored. See
[the experiment artifact guide](experiment_artifact_guide.md) for
classification and retention guidance.

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
