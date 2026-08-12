# Historical phased Cora pipeline

This directory contains the implementation of the staged Cora workflow used to
produce the older LHS and WGMM experiment families. The files retain their
historical defaults and scientific semantics; stable root-level compatibility
entry points preserve the original commands.

## Workflow

```text
generate_mini_data.py
        ↓
train_joint.py → joint D-VAE checkpoint
        ↓
bo_phase2.py   → architecture-only warm start
        ↓
bo_phase3.py   → joint architecture + HP warm start
        ↓
bo_phase4_tpe.py or the later root bo_phase4.py
        ↓
eval_joint.py / final_eval.py
```

Phase2 records architecture-only histories. Phase3 adds the hp-mode search
vector. The archived Phase4 TPE implementation uses Optuna and can persist an
SQLite storage database. The maintained formal workflow now uses root
`bo_phase4.py`, `cross_dataset_runner.py`, formal source gates, and seed-fair
`final_eval.py` instead.

## Compatibility commands

Continue to use the historical root commands:

```bash
python generate_mini_data.py --help
python train_joint.py --help
python eval_joint.py --help
python bo_phase2.py --help
python bo_phase3.py --help
python bo_phase4_tpe.py --help
```

Each root wrapper resolves the repository, changes to the repository root, and
executes exactly one implementation here with the original `sys.argv`. This
preserves relative defaults such as `data/`, `results/`, `logs/`, checkpoint
paths, warm starts, seeds, and output directories. The wrappers use `runpy`
because several historical scripts parse arguments at module scope and do not
expose a uniformly safe importable `main()`.

## Results and checkpoints

- `results-1-LHS/` and `logs-LHS/` contain earlier LHS, Phase2/3/4, TPE, and
  final-evaluation outputs.
- `results-wgmm1/` and `logs-wgmm1/` contain the later WGMM pipelines and the
  joint checkpoints used by several historical searches.
- Joint D-VAE checkpoints were produced by `train_joint.py` and are loaded via
  the still-rooted `models`, `util`, and `nas_space` module paths.
- `surrogate/`, `hp_modes.py`, `eval_utils.py`, and initialization modules
  remain in their maintained locations because both historical and current
  code depend on them.

None of those results, logs, datasets, or checkpoints are moved by this archive.
Do not rewrite saved state-dict keys or relocate model compatibility modules as
part of legacy cleanup.

## Support status

These programs remain available for reproduction and `--help`/syntax checks,
but they are not the current formal cross-dataset workflow and are outside the
default maintained test scope except for compatibility-wrapper tests. Some
commands can download Cora or start costly training; do not execute them during
local validation.
