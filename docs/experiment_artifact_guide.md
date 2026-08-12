# Experiment artifact classification and retention

This guide classifies persistent outputs without changing any existing result
path. A path being ignored by Git is not evidence that it is disposable.

## Artifact classes

| Class | Typical examples | Reproducibility role | Default policy |
| --- | --- | --- | --- |
| Search result | `history_final.json`, budget and convergence records, selected candidates | Primary scientific record of candidate order, observations, and budget | Keep with the run |
| Final evaluation | Top-k selections, per-replicate validation/test records, aggregate summaries | Separates search selection from held-out reporting | Keep with source histories |
| Posthoc/diagnostics | Tables, plots, diagnostics, contribution analyses | Often regenerable only while exact inputs and code remain | Keep until inputs and regeneration are verified |
| Log | Shell, Python, scheduler, and training logs | Operational evidence for failures, timing, and exact invocation | Keep with or index from the run |
| Checkpoint | D-VAE, GNN, best latent, and model-state files | May be expensive or impossible to reproduce exactly | Do not treat as cache |
| Resume/cache | GP online state, RNG state, optimizer state, partial histories | Required to continue with identical identity and seed semantics | Keep with the owning run |
| Manifest/provenance | Dataset/source/checkpoint manifests, fingerprints, exact argv, source IDs | Defines what was run and verifies frozen inputs | Never detach from the owning run |
| Database | SQLite/Optuna stores | May contain search ordering and resumable trial state | Keep and back up with the run |
| Archive | `tar.gz`, `zip`, checksums | Additional packaging or transfer copy | Verify checksum; do not replace originals implicitly |
| Data | downloaded/processed graph datasets and splits | Inputs may encode exact versions, transforms, or split identities | Keep unless reproducible replacement is proven |
| Rebuildable cache | Python bytecode and pytest cache | No scientific state | May be removed only by an explicit, narrowly scoped cleanup |

History, manifest, provenance, GP state, SQLite, and checkpoint files are not
ordinary caches. Candidate fingerprints, dataset identities, evaluation seeds,
and resume boundaries can depend on them even when a filename looks temporary.

## Inventory reports

Generate a read-only metadata inventory with an explicit output directory:

```bash
python scripts/maintenance/inventory_repository.py \
  --repo-root /path/to/D-VAE \
  --output-dir artifacts/indexes/repository_inventory_<timestamp>
```

The tool reads file metadata and Git classification only. It does not parse
large histories, open checkpoints, follow symlinks, move files, or delete
anything. JSON, CSV, and Markdown reports include the run-level path, artifact
type, Git status, file count, bytes, latest modification time, likely producer
and reader, rebuildability, critical-state flag, and retention policy.
Classification is conservative and path-based; `unknown` entries require user
review. Local inventory snapshots are ignored because they contain
machine-specific paths and live size/time observations.

## Current historical roots

`results/`, `results-1-LHS/`, `results-wgmm1/`, all `results_analysis_bundle*`
directories, `logs/`, `logs-LHS/`, and `logs-wgmm1/` remain in their historical
locations. `data/`, `Cora/`, and `raw/` retain existing loader semantics.
Moving them would risk breaking saved exact argv, history references, resume
identity, checkpoint provenance, and analysis commands, while also copying
several gigabytes without scientific benefit.

## Future layout

New CLI work may later opt into:

```text
results/
├── search/<run-tag>/
├── final_eval/<run-tag>/
└── posthoc/<run-tag>/

logs/<run-tag>/

artifacts/
├── archives/
└── indexes/
```

That convention is documentation only until a separate compatibility change
updates CLI defaults. Manifest, provenance, history, and resume state should
remain beside the run they describe. `artifacts/` is for additional archives
and indexes, not a destination for removing original metadata. Old result paths
must remain readable by final evaluation, resume, audits, and reproduction
tools.
