# Experiment artifact migration record

This record describes the local, same-filesystem archive operation. The large
payload directories, deletion manifest, and migration manifest are ignored by
Git; no historical result file was rewritten.

## Preserved archive

| Original top-level path | Files | Bytes |
| --- | ---: | ---: |
| `results/` | 7,920 | 2,558,185,696 |
| `results-1-LHS/` | 329 | 639,147,792 |
| `results-wgmm1/` | 92 | 144,670,358 |
| `logs/` | 543 | 16,393,428 |
| `logs-LHS/` | 69 | 640,250 |
| `logs-wgmm1/` | 31 | 534,941 |
| `results_analysis_bundle-wgmm1/` | 15 | 190,244 |
| `results_analysis_bundle_after_diagnostics_results-wgmm1/` | 18 | 207,349 |
| **Total** | **9,017** | **3,359,970,058** |

The before/after file counts and byte totals matched. Sampled history,
metadata, checkpoint, SQLite, log, and analysis-manifest SHA-256 values also
matched. Full details remain in the local
`legacy_artifacts/pre_20260813/migration_manifest.{json,md}`.

## Extracted current inputs

| Current path | SHA-256 |
| --- | --- |
| `checkpoints/joint_model_pipeline_global4_best.pth` | `e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4` |
| `artifacts/reference/cross_dataset_v1_preflight/training_mode_decisions.json` | `595442ac3486bab5261351075e9dfd65c7a698f4c67d181792cf0a5ca2e188e9` |

Eight dataset/split/download manifest files were extracted under
`artifacts/reference/cross_dataset_v1_preflight/manifests/`; their complete
hash list is in local `artifacts/reference/reference_manifest.json`.

## Removed regenerable state

The initial cleanup removed eleven untracked Python `__pycache__/` directories
(517,865 bytes total) and three verified empty run directories (30
directory-entry bytes). Required validation then regenerated 1,618,294 bytes of
pytest/bytecode cache, which was recorded and removed after the two full test
runs. A final focused regression check regenerated another 694,130 bytes of
cache, which was likewise recorded before removal. Total recorded removal was
2,830,319 bytes. Each cache was checked for tracked files
and each run directory was checked to be empty. No checkpoint, history,
manifest/provenance, SQLite state, archive, or unique diagnostic was deleted.
The pre-deletion path, type, size, reason, and replacement are recorded in the
local `legacy_artifacts/pre_20260813/deletion_manifest.{json,md}`.

A zero-byte `tracked_git_status.txt` inside a packaged DKL diagnostic was
retained because its enclosing package and checksums are provenance evidence.
Failed smoke, preflight, and interrupted runs were also retained whenever their
uniqueness or downstream value could not be proven safely.
