# Unified cross-dataset Phase4 protocol

`dataset_utils.py` is the single data-loading boundary for Phase4 search and
seed-fair final evaluation. The unified loader retains these canonical names:

`cora`, `citeseer`, `pubmed`, `dblp`, `flickr`, and `ogbn-arxiv`.

The frozen formal search matrix is limited to `citeseer`, `pubmed`, `dblp`,
and `flickr`: 4 datasets x 3 methods x 5 search seeds = 60 runs. Cora is a
historical reference only. `ogbn-arxiv` is fixed at
`blocked_resource_preflight_oom` after a real full-batch OOM on a 24 GB RTX
4500 Ada and is rejected by `cross_dataset_runner.py`.

Search seed roles are disjoint: `0--4` remain development/debug/history
seeds, while the only formal search seeds are `5--9`. Final-evaluation
replicate seeds retain their separate existing semantics.

- Cora, CiteSeer, and PubMed use `Planetoid(split="public")` with the same
  `NormalizeFeatures` transform as the historical Cora path.
- `dblp` means homogeneous `CitationFull(name="DBLP")`, never the
  heterogeneous PyG DBLP dataset. Its split uses split seed 0, 20 training
  nodes per class, 500 validation nodes, and 1000 test nodes. The PyG DBLP
  citation graph is already undirected, so only `NormalizeFeatures` is added.
- Flickr preserves its official features and masks.
- ogbn-arxiv uses the OGB official split, squeezes labels to `[N]`, and uses
  `to_undirected` for this experiment family.

All loaders validate homogeneous single-label data, nonempty disjoint masks,
label and edge ranges, and dynamic feature/class dimensions. Dataset content,
split, transform, dependency versions, and training mode are saved in
`dataset_manifest.json`. DBLP also saves `dblp_split_indices.json`.

Candidate fingerprints and candidate evaluation seeds are deliberately
dataset-independent. Dataset context is instead part of evaluation cache
identity, initialization/resume identity, GP metadata, history metadata, and
final-evaluation configuration.

The shared `JointSpaceVAE` contains an architecture DVAE and an HP VAE. It has
no Cora accuracy/performance head; target-dataset validation labels enter only
through `train_and_eval_arch`.

## Formal method definitions

The machine-readable source is
`configs/cross_dataset_methods.json`.

- S0/global_schur: 50 initial + 250 online = 300 full evaluations.
- G100/gmm_exp100: 50 seed + 100 expansion + 150 online = 300.
- G150/gmm_exp150: 50 seed + 150 expansion + 100 online = 300.

G100 and G150 are the same `gmm_fit_pool + wgmm_ted_lowfid` method family.
They differ only in expansion/online budget allocation.

Build a single formal command without executing it:

```bash
conda run -n nas python cross_dataset_runner.py \
  --run-tag cross_dataset_v1 \
  --dataset citeseer \
  --method G100 \
  --search-seed 5 \
  --checkpoint /absolute/path/joint_model_pipeline_global4_best.pth \
  --data-root /absolute/path/datasets \
  --expected-source-manifest /absolute/path/source_manifest.json \
  --frozen-source-id <SOURCE_ID> \
  --dry-run
```

Replace `--dry-run` with `--execute` only after the source/checkpoint hash gate,
dataset smoke tests, and the required full-batch GPU resource preflight pass.
Both modes verify source membership and bytes; a mismatch is never accepted by
updating the expected hash. Runs are isolated under:

```text
results/<run-tag>/<dataset>/<method>/search_seed<N>/
logs/<run-tag>/<dataset>/<method>/search_seed<N>/
```

Flickr requires both the historical static-pool evidence and
`flickr_continuous_extreme_preflight.py`. The latter deterministically searches
the real continuous decoder domain without labels, excludes all five formal
768-point pools, covers joint hidden/layer/GAT/parameter/activation extrema,
then runs a forward/backward probe and one real 150-epoch, patience-40
full-batch evaluation for the most dangerous valid candidate. Its outputs are
isolated from formal history and GP data. Any OOM blocks the dataset; there is
no CPU or sampling fallback. The retained `ogbn-arxiv` loader and preflight
path are for auditing the recorded blocked state only, not for launching
another full-batch probe.
