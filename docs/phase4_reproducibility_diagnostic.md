# Phase4 fixed-candidate reproducibility diagnostic

`analyse/analyze_candidate_reproducibility.py` diagnoses training-stage
reproducibility without entering BO, generating candidates, decoding the VAE,
or requesting test metrics.

The coordinator:

1. reads T1/C1 `history_final.json`;
2. verifies full-fidelity candidate identity, architecture, effective HP,
   decoder seed, and candidate evaluation seed across both histories;
3. fixes ten candidates: the known maximum-difference candidate, the shared
   best candidate, and eight deterministic equidistant positions over the
   sorted remaining overlap fingerprints;
4. checks that the requested physical GPU is idle before every worker;
5. launches every repeat in a fresh Python subprocess;
6. evaluates the persisted discrete architecture and HP directly with
   `track_test=False`.

The script refuses to download Cora. All raw Cora files must already exist
under the configured `--cora-root`.

## Read-only manifest validation

This validates the two histories and checkpoint fingerprint without creating
an output directory or using a GPU:

```bash
conda run -n nas python analyse/analyze_candidate_reproducibility.py \
  --validate-only
```

## Bounded GPU diagnostic

Run only after `nvidia-smi` reports an idle GPU:

```bash
conda run -n nas python analyse/analyze_candidate_reproducibility.py \
  --gpu-index 0 \
  --cora-root /tmp/Cora \
  --eval-epochs 150 \
  --patience 40
```

The default output is a new timestamped directory under
`results/diagnostics/`. Existing output directories are never overwritten.

Default mode runs ten candidates three times each. Strict mode first runs the
two required anchor candidates twice each with:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=<candidate evaluation seed>
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
```

Strict settings are applied only inside strict diagnostic workers, before any
CUDA context is established. They do not change the Phase4 production
defaults. If the pilot fails, is invalid, or is not exactly repeatable in
validation and early-stopping metadata, the strict branch is not expanded to
the other eight candidates.

The output directory contains at least:

```text
candidate_manifest.json
default_repeats.csv
strict_repeats.csv
runtime_provenance.json
reproducibility_summary.json
```

Per-worker JSON records are also retained under `worker_records/` so an
unsupported deterministic operator or other exception is not reduced to a
generic failure flag.
