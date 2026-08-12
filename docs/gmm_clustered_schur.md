# Fixed K=6 GMM clustered Schur with low fidelity

Phase 4 exposes the initializer:

```text
gmm_schur_lowfid
```

It implements the fixed K6-LF protocol only. It does not perform BIC model
selection, balanced k-means, a new TED strategy, or a new surrogate.

## Flow and budgets

```text
768 deterministic LHS candidates
-> ordinary diagonal GMM on 12-dimensional z_arch only
-> fixed K=6, five deterministic restarts
-> six non-empty canonical hard clusters
-> hybrid cluster quotas, equal_weight=0.5
-> one global full-z_search feature transform and condition mask
-> one global RBF lengthscale and one shared 768x768 kernel
-> same-cluster Schur conditional-variance selection of 50 seeds
-> 50 full-fidelity evaluations
-> scratch Exact GP fit on valid full seed observations
-> same-cluster Schur shortlist conditioned on all same-cluster seeds
-> 200 low-fidelity evaluations at 20 epochs with no early stopping
-> within-cluster promotion of 100 candidates
-> 100 full-fidelity promotion evaluations
-> Exact GP refit on valid full observations only
-> existing 150-step online qLogEI
```

The strict full-fidelity budget is:

```text
50 seed full + 100 promoted full + 150 online full = 300 full evaluations
```

Low-fidelity observations are kept out of Exact GP training. Their planned
epoch-ratio cost is recorded separately:

```text
200 * 20 / 150 = 26.67 equivalent full evaluations
```

The artifacts also record the real low-fidelity count, completed epochs, wall
time, optional GPU seconds, and invalid counts. The budget summary separately
records requested, completed, invalid, and replenished counts for the seed,
shortlist, and promotion stages, plus both actual and planned equivalent-full
low-fidelity cost.

## Fixed ordinary GMM

The GMM reads only `z_arch = z_search[:12]`. It reuses the ordinary
`gmm_fit_pool` global standardization and diagonal covariance semantics with
uniform candidate weights.

Each restart seed is derived from:

```text
search_seed
"gmm_fixed_v1"
candidate_pool_fingerprint
K=6
restart_id
```

The selected restart must converge, have finite parameters and likelihood, and
produce six non-empty hard clusters. The highest log-likelihood eligible
restart wins; numerical ties use the lower restart id. If none is eligible,
initialization fails explicitly.

Component labels are canonicalized by lexicographically sorting standardized
component means, with the original component index as the final tie-break.
Restart seeds, the selected restart, component order, assignments, capacities,
and fingerprints are deterministic for a fixed pool and search seed.

## One shared RBF kernel

Schur selection uses the complete `z_search`, not just `z_arch`:

```text
z_search
-> normalize_search_vector()
-> apply the existing inactive condition mask
-> global design features
-> one RBF lengthscale
-> one shared global RBF kernel
```

The feature transform, condition mask, lengthscale, and kernel are each
constructed once per initialization. Cluster selectors only index rows and
columns of this shared kernel. They do not standardize cluster data or estimate
cluster-specific lengthscales.

The initialization config records:

```text
design_feature_fingerprint
condition_mask_fingerprint
rbf_lengthscale
global_kernel_fingerprint
```

## Conditional-variance Schur

`initialization_gmm_schur.py` provides:

```python
fit_fixed_pool_gmm_restarts(...)
conditional_schur_variances(...)
greedy_conditional_schur_select(...)
assert_exact_cluster_quotas(...)
```

For candidate `j` and the same-cluster conditioning set `S`, the score is:

```text
K[j,j] - K[j,S] (K[S,S] + jitter * I)^-1 K[S,j]
```

The implementation uses Cholesky solves without explicitly forming an inverse.
Jitter is increased a bounded number of times and failure remains explicit.
Only small negative roundoff within the configured tolerance is clamped to
zero. Score ties use the original global LHS candidate index.

Seed selection starts from an empty conditioning set in each cluster.
Shortlist selection starts from every seed in the same cluster. Each newly
selected point is added before the next score calculation. Other clusters
never enter the conditioning set.

The enforced set relationships are:

```text
seed intersection shortlist = empty
promoted is a subset of shortlist
seed intersection promoted = empty
```

## Quotas and promotion

Seed, shortlist, and promotion quotas use the existing hybrid allocator with
largest remainder, minimum non-empty-cluster coverage when feasible, capacity
clipping, and deterministic redistribution.

Capacities are:

```text
seed: cluster size
shortlist: cluster size - selected seed count
promotion: completed valid low-fidelity shortlist count
```

All quota sums must exactly equal 50, 200, and 100 respectively. Insufficient
capacity is an error and reports the requested budget, total and per-cluster
capacity, and the unfilled count.

Promotion reuses the existing deterministic percentile implementation:

```text
0.50 * low-fidelity percentile
+ 0.25 * GP mean percentile
+ 0.25 * GP standard-deviation percentile
```

Only completed valid low-fidelity candidates can be promoted. Ties use the
existing candidate-index rule.

## Identity, history, and resume

Candidate identity and training seeds keep the existing derivations:

```text
stable_seed(search_seed, "candidate_evaluation",
            candidate_fingerprint, fidelity)
```

Method name, cluster, stage, selection rank, full-evaluation index, and output
directory do not affect the candidate evaluation seed. Low/full evaluations of
a promoted candidate retain the same `z_search`, HP, decoder seed, discrete
architecture, and candidate fingerprint; fidelity alone isolates the training
seed.

Low-fidelity rows are written to `low_fidelity_history.json`. Full seed,
promotion, and online evaluations remain in the normal full history consumed
by `final_eval.py`.

The K6-LF-specific artifacts are:

```text
gmm_fixed_restarts.csv
gmm_cluster_assignments.csv
gmm_cluster_quotas.json
schur_seed_trace.csv
schur_shortlist_trace.csv
gmm_schur_lowfid_promotion.csv
gmm_schur_lowfid_initialization_config.json
```

Resume regenerates the unlabeled design and rejects configuration fingerprint
or selected-index mismatches before reusing observations. The fingerprint
covers the fixed GMM, assignments, shared feature/kernel construction, Schur
jitter, quota inputs, 50/200/100/150 budgets, low/full fidelity settings,
promotion weights, and online configuration.

Legacy `schur`, `wgmm_ted`, and `wgmm_ted_lowfid` behavior and artifact names
remain unchanged. The CLI default remains `schur`.

## Command template

Run from the repository root and use a new output directory for every run.
Replace the example checkpoint filename and dataset root as needed; both paths
below are repository-relative.

```bash
conda run -n nas python bo_phase4.py \
  --checkpoint results-wgmm1/joint_search/vae_checkpoint.pth \
  --cora_root data/Cora \
  --hp_mode global4 \
  --initial_selection_strategy gmm_schur_lowfid \
  --wgmm_source gmm_fit_pool \
  --gmm_component_selection fixed \
  --wgmm_n_components 6 \
  --gmm_restarts 5 \
  --wgmm_quota_mode hybrid \
  --wgmm_equal_weight 0.5 \
  --n_lhs_candidates 768 \
  --n_init 50 \
  --initial_seed_evals 50 \
  --initial_shortlist_evals 200 \
  --initial_expand_evals 100 \
  --n_iter 150 \
  --max_total_full_evals 300 \
  --eval_epochs 150 \
  --patience 40 \
  --low_fidelity_epochs 20 \
  --low_fidelity_patience 0 \
  --low_fidelity_score_weight 0.50 \
  --gp_mean_score_weight 0.25 \
  --gp_std_score_weight 0.25 \
  --surrogate_type exact_gp \
  --gp_init_mode scratch \
  --warm_start "" \
  --online_candidate_strategy qlogei \
  --seed <SEARCH_SEED> \
  --version k6_lf_seed<SEARCH_SEED>_<TIMESTAMP> \
  --log_dir logs/k6_lf \
  --output results/k6_lf_seed<SEARCH_SEED>_<TIMESTAMP>
```
