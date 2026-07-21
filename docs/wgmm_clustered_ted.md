# Exact-GP WGMM-clustered TED initialization

This repository calls the method `wgmm_clustered_ted` or “NAS-adapted
MicroAL”. It is not a literal reproduction of BOOM-Explorer MicroAL.

## Scope and provenance

The sequential selection criterion is independently implemented from the
equations in Kai Yu, Jinbo Bi, and Volker Tresp, *Active Learning via
Transductive Experimental Design*, ICML 2006 (DOI
`10.1145/1143844.1143980`). BOOM-Explorer motivates adding domain-aware
partitions to TED: its MicroAL clusters a discrete microarchitecture space with
a distance that gives extra weight to `DecodeWidth`, then takes an equal target
count from each computed cluster. This NAS repository has no equivalent
hand-authored domain metric, so it uses posterior assignments from a diagonal
latent GMM on `z_arch` as the region definition.

No BOOM-Explorer or TED reference implementation source was copied. The new
implementation is original MIT-repository code based on the published
mathematics. The public BOOM-Explorer repository is GPL-3.0; its pinned commit
`0f3d6cbcdf5c9b506b1c87178345885a7014ede5` was used only for a read-only
behavioral audit. GPL source is not imported, vendored, or adapted into this MIT
repository.

## Relationship to BOOM-Explorer MicroAL

The borrowed ideas are cluster-aware initialization, representative selection
within clusters, TED-style RBF similarity, and true evaluation of the selected
initial designs. This is therefore described as **MicroAL-inspired
WGMM-clustered TED** or **NAS-adapted MicroAL**, never as the original MicroAL
or a line-by-line reproduction.

BOOM-Explorer's pinned implementation uses a hand-written weighted K-means in
which the `DecodeWidth` coordinate receives extra weight. It assigns
`batch // cluster` samples per cluster and uses RandomizedTED: every selection
draws a random candidate subset, mixes in prior selections, removes duplicates
through an unordered set, and computes its RBF after implicitly dropping the
last two input fields. It uses module-global Python and NumPy RNG state. Its
solver then truly evaluates the selected designs and proceeds with DKL-GP and
expected hypervolume improvement. None of that source code or control flow is
reused here.

The NAS replacements are `z_arch`-only checkpoint WGMM parameters when
available, or an explicitly requested unlabeled-pool diagonal GMM; deterministic
full-cluster greedy TED; equal/proportional/hybrid quotas with deterministic
capacity redistribution; SHA-256-derived isolated seeds; explicit full- versus
low-fidelity records; and Exact GP plus qLogEI. TED uses the normalized complete
`z_search`, with inactive conditional dimensions explicitly masked rather than
removed by positional slicing.

BOOM RandomizedTED and this greedy TED are not mathematically equivalent. The
NAS method maintains the full conditional covariance over the transductive set
and maximizes the exact rank-one trace reduction shown below. Random candidate
subsampling is not part of `wgmm_ted` or `wgmm_ted_lowfid`.

## Existing code versus new code

The legacy default remains:

```text
LHS pool -> pool min/max normalization -> RBF -> Schur conditional-diagonal
selection -> full GNN evaluation -> Exact GP -> unchanged qLogEI loop
```

`schur_greedy_select` is a greedy log-determinant/diversity heuristic. After
the first special far-from-center point, it selects the largest conditional
diagonal variance. It does not maximize reduction of variance over every point
in the cluster.

The new TED score is:

```text
C = K_UU - K_US (K_SS + lambda I)^-1 K_SU
score(j) = ||C[:, j]||^2 / (C[j, j] + lambda)
C <- C - C[:, j] C[j, :] / (C[j, j] + lambda)
```

Thus TED values a point for both its own uncertainty and its covariance with
the entire transductive set. The implementation uses Cholesky conditioning,
symmetric rank-one updates, finite-value checks, bounded numerical cleanup,
and original LHS index tie-breaking.

## WGMM source

The repository's existing model is `WeightedDiagonalGMM` in
`weighted_diag_gmm_init.py`: a deterministic NumPy EM implementation with
diagonal covariance and optional sample weights. The inspected VAE save path
appears to write only `model.state_dict()`, and the inspected model definition
does not expose mixture parameters. That is a **code-path inference**, not
evidence about any particular server checkpoint. The local workspace contains
no checkpoint, sidecar mixture, or saved latent artifact with which to verify
the server file. Every server checkpoint must therefore be inspected by the
loader or a separate read-only audit before its source is classified.

- `--wgmm_source checkpoint` requires recoverable mixture parameters in the
  specified VAE checkpoint or `--wgmm_checkpoint`; it fails clearly otherwise.
- `--wgmm_source gmm_fit_pool` explicitly fits the same diagonal GMM to only
  the unlabeled LHS pool's 12-dimensional `z_arch`, with a SHA-256-derived
  random state. Candidate sample weights are all one, so this is mathematically
  an **ordinary diagonal GMM**, not a weighted-data fit and not a checkpoint
  WGMM. It is an explicitly named fallback/ablation and must not silently stand
  in for a formal checkpoint-WGMM experiment.

Responsibilities, entropy, component weights, fit parameters, source-file or
fit-configuration fingerprints, and a separate fingerprint of the effective
WGMM parameter arrays are saved. HP dimensions are excluded from clustering. TED still
uses the complete Exact-GP-normalized `z_search`; inactive conditional HP
dimensions are multiplied by their existing masks.

## Two-stage flow and budget

For `wgmm_ted_lowfid` with the formal settings:

```text
one shared unlabeled LHS pool
  -> z_arch-only WGMM assignments
  -> deterministic cluster quotas
  -> per-cluster TED: 50 full-fidelity seed evaluations
  -> initial Exact GP fit on valid full-fidelity seed labels only
  -> conditioned per-cluster TED shortlist, at most 100 per nonempty cluster
  -> independent fixed-epoch low-fidelity evaluations
  -> within-cluster percentile score:
       0.50 low fidelity + 0.25 GP mean + 0.25 latent GP std
  -> select all 150 expansion indices before reading their full labels
  -> 150 fresh full-fidelity evaluations
  -> refit Exact GP on valid 50+150 full-fidelity labels only
  -> unchanged 100-step qLogEI BO
```

`50 + 150 + 100 = 300` counts only full-fidelity GNN evaluations. Low-fidelity
training is not free: candidate count, actual epochs, wall time, optional GPU
time, and epoch-ratio equivalent full evaluations are separate fields in
`budget_summary.json`. Low-fidelity observations never enter the Exact GP.
The same candidate uses the same `z_search`, HP fingerprint, candidate-specific
fidelity-independent decoder seed, and decoded-architecture fingerprint at low
and full fidelity. The GNN training seed remains fidelity/stage-specific.

Quota modes are `equal`, `proportional`, and `hybrid`. Hybrid uses
`alpha / K + (1-alpha) * pi_k`, where `pi_k` is the unlabeled pool's normalized
posterior responsibility mass and `alpha=0.5` by default. Largest remainder,
minimum nonempty-cluster coverage when feasible, capacity clipping, and
deterministic redistribution make the final quota sum exact.

The label-free numerical work stores an RBF matrix (quadratic in candidate
count) and the straightforward greedy updates are quadratic per selected point
within each cluster. Clustering reduces the sum of those per-cluster costs.
Exact GP fitting retains its existing cubic-in-training-size scaling; the
formal final initialization fit uses at most 200 valid full labels. The dominant
added experimental cost is up to 100 low-fidelity GNN trainings per nonempty
cluster, which is why their epochs and timings are reported separately.

## CLI and compatibility

The selector is optional:

```text
--initial_selection_strategy schur|wgmm_ted|wgmm_ted_lowfid
```

The default is `schur`. With no new arguments, the old `initial_points`, RNG
sequence, history schema, GP construction, and qLogEI loop remain on the old
branch. WGMM-TED requires `--surrogate_type exact_gp`,
`--gp_init_mode scratch`, an empty `--warm_start`, no frozen history, and no
legacy weighted-GMM trials. `--initial_seed_evals` must equal `--n_init` so
the existing meaning of `n_init` is preserved.

If `--max_total_full_evals` is supplied, startup requires exactly:

```text
initial_seed_evals + initial_expand_evals + n_iter == max_total_full_evals
```

The supported 300-full-evaluation comparisons are:

| Run | Seed | Expansion | Online |
|---|---:|---:|---:|
| Legacy Schur | 50 | 0 | 250 |
| WGMM-TED-50 | 50 | 0 | 250 |
| WGMM-TED-100 | 50 | 50 | 200 |
| WGMM-TED-150 | 50 | 100 | 150 |
| WGMM-TED-200 | 50 | 150 | 100 |
| WGMM-TED-LF-200 | 50 | 150 | 100 |

Use the same search seed, LHS pool size, Exact GP settings, data split,
full-fidelity settings, and qLogEI settings. Different initial points are
expected to produce naturally different online trajectories.

The formal comparison commands must explicitly retain the established
full-fidelity settings `--eval_epochs 150 --patience 40`; the parser's legacy
defaults of 100/20 are not a license to change the experiment matrix. Candidate
pool size is also part of the recorded configuration and fingerprint. Moving
from 768 candidates to 1000 defines a new experiment matrix and must not be
compared as though only the initializer had changed.

The canonical fair-comparison command invariant is therefore:

```text
--n_lhs_candidates 768 --eval_epochs 150 --patience 40
```

Every strategy in one comparison matrix must repeat that exact block. Formal
run labels must also include the WGMM source, using the recorded
`initialization_method_id` form `<strategy>__checkpoint` or
`<strategy>__gmm_fit_pool`; a fit-pool run must never be named as a checkpoint
WGMM run.

## RNG, history, and resume

Candidate evaluation seeds are derived from search seed, candidate SHA-256
fingerprint, fidelity, and stage. Decoder seeds retain the existing derivation.
Low- and full-fidelity evaluation contexts are isolated, so changing low-
fidelity order or failure behavior cannot change another candidate's full seed.
The decoder seed excludes fidelity and stage, while the training seed includes
both. Low/full consistency is checked before an expansion record is accepted.
TED and GMM utilities use local deterministic state and never Python `hash()`.

Full-fidelity records use explicit `initial_seed`, `initial_expand`, or
`online_bo` stages; low-fidelity records are kept only in
`low_fidelity_history.json`. Candidate pool, assignments, quotas, TED traces,
selected indices, scoring, budget, RNG provenance, and configuration are
written separately. Initialization artifacts and per-evaluation resume files
use temporary files plus atomic rename.

`--resume_initialization` regenerates the unlabeled design and rejects any
candidate-pool, checkpoint, strategy, quota, or configuration fingerprint
mismatch. Completed candidate/stage pairs are never reevaluated. Fixed-budget
online runs also restore the latest Exact GP and local qLogEI generator state.
Adaptive runs additionally persist the complete convergence-monitor state and
accumulated elapsed time. Resume from a clean committed iteration boundary is
supported and monitor continuity is unit-tested. A crash leaving a pending
pre-GP-update online record is still rejected for adaptive mode because its
post-update convergence check and exact time-budget state cannot be recovered;
fixed-budget pending-record recovery remains supported. Thus adaptive resume is
not claimed for every possible mid-transaction crash point.

## Analysis and scientific limits

`analyse/analyze_initialization_strategies.py` accepts explicit `LABEL=PATH`
runs and writes per-run metrics, best-so-far curves, and an interpretation
manifest. Geometry and coverage of selected points are label-free diagnostics.
Prediction, ranking, and best-so-far metrics use only each run's own observed
full-fidelity history.

Test accuracy is unavailable unless a matching `LABEL=PATH` is supplied through
`--final_results`. The path must point to an independent `final_results_*.json`
artifact, and only valid `source=history` rows are considered. Search validation
accuracy is never renamed or reused as test accuracy; missing or unusable final
evaluation evidence produces an explicit `test_metric_status=unavailable:*`.

Unless the complete shared candidate pool has full-fidelity labels, histories
contain labels only for selected candidates. They cannot support a strict
offline claim that another initializer would have performed better. Such a
claim requires new evaluations or a fully labeled, predeclared shared pool.

Performance-aware latent learning is deliberately absent. A later isolated
ablation may freeze the VAE/decoder and learn a metric only from observations
available before each decision under a prequential protocol; it must not be
introduced together with this initializer.
