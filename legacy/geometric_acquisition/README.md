# Geometric acquisition prototype

This directory archives an experimental geometric-acquisition branch developed
before the maintained Phase4 search settled on its current BoTorch
`qLogExpectedImprovement` workflow.

## Research purpose

The prototype explored differentiable D-VAE decoding, decoder-Jacobian geometry,
latent/reconstruction novelty, conditional GP kernels, and a novelty-augmented
LogEI score. Its original synthetic integration program compared Jacobian
methods and attempted a short geometric BO loop without a graph dataset.

## Original paths

| Archived path | Original repository path |
|---|---|
| `acqf_geometric.py` | `/acqf_geometric.py` |
| `dvae_differentiable.py` | `/dvae_differentiable.py` |
| `jacobian_utils.py` | `/jacobian_utils.py` |
| `integration_experiment.py` | `/test_integration.py` |

`test_integration.py` was renamed so default pytest collection cannot mistake
the historical experiment for a maintained test module.

## Relationship to the maintained workflow

No maintained entry point imports this package. `bo_phase4.py` implements its
formal acquisition path independently and selects candidates with the current
qLogEI implementation. This directory is excluded from current paper runs,
formal source freezes, and the maintained `tests/` suite.

## Known breakage and run status

The modules are importable and Python-compilable after replacing their former
machine-specific import paths with repository-relative package imports. That
does not make the old experiment operational under the current stack:

- the original integration program imported a function named
  `optimize_geometric_logei` that no longer exists;
- the archived `GeometricLogEI` constructor and the integration program use
  incompatible generations of the prototype API;
- the available `make_acqf_and_optimize` factory has a different call contract
  from the removed optimizer;
- model/search dimensions in parts of the integration program predate the
  maintained search-space configuration;
- historical dependency combinations were not captured as a supported current
  environment.

Accordingly, this directory is retained as research evidence and a starting
point for deliberate reconstruction, not advertised as a runnable experiment.
Do not use it for current formal NAS results without a separately reviewed port.

## Git recovery

The pre-archive versions remain available from Git. For example:

```bash
git log --follow -- legacy/geometric_acquisition/acqf_geometric.py
git show aa61050a:acqf_geometric.py
git show aa61050a:test_integration.py
```

Git history is the recovery mechanism; no root-level compatibility wrappers are
kept because the current formal workflow has no dependency on these paths.
