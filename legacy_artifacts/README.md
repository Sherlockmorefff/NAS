# Legacy experiment artifacts

`pre_20260813/` is the local, ignored archive of experiment outputs that
predate the structured protocol layout. Its result and log files were moved by
same-filesystem rename and were not rewritten. Historical analysis must opt in
explicitly, for example:

```bash
python analyse/collect_experiment_results.py \
  --legacy-root legacy_artifacts/pre_20260813 \
  --output /tmp/nas-legacy-analysis
```

The local migration and deletion manifests live inside the ignored archive.
Do not treat this directory as a cache or delete it merely because Git ignores
it: histories, source/provenance records, checkpoints, SQLite state, and
formal outputs may be unique and expensive or impossible to reproduce.
