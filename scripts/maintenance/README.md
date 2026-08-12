# Repository maintenance utilities

`inventory_repository.py` performs a metadata-only scan of experiment artifact
paths. It never reads artifact contents, follows no symlinks, and writes only to
an explicitly supplied output directory. Each JSON/CSV/Markdown row represents
one run-level path and artifact type, with Git status, file count, byte count,
latest modification time, likely producer/reader, rebuildability, critical-state
flag, and retention guidance.

Run it with:

```bash
python scripts/maintenance/inventory_repository.py \
  --repo-root /path/to/D-VAE \
  --output-dir /explicit/report/directory
```

Classification is path-based and deliberately conservative. In particular,
ignored artifacts are not considered disposable, and `unknown` entries remain
in place until a user classifies them.
