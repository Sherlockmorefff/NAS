# Reference checkpoints

Checkpoint payloads in this directory are intentionally ignored by Git. The
current cross-dataset workflow uses
`joint_model_pipeline_global4_best.pth`; its SHA-256 and origin are recorded in
`artifacts/reference/reference_manifest.json` (local) and in the artifact
layout documentation. A checkpoint is a frozen experiment input, not a cache.
