"""Download/load datasets and emit manifests without starting NAS training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dataset_utils import (
    canonicalize_dataset_name,
    load_dataset_from_request,
    resolve_dataset_request,
    write_dataset_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real dataset loading and write manifests"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = []
    for raw_name in args.datasets:
        canonical = canonicalize_dataset_name(raw_name)
        request = resolve_dataset_request(
            canonical,
            args.data_root,
            split_seed=0 if canonical == "dblp" else None,
            ogbn_arxiv_edge_mode=(
                "undirected" if canonical == "ogbn-arxiv" else None
            ),
        )
        bundle = load_dataset_from_request(request)
        artifact_dir = Path(args.output) / canonical
        write_dataset_artifacts(bundle, artifact_dir)
        summaries.append(
            {
                key: bundle.manifest[key]
                for key in (
                    "canonical_name",
                    "num_nodes",
                    "num_edges",
                    "num_features",
                    "num_classes",
                    "train_count",
                    "val_count",
                    "test_count",
                    "split_protocol",
                    "split_seed",
                    "split_fingerprint",
                    "dataset_content_fingerprint",
                    "graph_transform",
                )
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
