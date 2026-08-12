"""Compatibility module for the relocated cross-dataset cost estimator."""

from pathlib import Path
import os
import sys

from scripts.validation import cross_dataset_cost_estimate as _implementation


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
