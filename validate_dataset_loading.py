"""Compatibility module for :mod:`scripts.validation.validate_dataset_loading`."""

from pathlib import Path
import os
import sys

from scripts.validation import validate_dataset_loading as _implementation


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
