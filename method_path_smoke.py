"""Compatibility module for :mod:`scripts.validation.method_path_smoke`."""

from pathlib import Path
import os
import sys

from scripts.validation import method_path_smoke as _implementation


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
