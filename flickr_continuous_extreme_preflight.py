"""Compatibility module for the relocated Flickr extreme preflight."""

from pathlib import Path
import os
import sys

from scripts.validation import flickr_continuous_extreme_preflight as _implementation


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
