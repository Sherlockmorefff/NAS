"""Compatibility module for the relocated Flickr incident smoke."""

from pathlib import Path
import os
import sys

from scripts.validation import flickr_failed_candidate_resource_smoke as _implementation


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
