"""Compatibility entry point for the archived Phase3 Cora search."""

from pathlib import Path
import os
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    implementation = repo_root / "legacy/phased_cora_pipeline/bo_phase3.py"
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    runpy.run_path(str(implementation), run_name="__main__")


if __name__ == "__main__":
    main()
