"""Compatibility entry point for the archived phased Cora data generator."""

from pathlib import Path
import os
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    implementation = repo_root / "legacy/phased_cora_pipeline/generate_mini_data.py"
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    runpy.run_path(str(implementation), run_name="__main__")


if __name__ == "__main__":
    main()
