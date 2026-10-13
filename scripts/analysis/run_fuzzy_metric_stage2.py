#!/usr/bin/env python3
"""Run the frozen stage-2 entry with fresh Python worker processes.

The numerical source/config stay unchanged. This launcher repairs PyTorch's
autograd/fork incompatibility and does not alter fit accounting or budgets.
"""
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    from analyse.fuzzy_metric_offline import main
    main()
