"""
run_autoresearch_round4_loop.py

Driver script to execute Round 4 (35 cycles) of the AutoResearch loop.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.autoresearch.coevolution_round4 import run_round4_coevolution_loop

if __name__ == "__main__":
    results = run_round4_coevolution_loop(num_cycles=35)
