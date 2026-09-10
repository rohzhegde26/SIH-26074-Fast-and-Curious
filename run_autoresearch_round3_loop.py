"""
run_autoresearch_round3_loop.py

Driver script to execute Round 3 of the AutoResearch co-evolutionary loop.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.autoresearch.coevolution_round3 import run_round3_coevolution_loop

if __name__ == "__main__":
    results = run_round3_coevolution_loop(num_cycles=15)
