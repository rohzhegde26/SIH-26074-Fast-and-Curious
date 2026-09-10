"""
run_autoresearch_round2_loop.py

Driver script to execute Round 2 of the AutoResearch co-evolutionary loop.
"""

import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.autoresearch.coevolution_round2 import run_round2_coevolution_loop

if __name__ == "__main__":
    results = run_round2_coevolution_loop(num_cycles=15)
