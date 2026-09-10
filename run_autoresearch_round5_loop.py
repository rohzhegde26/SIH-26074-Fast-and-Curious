"""
run_autoresearch_round5_loop.py

Driver script to execute Round 5 (50 cycles) of the AutoResearch loop.
Synthesizes:
- Wavelet-Guided Convective Attention (W-GCA)
- Gradient-Isolated Multivariate Downscaling (Precip, Temp, RH)
- Exact-Conserved Multi-Quantiles (P10/P50/P90)
- Meso-Scale Cloudburst Vortex Dynamics
- Open-Ended Deep Loss Basin Optimization (through Cycle 50)
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.autoresearch.coevolution_round5 import run_round5_coevolution_loop

if __name__ == "__main__":
    results = run_round5_coevolution_loop(num_cycles=50)
