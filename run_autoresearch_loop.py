"""
run_autoresearch_loop.py

Entrypoint to run the 15-cycle AutoResearch co-evolutionary loop.
"""

from src.autoresearch.coevolution import run_coevolution_loop

if __name__ == "__main__":
    results = run_coevolution_loop(num_cycles=15)
