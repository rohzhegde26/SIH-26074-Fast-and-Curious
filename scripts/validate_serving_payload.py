from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.payload_validation import map_gpcodes, validate_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Sprint 1--3 forecast handoff.")
    parser.add_argument("--forecast", default="data/serving/mandya_forecasts.json")
    parser.add_argument("--map", default="frontend/mandya_simplified.topojson")
    args = parser.parse_args()
    records = json.loads(Path(args.forecast).read_text(encoding="utf-8"))
    validate_records(records, map_gpcodes(Path(args.map)))
    print(f"Valid: {len(records)} forecast records match the supplied map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
