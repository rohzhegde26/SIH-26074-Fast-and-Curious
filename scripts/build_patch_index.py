"""
scripts/build_patch_index.py

Build All-India patch index with Mandya holdout buffer exclusion and land fraction filtering.
Calculates usable patch yield for 1,708 JJAS monsoon days (2010-2023).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.data.patch_extraction import (
    HR_PATCH_SIZE,
    HR_STRIDE,
    LAND_FRACTION_THRESHOLD,
    LR_PATCH_SIZE,
    filter_spatial_windows,
    generate_monsoon_dates,
    generate_spatial_windows,
    get_temporal_split,
)


def build_index(
    output_dir: str = "data/cache",
    start_year: int = 2010,
    end_year: int = 2023,
    land_threshold: float = LAND_FRACTION_THRESHOLD,
    holdout_path: str = "data/processed/mandya_holdout_buffer.geojson",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Generating All-India 80x80 HR candidate spatial windows...")
    raw_windows = generate_spatial_windows()
    raw_window_count = len(raw_windows)
    print(f"  Generated {raw_window_count} spatial windows per day.")

    print(f"[2/4] Filtering windows (Mandya buffer holdout & land fraction >= {land_threshold*100:.0f}%)...")
    usable_windows, dropped_windows = filter_spatial_windows(
        raw_windows,
        holdout_geojson_path=holdout_path,
        land_threshold=land_threshold,
    )
    usable_spatial_count = len(usable_windows)
    dropped_count = len(dropped_windows)
    print(f"  Usable spatial windows per day: {usable_spatial_count}")
    print(f"  Dropped spatial windows per day: {dropped_count}")

    # Verify zero overlap with Mandya holdout buffer
    for w in usable_windows:
        assert not w["intersects_mandya_buffer"], "Critical Error: Mandya buffer leaked into usable windows!"
        assert w["land_fraction"] >= land_threshold, f"Land fraction {w['land_fraction']} < threshold"

    print(f"[3/4] Expanding over JJAS monsoon dates ({start_year}-{end_year})...")
    years = list(range(start_year, end_year + 1))
    dates = generate_monsoon_dates(years)
    total_days = len(dates)
    total_usable_patches = usable_spatial_count * total_days
    total_raw_patches = raw_window_count * total_days

    print(f"  Monsoon days (JJAS): {total_days} days across {len(years)} years.")
    print(f"  Raw total patches across India: {total_raw_patches:,}")
    print(f"  Usable patches after holdout & land filtering: {total_usable_patches:,}")

    # Build spatial index table for quick lookups
    spatial_df = pd.DataFrame([
        {
            "window_id": w["window_id"],
            "hr_row": w["hr_row"],
            "hr_col": w["hr_col"],
            "lr_row": w["lr_row"],
            "lr_col": w["lr_col"],
            "min_lon": w["min_lon"],
            "max_lon": w["max_lon"],
            "min_lat": w["min_lat"],
            "max_lat": w["max_lat"],
            "land_fraction": w["land_fraction"],
        }
        for w in usable_windows
    ])

    # Date / split distribution
    date_df = pd.DataFrame([
        {
            "date": str(d.date()),
            "year": d.year,
            "split": get_temporal_split(d.year),
        }
        for d in dates
    ])

    split_counts = date_df["split"].value_counts().to_dict()
    split_patch_counts = {k: v * usable_spatial_count for k, v in split_counts.items()}

    print("\n--- Temporal Splits Distribution ---")
    for split, count in split_patch_counts.items():
        print(f"  {split.upper():6s}: {count:,} patches ({split_counts[split]} days)")

    # Save metadata index
    spatial_parquet_path = out_dir / "spatial_patch_index.parquet"
    spatial_df.to_parquet(spatial_parquet_path, index=False)
    print(f"\n[4/4] Saved spatial patch index: {spatial_parquet_path}")

    summary = {
        "hr_patch_size": HR_PATCH_SIZE,
        "lr_patch_size": LR_PATCH_SIZE,
        "hr_stride": HR_STRIDE,
        "land_fraction_threshold": land_threshold,
        "raw_windows_per_day": raw_window_count,
        "usable_windows_per_day": usable_spatial_count,
        "total_monsoon_days": total_days,
        "total_usable_patches": total_usable_patches,
        "total_raw_patches": total_raw_patches,
        "splits_days": split_counts,
        "splits_patches": split_patch_counts,
        "spatial_parquet_path": str(spatial_parquet_path.as_posix()),
    }

    summary_json_path = out_dir / "patch_index_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved patch index summary: {summary_json_path}")
    print("[SUCCESS] All-India patch index built successfully!")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build All-India Patch Index")
    parser.add_argument("--output_dir", default="data/cache")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--land_threshold", type=float, default=0.70)
    parser.add_argument("--holdout", default="data/processed/mandya_holdout_buffer.geojson")
    args = parser.parse_args()

    build_index(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        land_threshold=args.land_threshold,
        holdout_path=args.holdout,
    )
