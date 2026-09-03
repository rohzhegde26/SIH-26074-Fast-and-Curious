"""
test_splits.py
Unit tests for the 4-way temporal split architecture:
    - Train: 2010-2020 JJAS
    - Val: 2021 JJAS
    - Cal: 2022 JJAS (conformal calibration)
    - Test: 2023 JJAS
"""

import json
from pathlib import Path
import pandas as pd
import pytest
from src.data.patch_extraction import (
    generate_monsoon_dates,
    get_temporal_split,
)


def test_four_way_splits_mutually_exclusive():
    years = list(range(2010, 2024))
    dates = generate_monsoon_dates(years)

    df = pd.DataFrame([
        {
            "date": d,
            "year": d.year,
            "month": d.month,
            "split": get_temporal_split(d.year),
        }
        for d in dates
    ])

    # Assert all dates fall within June to September (JJAS)
    assert set(df["month"].unique()).issubset({6, 7, 8, 9}), "Dates must be strictly within JJAS"

    train_years = set(df[df["split"] == "train"]["year"].unique())
    val_years = set(df[df["split"] == "val"]["year"].unique())
    cal_years = set(df[df["split"] == "cal"]["year"].unique())
    test_years = set(df[df["split"] == "test"]["year"].unique())

    # Assert exact year allocations
    assert train_years == set(range(2010, 2021)), f"Unexpected train years: {train_years}"
    assert val_years == {2021}, f"Unexpected val years: {val_years}"
    assert cal_years == {2022}, f"Unexpected cal years: {cal_years}"
    assert test_years == {2023}, f"Unexpected test years: {test_years}"

    # Assert zero overlap between any pairs
    assert len(train_years & val_years) == 0, "Leakage between Train and Val"
    assert len(train_years & cal_years) == 0, "Leakage between Train and Cal"
    assert len(train_years & test_years) == 0, "Leakage between Train and Test"
    assert len(val_years & cal_years) == 0, "Leakage between Val and Cal"
    assert len(val_years & test_years) == 0, "Leakage between Val and Test"
    assert len(cal_years & test_years) == 0, "Leakage between Cal and Test"


def test_monsoon_day_counts():
    """Verify exact 122 days per year for JJAS (June 30 + July 31 + Aug 31 + Sept 30 = 122)."""
    years = list(range(2010, 2024))
    dates = generate_monsoon_dates(years)

    assert len(dates) == 1708, f"Expected 1,708 monsoon days, got {len(dates)}"

    # Check each year has exactly 122 days
    df = pd.DataFrame({"year": [d.year for d in dates]})
    counts = df["year"].value_counts()
    assert (counts == 122).all(), "Every year must have exactly 122 JJAS days"


def test_patch_index_summary_splits():
    summary_path = Path("data/cache/patch_index_summary.json")
    assert summary_path.exists(), "patch_index_summary.json must exist"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    splits = summary["splits_patches"]
    assert "train" in splits
    assert "val" in splits
    assert "cal" in splits
    assert "test" in splits

    # Val, Cal, Test each have 122 days of patches
    assert splits["val"] == splits["cal"] == splits["test"]
    assert splits["train"] > splits["val"] * 10
