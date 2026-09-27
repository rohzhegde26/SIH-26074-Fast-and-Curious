"""
tests/models/test_sprint8_5_pipeline.py

Unit tests for Sprint 8.5 pipeline:
  1. Strict checkpoint verification failure modes.
  2. Out-of-sample alpha selection and evaluation.
  3. Factorial attribution structure.
  4. Aggregate spatial sharpness across all slices.
  5. Machine-readable selection artifact validation.
"""

import json
from pathlib import Path
import pytest
import torch
import numpy as np

from scripts.evaluate_sprint8_5_calibration import (
    load_model,
    evaluate_sprint8_5_full_pipeline,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PARAM_COUNT,
)


def test_load_model_missing_checkpoint():
    with pytest.raises(FileNotFoundError, match="Strict Checkpoint Verification Failed"):
        load_model(Path("non_existent_model_checkpoint.pt"), torch.device("cpu"))


def test_load_model_lfs_pointer(tmp_path):
    lfs_file = tmp_path / "fake_lfs.pt"
    lfs_file.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:123456\nsize 100\n")
    with pytest.raises(RuntimeError, match="unhydrated Git-LFS pointer"):
        load_model(lfs_file, torch.device("cpu"))


def test_load_model_sha256_mismatch(tmp_path):
    bad_ckpt = tmp_path / "corrupted_ckpt.pt"
    bad_ckpt.write_bytes(b"corrupted_binary_data_for_testing_purposes")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_model(bad_ckpt, torch.device("cpu"))


def test_pipeline_out_of_sample_split_and_factorial():
    # 4 cases, 4 members, 2 lead times, 6 channels, 16x16
    n, k, t, c, h, w = 4, 4, 2, 6, 16, 16
    torch.manual_seed(42)

    members_raw = torch.randn(n, k, t, c, h, w) * 5.0 + 5.0
    targets = torch.randn(n, t, c, h, w) * 5.0 + 5.0

    dummy_data = {
        "members_phys_raw": members_raw,
        "target_phys": targets,
        "num_cases": n,
    }

    dummy_det_data = {
        "members_phys_raw": members_raw + 0.1,
        "target_phys": targets,
        "num_cases": n,
    }

    # Split index = 2 (2 fit, 2 eval)
    results = evaluate_sprint8_5_full_pipeline(
        dummy_data,
        calib_split_idx=2,
        deterministic_eval_data=dummy_det_data,
    )
    selection_artifact = results["selection_artifact"]
    calibrators = results["calibrators"]

    # 1. Check alpha selection is recorded for fit block
    assert "phase2a_fit_grid" in results
    assert "selected_alpha" in results
    assert results["selected_alpha"] in [1.0, 1.25, 1.5, 2.0, 3.0]

    # 2. Check out-of-sample eval block results
    assert "phase2a_spread_rescaling" in results
    assert f"alpha_{results['selected_alpha']}" in results["phase2a_spread_rescaling"]

    # 3. Check Phase 2D factorial attribution
    assert "phase2d_factorial_attribution" in results
    p2d = results["phase2d_factorial_attribution"]
    assert "arm1_baseline" in p2d
    assert "arm2_sampler_only" in p2d
    assert "arm3_calibration_only" in p2d
    assert "arm4_combined" in p2d

    # 4. Check Phase 5 aggregate sharpness (2 eval cases * 2 lead times = 4 slices)
    p5 = results["phase5_spatial_sharpness"]
    assert p5["total_slices_evaluated"] == 4
    assert p5["laplacian_energy_ground_truth"] > 0.0
    assert p5["laplacian_energy_single_member"] > 0.0
    assert p5["laplacian_energy_ensemble_mean"] > 0.0

    # 5. Check selection artifact schema
    assert selection_artifact["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert selection_artifact["parameter_count"] == EXPECTED_PARAM_COUNT
    assert selection_artifact["alpha_selection"]["fit_block_cubes"] == 2
    assert selection_artifact["alpha_selection"]["eval_block_cubes"] == 2
    assert selection_artifact["holdout_protocol"]["test_split"] == "test"
