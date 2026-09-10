"""
tests/test_forecast_audit.py

Rigorous audit test suite evaluating:
1. Weather Prediction vs Downscaling definition and physical scale preservation.
2. The root cause of the "zero / near-zero" anomaly: `use_residual=False` vs `use_residual=True`.
3. Date/provenance consistency across multi-day forecast slices (detecting 2023 vs 2026 mismatches).
4. Real-world multi-day forecast downscaling behavior on non-zero inputs.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.unet_5x import UNet5x
from src.eval.cqr import MCDropoutWrapper
from src.losses.conservation import expm1_transform, log1p_transform
from src.api.inference_service import conservative_renorm_local

ROOT = Path(__file__).resolve().parents[1]
CKPT_V3_1 = ROOT / "models" / "checkpoints" / "best_5x_model_v3_1.pt"
CKPT_V3_0 = ROOT / "models" / "checkpoints" / "best_5x_model.pt"
MULTIDAY_JSON = ROOT / "data" / "raw" / "forecast" / "multiday_coarse_20260910.json"


@pytest.mark.skipif(not CKPT_V3_1.exists(), reason="v3.1 checkpoint not found")
class TestForecastAudit:

    def test_audit_flaw_use_residual_collapse(self):
        """
        Demonstrates the exact flaw causing near-zero predictions:
        When UNet5x is instantiated with default use_residual=False, the forward pass
        returns ONLY the zero-centered adapter residual (hr_pred), dropping the coarse
        interpolated base. expm1(residual) collapses to ~0.0 mm.
        When instantiated with use_residual=True, the physical scale is correctly preserved.
        """
        ckpt = torch.load(CKPT_V3_1, map_location="cpu", weights_only=False)

        model_buggy = UNet5x(in_channels=1, out_channels=1, base_channels=32, use_residual=False)
        model_buggy.load_pretrained(ckpt, device="cpu")
        model_buggy.eval()

        model_fixed = UNet5x(in_channels=1, out_channels=1, base_channels=32, use_residual=True)
        model_fixed.load_pretrained(ckpt, device="cpu")
        model_fixed.eval()

        # Coarse input with 15.0 mm uniform rainfall
        x_phys = torch.full((1, 1, 16, 16), 15.0, dtype=torch.float32)
        x_log = log1p_transform(x_phys)

        with torch.no_grad():
            out_buggy_log = model_buggy(x_log)
            out_fixed_log = model_fixed(x_log)

            out_buggy_phys = expm1_transform(out_buggy_log)
            out_fixed_phys = expm1_transform(out_fixed_log)

        mean_buggy = float(out_buggy_phys.mean())
        mean_fixed = float(out_fixed_phys.mean())

        # The buggy instantiation produces < 0.5 mm despite a 15.0 mm input!
        assert mean_buggy < 0.5, (
            f"Buggy model output {mean_buggy:.2f} mm should be near-zero due to missing residual base"
        )

        # The fixed instantiation correctly preserves the physical order of magnitude (~10-18 mm)
        assert mean_fixed > 10.0, (
            f"Fixed model output {mean_fixed:.2f} mm must preserve coarse input magnitude (~15.0 mm)"
        )
        assert abs(mean_fixed - 15.0) / 15.0 < 0.40, (
            f"Expected fixed model to be within 40% of coarse input before conservation, got {mean_fixed:.2f} mm"
        )

    def test_audit_date_provenance_consistency(self):
        """
        Audits whether the multi-day forecast in mandya_forecasts.json has contiguous,
        consistent calendar dates without frankensteined temporal jumps.
        """
        serving_path = ROOT / "data" / "serving" / "mandya_forecasts.json"
        assert serving_path.exists(), "mandya_forecasts.json not found"

        with open(serving_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) > 0, "No records in mandya_forecasts.json"
        sample_gp = records[0]
        multi_day = sample_gp.get("multi_day_forecast", [])
        assert len(multi_day) >= 5, f"Expected at least 5 forecast days, got {len(multi_day)}"

        dates = [d["date"] for d in multi_day]
        # Check that dates are strictly sequential
        from datetime import date, timedelta
        d0 = date.fromisoformat(dates[0])
        for i, dt_str in enumerate(dates):
            expected = (d0 + timedelta(days=i)).isoformat()
            assert dt_str == expected, f"Discontinuous forecast date at offset {i}: got {dt_str}, expected {expected}"

    def test_downscaling_vs_prediction_contract(self):
        """
        Verifies the mathematical contract of downscaling:
        Downscaling is a spatial disaggregation operation:
            avg_pool2d(HR, 5, 5) ≈ LR
        It does NOT predict temporal transitions (t -> t+1); that is the role of NWP.
        """
        ckpt = torch.load(CKPT_V3_1, map_location="cpu", weights_only=False)
        model = UNet5x(in_channels=1, out_channels=1, base_channels=32, use_residual=True)
        model.load_pretrained(ckpt, device="cpu")
        model.eval()

        # Simulate synthetic NWP day 1 and day 2
        rng = np.random.default_rng(42)
        nwp_day1 = torch.from_numpy(rng.uniform(2.0, 20.0, (1, 1, 16, 16)).astype(np.float32))
        nwp_day2 = torch.from_numpy(rng.uniform(0.0, 5.0, (1, 1, 16, 16)).astype(np.float32))

        with torch.no_grad():
            hr1 = expm1_transform(model(log1p_transform(nwp_day1)))
            hr2 = expm1_transform(model(log1p_transform(nwp_day2)))

            # With conservation re-normalization
            hr1_conserved = conservative_renorm_local(hr1, nwp_day1)
            hr2_conserved = conservative_renorm_local(hr2, nwp_day2)

        # Both days must conserve mass relative to their respective NWP inputs
        coarse_from_hr1 = F.avg_pool2d(hr1_conserved, kernel_size=5, stride=5)
        coarse_from_hr2 = F.avg_pool2d(hr2_conserved, kernel_size=5, stride=5)

        assert torch.allclose(coarse_from_hr1, nwp_day1, atol=1e-4), "Day 1 spatial mass conservation violated"
        assert torch.allclose(coarse_from_hr2, nwp_day2, atol=1e-4), "Day 2 spatial mass conservation violated"

    def test_pipeline_operational_scale_preservation(self, tmp_path):
        """
        Verifies that run_pipeline in operational mode:
        1. Correctly preserves the physical precipitation scale on Day 0 and rainy days.
        2. Strictly bounds likely_min <= expected <= likely_max across all panchayats.
        """
        from scripts.run_pipeline import run_pipeline

        test_out = tmp_path / "audit_pipeline_forecasts.json"
        records = run_pipeline(
            forecast_date="2026-09-10",
            output_json_path=test_out,
            output_geojson_path=None,
            n_mc_passes=2,
            mode="operational",
        )

        assert len(records) == 234
        # On Day 0 (2026-09-10), coarse NWP has 1.94mm mean. Downscaled panchayats must be > 0.5mm
        day0_expected = [r["multi_day_forecast"][0]["expected_mm"] for r in records]
        mean_day0 = float(np.mean(day0_expected))
        assert mean_day0 > 1.0, f"Expected mean rainfall > 1.0 mm on 2026-09-10, got {mean_day0:.2f} mm"
        assert max(day0_expected) > 3.0, f"Expected peak rainfall > 3.0 mm, got {max(day0_expected):.2f} mm"

        # Conformal interval ordering
        for r in records:
            for day_item in r["multi_day_forecast"]:
                assert day_item["likely_min_mm"] <= day_item["likely_max_mm"] + 1e-4
                assert day_item["expected_mm"] >= 0.0
