"""
src/autoresearch/evaluator.py

Agent B (Auditor) Multi-Metric Benchmark Evaluator.
Computes:
1. Wet-Day MAE (> 2.5 mm) & All-Day MAE
2. Local 5x5 Block Mass Conservation discrepancy
3. Spatial High-Frequency Energy Ratio (detects blurry conditional mean cheats)
4. Orographic Windward Correlation
5. Critical Success Index (CSI) at 15mm and 30mm (extreme cloudburst detection)
6. Composite Meteorological Fitness Score
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.conservation import coarsen_hr_to_lr_torch, expm1_transform


class AutoResearchEvaluator:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 2D discrete Laplacian filter for spatial texture energy
        self.laplacian_k = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

    def compute_hf_energy_ratio(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        """Computes ratio of high-frequency Laplacian variance to detect oversmoothing."""
        grad_pred = F.conv2d(pred, self.laplacian_k, padding=1)
        grad_true = F.conv2d(true, self.laplacian_k, padding=1)
        var_pred = float(torch.var(grad_pred).item())
        var_true = float(torch.var(grad_true).item())
        return var_pred / max(var_true, 1e-8)

    def evaluate(
        self,
        model: nn.Module,
        val_loader,
        is_log_model: bool = True,
        max_batches: int = 20,
    ) -> Tuple[Dict[str, float], bool, str]:
        """
        Evaluates model across validation loader and returns (metrics_dict, passed_invariants, message).
        """
        model.eval()
        all_preds = []
        all_trues = []
        all_lrs = []
        all_lats = []
        all_terrains = []

        with torch.no_grad():
            for idx, (lr, hr, terrain, lats) in enumerate(val_loader):
                if idx >= max_batches:
                    break
                lr = lr.to(self.device)
                terrain = terrain.to(self.device)

                if is_log_model:
                    pred_log = model(torch.log1p(torch.clamp(lr, min=0.0)), terrain_hr=terrain)
                    pred_phys = torch.clamp(torch.expm1(pred_log), min=0.0)
                else:
                    pred_phys = torch.clamp(model(lr, terrain_hr=terrain), min=0.0)

                all_preds.append(pred_phys.cpu())
                all_trues.append(hr.cpu())
                all_lrs.append(lr.cpu())
                all_lats.append(lats.cpu())
                all_terrains.append(terrain.cpu())

        preds = torch.cat(all_preds, dim=0)  # [N, 1, 80, 80]
        trues = torch.cat(all_trues, dim=0)  # [N, 1, 80, 80]
        lrs = torch.cat(all_lrs, dim=0)      # [N, 1, 16, 16]
        lats = torch.cat(all_lats, dim=0)    # [N, 80]
        terrains = torch.cat(all_terrains, dim=0)  # [N, 5, 80, 80]

        # Invariant 1: Numerical validity
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            return {}, False, "FAIL: Output contains NaNs or Infinities."

        # Invariant 2: Block mass conservation error
        # Coarsen predictions back to LR using latitude weights
        coarsened_preds = []
        for i in range(len(preds)):
            c_p = coarsen_hr_to_lr_torch(preds[i : i + 1], lats[i])
            coarsened_preds.append(c_p)
        coarsened_preds = torch.cat(coarsened_preds, dim=0)

        rel_mass_err = torch.abs(coarsened_preds - lrs) / torch.clamp(lrs, min=1e-4)
        mean_mass_err = float(rel_mass_err.mean().item())

        p_np = preds.numpy().flatten()
        t_np = trues.numpy().flatten()

        # All-day and Wet-day MAE
        all_mae = float(np.mean(np.abs(p_np - t_np)))
        wet_mask = t_np > 2.5
        if np.sum(wet_mask) > 0:
            wet_mae = float(np.mean(np.abs(p_np[wet_mask] - t_np[wet_mask])))
        else:
            wet_mae = all_mae

        # High frequency texture energy ratio
        hf_ratio = self.compute_hf_energy_ratio(preds.to(self.device), trues.to(self.device))

        # CSI at 15 mm and 30 mm
        def calc_csi(threshold):
            hits = np.sum((p_np > threshold) & (t_np > threshold))
            misses = np.sum((p_np <= threshold) & (t_np > threshold))
            fas = np.sum((p_np > threshold) & (t_np <= threshold))
            denom = hits + misses + fas
            return float(hits / denom) if denom > 0 else 0.0

        csi_15 = calc_csi(15.0)
        csi_30 = calc_csi(30.0)

        # Orographic correlation on windward slopes (terrain ch 4 is w_orog_norm)
        orog_w = terrains[:, 4:5, :, :].numpy().flatten()
        valid_orog = np.abs(orog_w) > 0.1
        if np.sum(valid_orog) > 50 and np.std(p_np[valid_orog]) > 1e-4:
            orog_corr = float(np.corrcoef(p_np[valid_orog], orog_w[valid_orog])[0, 1])
        else:
            orog_corr = 0.0

        # Composite score (higher is better, represented as negative loss)
        # Penalizes: wet MAE, low CSI_15, low CSI_30, low high-frequency energy
        blur_penalty = max(0.0, 0.65 - hf_ratio) * 5.0
        composite_score = -(wet_mae + 1.2 * (1.0 - csi_15) + 1.8 * (1.0 - csi_30) + 0.3 * all_mae + blur_penalty)

        metrics = {
            "composite_score": round(composite_score, 4),
            "wet_mae": round(wet_mae, 4),
            "all_mae": round(all_mae, 4),
            "mass_error": round(mean_mass_err, 6),
            "hf_energy_ratio": round(hf_ratio, 4),
            "csi_15": round(csi_15, 4),
            "csi_30": round(csi_30, 4),
            "orog_corr": round(orog_corr, 4),
        }

        # Physical Invariant check
        if mean_mass_err > 0.15:  # more than 15% relative mass discrepancy on unconstrained models
            return metrics, False, f"FAIL: Mass conservation error too high ({mean_mass_err:.2%})."

        return metrics, True, "PASS: Invariants satisfied."


if __name__ == "__main__":
    from src.autoresearch.data_proxy import get_proxy_dataloaders
    from src.models.unet_5x import UNet5x

    _, val_loader = get_proxy_dataloaders()
    model = UNet5x()
    evaluator = AutoResearchEvaluator()
    metrics, passed, msg = evaluator.evaluate(model, val_loader)
    print("Evaluation Result:", msg)
    print("Metrics:", metrics)
