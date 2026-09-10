import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.autoresearch.coevolution_round2 import Round2Downscaler
from src.autoresearch.data_proxy import get_proxy_dataloaders
from src.losses.conservation import coarsen_hr_to_lr_torch

def run_independent_audit():
    print("=" * 70)
    print("[*] INDEPENDENT FORWARD INFERENCE & VALIDATION AUDIT")
    print("=" * 70)

    ckpt_path = Path("models/checkpoints/autoresearch_round2_champion.pt")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    print(f"[+] Loaded checkpoint keys: {list(ckpt_data.keys())}")
    print(f"[+] Champion cycle: {ckpt_data.get('champion_cycle')}")
    print(f"[+] Champion score: {ckpt_data.get('champion_score')}")
    print(f"[+] Champion metrics: {ckpt_data.get('champion_metrics')}")

    assert ckpt_data.get("champion_cycle") == 6, f"Expected cycle 6, got {ckpt_data.get('champion_cycle')}"
    
    # Initialize Cycle 6 Champion architecture
    # Cycle 6 config: use_film=True, use_dilated_convnext=True, use_windward_lift=True
    model = Round2Downscaler(
        use_film=True,
        use_dilated_convnext=True,
        use_windward_lift=True,
    )
    
    # Load state dict
    state_dict = ckpt_data["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"[+] Model load_state_dict strict=True passed! Missing: {missing}, Unexpected: {unexpected}")
    
    model.eval()

    # Load validation data
    _, val_loader = get_proxy_dataloaders(batch_size=8)
    
    total_samples = 0
    all_mae_list = []
    mass_errors = []
    
    with torch.no_grad():
        for i, (lr, hr, terrain, lats) in enumerate(val_loader):
            b, c, h, w = lr.shape
            total_samples += b
            
            # Forward pass
            lr_log = torch.log1p(torch.clamp(lr, min=0.0))
            pred_log = model(lr_log, terrain_hr=terrain, lats_deg=lats, lr_phys=lr)
            pred_phys = torch.clamp(torch.expm1(pred_log), min=0.0)

            # Assert shape
            assert pred_phys.shape == (b, 1, h * 5, w * 5), f"Unexpected shape {pred_phys.shape}"
            # Assert no NaNs / Infs
            assert not torch.isnan(pred_phys).any(), "NaN in pred_phys"
            assert not torch.isinf(pred_phys).any(), "Inf in pred_phys"
            # Assert non-negative
            assert (pred_phys >= 0.0).all(), "Negative precipitation values found"
            
            # Check mass conservation
            coarsened = torch.cat([coarsen_hr_to_lr_torch(pred_phys[j:j+1], lats[j]) for j in range(b)], dim=0)
            rel_mass_err = (torch.abs(coarsened - lr) / torch.clamp(lr, min=1e-4)).mean().item()
            mass_errors.append(rel_mass_err)
            
            # MAE
            mae = torch.abs(pred_phys - hr).mean().item()
            all_mae_list.append(mae)

            if i == 0:
                print(f"[+] Sample Batch 0: Input LR mean={lr.mean():.4f}, Target HR mean={hr.mean():.4f}, Pred HR mean={pred_phys.mean():.4f}")
                print(f"[+] Batch 0 Mass conservation error: {rel_mass_err:.4%}")
                print(f"[+] Batch 0 MAE: {mae:.4f}")

    avg_mass_error = np.mean(mass_errors)
    avg_mae = np.mean(all_mae_list)
    print(f"\n[+] Total validation samples tested: {total_samples}")
    print(f"[+] Aggregate Mean Mass Conservation Error: {avg_mass_error:.4%}")
    print(f"[+] Aggregate Mean Absolute Error: {avg_mae:.4f}")

    assert avg_mass_error < 0.15, f"Mass conservation failed: {avg_mass_error:.2%}"
    print("\n[***] INDEPENDENT TEST & INFERENCE AUDIT PASSED 100% [***]")

if __name__ == "__main__":
    run_independent_audit()
