import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import numpy as np
from src.autoresearch.coevolution_round2 import Round2Downscaler
from src.autoresearch.data_proxy import get_proxy_dataloaders

# 1. Instantiate Cycle 6 Champion architecture
model = Round2Downscaler(
    use_film=True,
    use_dilated_convnext=True,
    use_windward_lift=True,
    use_hurdle_gate=False,
    use_lapse_rate=False
)

# 2. Load Champion weights
ckpt_path = "models/checkpoints/autoresearch_round2_champion.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("[+] Cycle 6 Champion model instantiated and loaded weights successfully!")

# 3. Load one batch from validation proxy
_, val_loader = get_proxy_dataloaders(batch_size=4)
lr, hr, terrain, lats = next(iter(val_loader))

print(f"[*] Input shapes: LR={lr.shape}, HR={hr.shape}, Terrain={terrain.shape}, Lats={lats.shape}")

# 4. Perform forward pass
with torch.no_grad():
    pred_log = model(
        torch.log1p(torch.clamp(lr, min=0.0)),
        terrain_hr=terrain,
        lats_deg=lats,
        lr_phys=lr
    )
    pred_phys = torch.clamp(torch.expm1(pred_log), min=0.0)

print(f"[+] Output shape: {pred_phys.shape}")
print(f"[*] Stats: Min={pred_phys.min().item():.4f}, Max={pred_phys.max().item():.4f}, Mean={pred_phys.mean().item():.4f}")
assert not torch.isnan(pred_phys).any(), "NaNs detected in inference!"
assert not torch.isinf(pred_phys).any(), "Infs detected in inference!"
print("[+] Champion model inference test PASSED with 0 NaNs and valid physical range!")
