import os
import sys
from pathlib import Path
import numpy as np
import torch

root_dir = Path(r"c:\Users\rohit\.gemini\antigravity\playground\SIH")
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.autoresearch.coevolution_round3 import Round3Downscaler
from src.autoresearch.data_proxy import get_proxy_dataloaders
from src.autoresearch.evaluator import AutoResearchEvaluator
from src.losses.conservation import coarsen_hr_to_lr_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

ckpt_path = str(root_dir / "models/checkpoints/autoresearch_round3_champion.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Loaded cycle:", ckpt.get("champion_cycle"))
claimed = ckpt.get("champion_metrics", {})
print("Saved claimed metrics:", claimed)

# Instantiate Cycle 14 architecture
model = Round3Downscaler(use_curvature=True, use_terrain_skip=True).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

_, val_loader = get_proxy_dataloaders(batch_size=8)
evaluator = AutoResearchEvaluator(device=device)

all_preds, all_trues, all_lrs, all_lats, all_terrains = [], [], [], [], []
with torch.no_grad():
    for lr, hr, terrain, lats in val_loader:
        lr = lr.to(device)
        terrain = terrain.to(device)
        lats = lats.to(device)
        pred_log = model(
            torch.log1p(torch.clamp(lr, min=0.0)),
            terrain_hr=terrain,
            lats_deg=lats,
            lr_phys=lr,
        )
        pred_phys = torch.clamp(torch.expm1(pred_log), min=0.0)
        all_preds.append(pred_phys.cpu())
        all_trues.append(hr.cpu())
        all_lrs.append(lr.cpu())
        all_lats.append(lats.cpu())
        all_terrains.append(terrain.cpu())

preds = torch.cat(all_preds, dim=0)
trues = torch.cat(all_trues, dim=0)
lrs = torch.cat(all_lrs, dim=0)
lats = torch.cat(all_lats, dim=0)
terrains = torch.cat(all_terrains, dim=0)

p_np = preds.numpy().flatten()
t_np = trues.numpy().flatten()
all_mae = float(np.mean(np.abs(p_np - t_np)))
wet_mask = t_np > 2.5
wet_mae = float(np.mean(np.abs(p_np[wet_mask] - t_np[wet_mask]))) if np.sum(wet_mask) > 0 else all_mae
hf_ratio = evaluator.compute_hf_energy_ratio(preds.to(device), trues.to(device))

hits15 = np.sum((p_np > 15.0) & (t_np > 15.0))
denom15 = hits15 + np.sum((p_np <= 15.0) & (t_np > 15.0)) + np.sum((p_np > 15.0) & (t_np <= 15.0))
csi_15 = float(hits15 / denom15) if denom15 > 0 else 0.0

hits30 = np.sum((p_np > 30.0) & (t_np > 30.0))
denom30 = hits30 + np.sum((p_np <= 30.0) & (t_np > 30.0)) + np.sum((p_np > 30.0) & (t_np <= 30.0))
csi_30 = float(hits30 / denom30) if denom30 > 0 else 0.0

hits50 = np.sum((p_np > 50.0) & (t_np > 50.0))
denom50 = hits50 + np.sum((p_np <= 50.0) & (t_np > 50.0)) + np.sum((p_np > 50.0) & (t_np <= 50.0))
csi_50 = float(hits50 / denom50) if denom50 > 0 else 0.0

coarsened_preds = [coarsen_hr_to_lr_torch(preds[i:i+1], lats[i]) for i in range(len(preds))]
coarsened_preds = torch.cat(coarsened_preds, dim=0)
rel_mass_err = float((torch.abs(coarsened_preds - lrs) / torch.clamp(lrs, min=1e-4)).mean().item())

orog_w = terrains[:, 4:5, :, :].numpy().flatten()
valid_orog = np.abs(orog_w) > 0.1
orog_corr = float(np.corrcoef(p_np[valid_orog], orog_w[valid_orog])[0, 1]) if (np.sum(valid_orog) > 50 and np.std(p_np[valid_orog]) > 1e-4) else 0.0

blur_pen = max(0.0, 0.65 - hf_ratio) * 5.0
score = -(wet_mae + 1.2 * (1.0 - csi_15) + 1.8 * (1.0 - csi_30) + 0.3 * all_mae + blur_pen)

print("\n=== INDEPENDENT RE-EVALUATION RESULTS ===")
print("Composite Score: {:.4f} (Claimed: {})".format(score, claimed.get("composite_score")))
print("Wet MAE: {:.4f} mm (Claimed: {})".format(wet_mae, claimed.get("wet_mae")))
print("All-Day MAE: {:.4f} mm (Claimed: {})".format(all_mae, claimed.get("all_mae")))
print("Mass Error: {:.6f} (Claimed: {})".format(rel_mass_err, claimed.get("mass_error")))
print("HF Texture Ratio: {:.4f} (Claimed: {})".format(hf_ratio, claimed.get("hf_energy_ratio")))
print("CSI @ 15mm: {:.4f} (Claimed: {})".format(csi_15, claimed.get("csi_15")))
print("CSI @ 30mm: {:.4f} (Claimed: {})".format(csi_30, claimed.get("csi_30")))
print("CSI @ 50mm: {:.4f} (Claimed: {})".format(csi_50, claimed.get("csi_50")))
print("Orographic Correlation: {:.4f} (Claimed: {})".format(orog_corr, claimed.get("orog_corr")))

claimed_score = claimed.get("composite_score", 0.0)
diff_score = abs(score - claimed_score)
print("Score Discrepancy: {:.6f}".format(diff_score))
if diff_score < 1e-3:
    print("VERDICT: EXACT MATCH! Claimed metrics independently verified.")
else:
    print("WARNING: Metrics deviate beyond tolerance.")
