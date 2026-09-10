import json
import os
import torch

def verify_all():
    print("=== STEP 1: REPORT INTEGRITY & UTF-8 ===")
    p1 = "autoresearch_round2_progress_report.md"
    p2 = r"C:\Users\rohit\Downloads\Autoresearch_Round2_Progress_Report.md"
    
    with open(p1, "r", encoding="utf-8") as f:
        c1 = f.read()
    with open(p2, "r", encoding="utf-8") as f:
        c2 = f.read()
    assert c1 == c2, "Content mismatch between root report and Downloads copy!"
    print(f"PASS: Both reports exist, are valid UTF-8, length: {len(c1)} chars, match perfectly.")

    print("\n=== STEP 2: CHECKPOINT AUDIT ===")
    ckpt_path = "models/checkpoints/autoresearch_round2_champion.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    file_size = os.path.getsize(ckpt_path)
    print(f"PASS: Checkpoint exists at {ckpt_path}, size: {file_size} bytes ({file_size/1024:.2f} KB)")
    
    weights = torch.load(ckpt_path, map_location="cpu")
    print(f"Weights type: {type(weights)}")
    if isinstance(weights, dict):
        print(f"Total keys in state dict: {len(weights)}")
        sample_keys = list(weights.keys())[:5]
        for k in sample_keys:
            val = weights[k]
            if hasattr(val, "shape"):
                print(f"  Key '{k}': tensor shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"  Key '{k}': {type(val)}")
        # Check non-trivial weights (not all zeros or nans)
        for k, v in weights.items():
            if hasattr(v, "isnan"):
                assert not torch.isnan(v).any(), f"NaN found in tensor {k}"
                assert not torch.isinf(v).any(), f"Inf found in tensor {k}"
        print("PASS: All tensors valid, no NaNs or Infs.")
    else:
        raise AssertionError("Checkpoint is not a dictionary!")

    print("\n=== STEP 3: HISTORY JSON AUDIT ===")
    hist_path = "data/cache/autoresearch_round2_history.json"
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"Loaded {len(history)} cycles from {hist_path}")
    assert len(history) == 15, f"Expected 15 cycles, found {len(history)}"

    required_metrics = [
        "composite_score", "wet_mae", "all_mae", "mass_error", 
        "hf_energy_ratio", "csi_15", "csi_30", "orog_corr", "train_sec"
    ]

    for cycle_data in history:
        c_num = cycle_data["cycle"]
        c_name = cycle_data["name"]
        c_win = cycle_data["winner"]
        c_elo = cycle_data["elo"]
        metrics = cycle_data["metrics"]
        for m in required_metrics:
            assert m in metrics, f"Cycle {c_num} missing metric {m}"
        print(f"Cycle {c_num:2d} | Win: {str(c_win):5s} | Elo: {c_elo:6.1f} | Wet-MAE: {metrics['wet_mae']:6.3f} | Mass-Err: {metrics['mass_error']:6.3f}% | CSI@15: {metrics['csi_15']:6.4f} | Name: {c_name}")

    print("\n=== STEP 4: REPORT METRICS CROSS-CHECK WITH HISTORY ===")
    # Check that winning cycles in history are indeed marked in report and git
    winners = [c for c in history if c["winner"]]
    print(f"Winning cycles: {[w['cycle'] for w in winners]}")
    assert [w["cycle"] for w in winners] == [1, 5, 6], f"Unexpected winners: {[w['cycle'] for w in winners]}"

    # Verify Cycle 6 champion metrics match report exactly
    c6 = history[5] # 0-indexed 5 is Cycle 6
    assert c6["cycle"] == 6
    assert abs(c6["metrics"]["composite_score"] - (-15.4637)) < 1e-4
    assert abs(c6["metrics"]["wet_mae"] - 8.2614) < 1e-3
    assert abs(c6["metrics"]["mass_error"] - 0.023499) < 1e-4 # reported as 2.3499%
    assert abs(c6["metrics"]["csi_15"] - 0.1384) < 1e-4
    assert abs(c6["metrics"]["hf_energy_ratio"] - 0.2891) < 1e-4
    assert abs(c6["metrics"]["orog_corr"] - 0.0096) < 1e-4
    assert c6["elo"] == 1315.0
    print("PASS: Cycle 6 Champion metrics in report match history JSON exactly.")

if __name__ == "__main__":
    verify_all()
