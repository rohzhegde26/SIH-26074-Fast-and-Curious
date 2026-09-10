import json
import os

path = r'c:\Users\rohit\.gemini\antigravity\playground\SIH\data\cache\autoresearch_round2_history.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"JSON loaded successfully! Number of records: {len(data)}")
for d in data:
    metrics = d.get("metrics", {})
    cycle = d.get("cycle")
    name = d.get("name")
    win = d.get("win")
    elo = d.get("elo_after")
    wet_mae = metrics.get("wet_mae")
    csi_15 = metrics.get("csi_15")
    mass_err = metrics.get("mass_conservation_error")
    texture = metrics.get("texture_spectrum_ratio")
    orog = metrics.get("orographic_corr")
    print(f"Cycle {cycle:2d} | Win: {str(win):5s} | Elo: {elo:6.1f} | Wet-MAE: {wet_mae:6.4f} | MassErr: {mass_err:6.4f} | CSI@15: {csi_15:6.4f} | TexRatio: {texture:6.4f} | OrographicCorr: {orog:6.4f} | Name: {name}")
