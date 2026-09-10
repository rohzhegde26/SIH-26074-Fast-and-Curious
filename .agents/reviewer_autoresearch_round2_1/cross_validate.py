import json
import re

with open('data/cache/autoresearch_round2_history.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)

with open('autoresearch_round2_progress_report.md', 'r', encoding='utf-8') as f:
    report = f.read()

print(f"Total history records in JSON: {len(hist)}")

# Parse report table rows
table_lines = [l.strip() for l in report.split('\n') if l.strip().startswith('|') and ('**C' in l or '**Seed**' in l)]
print(f"Total data rows in report table: {len(table_lines)}")

# Let's map each table line to cycle
cycle_rows = {}
for line in table_lines:
    parts = [p.strip() for p in line.split('|')[1:-1]]
    cycle_tag = parts[0]
    cycle_rows[cycle_tag] = parts

print("\n--- Cross Validation Table ---")
print(f"{'Cycle':6s} | {'JSON Win':8s} | {'Report Win':12s} | {'Wet-MAE (JSON/Rep)':20s} | {'MassErr (JSON/Rep)':22s} | {'CSI@15 (JSON/Rep)':20s} | {'Texture (JSON/Rep)':20s} | {'Elo (JSON/Rep)':16s}")

discrepancies = []

for h in hist:
    c_num = h['cycle']
    c_tag = f"**C{c_num}**"
    m = h['metrics']
    json_win = h['winner']
    json_elo = h['elo']
    json_wet = m['wet_mae']
    json_mass = m['mass_error'] * 100.0  # as percentage
    json_csi15 = m['csi_15']
    json_tex = m['hf_energy_ratio']
    json_orog = m['orog_corr']
    
    rep_row = cycle_rows.get(c_tag)
    if not rep_row:
        discrepancies.append(f"Cycle {c_num} missing in report table!")
        continue
    
    # Table columns:
    # 0: Cycle, 1: Name, 2: Hypothesis, 3: Win/Reject, 4: Wet MAE, 5: Mass Error, 6: CSI@15, 7: Texture, 8: Orog Corr, 9: Elo
    rep_name = rep_row[1]
    rep_hyp = rep_row[2]
    rep_win = rep_row[3]
    rep_wet = float(rep_row[4].replace('*', ''))
    rep_mass_str = rep_row[5].replace('*', '').replace('%', '')
    rep_mass = float(rep_mass_str)
    rep_csi15 = float(rep_row[6].replace('*', ''))
    rep_tex = float(rep_row[7].replace('*', ''))
    rep_orog = float(rep_row[8].replace('*', ''))
    rep_elo = float(rep_row[9].replace('*', ''))
    
    # Verify values match within tolerance
    wet_diff = abs(json_wet - rep_wet)
    mass_diff = abs(json_mass - rep_mass)
    csi_diff = abs(json_csi15 - rep_csi15)
    tex_diff = abs(json_tex - rep_tex)
    elo_diff = abs(json_elo - rep_elo)
    
    status = "OK"
    if wet_diff > 0.01 or mass_diff > 0.01 or csi_diff > 0.005 or tex_diff > 0.005 or elo_diff > 0.5:
        status = "MISMATCH"
        discrepancies.append(f"C{c_num} mismatch: wet_diff={wet_diff:.4f}, mass_diff={mass_diff:.4f}, csi_diff={csi_diff:.4f}, tex_diff={tex_diff:.4f}, elo_diff={elo_diff:.1f}")
        
    print(f"C{c_num:2d}     | {str(json_win):8s} | {rep_win:12s} | {json_wet:6.3f} / {rep_wet:6.3f}   | {json_mass:6.4f}% / {rep_mass:6.4f}%  | {json_csi15:6.4f} / {rep_csi15:6.4f}   | {json_tex:6.4f} / {rep_tex:6.4f}   | {json_elo:6.1f} / {rep_elo:6.1f} | {status}")

print(f"\nTotal Discrepancies: {len(discrepancies)}")
for d in discrepancies:
    print(" -", d)
