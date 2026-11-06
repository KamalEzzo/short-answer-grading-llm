"""Appendix D, Table D.22 (bottom): training loss (last logged value in each epoch) and
validation loss at each epoch end for the four learning-rate runs (r = 80).

Run from the repository root: python scripts/tableD22_losses.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import pandas as pd
B = M.B + 'ablation/'
runs = [('lr1e-06', r'LR = 1$\times$10$^{-6}$'), ('lr5e-06', r'LR = 5$\times$10$^{-6}$\textsuperscript{*}'),
        ('lr1e-05', r'LR = 1$\times$10$^{-5}$'), ('lr2e-05', r'LR = 2$\times$10$^{-5}$')]
for d, lab in runs:
    tr = pd.read_csv(B + d + '/train_loss.csv').set_index('step').train_loss
    ev = pd.read_csv(B + d + '/eval_loss.csv').eval_loss.values
    assert len(ev) == 3 and tr.index.max() == 280
    t = [tr[90], tr[190], tr[280]]
    print(f"        {lab} & " + " & ".join(f"{a:.3f} & {b:.3f}" for a, b in zip(t, ev)) + r" \\")
