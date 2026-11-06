"""Appendix D, Table D.22 (top): 13 ablation runs on the validation set with 95%
question-cluster intervals (17 questions, B = 10,000, seed 42).

Run from the repository root: python scripts/tableD22_ablation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import numpy as np
A = M.B + 'ablation/'
RUNS = [('r8', '$r = 8$'), ('r16', '$r = 16$'), ('r32', '$r = 32$'), ('r64', '$r = 64$'),
        ('r80', r'$r = 80$\textsuperscript{*}'), ('r96', '$r = 96$'), ('r128', '$r = 128$'),
        ('lr1e-06', r'LR = 1$\times$10$^{-6}$'), ('lr5e-06', r'LR = 5$\times$10$^{-6}$\textsuperscript{*}'),
        ('lr1e-05', r'LR = 1$\times$10$^{-5}$'), ('lr2e-05', r'LR = 2$\times$10$^{-5}$'),
        ('r32_5epochs', '$r = 32$, 5 ep.'), ('r80_dropout0.1', r'$r = 80$, drop.\ 0.1')]
for run, lab in RUNS:
    d = M.load(A + run + '/val_results.csv')
    assert len(d) == 510
    hT, pT = d.Human_Total.values.astype(int), np.clip(d.LLM_Total.values.astype(int), 0, 10)
    hF, pF = d.Human_Final.values, d.LLM_Final.values
    Q, rows = M.question_rows(d.student_id); assert len(Q) == 17
    pt = (M.acc1(hF, pF), M.mae(hF, pF), M.qwk(hT, pT))
    rng = np.random.RandomState(M.SEED); acc, mae, q = [], [], []
    for _ in range(M.N_BOOT):
        i = np.concatenate([rows[x] for x in Q[rng.randint(0, len(Q), len(Q))]])
        acc.append(M.acc1(hF[i], pF[i])); mae.append(M.mae(hF[i], pF[i])); q.append(M.qwk(hT[i], pT[i]))
    ci = [np.percentile(a, [2.5, 97.5]) for a in (acc, mae, q)]
    print(f"{run:15s} acc1 {pt[0]:.2f} [{ci[0][0]:.1f},{ci[0][1]:.1f}]  MAE {pt[1]:.3f} [{ci[1][0]:.3f},{ci[1][1]:.3f}]  QWK {pt[2]:.4f} [{ci[2][0]:.3f},{ci[2][1]:.3f}]")
    print(f"LATEX {lab} & {pt[0]:.1f} [{ci[0][0]:.1f}, {ci[0][1]:.1f}] & {pt[1]:.3f} [{ci[1][0]:.3f}, {ci[1][1]:.3f}] & {pt[2]:.3f} [{ci[2][0]:.3f}, {ci[2][1]:.3f}]")
