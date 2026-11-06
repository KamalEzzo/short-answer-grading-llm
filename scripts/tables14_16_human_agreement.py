"""Tables 14 and 16 (test set, N = 510): pairwise agreement among the four sets of human
grades (original examiners, AE1-AE3) and multi-rater reliability with 95% question-cluster
intervals (17 questions, B = 10,000, seed 42). Bias = second-named minus first-named grader.

Run from the repository root: python scripts/tables14_16_human_agreement.py
"""
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, krippendorff
import metrics as M
import reliability as rel

ft = M.load('finetuning/finetuned_results.csv')
tot = pd.DataFrame({'ID': ft.student_id.astype(str).str.strip(),
                    'Original': ft.Human_Total.astype(int), 'Model': ft.LLM_Total.astype(int)})
for e in ['AE1', 'AE2', 'AE3']:
    a = pd.read_excel(M.B + f'human_benchmark/Test_Set_{e}.xlsx', sheet_name='Test Set')
    tot = tot.merge(pd.DataFrame({'ID': a.ID.astype(str).str.strip(),
                                  e: a[[r + '_Score' for r in M.RUBRICS]].apply(pd.to_numeric).sum(1).astype(int)}), on='ID')
assert len(tot) == 510
tot = tot.sort_values('ID').reset_index(drop=True)
fin = tot.set_index('ID')[['Original', 'AE1', 'AE2', 'AE3', 'Model']] / 2

print('Table 14: pairwise agreement among the four sets of human grades')
rows = []
for x, y in combinations(['Original', 'AE1', 'AE2', 'AE3'], 2):
    d = fin[y] - fin[x]
    rows.append((f'{x} vs {y}', M.qwk(tot[x], tot[y]), d.abs().mean(), (d.abs() <= 1).mean() * 100, d.mean()))
for c, q, mae, a1, b in sorted(rows, key=lambda r: -r[1]):
    print(f'  {c:18s} QWK {q:.3f}  MAE {mae:.3f}  ±1 {a1:.2f}%  bias {b:+.3f}')
print(f"  Model vs Original  QWK {M.qwk(tot.Original, tot.Model):.3f}  MAE {M.mae(fin.Original, fin.Model):.3f}  "
      f"±1 {M.acc1(fin.Original, fin.Model):.2f}%")

print('\nTable 16: multi-rater reliability (final grade 0-5)')
Q, idx = M.question_rows(tot.ID)
for name, cols in [('AE1-AE3 (primary)', ['AE1', 'AE2', 'AE3']), ('Original + AE1-AE3 (secondary)', ['Original', 'AE1', 'AE2', 'AE3'])]:
    v = fin[cols].to_numpy(float)
    rng = np.random.RandomState(M.SEED); ic, al = [], []
    for _ in range(M.N_BOOT):
        s = np.concatenate([idx[x] for x in Q[rng.randint(0, len(Q), len(Q))]])
        ic.append(rel.icc(v[s])['ICC(2,1)']); al.append(rel.krippendorff_alpha_interval(v[s]))
    ci = lambda a: np.percentile(a, [2.5, 97.5]).round(3)
    ordinal = krippendorff.alpha(reliability_data=v.T, level_of_measurement='ordinal')
    print(f"  {name:32s} ICC(2,1) {rel.icc(v)['ICC(2,1)']:.3f} {ci(ic)}  alpha interval "
          f"{rel.krippendorff_alpha_interval(v):.3f} {ci(al)}  alpha ordinal {ordinal:.3f}")
