"""Tables 17-18 (test set, N = 510): metrics of every condition, the nine paired
question-cluster bootstrap comparisons (alpha = 0.05/9), and the descriptive few-shot minus
zero-shot differences.

Run from the repository root: python scripts/tables17_18_commercial.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import numpy as np, pandas as pd
P = dict(M.PATHS)
P.update({'BERT s42': M.B + 'deberta/test_results_seed42.csv', 'BERT s43': M.B + 'deberta/test_results_seed43.csv',
          'BERT s44': M.B + 'deberta/test_results_seed44.csv',
          'FS GPT-4o': M.B + 'commercial_fewshot/gpt-4o/test/results.csv', 'FS GPT-5.2': M.B + 'commercial_fewshot/gpt-5_2/test/results.csv',
          'FS Opus 4.6': M.B + 'commercial_fewshot/claude-opus-4-6/test/results.csv'})
D = {k: M.load(v) for k, v in P.items()}
print('=== Table metrics (±1 %, QWK 0-10, MAE 0-5, bias model-human 0-5) ===')
for k, d in D.items():
    M.check_aligned(D['Gemma FT'], d)
    t = np.clip(d.LLM_Total.astype(int), 0, 10)
    print(f'{k:16s} ±1 {M.acc1(d.Human_Final, d.LLM_Final):6.2f}  QWK {M.qwk(d.Human_Total.astype(int), t):.4f}'
          f'  MAE {M.mae(d.Human_Final, d.LLM_Final):.4f}  bias {(d.LLM_Final-d.Human_Final).mean():+.4f}  n={len(d)}')
b = [D[f'BERT s{s}'] for s in (42, 43, 44)]
vals = np.array([[M.acc1(x.Human_Final, x.LLM_Final), M.qwk(x.Human_Total.astype(int), x.LLM_Total.astype(int)), M.mae(x.Human_Final, x.LLM_Final)] for x in b])
print('BERT mean', vals.mean(0).round(4), 'min', vals.min(0).round(4), 'max', vals.max(0).round(4))

print('\n=== Family: FT minus each (alpha = 0.05/9 = %.5f) ===' % M.ALPHA)
fam = ['Gemma ZS', 'GPT-4o', 'GPT-5.2', 'Claude Sonnet 4', 'Claude Opus 4.6', 'BERT s42', 'FS GPT-4o', 'FS GPT-5.2', 'FS Opus 4.6']
for k in fam:
    for r in M.paired_cluster_bootstrap(D['Gemma FT'], D[k]):
        print(f"{k:16s} {r['metric']:6s} Δ={r['delta']:+.4f} CI [{r['lo']:+.4f}, {r['hi']:+.4f}] {M.fmt_p(r['p'])} "
              f"(raw {r['p']:.4f}, mcse {r['mc_se']:.4f}) {'SIG' if M.significant(r['p']) else 'ns'}")

print('\n=== Descriptive: few-shot minus zero-shot (same model) ===')
for fs, zs in [('FS GPT-4o', 'GPT-4o'), ('FS GPT-5.2', 'GPT-5.2'), ('FS Opus 4.6', 'Claude Opus 4.6')]:
    for r in M.paired_cluster_bootstrap(D[fs], D[zs]):
        print(f"{fs:12s} - {zs:16s} {r['metric']:6s} Δ={r['delta']:+.4f} CI [{r['lo']:+.4f}, {r['hi']:+.4f}]")
