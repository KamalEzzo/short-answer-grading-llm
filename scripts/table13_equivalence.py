"""Table 13 and the equivalence text (test set): mean difference from the original examiner
grades with question-cluster 90% / 95% intervals (TOST at ±0.25 and ±0.50), and 95%
question-cluster intervals for the fine-tuned model's metrics. The response-level t-test and
Wilcoxon p-values printed first are not reported in the paper.

Run from the repository root: python scripts/table13_equivalence.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import numpy as np
from scipy import stats
ft = M.load('finetuning/finetuned_results.csv'); zs = M.load('finetuning/zeroshot_finetuned_prompt_results.csv')
for d, n in [(zs, 'ZS multi-step'), (ft, 'Fine-tuned')]:
    diff = (d.LLM_Final - d.Human_Final).values
    t = stats.ttest_rel(d.LLM_Final, d.Human_Final); w = stats.wilcoxon(d.LLM_Final, d.Human_Final)
    print(f'{n}: mean diff {diff.mean():+.4f} sd {diff.std(ddof=1):.4f} d={diff.mean()/diff.std(ddof=1):.3f} t p={t.pvalue:.6f} W p={w.pvalue:.6f}')
    f = lambda t: f'mean {t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]'
    print('   cluster 90%:', f(M.cluster_mean_interval(diff, d.student_id, level=90)), '| 95%:', f(M.cluster_mean_interval(diff, d.student_id, level=95)))
# single-model 95% question-cluster CIs for FT metrics
Q, rows = M.question_rows(ft.student_id)
rng = np.random.RandomState(M.SEED); B = M.N_BOOT  # frozen generator, as metrics.py
acc, q, mae = [], [], []
for _ in range(B):
    idx = np.concatenate([rows[x] for x in Q[rng.randint(0, len(Q), len(Q))]])
    h, p = ft.Human_Final.values[idx], ft.LLM_Final.values[idx]
    acc.append(M.acc1(h, p)); mae.append(M.mae(h, p)); q.append(M.qwk(ft.Human_Total.values[idx], ft.LLM_Total.values[idx]))
pc = lambda a: np.percentile(a, [2.5, 97.5]).round(3)
print('FT 95% question-cluster CIs: ±1', pc(acc), 'QWK', pc(q), 'MAE', pc(mae))
