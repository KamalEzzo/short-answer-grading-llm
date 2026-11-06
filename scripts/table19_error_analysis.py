"""Table 19 (test set): disagreement patterns of the fine-tuned model, plus the facts used in
the Error Analysis text.

Run from the repository root: python scripts/table19_error_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import numpy as np, pandas as pd
ft = M.load('finetuning/finetuned_results.csv')
R = M.RUBRICS
d = ft.LLM_Final - ft.Human_Final; a = d.abs()
hot = ft.Human_OffTopic.isin(['Off-Topic', 'No Answer']); lot = ft.LLM_OffTopic.isin(['Off-Topic', 'No Answer'])
gap = {r: ft[f'LLM_{r}'] - ft[f'Human_{r}'] for r in R}
C = {'FP (on-topic flagged)': ~hot & lot, 'FN (off-topic missed)': hot & ~lot,
     'Over (d >= +2)': d >= 2, 'Under (d <= -2)': d <= -2,
     'Partial (0.5 < |d| <= 1)': (a > 0.5) & (a <= 1), 'AccConf (|gapAcc| >= 2)': gap['Accuracy'].abs() >= 2,
     'CovOver (gapCov >= 2)': gap['Coverage'] >= 2}
for k, m in C.items(): print(f'{k:28s} {m.sum():4d}  {m.mean()*100:.1f}%')
print('\nany disagreement |d|>0:', (a > 0).sum(), '| exact', (a == 0).sum(), '| |d|=0.5:', (a == 0.5).sum(), '| |d|>1:', (a > 1).sum())
both_on = ~hot & ~lot
for k in ['Over (d >= +2)', 'Under (d <= -2)']:
    s = C[k] & both_on
    print(f'{k} both on-topic n={s.sum()} (all {C[k].sum()}):', {r: round(gap[r][s].mean(), 2) for r in R},
          '| all cases:', {r: round(gap[r][C[k]].mean(), 2) for r in R})
p = C['Partial (0.5 < |d| <= 1)']
ndims = sum((gap[r][p] != 0).astype(int) for r in R)
print('\nPartial-credit: |d| values', a[p].value_counts().to_dict(), '| rubric dims differing:', ndims.value_counts().sort_index().to_dict())
ac = C['AccConf (|gapAcc| >= 2)']
largest = sum(all(abs(gap['Accuracy'][i]) > abs(gap[r][i]) for r in R[:3]) for i in ft.index[ac])
print('AccConf: Accuracy largest gap in', largest, 'of', ac.sum(), '| other rubric |gap|>=2 anywhere:', {r: int((gap[r].abs() >= 2).sum()) for r in R[:3]})
print('FN also Over:', (C['FN (off-topic missed)'] & C['Over (d >= +2)']).sum(), '| FN model grades', ft.LLM_Final[C['FN (off-topic missed)']].min(), '-', ft.LLM_Final[C['FN (off-topic missed)']].max())
print('FP human grades:', ft.Human_Final[C['FP (on-topic flagged)']].describe()[['min','50%','max']].to_dict(), '| FP model grades', ft.LLM_Final[C['FP (on-topic flagged)']].unique())
print('FP also Under:', (C['FP (on-topic flagged)'] & C['Under (d <= -2)']).sum())
print('CovOver rows:', ft.loc[C['CovOver (gapCov >= 2)'], ['student_id', 'Human_Coverage', 'LLM_Coverage', 'Human_Final', 'LLM_Final']].to_string())
print('\nHuman flag labels:', ft.Human_OffTopic.value_counts().to_dict(), '| model:', ft.LLM_OffTopic.value_counts().to_dict())
print('Off-topic-only FN:', ((ft.Human_OffTopic == 'Off-Topic') & ~lot).sum(), '| No Answer missed:', ((ft.Human_OffTopic == 'No Answer') & ~lot).sum())
