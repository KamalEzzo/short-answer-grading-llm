"""Tables 9-12 (test set, N = 510): zero-shot (multi-step prompt) vs fine-tuned Gemma 2 9B,
off-topic detection under three conditions, confusion matrix and per-criterion results.

Run from the repository root: python scripts/tables09_12_finetuning.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import pandas as pd, numpy as np
ft = M.load('finetuning/finetuned_results.csv'); zs = M.load('finetuning/zeroshot_finetuned_prompt_results.csv')
M.check_aligned(ft, zs)
test = set(ft.student_id)
ss = pd.read_csv(M.B + 'zeroshot/results_Gemma2-9B-IT_full_dataset.csv'); ss = ss[ss.student_id.isin(test)].sort_values('student_id').reset_index(drop=True)
print('single-step test N', len(ss), 'human flag', ss.Human_Off_Topic.astype(bool).sum())
def grading(d, name):
    h, p = d.Human_Final.values, d.LLM_Final.values
    w1 = np.abs(p - h) <= 1; band = np.floor(h)
    macro = np.mean([w1[band == b].mean() for b in np.unique(band)]) * 100
    print(f'{name}: pm1={w1.mean()*100:.2f} ({w1.sum()}) macro={macro:.2f} QWK={M.qwk(d.Human_Total, d.LLM_Total):.4f} r={np.corrcoef(h,p)[0,1]:.4f} MAE={M.mae(h,p):.4f} bias={(p-h).mean():+.4f}')
    for c in M.RUBRICS:
        hc, pc = d['Human_'+c].values, d['LLM_'+c].values
        print(f'   {c:12s} QWK={M.qwk(hc, pc, labels=M.RUBRIC_LABELS[c]):.3f} MAE={np.abs(pc-hc).mean():.3f} exact={(pc==hc).mean()*100:.1f} bias={(pc-hc).mean():+.3f}')
def offtopic(hpos, mpos, nonzero, name):
    tp=(hpos&mpos).sum(); fp=(~hpos&mpos).sum(); fn=(hpos&~mpos).sum(); tn=(~hpos&~mpos).sum()
    P=tp/(tp+fp); Rr=tp/(tp+fn); F=2*P*Rr/(P+Rr)
    print(f'{name}: TP={tp} FP={fp} FN={fn} TN={tn} P={P*100:.1f} R={Rr*100:.1f} F1={F*100:.1f} incons={(mpos&nonzero).sum()}/{mpos.sum()}={(mpos&nonzero).sum()/mpos.sum()*100:.1f}')
grading(zs, 'ZS multi-step'); grading(ft, 'Fine-tuned')
pos = lambda s: s.isin(['Off-Topic', 'No Answer']).values
for d, n in [(zs, 'ZS multi-step'), (ft, 'Fine-tuned')]:
    offtopic(pos(d.Human_OffTopic), pos(d.LLM_OffTopic) | ~d.LLM_OffTopic.isin(['On-Topic','Off-Topic','No Answer']).values, d.LLM_Total.values > 0, n)
    print('   LLM categories:', d.LLM_OffTopic.value_counts().to_dict())
offtopic(ss.Human_Off_Topic.astype(bool).values, ss.LLM_Off_Topic.astype(bool).values, ss.LLM_Final.values > 0, 'ZS single-step')
print('\n3x3 fine-tuned (rows human, cols model):'); print(pd.crosstab(ft.Human_OffTopic, ft.LLM_OffTopic))
print('\nhuman categories (test):', ft.Human_OffTopic.value_counts().to_dict())
