"""Table 8 (zero-shot model selection, non-test data, N = 2,040).
Re-reads the saved Phase 1 replies: decimal comma read as a decimal point, each criterion
score rounded to the nearest whole score (ties to even; half-up shown as a sensitivity row)
and clipped to the rubric range. See CHANGELOG.md.

Run from the repository root: python scripts/table08_zeroshot_reread.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import pandas as pd, numpy as np, re
MAX = {'Clarity': 2, 'Terminology': 2, 'Coverage': 2, 'Accuracy': 4}
test = set(pd.read_excel(M.B + 'human_benchmark/Test_Set_AE1.xlsx', sheet_name='Test Set')['ID'])

def read(raw, key, stored):
    m = re.search(rf'"?{key}_Score"?\s*:\s*(\d+(?:[.,]\d+)?)', str(raw))
    return float(m.group(1).replace(',', '.')) if m else float(stored)

def flag(raw, stored):
    m = re.search(r'"?Off_Topic"?\s*:\s*(true|false)', str(raw), re.I)
    return (m.group(1).lower() == 'true') if m else bool(stored)

for name in ['DeepSeek-7B-Chat', 'Llama3-8B-Instruct', 'Gemma2-9B-IT']:
    d = pd.read_csv(M.B + f'zeroshot/results_{name}_full_dataset.csv')
    d = d[~d.student_id.isin(test)].reset_index(drop=True)
    for ties in ['even', 'up']:
        sc = {}
        for c, mx in MAX.items():
            v = np.array([read(r, c, s) for r, s in zip(d.raw_response, d['LLM_' + c])])
            v = np.rint(v) if ties == 'even' else np.floor(v + 0.5)
            sc[c] = np.clip(v, 0, mx).astype(int)
        tot = sum(sc.values()); f = tot / 2
        htot = sum(d['Human_' + c] for c in MAX).values; h = htot / 2
        w1 = np.abs(f - h) <= 1; band = np.floor(h)
        macro = np.mean([w1[band == b].mean() for b in np.unique(band)]) * 100
        ho = d.Human_Off_Topic.astype(bool).values
        mo = np.array([flag(r, s) for r, s in zip(d.raw_response, d.LLM_Off_Topic)])
        tp = (ho & mo).sum()
        crit = {c: round(M.qwk(d['Human_' + c].values, sc[c], labels=list(range(MAX[c] + 1))), 3) for c in MAX}
        changed = int(sum((sc[c] != d['LLM_' + c].values) for c in MAX).astype(bool).sum())
        print(f'{name:20s} ties={ties:4s} changed={changed:4d} pm1={w1.mean()*100:.1f} macro={macro:.1f} '
              f'QWK={M.qwk(htot, tot):.3f} r={np.corrcoef(h, f)[0,1]:.3f} MAE={M.mae(h, f):.3f} bias={(f-h).mean():+.3f} '
              f'P={tp/mo.sum()*100:.1f} R={tp/ho.sum()*100:.1f} incons={((mo)&(tot>0)).sum()/mo.sum()*100:.1f} crit={crit}')
