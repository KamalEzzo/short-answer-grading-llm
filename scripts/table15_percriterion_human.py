"""Table 15 (test set): per-criterion QWK among the four sets of human grades and between the
fine-tuned model and each set.

Run from the repository root: python scripts/table15_percriterion_human.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import itertools, pandas as pd
from metrics import B, RUBRICS, RUBRIC_LABELS, TOTAL_LABELS, load, qwk

ft = load(B + 'finetuning/finetuned_results.csv')
ft['ID'] = ft.student_id.astype(str).str.strip()
g = {'Orig': ft.rename(columns={f'Human_{r}': r for r in RUBRICS})[['ID'] + RUBRICS],
     'Model': ft.rename(columns={f'LLM_{r}': r for r in RUBRICS})[['ID'] + RUBRICS]}
for e in ['AE1', 'AE2', 'AE3']:
    a = pd.read_excel(B + f'human_benchmark/Test_Set_{e}.xlsx', sheet_name='Test Set')
    d = pd.DataFrame({'ID': a.ID.astype(str).str.strip()})
    for r in RUBRICS: d[r] = pd.to_numeric(a[r + '_Score']).values
    g[e] = d
ids = g['Orig'].ID
for k in g: g[k] = g[k].set_index('ID').loc[ids]; g[k]['Total'] = g[k][RUBRICS].sum(1)

raters = ['Orig', 'AE1', 'AE2', 'AE3']
pairs = list(itertools.combinations(raters, 2))
rows = []
for r in RUBRICS + ['Total']:
    lab = TOTAL_LABELS if r == 'Total' else RUBRIC_LABELS[r]
    row = {'Criterion': r, 'Model-Orig': qwk(g['Orig'][r], g['Model'][r], lab)}
    for x, y in pairs: row[f'{x}-{y}'] = qwk(g[x][r], g[y][r], lab)
    hp = [row[f'{x}-{y}'] for x, y in pairs]
    row['Human min'], row['Human max'] = min(hp), max(hp)
    for e in ['AE1', 'AE2', 'AE3']: row[f'Model-{e}'] = qwk(g[e][r], g['Model'][r], lab)
    rows.append(row)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).set_index('Criterion').round(3).T)
