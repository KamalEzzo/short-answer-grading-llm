"""Appendix E, Table E.23: per-question QWK (rubric total, labels 0-10) of the fine-tuned model
and the nine comparison conditions; bold = highest unrounded value; wins and means.

Run from the repository root: python scripts/tableE23_per_question.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import pandas as pd
P = dict(M.PATHS)
P.update({'FS GPT-4o': M.B + 'commercial_fewshot/gpt-4o/test/results.csv', 'FS GPT-5.2': M.B + 'commercial_fewshot/gpt-5_2/test/results.csv',
          'FS Opus 4.6': M.B + 'commercial_fewshot/claude-opus-4-6/test/results.csv', 'DeBERTa s42': M.B + 'deberta/test_results_seed42.csv'})
D = {k: M.load(p) for k, p in P.items()}
for k in D: M.check_aligned(D['Gemma FT'], D[k])
qs = sorted(D['Gemma FT'].student_id.str.split().str[0].unique(), key=lambda s: int(s[1:]))
T = pd.DataFrame({k: [M.qwk(d.Human_Total[d.student_id.str.split().str[0] == x].astype(int),
                             d.LLM_Total[d.student_id.str.split().str[0] == x].astype(int).clip(0, 10)) for x in qs]
                  for k, d in D.items()}, index=qs)
cols = list(T.columns); print('columns:', cols)
f = lambda v: f"{v:.3f}"
ties = [x for x, r in T.iterrows() if (r == r.max()).sum() > 1]; print('exact ties:', ties)
for x, r in T.iterrows():
    print(f"        {x} & " + " & ".join(("\\textbf{%s}" % f(r[c])) if r[c] == r.max() else f(r[c]) for c in cols) + r" \\")
m = T.mean(); print("        Mean & " + " & ".join(f(m[c]) for c in cols) + r" \\")
w = {c: int((T.idxmax(axis=1) == c).sum()) for c in cols}
print("        Wins & " + " & ".join(str(w[c]) for c in cols) + r" \\", "| total", sum(w.values()))
print('means 4dp', m.round(4).to_dict()); print('lowest', T.min().round(3).to_dict())
