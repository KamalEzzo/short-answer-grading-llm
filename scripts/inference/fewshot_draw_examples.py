"""Draw the five few-shot examples from the training questions by a fixed rule (seed 42):
one on-topic answer each with a high (7-10), middle (4-6) and low (1-3) rubric total, one
Off-Topic and one No Answer response, each from a different question.
Output: data/fewshot_examples.json (contains student text: not published).

Run from the repository root: python scripts/inference/fewshot_draw_examples.py
"""
import json, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = pd.read_excel(os.path.join(ROOT, 'data', 'Data.xlsx'))
SPLITS = pd.read_csv(os.path.join(ROOT, 'data', 'splits.csv'))
TEST = [int(q[1:]) for q in SPLITS.question[SPLITS.split == 'test']]
VAL = [int(q[1:]) for q in SPLITS.question[SPLITS.split == 'validation']]
S = ['Clarity_Score', 'Terminology_Score', 'Coverage_Score', 'Accuracy_Score']

D['q'] = D.ID.str.extract(r'^Q(\d+)')[0].astype(int)
D['s'] = D.ID.str.extract(r'(\d+)$')[0].astype(int)
D['total'] = D[S].sum(axis=1)
tr = D[~D.q.isin(TEST + VAL)].sort_values(['q', 's']).reset_index(drop=True)   # sorted by student ID (question, student)
assert tr.q.nunique() == 51 and len(tr) == 1530

bands = [('high', lambda d: (d.off_topic == 'On-Topic') & d.total.between(7, 10)),
         ('middle', lambda d: (d.off_topic == 'On-Topic') & d.total.between(4, 6)),
         ('low', lambda d: (d.off_topic == 'On-Topic') & d.total.between(1, 3)),
         ('off-topic', lambda d: d.off_topic == 'Off-Topic'),
         ('no answer', lambda d: d.off_topic == 'No Answer')]
rng, used, out = np.random.RandomState(42), set(), []
for name, cond in bands:
    c = tr[cond(tr) & ~tr.q.isin(used)]
    print(f'{name:10} candidates: {len(c)}')
    r = c.iloc[rng.randint(len(c))]
    used.add(r.q)
    out.append({'band': name, 'id': r.ID, 'question': r.Question, 'model_answer': r['Model Answer'],
                'student_answer': '' if pd.isna(r['Answer Text']) else str(r['Answer Text']),   # No Answer rows hold the placeholder '(left blank)'
                'output': {'off_topic': r.off_topic, 'Off_Topic_Reason': r.Off_Topic_Reason,
                           **{k: int(r[k]) for k in S}}})
json.dump(out, open(os.path.join(ROOT, 'data', 'fewshot_examples.json'), 'w'), indent=2, ensure_ascii=False)
for e in out:
    print(f"[{e['band']}] {e['id']} total={sum(e['output'][k] for k in S)} {e['output']}")
