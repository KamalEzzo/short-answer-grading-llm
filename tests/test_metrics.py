"""Known-answer checks for src/metrics.py. Run from the repository root: python tests/test_metrics.py"""
import sys, warnings, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M

# 1. Small examples with answers worked out by hand
assert M.qwk([0, 5, 10], [0, 5, 10]) == 1.0                        # perfect agreement
assert abs(M.qwk([0, 10], [10, 0]) - (-1.0)) < 1e-12               # full-scale swap: 1 - 1/0.5 = -1
assert abs(M.qwk([0, 1, 5, 5], [1, 0, 5, 1]) - M.qwk([0, 1, 5, 5], [1, 0, 5, 1], labels=None)) > 0.05  # fixed labels matter
assert M.mae([1, 2, 3], [1.5, 2, 1]) == (0.5 + 0 + 2) / 3
assert M.acc1([1, 2, 3], [2, 2, 1]) == 2 / 3 * 100                 # 1.0 counts as within 1
assert M.fmt_p(0) == 'p < 0.0001' and M.fmt_p(0.0138) == 'p = 0.0138'
assert M.significant(0.0055) and not M.significant(0.0056)   # 0.05 / 9 = 0.00556
assert abs(M.mc_se(0.005, 10_000) - 2 * np.sqrt(0.005 * 0.995 / 10_000)) < 1e-15   # 0.00141
print('small examples: OK')

# 1b. Bootstrap on a tiny made-up set: 3 questions x 2 answers
import pandas as pd
def fake(llm):
    ids = [f'Q{q} S{s}' for q in (1, 2, 3) for s in (1, 2)]
    hum = [4, 6, 2, 8, 5, 5]
    return pd.DataFrame({'student_id': ids, 'Human_Total': hum, 'LLM_Total': llm,
                         'Human_Final': [x / 2 for x in hum], 'LLM_Final': [x / 2 for x in llm]})
base, other = fake([4, 6, 2, 8, 5, 5]), fake([6, 6, 2, 4, 5, 9])   # base perfect; other errs in Q1 (1), Q2 (2), Q3 (2)
with warnings.catch_warnings():               # model vs itself: QWK differences are 0/0 in some draws
    warnings.simplefilter('ignore', RuntimeWarning)
    r = {x['metric']: x for x in M.paired_cluster_bootstrap(base, base, n=200)}
assert r['MAE']['delta'] == 0 and r['MAE']['lo'] == r['MAE']['hi'] == 0 and r['MAE']['p'] == 1.0   # model vs itself
r = {x['metric']: x for x in M.paired_cluster_bootstrap(base, other, n=200)}
assert abs(r['MAE']['delta'] - (-5 / 6 / 2 * 2)) < 1e-12        # other's errors on 0-5 scale: 1,0,0,2,0,2 -> mean 5/6
assert r['MAE']['hi'] < 0 and r['MAE']['p'] == 0                 # every question favours base
# independent re-computation of the same draws: per-question MAE gaps, mean over sampled answers
gap = {'Q1': -0.5, 'Q2': -1.0, 'Q3': -1.0}                       # mean |error| gap per question (2 answers each)
rng = np.random.RandomState(42); Q = np.array(['Q1', 'Q2', 'Q3'])
ref = [np.mean([gap[x] for x in Q[rng.randint(0, 3, 3)]]) for _ in range(200)]
assert np.allclose([r['MAE']['lo'], r['MAE']['hi']], np.percentile(ref, [2.5, 97.5]))
# mismatched answers or human scores must be refused
for bad in (base.assign(student_id=base.student_id[::-1].values), base.assign(Human_Total=base.Human_Total + 1)):
    try: M.paired_cluster_bootstrap(base, bad, n=10); raise SystemExit('mismatch not caught')
    except AssertionError: pass
print('bootstrap examples: OK')

# 2. Reproduce the zero-shot rows of Table 18 exactly (printed precision)
EXPECTED = {  # paired question-cluster bootstrap, fine-tuned minus each zero-shot condition (Table 18)
    ('Gemma ZS', 'QWK'): ('+0.060', '+0.018', '+0.115', '0.0004'),
    ('Gemma ZS', 'MAE'): ('-0.162', '-0.281', '-0.064', '0.0000'),
    ('Gemma ZS', 'Acc±1'): ('+6.078', '+1.176', '+10.588', '0.0138'),
    ('GPT-4o', 'QWK'): ('+0.055', '+0.009', '+0.109', '0.0178'),
    ('GPT-4o', 'MAE'): ('-0.132', '-0.256', '-0.032', '0.0080'),
    ('GPT-4o', 'Acc±1'): ('+6.863', '+1.176', '+13.529', '0.0166'),
    ('GPT-5.2', 'QWK'): ('+0.050', '+0.007', '+0.090', '0.0224'),
    ('GPT-5.2', 'MAE'): ('-0.184', '-0.293', '-0.082', '0.0002'),
    ('GPT-5.2', 'Acc±1'): ('+10.392', '+5.882', '+15.098', '0.0000'),
    ('Claude Sonnet 4', 'QWK'): ('+0.034', '-0.011', '+0.080', '0.1386'),
    ('Claude Sonnet 4', 'MAE'): ('-0.100', '-0.211', '+0.006', '0.0662'),
    ('Claude Sonnet 4', 'Acc±1'): ('+0.980', '-3.725', '+6.078', '0.7494'),
    ('Claude Opus 4.6', 'QWK'): ('+0.008', '-0.026', '+0.042', '0.6556'),
    ('Claude Opus 4.6', 'MAE'): ('-0.036', '-0.128', '+0.060', '0.4562'),
    ('Claude Opus 4.6', 'Acc±1'): ('+1.373', '-4.706', '+7.059', '0.6766'),
}

ft = M.load(M.PATHS['Gemma FT'])
bad = 0
for other in ['Gemma ZS', 'GPT-4o', 'GPT-5.2', 'Claude Sonnet 4', 'Claude Opus 4.6']:
    for r in M.paired_cluster_bootstrap(ft, M.load(M.PATHS[other])):
        got = (f"{r['delta']:+.3f}", f"{r['lo']:+.3f}", f"{r['hi']:+.3f}", f"{r['p']:.4f}")
        exp = EXPECTED[(other, r['metric'])]
        bad += got != exp
        print(f"{other:16} {r['metric']:6} {'OK ' if got == exp else 'DIFF'} got {got} expected {exp}")
assert bad == 0, f'{bad} mismatches'
print('Table 18 zero-shot comparisons reproduced: OK')

# 3. TOST and power
ids = [f'Q{q} S{s}' for q in (1, 2, 3) for s in (1, 2)]
d = [0.5, 0.5, -0.5, 0.5, 0.0, 0.0]                              # question means 0.5, 0.0, 0.0
m, lo, hi = M.cluster_mean_interval(d, ids, 90, n=200)
rng = np.random.RandomState(42); Q = np.array(['Q1', 'Q2', 'Q3']); qmean = {'Q1': 0.5, 'Q2': 0.0, 'Q3': 0.0}
ref = [np.mean([qmean[x] for x in Q[rng.randint(0, 3, 3)]]) for _ in range(200)]
assert abs(m - 1 / 6) < 1e-12 and np.allclose([lo, hi], np.percentile(ref, [5, 95]))
assert M.tost(d, ids, 0.6, n=200)['equivalent'] and not M.tost(d, ids, 0.3, n=200)['equivalent']   # 90% upper limit = 1/3
p = M.tost_power(d, ids, biases=(0, 1.0), margins=(0.6,), n_sets=50, n=200)
assert p[1.0][0.6] == 0                                          # bias far outside the margin is never equivalent
print('TOST examples: OK')

dft = (ft.LLM_Final - ft.Human_Final).values
t = M.tost(dft, ft.student_id, 0.25)
got = tuple(f'{x:+.4f}' for x in (t['mean'], t['lo'], t['hi']))
assert got == ('+0.0853', '-0.0314', '+0.1922') and t['equivalent'], got     # equivalence interval (Section 5)
pw = M.tost_power(dft, ft.student_id)
got = [(b, round(pw[b][0.25], 1), round(pw[b][0.50], 1)) for b in pw]
assert got == [(0, 95.6, 100.0), (0.10, 78.0, 100.0), (0.15, 47.2, 100.0), (0.20, 19.8, 100.0)], got   # power (Section 5)
print('Equivalence interval and power reproduced: OK')
