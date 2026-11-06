"""Shared metrics used by every analysis script in this repository.

QWK always uses the full label list (0-10 for the rubric total, 0-2 / 0-4 per criterion);
MAE and ±1 accuracy are on the 0-5 final grade; uncertainty comes from a paired
question-cluster bootstrap (B = 10,000, seed 42); the Bonferroni family has nine
comparisons per metric (alpha = 0.05/9); TOST and its power simulation are at the end.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DATA = ROOT / 'data' / 'Data.xlsx'          # restricted dataset (see data/README.md)
B = str(RESULTS) + '/'                       # base folder for relative result paths

RUBRICS = ['Clarity', 'Terminology', 'Coverage', 'Accuracy']
RUBRIC_LABELS = {'Clarity': [0, 1, 2], 'Terminology': [0, 1, 2], 'Coverage': [0, 1, 2], 'Accuracy': [0, 1, 2, 3, 4]}
TOTAL_LABELS = list(range(11))   # rubric total 0-10
ALPHA = 0.05 / 9                 # Bonferroni family of nine comparisons per metric
N_BOOT, SEED = 10_000, 42

# Test-set result files (510 answers, 17 questions)
PATHS = {'Gemma FT': 'finetuning/finetuned_results.csv',
         'Gemma ZS': 'finetuning/zeroshot_finetuned_prompt_results.csv',
         'GPT-4o': 'commercial_zeroshot/gpt-4o/test_results.csv',
         'GPT-5.2': 'commercial_zeroshot/gpt-5_2/test_results.csv',
         'Claude Sonnet 4': 'commercial_zeroshot/claude_sonnet_4_20250514/test_results.csv',
         'Claude Opus 4.6': 'commercial_zeroshot/claude_opus_4_6/test_results.csv'}


def load(path):
    """Read one result file, add totals and 0-5 finals if missing, sort by answer ID."""
    d = pd.read_csv(path if Path(path).is_absolute() else RESULTS / path)
    for who in ['Human', 'LLM']:
        if f'{who}_Total' not in d: d[f'{who}_Total'] = sum(d[f'{who}_{r}'] for r in RUBRICS)
        if f'{who}_Final' not in d: d[f'{who}_Final'] = d[f'{who}_Total'] / 2
    return d.sort_values('student_id').reset_index(drop=True)


def check_aligned(a, b):
    """Two result files must cover the same answers with identical human scores."""
    assert (a.student_id.values == b.student_id.values).all(), 'answer IDs differ'
    assert (a.Human_Total.values == b.Human_Total.values).all(), 'human scores differ'


def qwk(h, p, labels=TOTAL_LABELS):
    """Quadratic weighted kappa, always with the full label list."""
    return cohen_kappa_score(h, p, weights='quadratic', labels=labels)


def mae(h_final, p_final):
    return np.abs(np.asarray(p_final) - np.asarray(h_final)).mean()


def acc1(h_final, p_final):
    """Share (%) of answers within 1 point on the 0-5 scale."""
    return (np.abs(np.asarray(p_final) - np.asarray(h_final)) <= 1).mean() * 100


def question_rows(ids):
    """Question of each answer (first token of student_id) -> row indices per question."""
    q = pd.Series(ids).str.split().str[0].values
    Q = np.unique(q)
    return Q, {x: np.where(q == x)[0] for x in Q}


def paired_cluster_bootstrap(base, other, n=N_BOOT, seed=SEED):
    """Paired question-cluster bootstrap of base minus other (QWK, MAE, Acc±1).

    Returns one dict per metric
    with the full-test-set delta, unadjusted 95% percentile interval and two-sided p.
    """
    check_aligned(base, other)
    h = base.Human_Total.values.astype(int)
    pb = np.clip(base.LLM_Total.values.astype(int), 0, 10)
    po = np.clip(other.LLM_Total.values.astype(int), 0, 10)
    eb = np.abs(base.LLM_Final - base.Human_Final).values
    eo = np.abs(other.LLM_Final - other.Human_Final).values

    def deltas(i):
        return (qwk(h[i], pb[i]) - qwk(h[i], po[i]),
                eb[i].mean() - eo[i].mean(),
                ((eb[i] <= 1).mean() - (eo[i] <= 1).mean()) * 100)

    Q, rows = question_rows(base.student_id)
    point = deltas(np.arange(len(h)))
    rng = np.random.RandomState(seed)
    bs = np.array([deltas(np.concatenate([rows[x] for x in Q[rng.randint(0, len(Q), len(Q))]]))
                   for _ in range(n)])
    out = []
    for k, m in enumerate(['QWK', 'MAE', 'Acc±1']):
        lo, hi = np.percentile(bs[:, k], [2.5, 97.5])
        q = min((bs[:, k] <= 0).mean(), (bs[:, k] >= 0).mean())   # smaller tail share
        out.append({'metric': m, 'delta': point[k], 'lo': lo, 'hi': hi, 'p': min(1.0, 2 * q),
                    'mc_se': mc_se(q, n)})
    return out


def mc_se(q, n=N_BOOT):
    """Monte Carlo standard error of p = 2q, where q is the smaller tail share."""
    return 2 * np.sqrt(q * (1 - q) / n)


def fmt_p(p):
    """Report p; never 'p = 0'."""
    return 'p < 0.0001' if p == 0 else f'p = {p:.4f}'


def significant(p, alpha=ALPHA):
    return p < alpha


def cluster_mean_interval(d, ids, level=90, n=N_BOOT, seed=SEED):
    """Mean of per-answer differences d with a question-cluster bootstrap percentile interval."""
    d = np.asarray(d, float)
    Q, rows = question_rows(ids)
    rng = np.random.RandomState(seed)
    bs = np.array([d[np.concatenate([rows[x] for x in Q[rng.randint(0, len(Q), len(Q))]])].mean()
                   for _ in range(n)])
    lo, hi = np.percentile(bs, [(100 - level) / 2, (100 + level) / 2])
    return d.mean(), lo, hi


def tost(d, ids, margin, n=N_BOOT, seed=SEED):
    """TOST at alpha 0.05: equivalent if the whole 90% cluster interval lies inside ±margin."""
    m, lo, hi = cluster_mean_interval(d, ids, 90, n, seed)
    return {'mean': m, 'lo': lo, 'hi': hi, 'equivalent': bool(-margin < lo and hi < margin)}


def tost_power(d, ids, biases=(0, 0.10, 0.15, 0.20), margins=(0.25, 0.50), n_sets=2000, n=N_BOOT, seed=SEED):
    """Power of the cluster TOST by simulation: centre d once, add each true bias,
    resample the questions into n_sets simulated test sets, 90% cluster interval from n draws each.
    The seed restarts for each bias, so every bias uses the same simulated sets.
    Needs equal answers per question: then a set mean equals the mean of its question means."""
    d = np.asarray(d, float)
    Q, rows = question_rows(ids)
    assert len({len(r) for r in rows.values()}) == 1, 'unequal answers per question'
    dc = d - d.mean()
    qm = np.array([dc[rows[x]].mean() for x in Q])
    k, out = len(Q), {}
    for b in biases:
        rng = np.random.RandomState(seed)
        hits = dict.fromkeys(margins, 0)
        for _ in range(n_sets):
            sim = qm[rng.randint(0, k, k)] + b
            lo, hi = np.percentile(sim[rng.randint(0, k, (n, k))].mean(1), [5, 95])
            for m in margins: hits[m] += (-m < lo) and (hi < m)
        out[b] = {m: hits[m] / n_sets * 100 for m in margins}
    return out
