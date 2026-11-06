"""Power of the question-cluster TOST by simulation (equivalence section).

Run from the repository root: python scripts/table13_power.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
ft = M.load('finetuning/finetuned_results.csv')
d = (ft.LLM_Final - ft.Human_Final).values
for b, p in M.tost_power(d, ft.student_id).items():
    print(f'true bias {b:.2f}: power ±0.25 = {p[0.25]:.1f}%, ±0.50 = {p[0.50]:.1f}%')
