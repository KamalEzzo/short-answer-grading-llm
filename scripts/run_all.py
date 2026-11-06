"""Run every analysis script that works from the saved results and write each one's printed
output to outputs/<script>.txt. Compare with results/expected_outputs/.

Run from the repository root: python scripts/run_all.py
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ['table08_zeroshot_reread.py', 'tables09_12_finetuning.py', 'table13_equivalence.py',
           'table13_power.py', 'tables14_16_human_agreement.py', 'table15_percriterion_human.py',
           'tables17_18_commercial.py', 'offtopic_fewshot.py', 'table19_error_analysis.py',
           'tableD22_ablation.py', 'tableD22_losses.py', 'tableE23_per_question.py',
           'figures/fig05_06_zeroshot.py']
out = ROOT / 'outputs'
out.mkdir(exist_ok=True)
failed = []
for s in SCRIPTS:
    t = time.time()
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / s)], cwd=ROOT, capture_output=True, text=True)
    (out / (Path(s).stem + '.txt')).write_text(r.stdout + r.stderr)
    print(f'{s:40s} {"ok" if r.returncode == 0 else "FAILED"}  {time.time() - t:5.0f} s')
    if r.returncode: failed.append(s)
sys.exit(1 if failed else 0)
