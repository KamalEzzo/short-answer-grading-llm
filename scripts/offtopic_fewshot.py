"""Off-topic precision and recall of each prompted condition (descriptive; positive class =
Off-Topic or No Answer).

Run from the repository root: python scripts/offtopic_fewshot.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import metrics as M
import pandas as pd
B = M.B
F = M.B + 'commercial_fewshot/{}/test/results.csv'
runs = {'FT Gemma': B + 'finetuning/finetuned_results.csv',
        'ZS Gemma': B + 'finetuning/zeroshot_finetuned_prompt_results.csv',
        'GPT-4o ZS': B + 'commercial_zeroshot/gpt-4o/test_results.csv', 'GPT-4o FS': F.format('gpt-4o'),
        'GPT-5.2 ZS': B + 'commercial_zeroshot/gpt-5_2/test_results.csv', 'GPT-5.2 FS': F.format('gpt-5_2'),
        'Opus 4.6 ZS': B + 'commercial_zeroshot/claude_opus_4_6/test_results.csv', 'Opus 4.6 FS': F.format('claude-opus-4-6')}
NEG = 'On-Topic'
print(f"{'model':12} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>9} {'recall':>7}")
for n, p in runs.items():
    d = pd.read_csv(p)
    h, m = d.Human_OffTopic != NEG, d.LLM_OffTopic != NEG
    tp, fp, fn = int((h & m).sum()), int((~h & m).sum()), int((h & ~m).sum())
    print(f"{n:12} {tp:4d} {fp:4d} {fn:4d} {tp/(tp+fp):9.3f} {tp/(tp+fn):7.3f}")
