# Changelog

## v2.0 - revision (November 2026)

Repository reorganised for the revised manuscript. History restarted (see "Privacy" below).

### Added
- `src/metrics.py`: one shared implementation of every metric: QWK with fixed labels (0-10 total,
  0-2 / 0-4 per criterion), MAE, ±1 accuracy, paired question-cluster bootstrap (B = 10,000, seed 42),
  Bonferroni alpha = 0.05/9, TOST via 90% question-cluster intervals, and the TOST power simulation.
- `tests/test_metrics.py`: known-answer tests and reproduction of saved comparisons.
- `scripts/`: one script per table or figure of the revised manuscript, with `run_all.py`;
  `notebooks/00_Reproduce_Paper.ipynb` shows all results in the order of the paper;
  `results/expected_outputs/` holds each script's output from the release run.
- Few-shot condition for GPT-4o, GPT-5.2 and Claude Opus 4.6 (October 2026): scripts in
  `scripts/inference/`, outputs and run logs in `results/commercial_fewshot/`.
- DeBERTa-v3-base encoder baseline (three seeds): `notebooks/12_DeBERTa_Baseline.ipynb`, `results/deberta/`.
- `data/splits.csv` and `data/fewshot_example_ids.csv`.

### Changed
- **Phase 1 (zero-shot model selection) scores are re-read from the saved replies.** The original
  parser kept only the whole-number part of decimal scores and did not convert a decimal comma
  (DeepSeek often wrote `1,5`), and LLaMA sometimes gave Coverage scores above the rubric maximum.
  The saved replies (not re-run) are now read with one rule: decimal comma = decimal point; each
  criterion score rounded to the nearest whole score (ties to even) and clipped to the rubric range;
  off-topic flag as written. Values change for DeepSeek and LLaMA (for example QWK 0.339 -> 0.435 and
  0.742 -> 0.762); Gemma 2 9B is unchanged (QWK 0.819) and remains the selected model.
  Code: `scripts/table08_zeroshot_reread.py`, `scripts/figures/fig05_06_zeroshot.py`.
- Confidence intervals and comparisons now resample whole questions (question-cluster bootstrap)
  instead of single answers; QWK always uses the full label list.
- Commercial comparison: one family of nine comparisons per metric (Claude Sonnet 4 has no few-shot run
  because it was retired in June 2026).
- Folder layout: `Notebooks/` -> `notebooks/`; result folders moved into `results/`
  (`Zeroshot/` -> `results/zeroshot/`, `Finetuning/` -> `results/finetuning/`,
  `Ablation Study/` -> `results/ablation/`, `GPT_Comparison/` and `Claude_Comparison/` ->
  `results/commercial_zeroshot/`, `Human Benchmark/` -> `results/human_benchmark/`,
  `Data Statistical Analysis/` -> `results/validation/`); notebook paths updated accordingly.
- Outputs of the original analysis notebooks 03, 05, 07, 09, 10 and 11 cleared; each notebook names the
  script that produces the reported values.
- README rewritten (short-answer grading, open-weight model, final results, reproduction table).

### Privacy
- Model replies that had copied part of a student's answer now show `[student text removed]` in those
  places; scores are unchanged.
- Notebook metadata added by Google Colab (user name and account ID) removed.
- The git history was restarted so that earlier versions of these files are no longer published.

## v1.0 - first submission (February 2026)

Original notebooks 01-11 and saved outputs.
