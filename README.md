# Fine-Tuned Open-Weight LLM for Rubric-Based Short-Answer Grading

Code, saved model outputs and analysis scripts for a study of automated, rubric-based grading of
short answers in business education. An open-weight model (Gemma 2 9B-IT) is fine-tuned with LoRA
to classify each answer (On-Topic / Off-Topic / No Answer), select a reason, and score it on four
criteria: Clarity (0-2), Terminology (0-2), Coverage (0-2) and Accuracy (0-4). The final grade is
the rubric total divided by two (0-5 scale).

- **Fine-tuned LoRA adapter:** [Hugging Face](https://huggingface.co/KamalEzzo/gemma2-9b-it-short-answer-grading-lora)
- **Dataset:** restricted access on [Zenodo (10.5281/zenodo.18856922)](https://doi.org/10.5281/zenodo.18856922); see [`data/README.md`](data/README.md)
- **No student answer text is stored in this repository.**

## Study design

| | |
|---|---|
| Data | 2,550 short answers to 85 auditing questions (30 per question) from undergraduate accounting courses |
| Split | question-level, 51 / 17 / 17 questions (1,530 / 510 / 510 answers); no question in more than one split ([`data/splits.csv`](data/splits.csv)) |
| Phase 1 | three open-weight models (Gemma 2 9B, LLaMA 3 8B, DeepSeek 7B), zero-shot, single-step prompt, non-test data (N = 2,040) |
| Phase 2 | Gemma 2 9B fine-tuned with LoRA (13 ablation runs on validation), multi-step prompt |
| Comparisons (test set) | zero-shot Gemma 2 9B; GPT-4o, GPT-5.2, Claude Sonnet 4, Claude Opus 4.6 (zero-shot); GPT-4o, GPT-5.2, Claude Opus 4.6 (few-shot, five examples); DeBERTa-v3-base encoder (three seeds) |
| Human reference | original examiner grades; three additional examiners (AE1-AE3) on the 510 test answers |

## Key results (test set, N = 510, 17 questions)

| Condition | ±1 Acc. | QWK | MAE |
|---|---|---|---|
| **Gemma 2 9B, fine-tuned** | **78.43%** | **0.821** | **0.713** |
| Gemma 2 9B, zero-shot (same multi-step prompt) | 72.35% | 0.761 | 0.875 |
| GPT-4o, zero-shot | 71.57% | 0.766 | 0.845 |
| GPT-5.2, zero-shot | 68.04% | 0.771 | 0.897 |
| Claude Sonnet 4, zero-shot | 77.45% | 0.787 | 0.813 |
| Claude Opus 4.6, zero-shot | 77.06% | 0.813 | 0.749 |
| GPT-4o, few-shot | 74.31% | 0.807 | 0.781 |
| GPT-5.2, few-shot | 67.84% | 0.761 | 0.900 |
| Claude Opus 4.6, few-shot | 80.78% | 0.835 | 0.702 |
| DeBERTa-v3-base, seeds 42 / 43 / 44 | 73.14 / 76.27 / 75.10% | 0.784 / 0.795 / 0.796 | 0.833 / 0.780 / 0.793 |

QWK is computed on the 0-10 rubric total; MAE and ±1 accuracy on the 0-5 final grade.
After Bonferroni correction (nine comparisons per metric, alpha = 0.05/9), no condition was
significantly better than the fine-tuned model on any metric; the fine-tuned model was significantly
better than zero-shot Gemma 2 9B (QWK, MAE) and than GPT-5.2 (MAE, ±1 accuracy; zero-shot and few-shot).
Equivalence testing (TOST, 90% question-cluster interval) placed the mean difference from the original
examiner grades within ±0.25 points.

## Repository structure

```
├── README.md, CHANGELOG.md, CITATION.cff, LICENSE, requirements.txt
├── data/                  # dataset NOT included; README, question split, few-shot example IDs
├── src/
│   ├── metrics.py         # shared metrics: QWK, MAE, ±1, question-cluster bootstrap, TOST, power
│   └── reliability.py     # ICC and Krippendorff's alpha
├── tests/test_metrics.py  # known-answer tests + reproduction of saved comparisons
├── scripts/               # one script per paper table or figure (run from the repository root)
│   ├── run_all.py         # runs every analysis script; output goes to outputs/
│   ├── figures/           # Figures 1-2 (needs data/), Figures 5-6
│   └── inference/         # few-shot API runs (needs data/ and API keys)
├── notebooks/             # 00 = all paper results in order; 01-11 original pipeline; 12 DeBERTa
├── results/               # saved model outputs and logs (IDs, grades, model replies)
│   ├── validation/  zeroshot/  ablation/  finetuning/
│   ├── commercial_zeroshot/  commercial_fewshot/  deberta/  human_benchmark/
│   └── expected_outputs/  # printed output of every script from the release run, for comparison
└── figures/
```

## Installation

Python 3.13; analysis packages are pinned in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_metrics.py          # about 1 minute
python scripts/run_all.py             # about 3 minutes; writes each script's output to outputs/
```

To read every result in the order of the paper without running anything, open
[`notebooks/00_Reproduce_Paper.ipynb`](notebooks/00_Reproduce_Paper.ipynb) (outputs included).

All analysis scripts run on a laptop (CPU) from the saved outputs in `results/`. Only
`scripts/figures/fig01_02_dataset.py` needs the restricted dataset (`data/Data.xlsx`).

## Reproducing the paper

| Paper item | What it shows | Code | Needs |
|---|---|---|---|
| Figures 1-2 | grade and criterion distributions | `scripts/figures/fig01_02_dataset.py` | data |
| Table 7 | dataset validation (alpha, KMO, PCA, DIF, mixed model) | `notebooks/01_Data_Validation.ipynb` | data |
| Table 8 | zero-shot model selection (N = 2,040) | `scripts/table08_zeroshot_reread.py` | results |
| Figures 5-6 | zero-shot grade distributions, per-criterion QWK | `scripts/figures/fig05_06_zeroshot.py` | results |
| Figure 7, ablation text | LoRA rank and learning-rate sweep (the plot is drawn when notebook 05 is run) | `notebooks/05_Ablation_Study.ipynb`; values in `results/ablation/summary_lora_rank.csv` and `summary_learning_rate.csv` | results |
| Tables 9-12 | fine-tuned vs zero-shot; off-topic detection; confusion matrix; per criterion | `scripts/tables09_12_finetuning.py` | results |
| Table 13 + power | mean difference, question-cluster intervals, TOST, power | `scripts/table13_equivalence.py`, `scripts/table13_power.py` | results |
| Tables 14, 16 | human-human agreement; ICC and Krippendorff's alpha | `scripts/tables14_16_human_agreement.py` | results |
| Table 15 | per-criterion QWK, humans and model | `scripts/table15_percriterion_human.py` | results |
| Tables 17-18 | all conditions; nine paired bootstrap comparisons | `scripts/tables17_18_commercial.py` | results |
| Off-topic text | precision and recall of prompted conditions | `scripts/offtopic_fewshot.py` | results |
| Table 19 | disagreement patterns | `scripts/table19_error_analysis.py` | results |
| Table D.22 | ablation runs with intervals; per-epoch losses | `scripts/tableD22_ablation.py`, `scripts/tableD22_losses.py` | results |
| Table E.23 | per-question QWK, ten conditions | `scripts/tableE23_per_question.py` | results |

Generating the model outputs themselves (not needed to reproduce the tables):

| Step | Code | Hardware / access |
|---|---|---|
| Question-level split | `notebooks/04_Data_Splitting.ipynb` | CPU, data |
| Zero-shot inference, three open-weight models | `notebooks/02_Zero_Shot_Inference.ipynb` | Colab GPU, data |
| LoRA fine-tuning, ablation, test predictions | `notebooks/06_Finetuning_Model.ipynb` | Colab GPU, data |
| Commercial models, zero-shot | `notebooks/08_Commercial_Models_Inference.ipynb` | API keys, data |
| Commercial models, few-shot | `scripts/inference/fewshot_draw_examples.py`, `scripts/inference/fewshot_run.py` | API keys, data |
| DeBERTa-v3-base baseline | `notebooks/12_DeBERTa_Baseline.ipynb` | Colab GPU (A100), data |

The GPU and API notebooks (02, 06, 08, 12) are included as run; they were not re-executed for this
release. Notebooks 03, 05, 07, 09, 10 and 11 belong to the original analysis pipeline; their outputs
were cleared, and each starts with a note naming the script that produces the reported values.
See [`CHANGELOG.md`](CHANGELOG.md).

## Metric conventions

- **QWK:** quadratic weighted kappa on the rubric total with the full label list 0-10 (per criterion: 0-2 or 0-4).
- **MAE, ±1 accuracy, bias:** on the 0-5 final grade; ±1 accuracy counts |model - examiner| <= 1; bias = model minus examiner.
- **Macro-averaged accuracy:** ±1 accuracy within six examiner-grade bands ([0,1), ..., [4,5), 5), averaged with equal weight.
- **Off-topic detection:** positive class = Off-Topic or No Answer.
- **Uncertainty:** paired question-cluster bootstrap (whole questions resampled; 17 test questions; B = 10,000; `numpy.random.RandomState(42)`); 95% percentile intervals; two-sided p = 2 x smaller tail share.
- **Multiplicity:** nine comparisons per metric, Bonferroni alpha = 0.05/9.
- **Equivalence:** TOST at alpha = 0.05 via the 90% question-cluster interval of the mean difference; margins ±0.25 and ±0.50 points.
- **Invalid outputs:** all commercial-model test replies (zero-shot and few-shot) were valid JSON; Phase 1 scores are read with the rule in `scripts/table08_zeroshot_reread.py` (decimal comma, rounding, clipping to the rubric range).

## Model and run details

| Item | Value |
|---|---|
| Base model | `google/gemma-2-9b-it` (Unsloth `unsloth/gemma-2-9b-it` for training) |
| LoRA | r = 80, alpha = 80, dropout 0, all attention and feed-forward layers |
| Training | 3 epochs, learning rate 5e-6 (cosine, 10% warm-up), effective batch 16, bfloat16, AdamW; checkpoint with the lowest validation loss |
| Decoding | open-weight models: greedy (seed 42); API models: temperature 0, 400-token limit, seed 42 for OpenAI (Claude API has no seed) |
| API models | `gpt-4o` (served as `gpt-4o-2024-08-06`), `gpt-5.2`, `claude-sonnet-4-20250514`, `claude-opus-4-6` |
| API dates | zero-shot February 2026; few-shot October 2026 (logs in `results/commercial_fewshot/*/*/run_log.json`) |
| Encoder | `microsoft/deberta-v3-base`, max length 512, batch 16, up to 10 epochs (patience 2), learning rate chosen from {1e-5, 2e-5, 3e-5} |

Claude Sonnet 4 was retired by its provider in June 2026, so it has no few-shot run. The few-shot
`validation/` runs (510 validation answers per model) were used only to confirm that every reply was
valid JSON before the single test run; they are not used in any reported comparison.

## Privacy

The dataset is not included and is shared only under restricted access (Zenodo). Result files contain
answer IDs (for example `Q4 Student1`), grades and model replies. Where a model copied part of a
student's answer into its reply, that text was replaced with `[student text removed]`; scores were not
changed. The few-shot examples are drawn from the dataset by `scripts/inference/fewshot_draw_examples.py`
and are not stored here; only their IDs are listed in `data/fewshot_example_ids.csv`.

## License and citation

Code: MIT (see `LICENSE`). Saved outputs and figures in this repository may be reused with attribution.
The dataset is under its own restricted-use terms on Zenodo, and the LoRA adapter under the Gemma terms of use.
Citation details are in [`CITATION.cff`](CITATION.cff).
