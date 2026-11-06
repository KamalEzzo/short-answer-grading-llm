"""Figures 5 and 6: zero-shot grade distributions and per-criterion QWK on the non-test data
(N = 2,040), from the re-read Phase 1 replies (same rule as scripts/table08_zeroshot_reread.py).

Run from the repository root: python scripts/figures/fig05_06_zeroshot.py  (writes to figures/)
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[2] / "results"
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

MODEL_FILES = {
    "DeepSeek-7B": ROOT / "zeroshot" / "results_DeepSeek-7B-Chat_full_dataset.csv",
    "LLaMA 3-8B": ROOT / "zeroshot" / "results_Llama3-8B-Instruct_full_dataset.csv",
    "Gemma 2-9B": ROOT / "zeroshot" / "results_Gemma2-9B-IT_full_dataset.csv",
}

MODEL_COLORS = {
    "Human": "#2b2b2b",
    "DeepSeek-7B": "#d95f02",
    "LLaMA 3-8B": "#1b9e77",
    "Gemma 2-9B": "#377eb8",
}

RUBRICS = [
    ("Clarity", "Human_Clarity", "LLM_Clarity", [0, 1, 2]),
    ("Terminology", "Human_Terminology", "LLM_Terminology", [0, 1, 2]),
    ("Coverage", "Human_Coverage", "LLM_Coverage", [0, 1, 2]),
    ("Accuracy", "Human_Accuracy", "LLM_Accuracy", [0, 1, 2, 3, 4]),
]


MAX_SCORE = {"Clarity": 2, "Terminology": 2, "Coverage": 2, "Accuracy": 4}


def reread_scores(df):
    """Read each criterion score from the saved raw output (decimal comma = point),
    round to the nearest whole score (ties to even) and clip to the rubric range."""
    for crit, top in MAX_SCORE.items():
        vals = []
        for raw, stored in zip(df["raw_response"], df[f"LLM_{crit}"]):
            m = re.search(rf'"?{crit}_Score"?\s*:\s*(\d+(?:[.,]\d+)?)', str(raw))
            vals.append(float(m.group(1).replace(",", ".")) if m else float(stored))
        df[f"LLM_{crit}"] = np.clip(np.rint(vals), 0, top).astype(int)
    df["LLM_Final"] = sum(df[f"LLM_{c}"] for c in MAX_SCORE) / 2
    return df


def load_non_test_results():
    test_path = ROOT / "human_benchmark" / "Test_Set_AE1.xlsx"
    test_ids = set(
        pd.read_excel(test_path, sheet_name="Test Set")["ID"].astype(str).str.strip()
    )

    results = {}
    for model, path in MODEL_FILES.items():
        df = pd.read_csv(path)
        df["student_id"] = df["student_id"].astype(str).str.strip()
        results[model] = reread_scores(df.loc[~df["student_id"].isin(test_ids)].copy())
        assert len(results[model]) == 2040
    return results


def plot_grade_distributions(results):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    first_df = next(iter(results.values()))
    sns.kdeplot(
        data=first_df,
        x="Human_Final",
        ax=ax,
        color=MODEL_COLORS["Human"],
        linewidth=2.4,
        linestyle="--",
        label="Human",
        bw_adjust=0.8,
        clip=(0, 5),
    )

    for model, df in results.items():
        sns.kdeplot(
            data=df,
            x="LLM_Final",
            ax=ax,
            color=MODEL_COLORS[model],
            linewidth=2.0,
            label=model,
            bw_adjust=0.8,
            clip=(0, 5),
        )

    ax.set_xlim(0, 5)
    ax.set_xlabel("Final grade")
    ax.set_ylabel("Density")
    ax.set_title("Zero-shot predicted grade distributions (non-test data)")
    ax.legend(frameon=False, ncol=2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_zeroshot_grade_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def rubric_qwk(df, human_col, model_col, labels):
    valid = df[[human_col, model_col]].dropna()
    return cohen_kappa_score(
        valid[human_col].astype(int),
        valid[model_col].astype(int),
        labels=labels,
        weights="quadratic",
    )


def plot_rubric_radar(results):
    sns.set_theme(style="white", context="paper", font_scale=1.1)

    labels = [item[0] for item in RUBRICS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw={"polar": True})

    all_values = {}
    for model, df in results.items():
        values = [
            rubric_qwk(df, human_col, model_col, score_labels)
            for _, human_col, model_col, score_labels in RUBRICS
        ]
        all_values[model] = values
        print(f'{model:12s} per-criterion QWK: ' + ', '.join(f'{lab} {v:.3f}' for lab, v in zip(labels, values)))
        values = values + values[:1]
        ax.plot(angles, values, color=MODEL_COLORS[model], linewidth=2.0, label=model)
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.08)

    # Numeric value labels (three decimals), in each model's colour. At each
    # criterion the highest value is labelled just outside its point and the
    # others just inside, set slightly beside the axis line.
    for i in range(len(labels)):
        order = sorted(all_values, key=lambda m: all_values[m][i], reverse=True)
        for k, model in enumerate(order):
            v = all_values[model][i]
            r = v + 0.09 if k == 0 else (v - 0.12 if k == 1 else v - 0.13)
            ax.text(angles[i] + 0.16, r, f"{v:.3f}", color=MODEL_COLORS[model],
                    fontsize=8, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(135)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Per-criterion QWK by zero-shot model (non-test data)", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_zeroshot_rubric_radar.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    results = load_non_test_results()
    print('non-test answers per model:', {m: len(d) for m, d in results.items()})
    plot_grade_distributions(results)
    plot_rubric_radar(results)
    print('written: figures/fig_zeroshot_grade_distributions.pdf, figures/fig_zeroshot_rubric_radar.pdf')


if __name__ == "__main__":
    main()
