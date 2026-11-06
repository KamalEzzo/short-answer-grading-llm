"""Rebuild the two Dataset figures (Figure 1 and Figure 2) directly from Data.xlsx.

Figure 1: final grade (sum of the four rubric scores / 2) grouped into six bands,
          [0,1), [1,2), [2,3), [3,4), [4,5), 5; same grouping as the
          macro-averaged accuracy.
Figure 2: score distribution of each rubric criterion.

Run from the repository root: python scripts/figures/fig01_02_dataset.py  (writes to figures/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Data.xlsx"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA)
crit = ["Clarity_Score", "Terminology_Score", "Coverage_Score", "Accuracy_Score"]
grade = df[crit].sum(axis=1) / 2
assert len(df) == 2550
assert (grade == df["Rubric_Grade"]).all(), "Rubric_Grade must equal sum / 2"

plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
BLUE = "#4472C4"

# ---------- Figure 1: final grade distribution ----------
band = np.floor(grade).astype(int)
counts = band.value_counts().reindex(range(6), fill_value=0)
pct = counts / len(df) * 100
labels = ["0\n[0, 1)", "1\n[1, 2)", "2\n[2, 3)", "3\n[3, 4)", "4\n[4, 5)", "5\n[5]"]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
bars = ax.bar(range(6), pct.values, color=BLUE, edgecolor="black", linewidth=0.5, width=0.7)
for b, n, p in zip(bars, counts.values, pct.values):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.5, f"{p:.1f}%\n({n})", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(6), labels)
ax.set_xlabel("Final grade band (0--5 scale)".replace("--", "–"))
ax.set_ylabel("Percentage of responses (%)")
ax.set_ylim(0, pct.max() * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"grade_distribution.{ext}", dpi=300)
plt.close(fig)
print("Figure 1 counts:", counts.to_dict(), "| %:", pct.round(1).to_dict())

# ---------- Figure 2: rubric criterion distributions ----------
names = ["Clarity (0–2)", "Terminology (0–2)", "Coverage (0–2)", "Accuracy (0–4)"]
fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0))
for ax, c, name in zip(axes.flat, crit, names):
    mx = 4 if c.startswith("Accuracy") else 2
    vc = df[c].value_counts().reindex(range(mx + 1), fill_value=0)
    p = vc / len(df) * 100
    bars = ax.bar(range(mx + 1), p.values, color=BLUE, edgecolor="black", linewidth=0.5, width=0.7)
    for b, v in zip(bars, p.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(name, fontsize=10)
    ax.set_xticks(range(mx + 1))
    ax.set_xlabel("Score")
    ax.set_ylabel("Percentage of responses (%)", fontsize=8.5)
    ax.set_ylim(0, p.max() * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    print(name, vc.to_dict(), "| %:", p.round(1).to_dict())
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"rubric_distribution.{ext}", dpi=300)
plt.close(fig)
