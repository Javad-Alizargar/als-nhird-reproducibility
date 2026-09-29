import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "str(Path(__file__).resolve().parent.parent)"
PROC = str(SITE / "data")
OUT = os.path.join(ROOT, "manuscripts", "ALS", "figures", "Figure_2_ALS_Transcriptomics.png")

ALS_CASE = "#D55E00"
CTRL = "#0072B2"
UP = "#C62828"
DOWN = "#1565C0"
GREY = "#B0BEC5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

expr = pd.read_csv(os.path.join(PROC, "ALS_TMM_log2CPM_expr.csv"), index_col=0)
meta = pd.read_csv(os.path.join(PROC, "ALS_sample_metadata.csv"))
scores = pd.read_csv(os.path.join(PROC, "ALS_PCA_scores.csv"), index_col=0)
res = pd.read_csv(os.path.join(ROOT, "manuscripts", "ALS", "tables",
                               "Supp_Table_1_ALS_DEGs.csv"))
ev = {"PC1": 0.172, "PC2": 0.040}

fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))

ax = axes[0, 0]
for g, c, lab in [("ALS", ALS_CASE, "ALS"), ("Control", CTRL, "Control")]:
    sub = scores.loc[meta.loc[meta["group"] == g, "sample"]]
    ax.scatter(sub["PC1"], sub["PC2"], s=26, c=c, alpha=0.85,
               edgecolors="white", linewidths=0.5, label=lab, rasterized=True)
ax.set_title("A  Principal component analysis (n=60)", loc="left", pad=6)
ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.3)
ax.set_xlabel(f"PC1 ({ev['PC1'] * 100:.0f}% variance)", labelpad=2)
ax.set_ylabel(f"PC2 ({ev['PC2'] * 100:.0f}% variance)", labelpad=2)
ax.set_xticks([])
ax.set_yticks([])

ax = axes[0, 1]
x = res["log2FC"].values
y = -np.log10(res["padj_BH"].clip(lower=1e-300).values)
sig = res["padj_BH"] < 0.05
sug = (res["padj_BH"] >= 0.05) & (res["padj_BH"] < 0.10)
ax.scatter(x[~(sig | sug)], y[~(sig | sug)], s=4, c=GREY, alpha=0.45,
           edgecolors="none", rasterized=True)
ax.scatter(x[sug], y[sug], s=6, c="#90A4AE", alpha=0.8, edgecolors="none",
           rasterized=True)
ax.scatter(x[sig], y[sig], s=9, c=DOWN, alpha=0.95, edgecolors="none",
           rasterized=True)
for i in res[sig].index:
    row = res.loc[i]
    ax.annotate(str(row["gene_symbol"]), (row["log2FC"], -np.log10(max(row["padj_BH"], 1e-300))),
                xytext=(6, 4), textcoords="offset points", fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888"))
ax.axhline(-np.log10(0.05), color="#777777", lw=0.5, ls="--")
ax.text(0.02, 0.97, "FDR<0.10 suggestive", transform=ax.transAxes, fontsize=6,
        color="#90A4AE", va="top")
ax.set_title("B  Differential expression: ALS vs Control", loc="left", pad=6)
ax.set_xlabel("log$_2$ fold change")
ax.set_ylabel("$-$log$_{10}$ adjusted $P$")

ax = axes[1, 0]
top3 = ["SRP72", "NPY2R", "WDR36"]
for pos, g in enumerate(top3):
    vals_als = expr.loc[g, meta.loc[meta["group"] == "ALS", "sample"]]
    vals_ctr = expr.loc[g, meta.loc[meta["group"] == "Control", "sample"]]
    jitter = np.random.default_rng(42).normal(0, 0.04, len(vals_als) + len(vals_ctr))
    xs = pos + jitter
    ax.scatter(xs[:len(vals_als)], vals_als, s=8, c=ALS_CASE, alpha=0.7,
               edgecolors="none", rasterized=True)
    ax.scatter(xs[len(vals_als):], vals_ctr, s=8, c=CTRL, alpha=0.7,
               edgecolors="none", rasterized=True)
    ax.plot([pos - 0.22, pos + 0.22], [vals_als.mean()] * 2, color="#222222", lw=1.4)
    ax.plot([pos - 0.22, pos + 0.22], [vals_ctr.mean()] * 2, color="#222222", lw=1.4, ls="--")
    p = res.set_index("gene_symbol").loc[g, "padj_BH"]
    ax.text(pos, ax.get_ylim()[1] * 0.99, f"q={p:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(range(3))
ax.set_xticklabels(top3, fontstyle="italic")
ax.set_xlim(-0.5, 2.5)
ax.set_ylabel("TMM log$_2$ CPM")
ax.set_title("C  Top-3 FDR-significant DEGs (EV-RNA)", loc="left", pad=6)

ax = axes[1, 1]
grp = ["ALS"] * 45 + ["Control"] * 15
bat = list(meta["batch"])
order = ["Study_1", "Study_2", "Study_4", "Study_5"]
for g_i, g in enumerate(["ALS", "Control"]):
    vals = []
    for b in order:
        vals.append(int(((meta["group"] == g) & (meta["batch"] == b)).sum()))
    xpos = np.arange(len(order)) + g_i * 0.42
    color = ALS_CASE if g == "ALS" else CTRL
    ax.bar(xpos, vals, width=0.4, color=color, alpha=0.9, label=g)
ax.set_xticks(np.arange(len(order)) + 0.21)
ax.set_xticklabels(order, rotation=15, ha="right")
ax.set_ylabel("Samples (n)")
ax.set_title("D  Batch composition by group", loc="left", pad=6)
ax.legend(frameon=False, fontsize=7)
ax.text(0.02, 0.03, "Batch fully confounded with group\n(45 ALS = Study_1)",
        transform=ax.transAxes, fontsize=6.3, color="#555555", va="bottom")

fig.tight_layout(pad=1.2)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", OUT, f"({os.path.getsize(OUT):,} bytes)")
