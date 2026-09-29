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
from sklearn.decomposition import PCA

ROOT = "str(Path(__file__).resolve().parent.parent)"
PROC = str(SITE / "data")
TABLES = os.path.join(ROOT, "manuscript", "tables")
FIGDIR = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIGDIR, exist_ok=True)

ALS_CASE = "#D55E00"
CTRL = "#0072B2"
UP = "#C62828"
DOWN = "#1565C0"
GREY = "#B0BEC5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "svg.fonttype": "none",
})


def pca_scores(expr, meta, n_top=500):
    var = expr.var(axis=1)
    top = expr.loc[var.nlargest(n_top).index]
    z = top.sub(top.mean(axis=1), axis=0).div(top.std(axis=1) + 1e-12, axis=0)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(z.T)
    return pd.DataFrame(scores, index=z.columns,
                        columns=["PC1", "PC2"]), pca.explained_variance_ratio_


def scatter_pca(ax, scores, meta, title):
    for grp, color, label in [("Control", CTRL, "Control"),
                              (None, None, None)]:
        pass
    groups = {"ALS": (ALS_CASE, "ALS"), "Control": (CTRL, "Control"),
              "WD": (ALS_CASE, "WD")}
    for g in meta["group"].unique():
        color, label = groups[g]
        sub = scores.loc[meta.loc[meta["group"] == g, "sample"]]
        ax.scatter(sub["PC1"], sub["PC2"], s=26, c=color, alpha=0.85,
                   edgecolors="white", linewidths=0.5, label=label, rasterized=True)
    ev1 = scores.attrs.get("ev1", 0)
    ax.set_title(title, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def volcano(ax, res, title, label_n=10, fdr_line=0.05):
    x = res["log2FC"].values
    y = -np.log10(res["padj_BH"].clip(lower=1e-300).values)
    sig = res["padj_BH"] < fdr_line
    up = sig & (x > 0)
    dn = sig & (x < 0)
    ax.scatter(x[~(up | dn)], y[~(up | dn)], s=4, c=GREY, alpha=0.5,
               edgecolors="none", rasterized=True)
    ax.scatter(x[up], y[up], s=7, c=UP, alpha=0.9, edgecolors="none",
               rasterized=True)
    ax.scatter(x[dn], y[dn], s=7, c=DOWN, alpha=0.9, edgecolors="none",
               rasterized=True)
    top = res[(res["padj_BH"] < 0.15)].nlargest(label_n, "log2FC").index.tolist()
    top += res[(res["padj_BH"] < 0.15)].nsmallest(label_n, "log2FC").index.tolist()
    top += res[res["padj_BH"] < fdr_line].index.tolist()
    seen = []
    for i in top:
        if i not in seen:
            seen.append(i)
    for i in seen[:label_n * 2]:
        row = res.loc[i]
        label = str(i)
        dx = 0.15 if row["log2FC"] >= 0 else -0.15
        ax.annotate(label, (row["log2FC"], -np.log10(max(row["padj_BH"], 1e-300))),
                    xytext=(6 * np.sign(dx), 4), textcoords="offset points",
                    fontsize=6.5, color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888"),
                    ha="left" if row["log2FC"] >= 0 else "right")
    n_up = int(up.sum())
    n_dn = int(dn.sum())
    ax.text(0.03, 0.97, f"{n_up} up", transform=ax.transAxes, fontsize=6.5,
            color=UP, va="top")
    ax.text(0.97, 0.97, f"{n_dn} down", transform=ax.transAxes, fontsize=6.5,
            color=DOWN, va="top", ha="right")
    ax.axhline(-np.log10(fdr_line), color="#777777", lw=0.5, ls="--")
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlabel("log$_2$ fold change")
    ax.set_ylabel("$-$log$_{10}$ FDR")


als_expr = pd.read_csv(os.path.join(PROC, "ALS_TMM_log2CPM_expr.csv"), index_col=0)
als_meta = pd.read_csv(os.path.join(PROC, "ALS_sample_metadata.csv"))
als_res = pd.read_csv(os.path.join(TABLES, "Supp_Table_1_ALS_DEGs.csv"))
als_res = als_res.set_index("gene_symbol")

wd_expr = pd.read_csv(os.path.join(PROC, "Wilson_log2_signal_matrix.csv"), index_col=0)
wd_meta = pd.read_csv(os.path.join(PROC, "Wilson_sample_metadata.csv"))
wd_res = pd.read_csv(os.path.join(TABLES, "Supp_Table_2_Wilson_DEGs.csv"))
wd_res = wd_res.set_index("gene_symbol")

als_scores, als_ev = pca_scores(als_expr, als_meta)
als_scores.attrs["ev1"], als_scores.attrs["ev2"] = als_ev
wd_scores, wd_ev = pca_scores(wd_expr, wd_meta)
wd_scores.attrs["ev1"], wd_scores.attrs["ev2"] = wd_ev

als_scores.to_csv(os.path.join(PROC, "ALS_PCA_scores.csv"))
wd_scores.to_csv(os.path.join(PROC, "Wilson_PCA_scores.csv"))
ev = {"als_pc1": float(als_ev[0]), "als_pc2": float(als_ev[1]),
      "wd_pc1": float(wd_ev[0]), "wd_pc2": float(wd_ev[1])}
print("PCA variance explained:", {k: round(v, 3) for k, v in ev.items()})

fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))

ax = axes[0, 0]
for g in als_meta["group"].unique():
    color, label = {"ALS": (ALS_CASE, "ALS"), "Control": (CTRL, "Control")}[g]
    sub = als_scores.loc[als_meta.loc[als_meta["group"] == g, "sample"]]
    ax.scatter(sub["PC1"], sub["PC2"], s=26, c=color, alpha=0.85,
               edgecolors="white", linewidths=0.5, label=label, rasterized=True)
ax.set_title("A  ALS cohort (GSE346896, n=60)", loc="left", pad=6)
ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.3)
ax.set_xlabel(f"PC1 ({als_ev[0] * 100:.0f}% var)", labelpad=2)
ax.set_ylabel(f"PC2 ({als_ev[1] * 100:.0f}% var)", labelpad=2)
ax.set_xticks([])
ax.set_yticks([])

ax = axes[0, 1]
volcano(ax, als_res, "B  ALS: differential expression (FDR<0.05)",
        label_n=6, fdr_line=0.05)

ax = axes[1, 0]
for g in wd_meta["group"].unique():
    color, label = {"WD": (ALS_CASE, "WD"), "Control": (CTRL, "Control")}[g]
    sub = wd_scores.loc[wd_meta.loc[wd_meta["group"] == g, "sample"]]
    ax.scatter(sub["PC1"], sub["PC2"], s=30, c=color, alpha=0.85,
               edgecolors="white", linewidths=0.5, label=label, rasterized=True)
ax.set_title("C  Wilson's disease cohort (GSE197406, n=15)", loc="left", pad=6)
ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.3)
ax.set_xlabel(f"PC1 ({wd_ev[0] * 100:.0f}% var)", labelpad=2)
ax.set_ylabel(f"PC2 ({wd_ev[1] * 100:.0f}% var)", labelpad=2)
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1, 1]
volcano(ax, wd_res, "D  Wilson's disease: differential expression (FDR<0.05)",
        label_n=6, fdr_line=0.05)

fig.tight_layout(pad=1.2)
out = os.path.join(FIGDIR, "Figure_2_DE_Analysis.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", out, f"({os.path.getsize(out):,} bytes)")

print("\nALS top DEGs:")
print(als_res.nsmallest(5, "padj_BH")[["log2FC", "p_value", "padj_BH"]].to_string())
print("\nWD top DEGs by |log2FC|:")
print(wd_res[wd_res["padj_BH"] < 0.05].reindex(
    wd_res[wd_res["padj_BH"] < 0.05]["log2FC"].abs().nlargest(8).index)
    [["log2FC", "padj_BH"]].to_string())
