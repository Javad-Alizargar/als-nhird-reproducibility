import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = "str(Path(__file__).resolve().parent.parent)"
ALS_DIR = str(SITE / "data" / "transcriptomics")
WD_DIR = os.path.join(ROOT, "download", "Wilson_GSE197406")
OUT_DIR = str(SITE / "data")
TABLES = str(SITE / "data" / "tables")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

summary = {}


def welch_de(expr, group, case, ctrl):
    m1 = expr.loc[:, group == case].values.astype(float)
    m2 = expr.loc[:, group == ctrl].values.astype(float)
    t, p = stats.ttest_ind(m1, m2, axis=1, equal_var=False, nan_policy="propagate")
    lfc = np.nanmean(m1, axis=1) - np.nanmean(m2, axis=1)
    t = np.where(np.isfinite(t), t, 0.0)
    p = np.where(np.isfinite(p), p, 1.0)
    return lfc, t, p


def ebayes_fitfdist(s2, df):
    from scipy import special, optimize
    s2 = np.maximum(s2, 1e-8)
    z = np.log(s2)
    n = len(s2)
    if df < 2:
        return np.full(n, s2.mean()), 1e6, s2.mean()
    m0 = z.mean() - special.digamma(df / 2.0) + np.log(df / 2.0)
    target = ((z - z.mean()) ** 2).mean() * n / (n - 1.0) - special.polygamma(1, df / 2.0)
    if target <= 0 or not np.isfinite(target):
        d0 = 1e6
    else:
        try:
            d0 = 2.0 * optimize.brentq(
                lambda d: special.polygamma(1, d / 2.0) - target, 0.05, 1e5)
        except ValueError:
            d0 = 1e6
    s02 = np.exp(m0 + special.digamma(d0 / 2.0) - np.log(d0 / 2.0))
    moderated = (d0 * s02 + df * s2) / (d0 + df)
    return moderated, d0, s02


def moderated_de(expr, group, case, ctrl):
    y = expr.values.astype(float)
    d = (group == case).astype(float)
    X = np.column_stack([np.ones(len(d)), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y.T
    resid = y.T - X @ beta
    df = len(d) - 2
    s2 = (resid ** 2).sum(axis=0) / df
    smod, d0, s02 = ebayes_fitfdist(s2, df)
    se = np.sqrt(smod * XtX_inv[1, 1])
    t = beta[1] / np.maximum(se, 1e-12)
    df_total = df + d0
    p = 2.0 * stats.t.sf(np.abs(t), df_total)
    p = np.clip(p, 0.0, 1.0)
    return beta[1], t, p, d0, s02


def tmm_factors(counts):
    counts = counts.values.astype(float)
    lib = counts.sum(axis=0)
    q75 = np.quantile(lib, 0.75)
    ref = int(np.argmin(np.abs(lib - q75)))
    f = np.ones(counts.shape[1])
    for j in range(counts.shape[1]):
        if j == ref:
            continue
        xj = counts[:, j]
        xr = counts[:, ref]
        pos = (xj > 0) & (xr > 0)
        m = np.log2((xj[pos] / lib[j]) / (xr[pos] / lib[ref]))
        a = 0.5 * np.log2((xj[pos] / lib[j]) * (xr[pos] / lib[ref]))
        w = (lib[j] - xj[pos]) / (lib[j] * xj[pos]) + \
            (lib[ref] - xr[pos]) / (lib[ref] * xr[pos])
        keep = np.ones(len(m), dtype=bool)
        if len(m) > 2:
            keep &= np.abs(m) <= np.quantile(np.abs(m), 0.7)
            keep &= a <= np.quantile(a, 0.975)
            keep &= a >= np.quantile(a, 0.025)
        if keep.sum():
            f[j] = 2 ** ((w[keep] * m[keep]).sum() / w[keep].sum())
    f = f * np.exp(-np.mean(np.log(f[f > 0])))
    return f, lib


def normalize_tmm(counts):
    f, lib = tmm_factors(counts)
    cpm = counts.values.astype(float) / (lib * f) * 1e6
    expr = np.log2(cpm + 0.5)
    return pd.DataFrame(expr, index=counts.index, columns=counts.columns), f


def bh(pvals):
    p = np.array(pvals, dtype=float)
    mask = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if mask.any():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return q


print("=" * 60)
print("ALS (GSE346896)")
print("=" * 60)

with gzip.open(os.path.join(ALS_DIR, "GSE346896_discovery_mRNA_raw_counts.csv.gz"),
               "rt") as f:
    counts = pd.read_csv(f, index_col=0)
print("raw counts:", counts.shape)

ann = pd.read_csv(os.path.join(ALS_DIR, "GSE346896_gencode_hg38_gene_annotation.csv.gz"))
ann = ann.rename(columns={ann.columns[0]: "Ensembl_ID"})
pc = set(ann.loc[ann["gene_biotype"] == "protein_coding", "Ensembl_ID"])
sym = dict(zip(ann["Ensembl_ID"], ann["gene_name"]))

counts = counts.loc[[i for i in counts.index if i in pc]]
print("protein-coding genes:", counts.shape[0])

series_txt = gzip.open(
    os.path.join(ALS_DIR, "GSE346896_series_matrix.txt.gz"),
    "rt", encoding="utf-8", errors="replace").read()
status = {}
for line in series_txt.splitlines():
    if line.startswith("!Sample_description\t") and "disease status" in line:
        for part in re.findall(r'"([^"]*)"', line):
            m = re.search(r"Participant (\S+); disease status (\S+);", part)
            if m:
                status[m.group(1)] = m.group(2)
batch = []
for line in series_txt.splitlines():
    if line.startswith("!Sample_characteristics_ch1\t") and "batch:" in line:
        batch += [p.split(": ")[1] for p in re.findall(r'"([^"]*)"', line)]
titles = [p.split(" plasma")[0] for p in
          re.findall(r'"([^"]* plasma EV total RNA-seq[^"]*)"', series_txt)]
assert len(titles) == counts.shape[1] == len(batch)
meta_als = pd.DataFrame({"sample": titles,
                         "group": [status.get(t, "?") for t in titles],
                         "batch": batch})
meta_als.to_csv(os.path.join(OUT_DIR, "ALS_sample_metadata.csv"), index=False)
print("group counts:\n", meta_als["group"].value_counts().to_string())
print("batch x group:\n",
      pd.crosstab(meta_als["group"], meta_als["batch"]).to_string())

cpm = counts / counts.sum(axis=0) * 1e6
keep = (cpm >= 1).sum(axis=1) >= 15
counts = counts.loc[keep]
print("after expression filter:", counts.shape[0])

expr_own, tmm_f = normalize_tmm(counts)
print("TMM factors:", np.round(tmm_f, 3))

tmm_dep = pd.read_csv(
    os.path.join(ALS_DIR, "GSE346896_discovery_mRNA_TMM_log2CPM_protein_coding.csv.gz"),
    index_col=0)
tmm_dep.index.name = "gene_name"
expr = tmm_dep.loc[:, titles]
print("deposited TMM matrix (primary input):", expr.shape)

sym2id = {}
for eid, s in zip(ann["Ensembl_ID"], ann["gene_name"]):
    sym2id.setdefault(s, eid)

common = expr_own.index.intersection(expr.index)
corr = pd.Series({
    c: np.corrcoef(expr_own.loc[common, c], expr.loc[common, c])[0, 1]
    for c in expr.columns})
print("cor(own TMM expr vs deposited) per sample: median",
      round(float(corr.median()), 4))
expr.to_csv(os.path.join(OUT_DIR, "ALS_TMM_log2CPM_expr.csv"))

group = meta_als["group"].values
lfc, t, p, d0, s02 = moderated_de(expr, group, case="ALS", ctrl="Control")
q = bh(p)
res = pd.DataFrame({
    "Ensembl_ID": [sym2id.get(s, "") for s in expr.index],
    "gene_symbol": expr.index,
    "log2FC": lfc,
    "t_moderated": t,
    "p_value": p,
    "padj_BH": q,
    "mean_log2expr_ALS": expr.loc[:, group == "ALS"].mean(axis=1).values,
    "mean_log2expr_Control": expr.loc[:, group == "Control"].mean(axis=1).values,
})
res = res.sort_values("padj_BH").reset_index(drop=True)
res.to_csv(os.path.join(TABLES, "Supp_Table_1_ALS_DEGs.csv"), index=False)
n_sig = int((res["padj_BH"] < 0.05).sum())
n_sig_fc = int(((res["padj_BH"] < 0.05) & (res["log2FC"].abs() > 1)).sum())
n_sig10 = int((res["padj_BH"] < 0.10).sum())
print(f"ALS: tested={len(res)} padj<0.05={n_sig} padj<0.10={n_sig10} "
      f"|log2FC|>1 & padj<0.05={n_sig_fc} (ebayes d0={d0:.1f}, s02={s02:.4f})")
summary["als"] = {
    "dataset": "GSE346896", "assay": "RNA-seq (plasma GLAST+ sEV total RNA)",
    "n_cases": int((group == "ALS").sum()),
    "n_controls": int((group == "Control").sum()),
    "normalization": "deposited edgeR TMM log2-CPM matrix (protein-coding); "
                     "independent TMM reimplementation cross-check (median r=0.97)",
    "test": "per-gene OLS log2FC with limma-style empirical Bayes moderated t (fitFDist); BH FDR",
    "genes_tested": int(len(res)),
    "deg_padj05": n_sig, "deg_padj10": n_sig10, "deg_padj05_lfc1": n_sig_fc,
    "ebayes_d0": float(d0), "ebayes_s02": float(s02),
    "tmm_cor_deposited_median": float(corr.median()),
    "note": "Batch (Study_1 vs Study_2/4/5) is fully confounded with group; "
            "results should be interpreted with this limitation.",
}

print()
json.dump(summary, open(os.path.join(OUT_DIR, "DE_summary.json"), "w"), indent=2, ensure_ascii=False)
print("DE_summary.json written")
