import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("str(Path(__file__).resolve().parent.parent)")
PKG = ROOT / "codebase_pkg"
KNOW = ROOT / "download" / "nhri_knowledge"
PROC = ROOT / "download" / "processed" / "proposal_validation"
os.makedirs(PROC, exist_ok=True)
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "nhri_app"))
os.environ["NHRI_SOURCE_DIR"] = str(KNOW)

from openai import OpenAI
from nhri_app.catalog import parse_catalog
from nhri_app.documents import load_knowledge_documents
from nhri_app.retrieval import Retriever
from nhri_app import planner as pl
from nhri_app.llm import PLAN_PROMPT, generate_plan_with_deepseek

key = open(ROOT / "apis" / "deepseek.txt").read().strip().split()[0]
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
MODEL = "deepseek-flash"

datasets = parse_catalog(KNOW / "nhri_dataset_categories_expanded.txt")
chunks = load_knowledge_documents(KNOW)
ret = Retriever(chunks)
valid_codes = set(d.code for d in datasets)

TOPICS = {
    "ALS": {
        "title": "ALS disease progression, functional decline and survival in the NHRID",
        "brief": ("Study cohort of patients with amyotrophic lateral sclerosis (ICD-10 G12.21). "
                  "Objectives: (1) estimate incidence and prevalence; (2) model ALSFRS-R-equivalent "
                  "functional decline using healthcare utilization trajectories; (3) estimate overall "
                  "survival and time to invasive ventilation. Linkage plan: Health-01 (outpatient), "
                  "Health-02 (inpatient), Health-08 (catastrophic illness), Health-10 (cause of death). "
                  "Covariates: age at onset, sex, riluzole use, NIV/tracheostomy. "
                  "Statistical plan: Kaplan-Meier, Cox regression, mixed models for longitudinal visits."),
        "icd": "G12.21",
    },
    "Wilson": {
        "title": "Wilson disease hepatic outcomes and surveillance economics in the NHRID",
        "brief": ("Study cohort of patients with Wilson disease (ICD-10 E83.01). "
                  "Objectives: (1) characterize hepatic decompensation trajectory; (2) compare outcomes "
                  "by chelation strategy (penicillamine vs trientine vs zinc); (3) model long-term "
                  "surveillance costs. Linkage plan: Health-01, Health-02, Health-08, Health-51 "
                  "(rare disease registry), Health-10. Covariates: age at diagnosis, initial presentation "
                  "(hepatic vs neurologic), adherence proxy (medication refill). "
                  "Statistical plan: competing-risk models, cost-effectiveness, landmark analysis."),
        "icd": "E83.01",
    },
}

UNCONSTRAINED_SYS = (
    "You are a senior epidemiology researcher writing NHRI data-application proposals."
)


def skeleton_for(disease):
    t = TOPICS[disease]
    phases = pl.build_project_plan(t["title"], t["brief"], datasets,
                                   cost_strategy="Lean / gated")
    skeleton = pl.plan_to_markdown(phases)
    hits = ret.search(t["title"] + " " + t["brief"], k=8)
    context = "\n\n".join(f"[{h.chunk.source}]\n{h.chunk.text}" for h in hits)
    return skeleton, context


def gen_unconstrained(disease):
    t = TOPICS[disease]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": UNCONSTRAINED_SYS},
            {"role": "user",
             "content": (f"Write an NHRI (Taiwan National Health Insurance Research Database) "
                         f"data-application research proposal on: {t['title']}.\n"
                         f"Research brief:\n{t['brief']}\n"
                         "Include: background, cohort definition, exposure/outcome, and "
                         "statistical plan.")},
        ],
        temperature=0.7, max_tokens=12000)
    return resp.choices[0].message.content or "", resp.usage


def gen_skeleton_first(disease, skeleton, context):
    t = TOPICS[disease]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": (
                f"Output language: English. Do not mix languages unless an official "
                f"dataset field or rule is quoted.\n\n"
                f"Project title:\n{t['title']}\n\n"
                f"Project description:\n{t['brief']}\n\n"
                f"Deterministic NHRI-ordered skeleton:\n{skeleton}\n\n"
                f"Retrieved NHRI rules/context:\n{context}")},
        ],
        temperature=0.15, max_tokens=12000)
    return resp.choices[0].message.content or "", resp.usage


def check(present):
    return 1 if present else 0


CODE_RE = re.compile(r"(?:Health|Society|Welfare|Cancer)-\d{1,3}(?:-\d+)?")
BUDGET_RE = re.compile(r"(NT\$[\d,]+|NTD\s*[\d,]+|[\d][\d,]*\s*(?:元|NTD)|budget|cost)", re.I)
SS_RE = re.compile(r"(sample size|n\s*=\s*\d|N\s*=\s*\d|樣本數|人數|样本)", re.I)
VAR_RE = re.compile(r"(Health-\d+|欄位|variables|fields)", re.I)
PHASE_RE = re.compile(r"(Phase\s*\d|階段)", re.I)
BG_RE = re.compile(r"(background|introduction|研究背景|簡介)", re.I)
COH_RE = re.compile(r"(cohort|世代|納入|收案|inclusion)", re.I)
EO_RE = re.compile(r"(exposure|outcome|暴露|結果)", re.I)
STAT_RE = re.compile(r"(statistical|統計|cox|kaplan|regression|迴歸|sample size)", re.I)


def metrics(disease, text, usage):
    m = {}
    m["disease"] = disease
    mentions = CODE_RE.findall(text)
    invalid = [c for c in mentions if c not in valid_codes]
    m["dataset_mentions"] = len(mentions)
    m["invalid_dataset_mentions"] = len(invalid)
    m["hallucination_rate"] = round(len(invalid) / len(mentions), 4) if mentions else 0.0
    m["budget_compliance"] = check(bool(BUDGET_RE.search(text)))
    m["sample_size_justified"] = check(bool(SS_RE.search(text)))
    m["variable_inclusion"] = check(bool(VAR_RE.search(text)))
    m["phase_structure"] = check(bool(PHASE_RE.search(text)))
    m["has_background"] = check(bool(BG_RE.search(text)))
    m["has_cohort_definition"] = check(bool(COH_RE.search(text)))
    m["has_exposure_outcome"] = check(bool(EO_RE.search(text)))
    m["has_statistical_plan"] = check(bool(STAT_RE.search(text)))
    m["structural_score"] = round(sum(
        m[k] for k in ["has_background", "has_cohort_definition",
                       "has_exposure_outcome", "has_statistical_plan"]) / 4.0, 2)
    if usage:
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if pt is None:
            pt = usage.get("prompt_tokens")
        if ct is None:
            ct = usage.get("completion_tokens")
        m["prompt_tokens"] = pt
        m["completion_tokens"] = ct
    else:
        m["prompt_tokens"] = None
        m["completion_tokens"] = None
    m["completion_chars"] = len(text)
    m["content_tokens_est"] = int(round(len(text) / 4.0))
    return m


def run_one(args):
    disease, paradigm, run_id, skeleton, context = args
    try:
        for attempt in range(3):
            if paradigm == "skeleton_first":
                text, usage = gen_skeleton_first(disease, skeleton, context)
            else:
                text, usage = gen_unconstrained(disease)
            if text.strip():
                break
        m = metrics(disease, text, usage)
        m["paradigm"] = paradigm
        m["run_id"] = run_id
        m["text"] = text
        return m
    except Exception as e:
        return {"disease": disease, "paradigm": paradigm, "run_id": run_id,
                "error": str(e), "text": ""}


N_RUNS = 30
all_rows = []
for disease in ["ALS", "Wilson"]:
    print("=" * 60)
    print(f"{disease}: building skeleton + context")
    skeleton, context = skeleton_for(disease)
    print(f"  skeleton {len(skeleton):,} chars, context {len(context):,} chars")
    tasks = []
    for r in range(N_RUNS):
        tasks.append((disease, "skeleton_first", r, skeleton, context))
        tasks.append((disease, "unconstrained", r, skeleton, context))
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(run_one, tasks):
            results.append(res)
    rows = []
    for r in results:
        if "error" in r and not r.get("text"):
            print("  error:", r["error"][:100])
            continue
        rows.append({k: v for k, v in r.items() if k != "text"})
    df = pd.DataFrame(rows)
    df = df.sort_values(["paradigm", "run_id"]).reset_index(drop=True)
    df.to_csv(ROOT / "manuscripts" / disease / "tables" /
              "Supp_Table_5_Generated_Proposal_Comparison.csv", index=False)
    with open(PROC / f"{disease}_generations.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)
    all_rows.append(df)
    s1 = df[df["paradigm"] == "skeleton_first"]
    u1 = df[df["paradigm"] == "unconstrained"]
    print(f"  skeleton_first n={len(s1)}: completion_tokens "
          f"med={s1['completion_tokens'].median():.0f}, "
          f"budget={s1['budget_compliance'].mean():.2f}, "
          f"struct={s1['structural_score'].mean():.2f}, "
          f"hall={s1['hallucination_rate'].mean():.3f}")
    print(f"  unconstrained n={len(u1)}: completion_tokens "
          f"med={u1['completion_tokens'].median():.0f}, "
          f"budget={u1['budget_compliance'].mean():.2f}, "
          f"struct={u1['structural_score'].mean():.2f}, "
          f"hall={u1['hallucination_rate'].mean():.3f}")

print("=" * 60)
print("FINAL FULL PROPOSALS (skeleton-first, long-form)")
print("=" * 60)
for disease in ["ALS", "Wilson"]:
    t = TOPICS[disease]
    skeleton, context = skeleton_for(disease)
    resp = None
    final_text = ""
    for attempt in range(4):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PLAN_PROMPT},
                {"role": "user", "content": (
                    f"Output language: English. Do not mix languages unless an official "
                    f"dataset field or rule is quoted.\n\n"
                    f"Project title:\n{t['title']}\n\n"
                    f"Project description:\n{t['brief']}\n\n"
                    f"Deterministic NHRI-ordered skeleton:\n{skeleton[:28000]}\n\n"
                    f"Retrieved NHRI rules/context:\n{context[:18000]}")},
            ],
            temperature=0.2, max_tokens=16000)
        final_text = resp.choices[0].message.content or ""
        if final_text.strip():
            break
    md = (f"# Final Generated Research Proposal — {disease}\n\n"
          f"## Project title\n{t['title']}\n\n"
          f"## ICD-10 focus\n{t['icd']}\n\n"
          f"## Deterministic platform skeleton (dataset/variable truth)\n\n"
          f"{skeleton[:12000]}\n\n"
          f"## AI-generated application narrative\n\n{final_text}\n")
    out = ROOT / "manuscripts" / disease / "drafts" / "final_generated_proposal.md"
    out.write_text(md, encoding="utf-8")
    print(f"saved {out} ({len(md):,} chars)")

    m = metrics(disease, final_text, dict(resp.usage))
    json.dump(m, open(PROC / f"{disease}_final_metrics.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"  final: {resp.usage.prompt_tokens} in / "
          f"{resp.usage.completion_tokens} out, "
          f"invalid refs={m['invalid_dataset_mentions']}/{m['dataset_mentions']}")

print("=" * 60)
print("FIGURE 4")
print("=" * 60)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})
SF = "#1565C0"
UC = "#C62828"
for disease, df in zip(["ALS", "Wilson"], all_rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))

    ax = axes[0, 0]
    width = 0.35
    x = np.arange(2)
    for label, sub in [("Skeleton-first", df[df["paradigm"] == "skeleton_first"]),
                       ("Unconstrained", df[df["paradigm"] == "unconstrained"])]:
        ax.bar(x[0] if label == "Skeleton-first" else x[1],
               sub["content_tokens_est"].median(), width,
               color=SF if label == "Skeleton-first" else UC, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Skeleton-first", "Unconstrained"])
    ax.set_ylabel("Median content tokens")
    ax.set_title("A  Token efficiency of generation", loc="left", pad=6)
    s1 = df[df["paradigm"] == "skeleton_first"]["content_tokens_est"]
    u1 = df[df["paradigm"] == "unconstrained"]["content_tokens_est"]
    red = (1 - s1.median() / u1.median()) * 100
    ax.annotate(f"{abs(red):.0f}% {'fewer' if red > 0 else 'more'} content tokens "
                f"vs unconstrained",
                xy=(0.5, 1.06), xycoords="axes fraction", fontsize=6.3,
                ha="center", color="#444444")

    ax = axes[0, 1]
    checks = ["budget_compliance", "sample_size_justified",
              "variable_inclusion", "phase_structure"]
    labels = ["Budget", "Sample\nsize", "Variable\ninclusion", "Phase\nstructure"]
    x = np.arange(len(checks))
    for i, paradigm in enumerate(["skeleton_first", "unconstrained"]):
        sub = df[df["paradigm"] == paradigm]
        vals = [sub[c].mean() * 100 for c in checks]
        off = (i - 0.5) * 0.36
        ax.bar(x + off, vals, 0.34, label="Skeleton-first" if i == 0 else "Unconstrained",
               color=SF if i == 0 else UC, alpha=0.9)
        for j, v in enumerate(vals):
            ax.text(j + off, v + 2, f"{v:.0f}%", ha="center", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Adherence rate (%)")
    ax.set_title("B  Constraint adherence (30 runs each)", loc="left", pad=6)
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1, 0]
    data = [df[df["paradigm"] == "skeleton_first"]["structural_score"].values,
            df[df["paradigm"] == "unconstrained"]["structural_score"].values]
    bp = ax.boxplot(data, tick_labels=["Skeleton-first", "Unconstrained"],
                    widths=0.45, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [SF, UC]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for i, d in enumerate(data):
        xj = np.random.default_rng(i).normal(i + 1, 0.05, len(d))
        ax.scatter(xj, d, s=8, alpha=0.5, color="white",
                   edgecolors="none")
        ax.scatter(xj, d, s=10, alpha=0.35,
                   color=SF if i == 0 else UC, edgecolors="none")
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel("Structural completeness (0-1)")
    ax.set_title("C  Structural completeness score", loc="left", pad=6)

    ax = axes[1, 0].get_position()
    ax = axes[1, 1]
    for paradigm, color, lab in [("skeleton_first", SF, "Skeleton-first"),
                                 ("unconstrained", UC, "Unconstrained")]:
        sub = df[df["paradigm"] == paradigm].sort_values("run_id")
        ax.plot(sub["run_id"], sub["hallucination_rate"] * 100,
                "o-", ms=3, lw=1.1, color=color, alpha=0.85, label=lab)
    ax.set_xlabel("Generation run")
    ax.set_ylabel("Invalid dataset-reference rate (%)")
    ax.set_title("D  Hallucination rate across 30 runs", loc="left", pad=6)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout(pad=1.3)
    out = ROOT / "manuscripts" / disease / "figures" / "Figure_4_Proposal_Optimization.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out, f"({os.path.getsize(out):,} bytes)")
print("DONE")
