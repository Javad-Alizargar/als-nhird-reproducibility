#!/usr/bin/env python3
"""verify_package.py - offline verification of the ALS reproducibility package.

Recomputes the manuscript's key reported quantities DIRECTLY from the archived
records (no private workspace, no API keys, no network). Exit 0 = all checks
pass. Print-tolerant: each check prints PASS/FAIL with the recomputed value
versus the recorded/manuscript value.

What this verifies (and what it does not):
  * verifies recorded results against the archived records (recomputation);
  * verifies that saved model weights reproduce the recorded PINN validation;
  * does NOT reproduce historical latency measurements or stochastic online
    LLM responses (those are archived observations, documented as such).
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SITE = Path(__file__).resolve().parent.parent
DATA = SITE / "data"
FAILS = []


def check(name, got, expected, tol=1e-9, note=""):
    ok = abs(got - expected) <= tol
    flag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"[{flag}] {name}: recomputed={got} expected={expected}{note}")


def load_gz_csv(p):
    return pd.read_csv(p, compression="gzip", index_col=0) \
        if str(p).endswith(".gz") else pd.read_csv(p)


def main():
    print("== 1. Digital twin ==")
    d = np.load(DATA / "digital_twin" / "als_sim.npz")
    ev = d["event"]
    check("twin count", len(ev), 1000)
    check("simulated events", int(ev.sum()), 391)
    check("event rate", float(ev.mean()), 0.391, tol=1e-6)
    check("E proxy mean", float(d["E"].mean()), 1.163, tol=1e-3)
    km = pd.read_csv(DATA / "digital_twin" / "als_km.csv")
    tiers = dict(km["risk_tier"].value_counts())
    check("high-risk twins", int(tiers.get("high_risk", 0)), 486)
    check("low-risk twins", int(tiers.get("low_risk", 0)), 514)

    print("== 2. PINN saved weights reproduce recorded validation ==")
    try:
        import torch
        import torch.nn as nn
        from scipy.integrate import solve_ivp
    except Exception as e:
        print(f"[SKIP] PINN check (torch unavailable: {e})")
        torch = None
    if torch is not None:
        state = torch.load(DATA / "digital_twin" / "als_pinn_state.pt",
                           map_location="cpu", weights_only=True)

        class PINN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(14, 128), nn.Tanh(),
                    nn.Linear(128, 128), nn.Tanh(),
                    nn.Linear(128, 128), nn.Tanh(),
                    nn.Linear(128, 64), nn.Tanh(),
                    nn.Linear(64, 3))

            def forward(self, x):
                return self.net(x)

        def featurize(t, E, a0):
            t = t.unsqueeze(1)
            feats = [torch.cos(k * np.pi * t / 60.0) for k in range(1, 7)]
            feats += [torch.sin(k * np.pi * t / 60.0) for k in range(1, 7)]
            return torch.cat(feats + [(E / 3.0).unsqueeze(1),
                                      ((a0 - 30.0) / 50.0).unsqueeze(1)], 1)

        model = PINN(); model.load_state_dict(state); model.eval()
        k1, k2, k3 = 0.012, 0.02, 0.015
        k4, k5, k6 = 0.01, 40.0, 0.05
        k7, k8, k9 = 0.08, 0.1, 0.5

        def ode(t, y):
            M, F, N = y
            age = max(60.0 - 55.0, 0.0)
            dM = -(k1 * (1.0 + k4 * age) + k2 * N + k3 * 1.0) * M
            dF = -k5 * (-dM) + k6 * (48.0 - F) * M
            dN = k7 * (1.0 + k9 * 1.0) * (1.0 - M) - k8 * N
            return [dM, dF, dN]

        sol = solve_ivp(ode, (0, 60), [1.0, 48.0, 0.0],
                        t_eval=np.arange(0, 61), method="DOP853",
                        rtol=1e-8, atol=1e-10)
        t_v = torch.tensor(np.arange(0, 61), dtype=torch.float32)
        E_v = torch.full((61,), 1.0); a0_v = torch.full((61,), 60.0)
        t_v.requires_grad_(True)
        outv = model(featurize(t_v, E_v, a0_v))
        M_v, F_v, N_v = outv[:, 0], outv[:, 1] * 48.0, outv[:, 2] * 2.0
        dM = torch.autograd.grad(M_v.sum(), t_v, retain_graph=True)[0]
        dF = torch.autograd.grad(F_v.sum(), t_v, retain_graph=True)[0]
        dN = torch.autograd.grad(N_v.sum(), t_v)[0]
        age = torch.clamp(a0_v - 55.0, min=0.0)
        r1 = (dM + (k1 * (1.0 + k4 * age) + k2 * N_v + k3 * E_v) * M_v)
        r2 = dF + k5 * (-dM) - k6 * (48.0 - F_v) * M_v
        r3 = dN - k7 * (1.0 + k9 * E_v) * (1.0 - M_v) + k8 * N_v
        rec = json.load(open(DATA / "digital_twin" / "als_pinn_validation.json"))
        check("residual r1", float(r1.abs().mean()), rec["per_equation_residual_mean_abs"]["r1_dMdt"], tol=1e-4)
        check("residual r2", float(r2.abs().mean()), rec["per_equation_residual_mean_abs"]["r2_dFdt"], tol=1e-4)
        check("residual r3", float(r3.abs().mean()), rec["per_equation_residual_mean_abs"]["r3_dNdt"], tol=1e-4)
        check("DOP853 max |dF|", float(np.max(np.abs(F_v.detach().numpy() - sol.y[1]))),
              rec["vs_high_accuracy_solver_max_abs_error"]["F"], tol=1e-4)
        print("  forward deviation pct:", round(
            float(np.max(np.abs(F_v.detach().numpy() - sol.y[1]))) / 48 * 100, 3))

    print("== 3. Transcriptomics (descriptive, batch-confounded) ==")
    meta = pd.read_csv(DATA / "ALS_sample_metadata.csv")
    check("ALS samples", int((meta["group"] == "ALS").sum()), 45)
    check("control samples", int((meta["group"] == "Control").sum()), 15)
    tab = pd.crosstab(meta["group"], meta["batch"])
    check("Study_1 ALS", int(tab.loc["ALS", "Study_1"]), 45)
    check("Study_2 controls", int(tab.loc["Control", "Study_2"]), 9)
    check("Study_4 controls", int(tab.loc["Control", "Study_4"]), 1)
    check("Study_5 controls", int(tab.loc["Control", "Study_5"]), 5)
    X = np.column_stack([np.ones(60), (meta["group"] == "ALS").astype(float),
                         (meta["batch"] == "Study_2").astype(float),
                         (meta["batch"] == "Study_4").astype(float),
                         (meta["batch"] == "Study_5").astype(float)])
    check("design rank (contrast not identifiable)", int(np.linalg.matrix_rank(X)), 4)
    deg = pd.read_csv(DATA / "tables" / "Supp_Table_2_ALS_DEGs.csv")
    check("genes tested", len(deg), 19885)
    check("FDR<0.05 genes", int((deg["padj_BH"] < 0.05).sum()), 3)
    check("FDR<0.10 genes", int((deg["padj_BH"] < 0.10).sum()), 16)
    for g, lfc in [("SRP72", -1.9995), ("NPY2R", -1.7349), ("WDR36", -2.2822)]:
        check(f"{g} log2FC", float(deg[deg["gene_symbol"] == g]["log2FC"].iloc[0]), lfc, tol=1e-3)

    print("== 4. Benchmarks (archived measurements) ==")
    b = pd.read_csv(DATA / "tables" / "Supp_Table_1_System_Benchmarks.csv")
    lat = b[(b.metric == "latency_ms") &
            b.detail.str.match(r"^(tfidf|faiss), query_len=\d+$")].copy()
    lat["mode"] = lat.detail.str.extract(r"^([a-z]+)")[0]
    check("TF-IDF mean (ms)", float(lat[lat["mode"] == "tfidf"].value.mean()),
          2.28, tol=0.05)
    check("FAISS mean (ms)", float(lat[lat["mode"] == "faiss"].value.mean()),
          7.00, tol=0.05)
    q = b[(b.benchmark == "concurrency") &
          (b.metric.isin(["p50_ms", "p95_ms", "throughput", "peak_memory"]))]
    check("p50 min (ms)", float(q[q.metric == "p50_ms"].value.min()), 510, tol=1)
    check("p50 max (ms)", float(q[q.metric == "p50_ms"].value.max()), 1012, tol=2)
    check("p95 max (ms)", float(q[q.metric == "p95_ms"].value.max()), 1471, tol=1)
    check("peak memory (MB)", float(q[q.metric == "peak_memory"].value.max()), 367, tol=1)
    check("benchmark rows", len(b), 324)

    print("== 5. Proposal generation (archived runs) ==")
    t5 = pd.read_csv(DATA / "tables" / "Supp_Table_5_Generated_Proposal_Comparison.csv")
    for par in ["skeleton_first", "unconstrained"]:
        s = t5[t5.paradigm == par]
        check(f"{par} runs", len(s), 30)
        check(f"{par} invalid refs (all runs)", int(s.invalid_dataset_mentions.sum()), 0)
    check("skeleton-first budget presence", float(t5[t5.paradigm == "skeleton_first"].budget_compliance.mean()), 0.767, tol=0.001)
    check("unconstrained budget presence", float(t5[t5.paradigm == "unconstrained"].budget_compliance.mean()), 0.100, tol=0.001)

    print("== 6. Simulated human-factors profiles ==")
    sus = pd.read_csv(DATA / "platform_eval" / "sus_scores.csv")
    check("usability platform", float(sus[sus.group == "Platform"].sus.mean()), 82.3, tol=0.3)
    check("usability manual", float(sus[sus.group == "Manual"].sus.mean()), 54.5, tol=0.3)
    tlx = pd.read_csv(DATA / "platform_eval" / "nasa_tlx.csv")
    check("workload manual", float(tlx[tlx.group == "Manual"].composite_tlx.mean()), 68.0, tol=0.5)
    check("workload platform", float(tlx[tlx.group == "Platform"].composite_tlx.mean()), 28.3, tol=0.5)
    tele = pd.read_csv(DATA / "platform_eval" / "platform_telemetry_stats.csv", index_col="metric")
    check("LLM short-probe mean (ms)", float(tele.loc["llm_latency_mean_ms", "value"]), 663.9, tol=1)
    check("LLM short-probe n", float(tele.loc["llm_latency_n", "value"]), 3)

    print("== 7. Power and fees (deterministic) ==")
    def cox_power(n, hr, alpha, p_event):
        d = n * p_event
        z = np.abs(np.log(hr)) * np.sqrt(d / 4.0)
        return float(stats.norm.cdf(z - stats.norm.ppf(1 - alpha / 2)))
    check("power n=500 HR=1.5 a=0.05 (55% events)", cox_power(500, 1.5, 0.05, 0.55), 0.9195, tol=1e-4)
    check("events needed n=500", 500 * 0.55, 275.0)
    check("cost 31 fields x 5 y x 210", 31 * 5 * 210, 32550)
    t6 = pd.read_csv(DATA / "tables" / "Supp_Table_6_Power_and_Cost_Matrix.csv")
    check("table rows", len(t6), 70)
    sys.path.insert(0, str(SITE / "code" / "platform"))
    from nhri_app import rules
    ds = type("DS", (), {})()
    ds.code = "Health-01"; ds.total_fields = 40
    fld = type("F", (), {"number": 1})()
    check("fee at 11 files", int(rules.dataset_processing_fee(ds, [fld], 1, 11)), 210)
    check("fee at 12 files", int(rules.dataset_processing_fee(ds, [fld], 1, 12)), 240)

    print()
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)} failed: {FAILS})")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
