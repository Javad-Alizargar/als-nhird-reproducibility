import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("str(Path(__file__).resolve().parent.parent)")
OUT_PROC = ROOT / "download" / "processed" / "economics"
os.makedirs(OUT_PROC, exist_ok=True)

CONFIG = {
    "ALS": {
        "p_event": 0.55,
        "endpoint": "death or tracheostomy",
        "pack": [("Health-01", 10), ("Health-02", 12), ("Health-08", 3),
                 ("Health-10", 4), ("Health-51", 2)],
        "field_rate": 210,
        "seed": 601,
    },
    "Wilson": {
        "p_event": 0.30,
        "endpoint": "hepatic decompensation",
        "pack": [("Health-01", 9), ("Health-02", 10), ("Health-08", 3),
                 ("Health-10", 4), ("Health-51", 2)],
        "field_rate": 210,
        "seed": 602,
    },
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def cox_power(n, hr, alpha, p_event):
    d = n * p_event
    if d <= 0 or hr <= 1:
        return 0.0
    z = np.abs(np.log(hr)) * np.sqrt(d / 4.0)
    z_alpha = stats.norm.ppf(1 - alpha / 2.0)
    return float(stats.norm.cdf(z - z_alpha))


def pack_cost(years, pack, rate):
    fields = sum(f for _, f in pack)
    return fields * years * rate


def att_sim(seed, p_event, n=1000, tmax=120):
    r = np.random.default_rng(seed)
    h_ev = -np.log(1 - p_event) / 60.0
    h_comp = 0.0015
    h_cens = 0.003
    ev_t = r.exponential(1 / h_ev, n) if h_ev > 0 else np.full(n, np.inf)
    comp_t = r.exponential(1 / h_comp, n)
    cens_t = r.exponential(1 / h_cens, n)
    out_t = np.minimum.reduce([ev_t, comp_t, cens_t, np.full(n, tmax)])
    kind = np.full(n, "at_risk")
    kind[(out_t == ev_t)] = "endpoint"
    kind[(out_t == comp_t)] = "competing_mortality"
    kind[(out_t == cens_t)] = "censored"
    return out_t, kind


def main():
    _subs = os.environ.get("FIG_SUBS", ",".join(CONFIG.keys())).split(",")
    for sub in _subs:
        cfg = CONFIG[sub.strip()]
        print("=" * 60)
        print(sub, "Figure 5 economics")
        print("=" * 60)
        rng = np.random.default_rng(cfg["seed"])

        n_grid = np.linspace(100, 5000, 200)
        hrs = [1.2, 1.5, 2.0, 2.5]
        alphas = [0.05, 0.01]
        colors = ["#1565C0", "#2E7D32", "#EF6C00", "#C62828"]
        power_rows = []
        for hr in hrs:
            for alpha in alphas:
                for n in [200, 500, 1000, 2000, 5000]:
                    power_rows.append({
                        "disease": sub, "n": n, "hr": hr, "alpha": alpha,
                        "p_event": cfg["p_event"],
                        "events_needed": round(n * cfg["p_event"], 1),
                        "power": round(cox_power(n, hr, alpha, cfg["p_event"]), 4),
                        "datasets": "+".join(d for d, _ in cfg["pack"]),
                        "total_fields": sum(f for _, f in cfg["pack"]),
                        "years": 5, "cost_ntd": pack_cost(5, cfg["pack"],
                                                          cfg["field_rate"]),
                    })

        cost_rows = []
        for years in range(1, 11):
            for n in [500, 1000, 2000]:
                cost_rows.append({
                    "disease": sub, "n_cohort": n, "observation_years": years,
                    "datasets": "+".join(d for d, _ in cfg["pack"]),
                    "total_fields": sum(f for _, f in cfg["pack"]),
                    "cost_ntd": pack_cost(years, cfg["pack"],
                                          cfg["field_rate"]),
                    "power_at_hr15": round(
                        cox_power(n, 1.5, 0.05,
                                  min(0.9, cfg["p_event"] * min(years / 5, 2))),
                        4),
                })
        table = pd.DataFrame(power_rows + cost_rows)
        table.to_csv(ROOT / "manuscripts" / sub / "tables" /
                     "Supp_Table_6_Power_and_Cost_Matrix.csv", index=False)
        print(f"saved Supp_Table_6_{sub} ({len(table)} rows)")

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

        ax = axes[0, 0]
        for j, hr in enumerate(hrs):
            p05 = [cox_power(n, hr, 0.05, cfg["p_event"]) for n in n_grid]
            ax.plot(n_grid, p05, lw=1.5, color=colors[j],
                    label=f"HR={hr}, α=0.05")
        p10 = [cox_power(n, 1.5, 0.01, cfg["p_event"]) for n in n_grid]
        p20 = [cox_power(n, 2.0, 0.01, cfg["p_event"]) for n in n_grid]
        ax.plot(n_grid, p10, lw=1.1, ls="--", color=colors[1],
                label="HR=1.5, α=0.01")
        ax.plot(n_grid, p20, lw=1.1, ls="--", color=colors[2],
                label="HR=2.0, α=0.01")
        ax.axhline(0.80, color="#777777", lw=0.6, ls=":")
        ax.text(5050, 0.80, "80% power", fontsize=6, va="center", ha="right",
                color="#555555")
        ax.set_xlabel("Cohort size n")
        ax.set_ylabel("Statistical power")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"A  Power curves ({cfg['endpoint']}; "
                     f"event rate {cfg['p_event']:.0%})", loc="left", pad=6)
        ax.legend(frameon=False, fontsize=5.8, loc="lower right")

        ax = axes[0, 1]
        years = np.arange(1, 11)
        n_fix = 800
        costs = [pack_cost(y, cfg["pack"], cfg["field_rate"]) for y in years]
        powers = [cox_power(n_fix, 1.5, 0.05,
                            min(0.95, cfg["p_event"] * min(y / 5, 2)))
                  for y in years]
        sc = ax.scatter(costs, powers, c=years, cmap="viridis", s=34,
                        edgecolors="white", linewidths=0.5, zorder=3)
        dominated = []
        for i in range(len(years)):
            for j in range(len(years)):
                if i != j and costs[j] <= costs[i] and powers[j] >= powers[i] \
                        and (costs[j] < costs[i] or powers[j] > powers[i]):
                    dominated.append(i)
                    break
        frontier = [i for i in range(len(years)) if i not in dominated]
        ax.plot([costs[i] for i in frontier], [powers[i] for i in frontier],
                "-", color="#C62828", lw=1.3, label="Pareto frontier", zorder=2)
        for i in frontier:
            ax.annotate(f"{years[i]}y", (costs[i], powers[i]),
                        xytext=(6, 6), textcoords="offset points", fontsize=6,
                        color="#C62828")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Observation window (years)", fontsize=6.5)
        ax.set_xlabel("Total NHIRD acquisition cost (NTD)")
        ax.set_ylabel("Power (HR=1.5, n=" + str(n_fix) + ")")
        ax.set_title("B  Cost-power Pareto frontier", loc="left", pad=6)
        ax.legend(frameon=False, fontsize=6)

        ax = axes[1, 0]
        out_t, kind = att_sim(rng, cfg["p_event"])
        from collections import Counter
        cnt = Counter(kind)
        labels = {"endpoint": f"{cfg['endpoint']} (n={cnt['endpoint']})",
                  "competing_mortality": "competing mortality "
                  f"(n={cnt['competing_mortality']})",
                  "censored": f"loss to follow-up (n={cnt['censored']})",
                  "at_risk": f"administrative censor (n={cnt['at_risk']})"}
        t_grid = np.arange(0, 121)
        curves = {}
        for k in ["endpoint", "competing_mortality", "censored"]:
            surv = np.array([float(np.sum(out_t > t)) / len(out_t)
                             for t in t_grid])
            curves[k] = 1 - surv
        for k, color in zip(["endpoint", "competing_mortality", "censored"],
                            ["#C62828", "#1565C0", "#607D8B"]):
            ax.fill_between(t_grid, 0, curves[k], color=color, alpha=0.55,
                            label=labels[k], lw=0)
        ax.set_xlabel("Months of follow-up")
        ax.set_ylabel("Cumulative proportion")
        ax.set_ylim(0, 1.0)
        ax.set_title("C  Simulated first-event attrition (n=1,000)",
                     loc="left", pad=6)
        ax.legend(frameon=False, fontsize=5.8, loc="upper left")

        ax = axes[1, 1]
        delay = np.linspace(0, 24, 100)
        lag = np.linspace(0, 12, 100)
        DD, LL = np.meshgrid(delay, lag)
        acc = 1 / (1 + (DD + LL) / 10.0) ** 1.5
        cs = ax.contourf(DD, LL, acc, levels=12, cmap="RdYlGn", vmin=0.3,
                         vmax=1.0)
        cs2 = ax.contour(DD, LL, acc, levels=[0.4, 0.5, 0.7, 0.8, 0.9],
                         colors="black", linewidths=0.5, linestyles="--")
        ax.clabel(cs2, fmt="%.0f%%", fontsize=5.5)
        cb = fig.colorbar(cs, ax=ax, pad=0.02)
        cb.set_label("Cohort identification accuracy", fontsize=6.5)
        ax.set_xlabel("Diagnostic delay (months)")
        ax.set_ylabel("Regional referral lag (months)")
        ax.set_title("D  Delay/lag ascertainment sensitivity "
                     "(illustrative formula)", loc="left", pad=6)

        fig.tight_layout(pad=1.3)
        out = ROOT / "manuscripts" / sub / "figures" / \
            "Figure_5_NHIRD_Economics.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("saved", out, f"({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    main()
