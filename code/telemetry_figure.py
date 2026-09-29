import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("str(Path(__file__).resolve().parent.parent)")
OUT_PROC = SITE / "data" / "platform_eval"
os.makedirs(OUT_PROC, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
})

rng = np.random.default_rng(20260924)


def live_llm_latency(n=3):
    from openai import OpenAI
    key = open(Path(os.environ.get("DEEPSEEK_KEY_FILE", "/dev/null")), encoding="utf-8").read().strip().split()[0]
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    lats = []
    for i in range(n):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "pong?"}],
            max_tokens=8, temperature=0)
        lats.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.3)
    return np.array(lats)


def levenshtein(a, b):
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1):
        dp[i, 0] = i
    for j in range(len(b) + 1):
        dp[0, j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(dp[len(a), len(b)])


SECTIONS = ["Title", "Background", "Cohort definition", "Statistical plan",
            "Budget", "Checklist"]
BASE_LEN = {"Title": 90, "Background": 700, "Cohort definition": 520,
            "Statistical plan": 420, "Budget": 260, "Checklist": 380}
EDIT_PROB = {"Novice": 0.30, "Senior": 0.10}
N_NOV = 18
N_SEN = 12

CHAR_POOL = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:-()[]"


def rand_text(n, r):
    return "".join(r.choice(list(CHAR_POOL), size=n))


def sim_edits(r, base_len, p_edit):
    draft = rand_text(base_len, r)
    final = "".join(
        c if r.random() > p_edit else r.choice(list(CHAR_POOL))
        for c in draft)
    d = levenshtein(draft, final)
    return 1 - d / max(len(draft), len(final), 1)


def main():
    _stats_file = OUT_PROC / "platform_telemetry_stats.csv"
    if _stats_file.exists() and os.environ.get("FIG6_LIVE", "0") != "1":
        _stats = pd.read_csv(_stats_file, index_col="metric")
        llm_lat = np.array([float(_stats.loc["llm_latency_mean_ms", "value"])])
        llm_n = int(float(_stats.loc["llm_latency_n", "value"]))
        print(f"using cached LLM latency: {llm_lat[0]:.0f} ms "
              f"(n={llm_n}; FIG6_LIVE=1 to re-measure)")
    else:
        print("Measuring live DeepSeek endpoint latency (3 calls)...")
        llm_lat = live_llm_latency()
        llm_n = len(llm_lat)
        print(f"  live LLM latency: mean {llm_lat.mean():.0f} ms "
              f"(n={len(llm_lat)})")

    stages = [
        ("Auth (SSO)", 52, 6),
        ("Deterministic skeleton", 15.4, 1.2),
        ("Hybrid retrieval", 7.1, 0.5),
        ("LLM proposal generation", llm_lat.mean(), llm_lat.std()),
        ("SQLite persist", 4.2, 0.4),
        ("Email notification", 82, 15),
    ]

    edit_rows = []
    r_ed = np.random.default_rng(20260924)
    for group, n_ in [("Novice", N_NOV), ("Senior", N_SEN)]:
        for sec in SECTIONS:
            for i in range(n_):
                ratio = sim_edits(r_ed, BASE_LEN[sec], EDIT_PROB[group])
                edit_rows.append({"group": group, "section": sec,
                                  "similarity_ratio": round(ratio, 4)})
    edit_df = pd.DataFrame(edit_rows)

    mw_results = {}
    for sec in SECTIONS:
        a = edit_df[(edit_df["group"] == "Novice") &
                    (edit_df["section"] == sec)]["similarity_ratio"]
        b = edit_df[(edit_df["group"] == "Senior") &
                    (edit_df["section"] == sec)]["similarity_ratio"]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        mw_results[sec] = {"U": float(u), "p": float(p)}

    sus_rows = []
    r_sus = np.random.default_rng(7)
    for g, mu, sd in [("Manual", 56, 10), ("Platform", 83, 7)]:
        for dom in range(1, 11):
            vals = np.clip(r_sus.normal(mu, sd, 24), 0, 100)
            for v in vals:
                sus_rows.append({"group": g, "domain": dom, "sus": round(v, 1)})
    sus_df = pd.DataFrame(sus_rows)
    sus_p = stats.mannwhitneyu(
        sus_df[sus_df["group"] == "Manual"]["sus"],
        sus_df[sus_df["group"] == "Platform"]["sus"]).pvalue

    tlx_rows = []
    r_tlx = np.random.default_rng(9)
    for g, mu, sd in [("Manual", 66, 12), ("Platform", 30, 9)]:
        vals = np.clip(r_tlx.normal(mu, sd, 24), 0, 100)
        for v in vals:
            tlx_rows.append({"group": g, "composite_tlx": round(v, 1)})
    tlx_df = pd.DataFrame(tlx_rows)
    tlx_p = stats.mannwhitneyu(
        tlx_df[tlx_df["group"] == "Manual"]["composite_tlx"],
        tlx_df[tlx_df["group"] == "Platform"]["composite_tlx"]).pvalue

    time_rows = []
    r_t = np.random.default_rng(11)
    for g, mu, sd in [("Manual", 210, 55), ("Platform", 27, 9)]:
        vals = np.clip(r_t.normal(mu, sd, 24), 5, None)
        for v in vals:
            time_rows.append({"group": g, "time_min": round(v, 1)})
    time_df = pd.DataFrame(time_rows)
    time_p = stats.mannwhitneyu(
        time_df[time_df["group"] == "Manual"]["time_min"],
        time_df[time_df["group"] == "Platform"]["time_min"]).pvalue

    for sub in os.environ.get("FIG_SUBS", "ALS,Wilson").split(","):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

        ax = axes[0, 0]
        names = [s[0] for s in stages]
        means = [s[1] for s in stages]
        errs = [s[2] for s in stages]
        colors = ["#90A4AE", "#1565C0", "#2E7D32", "#C62828",
                  "#1565C0", "#607D8B"]
        y = np.arange(len(names))[::-1]
        ax.barh(y, means, xerr=errs, color=colors, alpha=0.9, height=0.55,
                capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.8)
        ax.set_xlabel("Latency (ms)")
        ax.set_xscale("log")
        ax.set_title("A  Session-latency swimlane", loc="left", pad=6)
        ax.annotate("LLM stage mean measured live on api.deepseek.com "
                    "(short probe, n=3);\nother stages: nominal constants",
                    xy=(0.5, 0.02), xycoords="axes fraction", fontsize=6,
                    color="#555555")

        ax = axes[0, 1]
        xpos = np.arange(len(SECTIONS))
        for i, group in enumerate(["Novice", "Senior"]):
            subd = edit_df[edit_df["group"] == group]
            meds = [subd[subd["section"] == s]["similarity_ratio"].median()
                    for s in SECTIONS]
            q1 = [subd[subd["section"] == s]["similarity_ratio"].quantile(0.25)
                  for s in SECTIONS]
            q3 = [subd[subd["section"] == s]["similarity_ratio"].quantile(0.75)
                  for s in SECTIONS]
            off = (i - 0.5) * 0.36
            color = "#EF6C00" if group == "Novice" else "#1565C0"
            ax.bar(xpos + off, meds, 0.34, color=color, alpha=0.85,
                   label=group)
            ax.vlines(xpos + off, q1, q3, color="#333333", lw=0.8)
        for j, sec in enumerate(SECTIONS):
            p = mw_results[sec]["p"]
            ax.text(j, 1.0, "*" if p < 0.05 else "n.s.", ha="center",
                    fontsize=6.5, color="#C62828" if p < 0.05 else "#777777")
        ax.set_xticks(xpos)
        ax.set_xticklabels(SECTIONS, rotation=90, ha="center", fontsize=6)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("Draft-final similarity (1 − edit distance)")
        ax.set_title("B  Simulated revision load by section", loc="left", pad=6)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")

        ax = axes[1, 0]
        dom_pos = []
        for i, g in enumerate(["Manual", "Platform"]):
            subd = sus_df[sus_df["group"] == g]
            vals = [subd[subd["domain"] == d]["sus"].values for d in range(1, 11)]
            pos = np.arange(1, 11) + (i - 0.5) * 0.38
            bp = ax.boxplot(vals, positions=pos, widths=0.3,
                            patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("#607D8B" if g == "Manual" else "#2E7D32")
                patch.set_alpha = 0.4
            dom_pos.append(pos)
        ax.set_xticks(np.arange(1, 11))
        ax.set_xticklabels([f"D{i}" for i in range(1, 11)], fontsize=6.5)
        ax.set_xlabel("Usability domain")
        ax.set_ylabel("Domain score (0-100)")
        ax.set_ylim(0, 105)
        ax.set_title("C  Simulated usability profile (10 domains)",
                     loc="left", pad=6)
        ax.annotate(f"Manual mean {sus_df[sus_df['group']=='Manual']['sus'].mean():.0f} "
                    f"vs Platform {sus_df[sus_df['group']=='Platform']['sus'].mean():.0f} "
                    f"(simulated; U-test P={sus_p:.2e})", xy=(0.5, 1.14),
                    xycoords="axes fraction", fontsize=6.2, ha="center",
                    color="#444444")

        ax = axes[1, 1]
        data_t = [tlx_df[tlx_df["group"] == "Manual"]["composite_tlx"],
                  tlx_df[tlx_df["group"] == "Platform"]["composite_tlx"]]
        vp = ax.violinplot(data_t, positions=[1, 2], showmedians=True,
                           widths=0.6)
        for body, color in zip(vp["bodies"], ["#607D8B", "#2E7D32"]):
            body.set_facecolor(color)
            body.set_alpha(0.45)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Manual", "Platform"])
        ax.set_ylabel("Workload composite (0-100)")
        ax.set_ylim(0, 105)
        ax.annotate(f"TLX P={tlx_p:.2e} (simulated unweighted composite)",
                    xy=(0.5, 1.14),
                    xycoords="axes fraction", fontsize=6.2, ha="center",
                    color="#444444")
        ax2 = ax.twinx()
        tm = [time_df[time_df["group"] == "Manual"]["time_min"].mean(),
              time_df[time_df["group"] == "Platform"]["time_min"].mean()]
        ts = [time_df[time_df["group"] == "Manual"]["time_min"].std(),
              time_df[time_df["group"] == "Platform"]["time_min"].std()]
        ax2.errorbar([1, 2], tm, yerr=ts, fmt="D", ms=5, color="#C62828",
                     capsize=3, lw=1.2)
        ax2.set_ylabel("Completion time (min)", color="#C62828")
        ax2.tick_params(axis="y", labelcolor="#C62828")
        ax.set_title("D  Simulated workload and completion time", loc="left",
                     pad=6)
        ax.annotate(f"Time: manual {tm[0]:.0f}±{ts[0]:.0f} min vs "
                    f"platform {tm[1]:.0f}±{ts[1]:.0f} min "
                    f"(P={time_p:.2e})", xy=(0.5, -0.18),
                    xycoords="axes fraction", fontsize=6.2, ha="center",
                    color="#C62828")

        fig.tight_layout(pad=1.3)
        out = SITE / "figures" / \
            "Figure_6_Platform_Evaluation.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print("saved", out, f"({os.path.getsize(out):,} bytes)")

    edit_df.to_csv(OUT_PROC / "edit_distance_simulation.csv", index=False)
    sus_df.to_csv(OUT_PROC / "sus_scores.csv", index=False)
    tlx_df.to_csv(OUT_PROC / "nasa_tlx.csv", index=False)
    time_df.to_csv(OUT_PROC / "completion_time.csv", index=False)
    pd.DataFrame({
        "metric": ["llm_latency_mean_ms", "llm_latency_n",
                   "sus_p", "tlx_p", "time_p"],
        "value": [llm_lat.mean(), llm_n, sus_p, tlx_p, time_p],
    }).to_csv(OUT_PROC / "platform_telemetry_stats.csv", index=False)
    print("evaluation data saved to", OUT_PROC)


if __name__ == "__main__":
    main()
