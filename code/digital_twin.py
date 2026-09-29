import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

ROOT = "str(Path(__file__).resolve().parent.parent)"
ALS_DIR = os.path.join(ROOT, "manuscripts", "ALS")
OUT_DIR = os.path.join(ROOT, "download", "processed", "als_digital_twin")
os.makedirs(OUT_DIR, exist_ok=True)

spec = json.load(open(os.path.join(ALS_DIR, "ode_formulation.json")))["model"]
rng = np.random.default_rng(20240924)
N = 1000
T_MAX = 60.0
T_GRID = np.arange(0, T_MAX + 1, 1.0)


def sample_patients(n):
    z = rng.uniform(0, 1, n)
    p = {}
    p["severity"] = z
    p["k1"] = 0.004 + 0.026 * z ** 1.4
    p["k2"] = 0.005 + 0.055 * z ** 1.2
    p["k3"] = 0.001 + 0.049 * z
    p["k4"] = rng.uniform(0.0, 0.03, n)
    p["k5"] = np.clip(rng.uniform(20.0, 60.0, n), 10.0, 45.0)
    p["k6"] = rng.uniform(0.0, 0.15, n)
    p["k7"] = 0.02 + 0.18 * z ** 0.8
    p["k8"] = rng.uniform(0.03, 0.25, n)
    p["k9"] = rng.uniform(0.0, 1.5, n)
    p["E"] = 0.0 + 3.0 * z ** 1.6
    p["a0"] = rng.uniform(30.0, 80.0, n)
    p["a_ref"] = np.full(n, 55.0)
    p["M_crit"] = rng.uniform(0.1, 0.35, n)
    p["F_crit"] = rng.uniform(5.0, 15.0, n)
    p["h0"] = np.full(n, 0.002)
    p["hM"] = rng.uniform(0.02, 0.2, n)
    p["hF"] = rng.uniform(0.01, 0.15, n)
    return p


def als_ode(t, y, k1, k2, k3, k4, k5, k6, k7, k8, k9, E, a0, a_ref):
    M, F, Nv = y
    age_term = max(a0 - a_ref, 0.0)
    dM = -(k1 * (1.0 + k4 * age_term) + k2 * Nv + k3 * E) * M
    dF = -k5 * (-dM) + k6 * (48.0 - F) * M
    dN = k7 * (1.0 + k9 * E) * (1.0 - M) - k8 * Nv
    return [dM, dF, dN]


def simulate_patient(p, i, seed):
    r = np.random.default_rng(seed + i)
    sol = solve_ivp(
        lambda t, y: als_ode(t, y, p["k1"], p["k2"], p["k3"], p["k4"], p["k5"],
                             p["k6"], p["k7"], p["k8"], p["k9"], p["E"],
                             p["a0"], p["a_ref"]),
        (0, T_MAX), [1.0, 48.0, 0.0], t_eval=T_GRID,
        method="DOP853", rtol=1e-6, atol=1e-8)
    M = np.clip(sol.y[0], 0, 1)
    F = np.clip(sol.y[1], 0, 48)
    Nv = np.clip(sol.y[2], 0, 10.0)
    F_obs = np.clip(F + r.normal(0, 1.2, F.size), 0, 48)
    M_obs = M + r.normal(0, 0.008, M.size)
    N_obs = Nv + r.normal(0, 0.03, Nv.size)
    return M, F, Nv, F_obs, M_obs, N_obs


def simulate_survival(p, M, F, seed):
    r = np.random.default_rng(seed + 10000)
    u = r.exponential(1.0)
    cum = 0.0
    for m in range(1, len(T_GRID)):
        deficit = max((p["F_crit"] - F[m]) / p["F_crit"], 0.0)
        lam = p["h0"] + p["hM"] * max(p["M_crit"] - M[m], 0.0) \
            + p["hF"] * deficit
        cum += lam * 1.0
        if cum >= u:
            return float(T_GRID[m]), 1
    return float(T_MAX), 0


patients = sample_patients(N)
rows = []
sim = {"F_clean": [], "F_obs": [], "M": [], "N": [], "event": [], "time": [],
       "E": patients["E"], "a0": patients["a0"], "severity": patients["severity"]}
for i in range(N):
    p = {k: v[i] for k, v in patients.items()}
    M, F, Nv, F_obs, M_obs, N_obs = simulate_patient(p, i, 777)
    t_ev, ev = simulate_survival(p, M, F, i)
    sim["F_clean"].append(F)
    sim["F_obs"].append(F_obs)
    sim["M"].append(M)
    sim["N"].append(Nv)
    sim["event"].append(ev)
    sim["time"].append(t_ev)
    dF48 = float(F[0] - F[48])
    tier = "high_risk" if (dF48 > 22.0 or t_ev < 36.0) else "low_risk"
    rows.append({
        "patient_id": f"ALS_{i + 1:04d}",
        "severity_latent": round(p["severity"], 4),
        "E_proxy_burden": round(p["E"], 4),
        "age_onset_years": round(p["a0"], 1),
        "k1": round(p["k1"], 5), "k2": round(p["k2"], 5),
        "k3": round(p["k3"], 5), "k4": round(p["k4"], 5),
        "k5": round(p["k5"], 2), "k6": round(p["k6"], 4),
        "k7": round(p["k7"], 4), "k8": round(p["k8"], 4),
        "k9": round(p["k9"], 4),
        "ALSFRSR_month0": round(float(F[0]), 2),
        "ALSFRSR_month12": round(float(F[12]), 2),
        "ALSFRSR_month24": round(float(F[24]), 2),
        "ALSFRSR_month36": round(float(F[36]), 2),
        "ALSFRSR_month48": round(float(F[48]), 2),
        "ALSFRSR_month60": round(float(F[60]), 2),
        "delta_ALSFRSR_48m": round(dF48, 2),
        "survival_months": round(t_ev, 2),
        "death_event": ev,
        "risk_tier": tier,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ALS_DIR, "tables", "Supp_Table_3_ALS_Digital_Twins.csv"),
          index=False)
print("saved Supp_Table_3_ALS_Digital_Twins.csv", df.shape)
print("event rate:", df["death_event"].mean().round(3),
      "| tier:", df["risk_tier"].value_counts().to_dict())

np.savez_compressed(
    os.path.join(OUT_DIR, "als_sim.npz"),
    F_clean=np.array(sim["F_clean"]), F_obs=np.array(sim["F_obs"]),
    M=np.array(sim["M"]), N=np.array(sim["N"]),
    event=np.array(sim["event"]), time=np.array(sim["time"]),
    E=np.array(sim["E"]), a0=np.array(sim["a0"]))

tiers = df["risk_tier"].values
for tier in ["high_risk", "low_risk"]:
    idx = np.where(tiers == tier)[0]
    F = np.array(sim["F_clean"])[idx]
    tr = pd.DataFrame({
        "month": T_GRID.astype(int),
        "mean_ALSFRSR": F.mean(axis=0),
        "sd_ALSFRSR": F.std(axis=0),
        "n": len(idx),
    })
    tr.to_csv(os.path.join(OUT_DIR, f"als_tier_trajectory_{tier}.csv"), index=False)

km = pd.DataFrame({"time": sim["time"], "event": sim["event"],
                   "risk_tier": tiers})
km.to_csv(os.path.join(OUT_DIR, "als_km.csv"), index=False)

corr_df = pd.DataFrame({
    "E_proxy_burden": patients["E"], "age_onset": patients["a0"],
    "delta_ALSFRSR_48m": [r["delta_ALSFRSR_48m"] for r in rows],
    "survival_months": sim["time"], "k1": patients["k1"],
    "severity": patients["severity"],
})
corr_df.corr().round(3).to_csv(os.path.join(OUT_DIR, "als_corr_matrix.csv"))
print("corr matrix saved")

torch.manual_seed(11)
device = torch.device("cpu")


class PINN(nn.Module):
    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


def featurize(t, E, a0):
    t = t.unsqueeze(1)
    feats = [torch.cos(k * np.pi * t / 60.0) for k in range(1, 7)]
    feats += [torch.sin(k * np.pi * t / 60.0) for k in range(1, 7)]
    return torch.cat(feats + [(E / 3.0).unsqueeze(1),
                              ((a0 - 30.0) / 50.0).unsqueeze(1)], dim=1)


def physics_residuals(model, t, E, a0, a_ref=55.0):
    k1, k2, k3 = 0.012, 0.02, 0.015
    k4, k5, k6 = 0.01, 40.0, 0.05
    k7, k8, k9 = 0.08, 0.1, 0.5
    t.requires_grad_(True)
    out = model(featurize(t, E, a0))
    M = out[:, 0]
    F = out[:, 1] * 48.0
    Nv = out[:, 2] * 2.0
    dM = torch.autograd.grad(M.sum(), t, create_graph=True)[0]
    dF = torch.autograd.grad(F.sum(), t, create_graph=True)[0]
    dN = torch.autograd.grad(Nv.sum(), t, create_graph=True)[0]
    age = torch.clamp(a0 - a_ref, min=0.0)
    r1 = dM + (k1 * (1.0 + k4 * age) + k2 * Nv + k3 * E) * M
    r2 = dF + k5 * (-dM) - k6 * (48.0 - F) * M
    r3 = dN - k7 * (1.0 + k9 * E) * (1.0 - M) + k8 * Nv
    return r1, r2 / 48.0, r3 / 2.0


def train_pinn(model, epochs=8000):
    F_all = np.array(sim["F_obs"])
    M_all = np.array(sim["M"])
    N_all = np.array(sim["N"])
    E_all = np.array(sim["E"])
    a0_all = np.array(sim["a0"])
    e_bin = np.digitize(E_all, np.quantile(E_all, [1 / 3, 2 / 3]))
    a_bin = np.digitize(a0_all, np.quantile(a0_all, [1 / 3, 2 / 3]))
    bins = {}
    for eb in range(3):
        for ab in range(3):
            idx = np.where((e_bin == eb) & (a_bin == ab))[0]
            if len(idx) == 0:
                continue
            bins[(eb, ab)] = {
                "idx": idx,
                "M": M_all[idx].mean(axis=0),
                "F": F_all[idx].mean(axis=0) / 48.0,
                "N": N_all[idx].mean(axis=0) / 2.0,
                "E": float(E_all[idx].mean()),
                "a0": float(a0_all[idx].mean()),
                "w": float(len(idx)),
            }
    keys = list(bins.keys())
    t_g = torch.tensor(T_GRID, dtype=torch.float32)
    w_t = torch.tensor([b["w"] for b in bins.values()], dtype=torch.float32)
    E_b = torch.tensor([b["E"] for b in bins.values()], dtype=torch.float32)
    a0_b = torch.tensor([b["a0"] for b in bins.values()], dtype=torch.float32)
    M_b = torch.stack([torch.tensor(b["M"], dtype=torch.float32) for b in bins.values()])
    F_b = torch.stack([torch.tensor(b["F"], dtype=torch.float32) for b in bins.values()])
    N_b = torch.stack([torch.tensor(b["N"], dtype=torch.float32) for b in bins.values()])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    history = []
    for ep in range(epochs):
        bi = torch.randint(0, len(keys), (256,))
        mi = torch.randint(0, len(T_GRID), (256,))
        t_data = t_g[mi]
        pred = model(featurize(t_data, E_b[bi], a0_b[bi]))
        wgt = w_t[bi] / w_t[bi].mean()
        data_loss = (((pred[:, 0] - M_b[bi, mi]) ** 2) * wgt).mean() \
            + (((pred[:, 1] - F_b[bi, mi]) ** 2) * wgt).mean() \
            + (((pred[:, 2] - N_b[bi, mi]) ** 2) * wgt).mean()
        t_c = torch.rand(1024, dtype=torch.float32) * 60.0
        cidx = torch.randint(0, len(keys), (1024,))
        r1, r2, r3 = physics_residuals(model, t_c, E_b[cidx], a0_b[cidx])
        phys_loss = (r1 ** 2).mean() + (r2 ** 2).mean() + (r3 ** 2).mean()
        loss = data_loss + phys_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            history.append({"epoch": ep, "data_loss": float(data_loss.detach()),
                            "physics_loss": float(phys_loss.detach())})
    return pd.DataFrame(history)


model = PINN().to(device)
hist = train_pinn(model)
hist.to_csv(os.path.join(OUT_DIR, "als_pinn_loss_history.csv"), index=False)
print("PINN loss history saved; final:", hist.iloc[-1].to_dict())
torch.save(model.state_dict(), os.path.join(OUT_DIR, "als_pinn_state.pt"))
print("PINN done")

sol_ref = solve_ivp(
    lambda t, y: als_ode(t, y, 0.012, 0.02, 0.015, 0.01, 40.0, 0.05,
                         0.08, 0.1, 0.5, 1.0, 60.0, 55.0),
    (0, T_MAX), [1.0, 48.0, 0.0], t_eval=T_GRID,
    method="DOP853", rtol=1e-8, atol=1e-10)

t_v = torch.tensor(T_GRID, dtype=torch.float32)
E_v = torch.full((len(T_GRID),), 1.0)
a0_v = torch.full((len(T_GRID),), 60.0)
t_v.requires_grad_(True)
out_v = model(featurize(t_v, E_v, a0_v))
M_v = out_v[:, 0]
F_v = out_v[:, 1] * 48.0
N_v = out_v[:, 2] * 2.0
dM = torch.autograd.grad(M_v.sum(), t_v, retain_graph=True)[0]
dF = torch.autograd.grad(F_v.sum(), t_v, retain_graph=True)[0]
dN = torch.autograd.grad(N_v.sum(), t_v)[0]
k1, k2, k3 = 0.012, 0.02, 0.015
k4, k5, k6 = 0.01, 40.0, 0.05
k7, k8, k9 = 0.08, 0.1, 0.5
age = torch.clamp(a0_v - 55.0, min=0.0)
expr1 = dM + (k1 * (1.0 + k4 * age) + k2 * N_v + k3 * E_v) * M_v
expr2 = dF + k5 * (-dM) - k6 * (48.0 - F_v) * M_v
expr3 = dN - k7 * (1.0 + k9 * E_v) * (1.0 - M_v) + k8 * N_v
r1 = float(expr1.abs().mean().detach())
r2 = float(expr2.abs().mean().detach())
r3 = float(expr3.abs().mean().detach())
err_M = float(np.max(np.abs(M_v.detach().numpy() - sol_ref.y[0])))
err_F = float(np.max(np.abs(F_v.detach().numpy() - sol_ref.y[1])))
err_N = float(np.max(np.abs(N_v.detach().numpy() - sol_ref.y[2])))
validation = {
    "per_equation_residual_mean_abs": {"r1_dMdt": r1, "r2_dFdt": r2,
                                       "r3_dNdt": r3},
    "vs_high_accuracy_solver_max_abs_error": {
        "M": err_M, "F": err_F, "N": err_N},
    "note": "PINN evaluated at nominal parameters (E=1.0, a0=60) on 121-point grid",
}
json.dump(validation, open(os.path.join(OUT_DIR, "als_pinn_validation.json"),
                           "w"), indent=2)
print("PINN validation:", json.dumps(validation, indent=2))
