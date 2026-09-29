import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from siteconfig import SITE
import json
import os
import re

from openai import OpenAI

ROOT = "str(Path(__file__).resolve().parent.parent)"
APIS = os.path.join(ROOT, "apis")

key = open(os.path.join(APIS, "deepseek.txt")).read().strip().split()[0]
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

SYSTEM = (
    "You are a mathematical biologist expert in ordinary differential equation (ODE) "
    "modeling and physics-informed neural networks for rare diseases. You output strict JSON only."
)

ALS_PROMPT = """Derive a biologically grounded system of ODEs for Amyotrophic Lateral Sclerosis (ALS)
disease progression, to be used as the physics backbone of a physics-informed neural network (PINN)
digital twin.

MODEL REQUIREMENTS:
- Time domain t in [0, 60] months.
- State variables must include:
  M(t) = surviving motor neuron fraction (0..1),
  F(t) = ALSFRS-R functional score (0..48),
  N(t) = neuroinflammation index driven by EV-RNA proxy burden.
- The transcriptomic proxy covariates (from our DE analysis of plasma GLAST+ sEV RNA, GSE346896)
  are: SRP72 (down, log2FC=-2.0), NPY2R (down, log2FC=-1.7), WDR36 (down, log2FC=-2.3).
  Model their joint burden as a per-patient constant E = E(SRP72, NPY2R, WDR36) that modulates
  the motor neuron death rate (e.g., loss of neuroprotective/RNA-handling function).
- Age of onset (a0, years) must appear as a covariate accelerating baseline degeneration.
- The system must be able to produce heterogeneous outcomes: fast vs slow progressors
  (survival endpoints) depending on parameters.

OUTPUT STRICT JSON with exactly these keys:
{
 "title": "...",
 "state_variables": [{"symbol":"M","name":"surviving motor neuron fraction","unit":"dimensionless","range":"[0,1]"} , ...],
 "parameters": [{"symbol":"k1","name":"...","nominal":0.0,"range":[..,..],"unit":"month^-1","biological_meaning":"..."}, ...],
 "equations": [{"latex":"\\frac{dM}{dt} = - ...","number":"(Eq. ALS-1)","explanation":"..."}, ...],
 "initial_conditions": {"M(0)":1.0,"F(0)":48.0,"N(0)":0.0},
 "hazard_model": {"latex":"\\lambda(t) = ...","explanation":"how survival/death is linked to F(t) and M(t)"},
 "heterogeneity": "how per-patient parameters are sampled (which parameters vary, with ranges)",
 "assumptions": ["..."],
 "literature_rationale": ["term-by-term justification; use [CITE-NEEDED: brief description] placeholders, NEVER invent PMID/DOI numbers"]
}
RULES:
- LaTeX must be pure ASCII (no Unicode math glyphs).
- Every symbol used must be defined.
- Equations numbered (Eq. ALS-1), (Eq. ALS-2) ...
- Do NOT fabricate citations, PMIDs or DOIs. Use [CITE-NEEDED: ...] where literature support is
  required but cannot be verified here.
- Parameter values must be plausible (ALSFRS-R declines roughly 0.5-1.5 points/month in fast progressors;
  median survival from onset 20-48 months).
- Return ONLY the JSON object, no markdown fences."""

WILSON_PROMPT = """Derive a biologically grounded system of ODEs for Wilson's disease (WD) progression,
to be used as the physics backbone of a physics-informed neural network (PINN) digital twin.

MODEL REQUIREMENTS:
- Time domain t in [0, 120] months.
- State variables must include:
  Cu(t) = hepatic copper concentration (normalized index, 1 = upper normal limit),
  Fb(t) = liver fibrosis stage index (0..4, METAVIR-like continuous),
  D(t) = hepatic decompensation index combining transaminase elevation and synthetic failure.
- Transcriptomic covariates from our DE analysis (GSE197406, WD liver vs control):
  BCHE downregulated (log2FC=-6.0), AKR1B10 upregulated (log2FC=+5.1), CCL20 upregulated (log2FC=+4.2).
  Model: BCHE deficiency reduces copper handling / antioxidant defense (accelerates toxic free copper),
  AKR1B10 marks hepatocyte injury/proliferation (proportional to damage flux),
  CCL20 drives inflammatory recruitment that accelerates fibrosis.
- The system must produce heterogeneous outcomes: compensated vs decompensated cirrhosis,
  with time-to-decompensation endpoints over 10 years.

OUTPUT STRICT JSON with exactly these keys:
{
 "title": "...",
 "state_variables": [{"symbol":"Cu","name":"hepatic copper index","unit":"dimensionless","range":"[0,10]"}, ...],
 "parameters": [{"symbol":"k_in","name":"copper influx rate","nominal":0.0,"range":[..,..],"unit":"month^-1","biological_meaning":"..."}, ...],
 "equations": [{"latex":"\\frac{dCu}{dt} = ...","number":"(Eq. WD-1)","explanation":"..."}, ...],
 "initial_conditions": {"Cu(0)":..,"Fb(0)":..,"D(0)":..},
 "hazard_model": {"latex":"...","explanation":"how time-to-decompensation/cirrhosis event is linked to D(t) and Fb(t)"},
 "heterogeneity": "how per-patient parameters are sampled (which parameters vary, with ranges)",
 "assumptions": ["..."],
 "literature_rationale": ["term-by-term justification; use [CITE-NEEDED: brief description] placeholders, NEVER invent PMID/DOI numbers"]
}
RULES:
- LaTeX must be pure ASCII (no Unicode math glyphs).
- Every symbol used must be defined.
- Equations numbered (Eq. WD-1), (Eq. WD-2) ...
- Do NOT fabricate citations, PMIDs or DOIs. Use [CITE-NEEDED: ...] where literature support is
  required but cannot be verified here.
- Parameter values must be plausible (untreated WD typically progresses to decompensated cirrhosis
  within 5-15 years depending on onset; Kayser-Fleischer rings appear as copper accumulates).
- Return ONLY the JSON object, no markdown fences."""


def ask(prompt, out_path):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=4000)
    text = resp.choices[0].message.content or ""
    clean = re.sub(r"^```(json)?\s*", "", text.strip())
    clean = re.sub(r"\s*```$", "", clean)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.S)
        data = json.loads(m.group(0)) if m else {"raw": text, "parse_error": True}
    with open(out_path, "w") as f:
        json.dump({"raw_response": text, "model": data}, f,
                  ensure_ascii=False, indent=2)
    print("saved", out_path)
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1800])
    return data


als = ask(ALS_PROMPT,
          str(SITE / "data" / "digital_twin" / "ode_formulation.json"))
wil = ask(WILSON_PROMPT,
          os.path.join(ROOT, "manuscripts", "Wilson", "ode_formulation.json"))
print("\nDONE: ALS keys:", list(als.keys()) if isinstance(als, dict) else "parse issue")
print("DONE: Wilson keys:", list(wil.keys()) if isinstance(wil, dict) else "parse issue")
