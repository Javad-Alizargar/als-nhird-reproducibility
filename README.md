# als-nhird-reproducibility

Reproducibility package for the study: *A Simulation-Assisted
Proposal-Planning and Decision-Support Platform for Amyotrophic Lateral
Sclerosis Research in the Taiwan National Health Insurance Research Database*.

This package contains the code and underlying data needed to **verify the
reported numbers and regenerate the scientific outputs** of that study. It is
a curated export; it is **not** a mirror of the private working repository and
does not contain the manuscript or submission documents.

## What is included

- `data/` — archived records used by the analyses:
  - `data/tables/Supp_Table_1_System_Benchmarks.csv` — 324 measurement rows:
    retrieval latency, Precision@5, concurrency percentiles, throughput,
    tracemalloc peak, fee-engine and review-rule results.
  - `transcriptomics/` — GSE346896 deposited matrices and annotation (gzip),
    sample metadata, the analysis-ready TMM log2-CPM matrix, PCA scores,
    and the complete differential-expression table
    (`data/tables/Supp_Table_2_ALS_DEGs.csv`, 19,885 genes) and top-100
    statistics (`data/tables/Supp_Table_3_…`).
  - `digital_twin/` — the 1,000-patient simulation (`als_sim.npz`), Kaplan-Meier
    records, tier trajectories, the saved PINN weights
    (`als_pinn_state.pt`), training-loss history, and the recorded validation
    results (`als_pinn_validation.json`).
  - `proposals/` — the complete 60-run generation record
    (`data/tables/Supp_Table_5_…` plus per-run prompt metrics and generated
    texts in `ALS_generations.json`).
  - `tables/` — the six supplementary tables exactly as cited in the paper
    (S1 benchmarks, S2 DEGs, S3 top-100 statistics, S4 twins, S5 proposals,
    S6 power/cost).
  - `platform_eval/` — simulated human-factors records (usability profile,
    workload composite, completion time, edit-distance simulation) and the
    live-telemetry summary (LLM short-probe mean, n=3).
  - `nhird_documents/` — the official NHIRD application documents used to
    encode the fee and review rules (fee standard, application regulations
    2026-05-28, review principles, data catalogue). These are official
    documents reproduced for verification; see `THIRD_PARTY_NOTICES.md`.
- `code/` — the analysis and figure generators (adapted to this layout via
  `code/siteconfig.py`) and the platform package `code/platform/nhri_app/`
  (catalogue, retrieval, fee/rule engines, planner, LLM helpers).
- `figures/` — the six figures as rendered in the manuscript.
- `verification/` — the offline verification harness and figure-regeneration
  commands.

## What is not included (by design)

Manuscript prose, drafts, submission PDF/DOCX files, administrative
declarations, editorial disclosures, private audit records, API credentials,
and unrelated research projects. Transcriptomic raw-count and deposited
matrices are redistributed from the NCBI GEO accession GSE346896 (see
`data/transcriptomics/` and `THIRD_PARTY_NOTICES.md` for provenance and
checksums).

## Installation

```
python3 -m venv .venv && . .venv/bin/activate
pip install numpy pandas scipy statsmodels matplotlib torch scikit-learn openai
```

Recorded environment (Python 3.9.6, macOS arm64): numpy 2.0.2, pandas 2.x,
scipy 1.13, torch 2.8.0, statsmodels 0.14, matplotlib 3.9.4, scikit-learn 1.x.
See `environment.md` for the complete recorded list.

## Quick verification (offline, no API keys)

```
python3 verification/verify_package.py
```

This recomputes every headline quantity from the archived records — digital
twin (1,000 patients, 391 events, 486/514 tiers), PINN saved weights against
the recorded validation (residuals and DOP853 deviations), transcriptomic
counts (19,885 genes, 3 at FDR<0.05) and the batch–group confound (design rank
4 < 5), benchmark means, proposal metrics (0 invalid references in both arms),
simulated profile means, power (0.9195) and fee boundary values. Expected
output: `RESULT: ALL CHECKS PASSED` with exit code 0.

## Regenerating figures and tables

```
python3 code/figure_refinements.py        # Figures 1-4 (FIG_SUBS=ALS by default)
python3 code/economics_figure.py          # Figure 5
FIG_SUBS=ALS python3 code/telemetry_figure.py  # Figure 6
python3 code/de_pipeline.py               # differential expression + metadata
```

`code/digital_twin.py` re-runs the full 1,000-patient simulation and PINN
training (8,000 epochs; tens of minutes on CPU) — the archived outputs are
already present in `data/digital_twin/`, so this is optional.
`code/benchmarking.py` and `code/proposal_validation.py` re-run live
measurements and LLM calls; they require API keys via environment variables
and are **not** part of the offline verification.

## Reproducibility scope (honest limits)

- **Recomputable from inputs:** differential expression, PCA, design-rank and
  confound assessment, power/cost/fee computations, every summary statistic,
  and the PINN validation against the saved weights.
- **Verifiable against archived records:** all measurements that were recorded
  once (retrieval latencies, concurrency percentiles, tracemalloc peaks, the
  live LLM short-probe mean, proposal-generation runs).
- **Regenerable plots:** all six figures from the archived tables/records.
- **Not reproducible by re-running:** historical latency measurements
  (machine-dependent), individual short-probe timing records (only mean and
  n=3 were archived), and stochastic online LLM responses.
- **Scientific qualifications preserved:** the transcriptomic re-analysis is
  an independent illustration whose outputs feed no downstream computation;
  processing batch and group are completely confounded (the ALS–control
  contrast is not identifiable); gene-level outputs are descriptive, not
  biomarkers; the human-factors evaluation is simulated; risk tiers are
  outcome-defined; the PINN forward-trajectory discrepancy (≈4.3% of the
  48-point scale) is distinct from the per-equation residuals; the plotted
  attrition curves are 1−KM-of-first-event curves, not Aalen–Johansen
  cumulative-incidence estimates; the 664 ms LLM timing is a short-probe
  completion, not proposal-generation latency.

## Data dictionary and provenance

See `docs/data_dictionary.md`, `docs/provenance.md`,
`docs/result_to_code_map.md`, and `docs/limitations.md`.

## License

Original code and software documentation are licensed under the MIT License
(see `LICENSE`, Copyright (c) 2026 Javad Alizargar). Third-party data and
official documents retain their own terms (see `LICENSING.md` and
`THIRD_PARTY_NOTICES.md`).

## Citing this package

Version 1.4.0 is archived on Zenodo. Cite the archived version with its
version DOI:

> Alizargar, J. (2026). *als-nhird-reproducibility: Code and data for the ALS
> NHIRD proposal-planning and decision-support platform study* (Version 1.4.0)
> [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23053558

- Version DOI: https://doi.org/10.5281/zenodo.23053558
- Concept DOI (all versions): https://doi.org/10.5281/zenodo.23053557
- Record: https://zenodo.org/records/23053558
- Release: https://github.com/Javad-Alizargar/als-nhird-reproducibility/releases/tag/v1.4.0

Citation metadata is also provided in `CITATION.cff`. Manuscript metadata is
intentionally minimal: no publication venue or acceptance status is claimed.
