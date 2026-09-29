# Data dictionary

## data/tables/Supp_Table_1_System_Benchmarks.csv (324 rows)
Columns: benchmark (retrieval|concurrency|fee_engine|review_enforcement|…),
metric (latency_ms, precision_at5, p50_ms, p95_ms, throughput, peak_memory,
accuracy, mismatches, …), value, unit, detail (condition labels). Historical
measurements; not re-run in verification.

## data/tables/Supp_Table_2_ALS_DEGs.csv (19,885 rows)
Ensembl_ID, gene_symbol, log2FC (ALS vs control on deposited TMM log2-CPM),
t_moderated, p_value, padj_BH, mean_log2expr_ALS, mean_log2expr_Control,
gini_importance, permutation_p. Descriptive only; batch and group fully
confounded.

## data/tables/Supp_Table_3_ALS_Top100_DEG_Statistics.csv (100 rows)
Same columns for the top-100 genes by BH-adjusted p.

## data/tables/Supp_Table_4_ALS_Digital_Twins.csv (1,000 rows)
patient_id, severity_latent, E_proxy_burden (severity-sampled, NOT derived
from transcriptomics), age_onset_years, k1-k9, ALSFRSR_month0/12/24/36/48/60,
delta_ALSFRSR_48m, survival_months, death_event, risk_tier (outcome-defined:
48-month decline > 22 points or event < 36 months), network fields.

## data/tables/Supp_Table_5_Generated_Proposal_Comparison.csv (60 rows)
run_id, paradigm (skeleton_first|unconstrained), dataset_mentions,
invalid_dataset_mentions, hallucination_rate, budget_compliance (regex
presence), sample_size_justified, variable_inclusion, phase_structure,
has_background, has_cohort_definition, has_exposure_outcome,
has_statistical_plan, structural_score (0-1), prompt_tokens,
completion_tokens, completion_chars, content_tokens_est (chars/4),
skeleton_alignment. Full generated texts: data/proposals/ALS_generations.json.

## data/tables/Supp_Table_6_Power_and_Cost_Matrix.csv (70 rows)
Power rows: n, hr, alpha, p_event (0.55 assumption), events_needed, power
(PH normal approximation), datasets, total_fields (31), years (5),
cost_ntd (32,550). Cost rows add n_cohort, observation_years, power_at_hr15.

## data/digital_twin/als_sim.npz
Arrays (each length 1,000): F_clean, F_obs (61 months), M, N, event (391
ones), time (months), E (mean 1.163), a0 (mean 55.3).

## data/platform_eval/*.csv
sus_scores.csv: group (Manual|Platform), domain (1-10), sus (0-100 sampled
directly; NOT SUS item responses; n=24 per arm per domain).
nasa_tlx.csv: group, composite_tlx (0-100, unweighted, sampled directly).
completion_time.csv: group, time_min (simulated).
edit_distance_simulation.csv: group (Novice n=18|Senior n=12), section,
similarity_ratio (1 − edit distance of simulated character substitutions).
platform_telemetry_stats.csv: llm_latency_mean_ms=663.94 (live short probe,
n=3; not proposal-generation latency), llm_latency_n=3, sus_p, tlx_p, time_p.
