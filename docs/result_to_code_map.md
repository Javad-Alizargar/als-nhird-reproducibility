# Result → code → data map

| Reported result | Code (public) | Data (public) |
|---|---|---|
| 45 ALS / 15 controls; batch×group table; design rank 4<5 | code/de_pipeline.py | data/ALS_sample_metadata.csv, data/transcriptomics/*series_matrix* |
| 19,885 genes; SRP72/NPY2R/WDR36 log2FC and FDR; 3/16 counts | code/de_pipeline.py | data/tables/Supp_Table_2_ALS_DEGs.csv |
| PCA 17%/4% | code/transcriptomics_figure.py | data/ALS_PCA_scores.csv |
| 1,000 twins; 391 events; 486/514 tiers; KM/log-rank | code/digital_twin.py | data/digital_twin/als_sim.npz, als_km.csv |
| PINN losses, residuals, DOP853 deviations (≈4.3% forward) | code/digital_twin.py; verification/verify_package.py | data/digital_twin/als_pinn_state.pt, als_pinn_validation.json |
| Retrieval/concurrency/fee/rules benchmarks | code/benchmarking.py (re-run) | data/tables/Supp_Table_1_System_Benchmarks.csv |
| 60 proposals; 0 invalid refs both arms; budget 77%/10%; structural 0.56/1.00 | code/proposal_validation.py (re-run) | data/tables/Supp_Table_5_…, data/proposals/ALS_generations.json |
| Power 0.9195 @ n=500/HR1.5; cost 32,550; 70-row matrix | code/economics_figure.py | data/tables/Supp_Table_6_… |
| Telemetry 664 ms (n=3); simulated 82.3/54.5; 28/68; 24/213 | code/telemetry_figure.py | data/platform_eval/*.csv |
| Figures 1-6 | code/figure_refinements.py, economics_figure.py, telemetry_figure.py | figures/*.png |
