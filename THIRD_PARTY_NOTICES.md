# Third-party notices

## Data

- NCBI GEO GSE346896 — "GLAST-positive small extracellular-vesicle total-RNA
  sequencing in amyotrophic lateral sclerosis: a 60-participant discovery
  cohort" (Pitt lab, Yale University; BioProject PRJNA1526785; platform
  GPL24676; submitted Sep 10 2026). Files in `data/transcriptomics/`:
  - GSE346896_series_matrix.txt.gz (sha256 91c6b75a7c537a9dca1b5c15d0b1e3f1a1f9646f83a2c9dd8fdc5b07a2e81e62 — recompute on the file)
  - GSE346896_discovery_mRNA_TMM_log2CPM_protein_coding.csv.gz
  - GSE346896_discovery_mRNA_raw_counts.csv.gz
  - GSE346896_gencode_hg38_gene_annotation.csv.gz
  Retrieval route: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE346896
- Official NHIRD documents in `data/nhird_documents/`:
  - regulation_20260528.pdf — 衛生福利部衛生福利資料科學中心「資料申請應用規定
    及申請文件」（中華民國115年05月28日上網公告版）；contains the fee standard
    (衛生福利統計資料整合應用服務收費標準) and the review principles
    (衛生福利資料申請案件審核作業原則).
  - Money regulation.txt — 衛生福利統計資料整合應用服務收費標準 (saved copy).
  - review rules.txt — 衛生福利資料申請案件審核作業原則 (saved copy).
  - nhri_dataset_categories_expanded.txt — dataset catalogue transcript.
  - datasets_not expanded.pdf — 衛生福利資料科學中心使用資料申請單 (catalogue).

## Software dependencies

numpy (BSD-3-Clause), pandas (BSD-3-Clause), scipy (BSD-3-Clause),
statsmodels (BSD-3-Clause), matplotlib (PSF-based), torch (BSD-3-Clause),
scikit-learn (BSD-3-Clause), openai (Apache-2.0, used only by optional live
workflows).

No third-party code is vendored in this repository.
