# Limitations (reproducibility)

1. Historical latency/concurrency measurements were recorded once on the
   original machine; the package verifies the archived records but does not
   claim they can be reproduced exactly elsewhere.
2. Individual LLM short-probe timing runs were not archived (only mean and
   n=3); the 664 ms figure is verifiable as the recorded mean only.
3. Proposal-generation runs and benchmark records are archived outputs;
   re-running them requires paid API access (optional, environment-variable
   keys) and yields different stochastic outputs.
4. PINN training re-runs are expensive; the saved weights and validation
   record are provided instead.
5. The transcriptomic re-analysis is an independent illustration; batch and
   group are completely confounded, and the gene-level outputs are
   descriptive (not biomarkers).
6. Official NHIRD documents are saved copies; their official web locations
   were not retrievable at export time (bot-blocked) and are documented from
   the saved originals.
7. This package contains no manuscript text and no submission documents.
