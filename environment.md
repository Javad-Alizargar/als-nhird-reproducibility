# Recorded environment

- OS: macOS (arm64), Darwin
- Python: 3.9.6 (system) — recorded working interpreter
  `/Library/Developer/CommandLineTools/usr/bin/python3`
- Key package versions recorded at build time:
  - numpy 2.0.2
  - pandas 2.2.3
  - scipy 1.13.1
  - statsmodels 0.14.2
  - matplotlib 3.9.4
  - scikit-learn 1.5.2
  - torch 2.8.0
  - openai (client library; optional live workflows only)
- LaTeX build: tectonic 0.x (used for the manuscript build; not required for
  package verification)

The offline verification (`verification/verify_package.py`) requires only
numpy, pandas, scipy; the PINN check additionally requires torch.
