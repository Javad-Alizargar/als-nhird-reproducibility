#!/bin/sh
# Regenerate the six figures from the archived data (offline).
set -e
cd "$(dirname "$0")/.."
python3 code/figure_refinements.py      # Figures 1-4 (FIG_SUBS=ALS by default)
python3 code/economics_figure.py        # Figure 5
FIG_SUBS=ALS python3 code/telemetry_figure.py   # Figure 6
echo "figures regenerated in figures/"
