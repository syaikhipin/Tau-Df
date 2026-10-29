#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"

"$PY" experiments/E1/run_r003_crosscheck.py
"$PY" experiments/E1/run_m1_seeds.py
"$PY" experiments/E1/run_r017_reduced_n.py
"$PY" experiments/E1/run_r018_capacity.py
"$PY" experiments/E2/run_m2_gate.py
"$PY" experiments/E2/run_m3_real_data.py
"$PY" experiments/E2/run_r022_ablation.py
"$PY" experiments/figures/make_paper_figures.py
