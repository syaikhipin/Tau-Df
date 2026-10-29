# Tau-df reproducibility artifact

Data, experiment code, and frozen results for:

> Hidden Degrees of Freedom: Predicting When SURE Over-Credits Label-Fitted Kernels

This repository contains what is needed to reproduce the paper's experiments. The
manuscript source is not included.

## Contents

```
data/
  taiwan_credit.npz        Taiwan credit-default arrays (Yeh & Lien 2009)
  LICENSE-lkbook           MIT license of the package the data came from
experiments/
  E1/                      ledger, fusion, robustness, capacity
    exp1.py                independent re-implementation
    run_r003_crosscheck.py  ours vs the reference implementation
    run_m1_seeds.py        ledger + fusion, 3 seeds x 2 datasets
    run_r017_reduced_n.py  reduced-support ablations
    run_r018_capacity.py   capacity map
    results/               frozen JSON/CSV outputs
  E2/                      tau-df diagnostics
    exp2.py                synthetic harness + tau statistics
    run_m2_gate.py         synthetic gate
    run_m3_real_data.py    real support folds + threshold rule
    run_r022_ablation.py   frozen-kernel + noise-variance ablations
    results/               frozen JSON outputs and notes
  figures/
    make_paper_figures.py  regenerates figures from frozen results
figures/                   generated PDF/PNG figures
```

## Data

`data/taiwan_credit.npz` is the Taiwan credit-card default dataset (UCI Machine
Learning Repository ID 350; Yeh & Lien 2009). It is copied from the
MIT-licensed `lkbook` package; see `data/LICENSE-lkbook`.

California housing (Pace & Barry 1997) is fetched at runtime by
`sklearn.datasets.fetch_california_housing`; no local copy is needed.

## Environment

Tested with Python 3.13 on one CPU (no GPU):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The reference cross-check (`run_r003_crosscheck.py`) also needs the upstream
`lkbook` package:

```bash
.venv/bin/pip install "lkbook @ git+https://github.com/asudjianto-xml/Learned-Kernel"
```

## Reproduce

Run from the repository root:

```bash
.venv/bin/python experiments/E1/run_r003_crosscheck.py    # ours vs upstream
.venv/bin/python experiments/E1/run_m1_seeds.py           # ledger + fusion, 3 seeds
.venv/bin/python experiments/E1/run_r017_reduced_n.py     # reduced-support folds
.venv/bin/python experiments/E1/run_r018_capacity.py      # capacity map
.venv/bin/python experiments/E2/run_m2_gate.py            # synthetic gate
.venv/bin/python experiments/E2/run_m3_real_data.py       # real support folds
.venv/bin/python experiments/E2/run_r022_ablation.py      # frozen/variance ablations
.venv/bin/python experiments/figures/make_paper_figures.py
```

Each driver writes into its own `results/` directory. Split sizes, seeds, grids,
and bootstrap counts are config constants at the top of `exp1.py` and `exp2.py`.
The committed `results/` files are the exact inputs used by the figures and
tables in the paper.

## Result map

- `E1/results/table1_california_ledger.csv`, `table2_taiwan_ledger.csv`,
  `m1_seeds.json`: support/query ledger and fusion selection.
- `E1/results/r003_crosscheck.json`: cross-check against the reference
  implementation.
- `E1/results/r017_reduced_n.json`: reduced-support ablations.
- `E1/results/r018_capacity_map.json`: capacity map.
- `E2/results/m2_gate.json`: synthetic provenance test.
- `E2/results/m3_real_data.json`: tau-df on real support folds, seed 0.
- `E2/results/r022_ablation.json`: frozen-kernel and noise-variance ablations,
  plus seeds 0-2.

## Sources

- A. Sudjianto, *The Learned Kernel*, Ch. 7.
- Reference implementation (MIT):
  https://github.com/asudjianto-xml/Learned-Kernel
- Theory preprints: SSRN 6823639 and SSRN 6815720.
