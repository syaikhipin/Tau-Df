# Tau-df reproducibility artifact

Data, experiment code, and frozen results for:

> Hidden Degrees of Freedom: Predicting When SURE Over-Credits Label-Fitted Kernels

This repository contains what is needed to reproduce the paper's experiments. The
manuscript source is not included.

## Contents

```
data/                     Taiwan credit-default arrays and license
experiments/E1/           verification drivers and frozen results
experiments/E2/           diagnostic drivers and frozen results
experiments/figures/      figure generator
figures/                  generated PDF/PNG figures
reproduce.sh              runs every driver and regenerates the figures
```

## Data

The `data/` directory holds the Taiwan credit-card default dataset (UCI Machine
Learning Repository ID 350; Yeh & Lien 2009), copied from the MIT-licensed
reference package. See the note and license in that directory.

California housing (Pace & Barry 1997) is fetched at runtime by scikit-learn; no
local copy is needed.

## Environment

Tested with Python 3.13 on one CPU (no GPU):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The reference cross-check also needs the upstream package:

```bash
.venv/bin/pip install "lkbook @ git+https://github.com/asudjianto-xml/Learned-Kernel"
```

## Reproduce

Run from the repository root:

```bash
bash reproduce.sh
```

The script runs the cross-check, ledger and fusion experiments, reduced-support
ablations, capacity map, synthetic gate, real-data diagnostic, reviewer
ablations, and figure regeneration. Each driver writes into its own results
directory. Split sizes, seeds, grids, and bootstrap counts are config constants
at the top of the two experiment modules.

The committed result files are the exact inputs used by the figures and tables
in the paper. Result directories contain the ledger and fusion tables, the
cross-check, reduced-support and capacity outputs, the synthetic gate, real-data
diagnostics, and the frozen-kernel and noise-variance ablations.

## Sources

- A. Sudjianto, *The Learned Kernel*, Ch. 7.
- Reference implementation (MIT):
  https://github.com/asudjianto-xml/Learned-Kernel
- Theory preprints: SSRN 6823639 and SSRN 6815720.
