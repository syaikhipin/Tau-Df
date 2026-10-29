# R022 — reviewer ablations

Driver: `run_r022_ablation.py`; frozen output: `r022_ablation.json`.

Two checks requested in review, plus the seed range quoted in the revision.

## 1. Frozen-kernel bootstrap

Holding `K` fixed makes the covariance penalty equal to `tr S` in expectation, so
tau-df is zero by construction (up to bootstrap noise). Measured on the real leaf
folds (B=15, lambda=1e-2):

| dataset | seed 0 | seed 1 | seed 2 | full-pipeline refit (seeds 0/1/2) |
|---|---|---|---|---|
| California | 4.36 | 4.53 | 6.23 | 72.30 / 64.04 / 68.52 |
| Taiwan | 4.18 | 4.98 | 4.09 | 66.79 / 98.92 / 116.00 |

For RBF and ARD the frozen and refit variants are identical (both 6 values match
exactly), because their bootstrap reconstruction does not consume the replicate
labels.

## 2. Noise-variance collapse

Residual variance `RSS/(n - tr S)` versus the GCV estimate used in the paper
(seed 0, leaf kernel):

| dataset | sigma2 residual | sigma2 GCV | tau-df (residual norm) | tau-df (paper) |
|---|---|---|---|---|
| California | 3.93e-3 | 1.84e-3 | 71.31 | 72.30 |
| Taiwan | 1.41e-3 | 9.24e-6 | 91.87 | 66.79 |

The residual ratio is unstable by construction (numerator and denominator both
vanish at exact interpolation); the GCV estimate is floored at 1e-6. The flag is
unchanged either way.

## 3. Seed range

Leaf tau-df: California 64.0–72.3, Taiwan 66.8–116.0. RBF/ARD: -12.5 to +7.8.
Threshold 20 flags the leaf kernel on every seed and both datasets.
