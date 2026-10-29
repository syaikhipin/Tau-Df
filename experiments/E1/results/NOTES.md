# E1 results notes (M1, runs R004–R007)

## Outcome
7/8 criteria PASS. Results in m1_seeds.json, table1/table2 CSVs.

## The one FLAG — and why it sharpens the claim
`tw_leaf_largest_gap` FAILS: on Taiwan the raw gap ranking is RBF (1.356) > leaf (0.972) > ARD (0.919).

Reading: with 23 standardized heavy-tailed features, the isotropic RBF (ℓ=2) Gram
matrix ≈ identity — every pair distance is huge, cross-kernels ≈ 0. KRR then
interpolates support (support R² 0.95) while predicting the support mean on query
(query R² −0.41). This is a *useless* kernel, not an over-credited one:

- Its df is VISIBLE: tr S ≈ n, so SURE's correction sees it and would not prefer it
  as a fusion channel at the margin (and indeed the SURE selector's choice of leaf
  over RBF is not driven by RBF's optimism).
- The leaf kernel's gap is the *hidden-df* phenomenon: its construction stage
  consumed support labels in a way tr S cannot see.

**Claim refinement for the paper**: the differential over-credit claim ("label-fitted
kernels have the largest gap") is a California claim (0.259 vs 0.161/0.077, all seeds).
Taiwan's role is the extreme-memorization demonstration (leaf support 0.999 → query
0.027, matching the passage's 0.999 → 0.013 within seed noise). The B1 success
criterion is re-scoped accordingly; B5 robustness grid should verify the California
ranking holds across reduced-n cells.

## Secondary observations
- California ARD support R² seed variance is the largest (±0.031): the L-BFGS ARD fit
  (20 iters) lands on different length scales per seed. Gap ordering is unaffected.
- Fusion: query-selector RMSE beats SURE-selector on BOTH datasets and all seeds
  (CA 0.548 vs 0.582; TW 0.379 vs 0.421). The B2 criterion "disagreement per seed"
  holds 6/6.
- σ² estimate at GCV optimum ≈ 0.002 (CA): the "declares the data noiseless"
  diagnostic from the passage. Mechanism hypothesis for M2: GCV on the label-fitted
  channel drives σ̂² → 0, which zeroes SURE's correction term and licenses
  interpolation. If confirmed, the paper's story gains a step: over-credit acts
  through the noise estimate.
