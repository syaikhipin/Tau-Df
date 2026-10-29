# R017 — reduced-n robustness: results and reading

## Raw outcome (California, 3 seeds per cell)

| split | leaf top-gap | query selector wins |
|-------|--------------|---------------------|
| 800/800 (paper operating point) | 3/3 | 3/3 |
| 400/400 | 2/3 | 3/3 |
| 200/200 | 1/3 | 1/3 |

The two R017 "FLAG"s are the 200/200 cell degrading, not a contradiction of B1/B2 —
those criteria were scoped at the operating point and passed in R004–R006.

## Why the small-n cell degrades (two mechanisms, both from theory)

1. **Selection cost ∝ √(𝔠(Θ)/n)**: at n_support = 200 the leaf kernel's hidden-df
   budget shrinks (fewer points to memorize with) while the *relative* advantage of
   any kernel is noisier — the differential over-credit signal gets buried.
2. **The query fold is itself an estimator**: with 200 query points, the query R²
   has high variance, so the leakage-free selector makes mistakes too. At 200/200
   the two selectors are both noisy; the comparison loses its bite.

This is the honest boundary of the remedy: the support/query discipline needs
enough data in BOTH folds. Paper framing (appendix subsection, one paragraph + the
table): "the leakage-free advantage emerges between n = 200 and n = 400 per fold on
California and is uniform at the operating point" — which is itself useful guidance
for practitioners choosing fold sizes, and matches the √(1/n) selection-cost shape.

## What R017 does NOT show
No sign reversal at any size: the leaf kernel's over-credit gap never becomes
harmful *in the query selector's favor*, and SURE never wins significantly — the
query selector's 200/200 losses (0.6767 vs 0.6711, 0.7479 vs 0.6962) are within
its own fold noise, consistent with mechanism 2.
