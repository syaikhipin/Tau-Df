# M3 — τ on real data (R013/R015): results and reading

## Numbers (seed 0, support folds; τ-boot over 15 residual-bootstrap refits)

| channel | CA τ-df | CA τ-boot | TW τ-df | TW τ-boot | CA gap | TW gap |
|---------|---------|-----------|---------|-----------|--------|--------|
| RBF (chosen) | −12.5 | 0.000 | 4.6 | 0.000 | 0.164 | 1.380 |
| ARD (learned) | −10.0 | 0.000 | 5.7 | 0.000 | 0.122 | 0.966 |
| leaf / tree | **72.3** | 0.008 | **66.8** | 0.022 | 0.281 | 0.987 |

California df detail for the leaf kernel: df_boot = 788.6 vs tr S = 716.3 — the
construction stage's hidden df, measured directly.

## The deliverable diagnostic (R015)
A τ-df threshold of 20 flags exactly the leaf kernel on both datasets, and exactly
the label-fitted settings on synthetic (fixed ≤ 3.6, control −3.8, label-fitted ≥ 34.6).
τ-boot at its 0-floor flags the same set even more starkly (0.000 vs ≥ 0.008), but
carries no magnitude information (see m2_gate_notes.md).

## The Taiwan "disagreement" is the paper's second finding
Top-gap on Taiwan is RBF (1.380) while top-τ is leaf. Not a failure of the
diagnostic — the gap conflates two mechanisms:

1. **Useless-kernel gap** (Taiwan RBF): Gram ≈ identity under 23 standardized
   heavy-tailed features → interpolates support, predicts the mean; huge gap, but
   tr S ≈ n is VISIBLE, so SURE prices it correctly. τ-df correctly stays ≈ 0 (4.6).
2. **Hidden-df over-credit** (leaf on both datasets): the construction stage
   consumes labels in a way tr S cannot see. τ-df catches exactly this (66.8–72.3).

So: gap ranking agrees with τ ranking on California (where all three kernels carry
signal), and *disagrees* on Taiwan precisely because the gap alone cannot tell
SURE-unsafe kernels from SURE-safe ones. τ-df can. This upgrades the paper's claim
from "τ predicts the gap" to "τ isolates the SURE-relevant component of the gap" —
with the Taiwan ledger as the separating example.
