"""M3 — the chosen τ statistics on real data (runs R013–R015).

Computes τ-df and τ-boot for the three E1 channels (RBF chosen, ARD learned, leaf
label-fitted) on California and Taiwan support folds, and compares the τ ranking
against the measured support-query gap ranking from E1.

Expectation if the hidden-df story holds: RBF and ARD ≈ 0 (their construction either
doesn't consume labels or its df is visible), leaf ≫ 0 on both datasets — and τ
ranking agrees with the gap ranking among label-fitted kernels.

Writes experiments/E2/results/m3_real_data.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.E1.exp1 import (  # noqa: E402
    build_candidates, load_california, load_taiwan, support_query_split,
    iso_rbf, ard_gram, LeafKernel, GradientBoostingRegressor, T_TREES, DEPTH, LR)
from experiments.E2.exp2 import tau_df, tau_boot  # noqa: E402

OUT = Path(__file__).parent / "results"


def channel_builders(d, seed=0):
    """Rebuild the E1 candidate kernels; return name -> build(X, y) closures that
    reconstruct each channel's Gram on (X, y). The leaf channel consumes y; RBF is
    label-free; ARD's length scales were fit on the query fold in E1 (leakage-free
    construction) — for τ we refit them inside the bootstrap, i.e. τ measures the
    full procedure as a practitioner would run it, minus the query fold (which the
    τ user wouldn't have)."""
    Xs, ys, Xq, yq = support_query_split(d, seed=seed)

    # re-run the E1 ARD fit once to get its length scales (fixed thereafter, as in E1)
    from experiments.E1.exp1 import fit_ard_ell
    ard_ell = fit_ard_ell(Xs, ys, Xq, yq)

    builders = {
        "RBF (chosen)": lambda X, y: iso_rbf(X, X),
        "ARD (learned)": lambda X, y: ard_gram(X, X, ard_ell),
        "leaf / tree": lambda X, y: LeafKernel(
            GradientBoostingRegressor(n_estimators=T_TREES, max_depth=DEPTH,
                                      learning_rate=LR, random_state=seed).fit(X, y)
        ).gram(X, X),
    }
    return Xs, ys, Xq, yq, builders


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    for dname, loader in [("california", load_california), ("taiwan", load_taiwan)]:
        d = loader()
        Xs, ys, Xq, yq, builders = channel_builders(d, seed=0)
        # measured gaps from E1 seed-0 ledger for the ranking comparison
        gaps = {r["name"]: r["gap"] for r in
                __import__("experiments.E1.exp1", fromlist=["per_channel_credit"])
                .per_channel_credit(d, seed=0)}
        rows = []
        print(f"\n[{dname}]")
        for name, build in builders.items():
            td, dfb, trs = tau_df(build, Xs, ys, lam=1e-2, n_boot=15)
            tb, tbs = tau_boot(build, Xs, ys, lam=1e-2, n_boot=15)
            gap = gaps.get(name, gaps.get(next((k for k in gaps if k.split(" (")[0] in name), None)))
            rows.append(dict(channel=name, tau_df=td, df_boot=dfb, tr_S=trs,
                             tau_boot=tb, tau_boot_std=tbs, measured_gap=gap))
            print(f"  {name:20s} τ-df={td:8.1f}  τ-boot={tb:.3f}±{tbs:.3f}  "
                  f"df_boot={dfb:7.1f}  tr S={trs:6.1f}  measured gap={gap:.3f}")
        report[dname] = rows

    # ranking comparisons (write into a separate dict — report is being read here)
    ranking = {}
    print("\n--- ranking: τ vs measured gap ---")
    for dname, rows in report.items():
        for tau in ("tau_df", "tau_boot"):
            tau_rank = {r["channel"]: r[tau] for r in rows}
            gap_rank = {r["channel"]: r["measured_gap"] for r in rows}
            leaf_top_tau = max(tau_rank, key=tau_rank.get)
            leaf_top_gap = max(gap_rank, key=gap_rank.get)
            agree = "leaf" in leaf_top_tau and "leaf" in leaf_top_gap
            print(f"  [{dname}] {tau}: top-τ={leaf_top_tau}, top-gap={leaf_top_gap}, "
                  f"agree={agree}")
            ranking[f"{dname}_{tau}_top_agrees"] = bool(agree)
    report["ranking"] = ranking

    (OUT / "m3_real_data.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nresults: {OUT / 'm3_real_data.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
