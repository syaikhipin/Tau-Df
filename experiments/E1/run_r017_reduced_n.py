"""R017 — reduced-n robustness: support/query splits at 25% and 50% of the standard
800/800 size, on California. Question: do the B1/B2 conclusions (leaf gap largest;
query selector beats SURE) survive at smaller n, where the selection-cost term
sqrt(c(Theta)/n) is larger?

Writes experiments/E1/results/r017_reduced_n.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.E1.exp1 as exp1  # noqa: E402
from experiments.E1.exp1 import load_california, r2, rmse  # noqa: E402

OUT = Path(__file__).parent / "results"


def run_size(d, n_sup, n_qry, seed):
    """Ledger + fusion at a custom split size (monkeypatched module constants)."""
    old = exp1.N_SUP, exp1.N_QRY
    exp1.N_SUP, exp1.N_QRY = n_sup, n_qry
    try:
        led = exp1.per_channel_credit(d, seed=seed)
        fus = exp1.select_in_sample_vs_query(d, seed=seed)
    finally:
        exp1.N_SUP, exp1.N_QRY = old
    return led, fus


def main():
    OUT.mkdir(exist_ok=True)
    d = load_california()
    report = {}
    for label, (n_sup, n_qry) in [("25pct", (200, 200)), ("50pct", (400, 400)),
                                  ("100pct", (800, 800))]:
        print(f"\n=== support/query = {n_sup}/{n_qry} ({label}) ===")
        cells = []
        for seed in (0, 1, 2):
            led, fus = run_size(d, n_sup, n_qry, seed)
            gaps = {r["name"]: r["gap"] for r in led}
            leaf_top = max(gaps, key=gaps.get)
            rm_in = fus["in_sample"]["test_rmse"]
            rm_q = fus["query"]["test_rmse"]
            cells.append(dict(
                seed=seed, gaps=gaps, leaf_is_top=bool("leaf" in leaf_top),
                leaf_weight_sure=fus["in_sample"]["weights"],
                leaf_weight_query=fus["query"]["weights"],
                rmse_insample=rm_in, rmse_query=rm_q,
                query_beats=bool(rm_q < rm_in)))
            print(f"  seed {seed}: leaf top-gap={cells[-1]['leaf_is_top']}  "
                  f"query RMSE {rm_q:.4f} vs SURE {rm_in:.4f} ({'wins' if rm_q < rm_in else 'loses'})")
        report[label] = dict(
            n_sup=n_sup, n_qry=n_qry, cells=cells,
            leaf_top_rate=float(np.mean([c["leaf_is_top"] for c in cells])),
            query_wins_rate=float(np.mean([c["query_beats"] for c in cells])))

    checks = {
        "leaf_top_gap_all_cells": all(report[k]["leaf_top_rate"] == 1.0
                                      for k in report),
        "query_selector_wins_all_cells": all(report[k]["query_wins_rate"] == 1.0
                                             for k in report),
    }
    print("\n--- R017 checks ---")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FLAG'}  {k}")
    report["checks"] = checks
    (OUT / "r017_reduced_n.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"results: {OUT / 'r017_reduced_n.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
