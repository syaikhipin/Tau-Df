"""R018/R019 — capacity map (B4, appendix): free atoms vs convex banks.

FREE ATOMS axis: GBDT leaf-kernel capacity grows with tree depth (leaf count K);
train-test RMSE gap should grow (Cor. D.2, sqrt(K) selection term).
BANKS axis: convex (uniform) mixture of H RBF bandwidth banks; gap should stay
flat in H once a useful scale is in the bank (Cor. D.3, sqrt(log H)).

Writes experiments/E1/results/r018_capacity_map.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.E1.exp1 import (  # noqa: E402
    LeafKernel, iso_rbf, load_california, rmse, GradientBoostingRegressor, LR)
from experiments.E2.exp2 import smoother  # noqa: E402

OUT = Path(__file__).parent / "results"
N_THREADS = 4


def _mix(blocks, w):
    return sum(wi * b for wi, b in zip(w, blocks))


def main():
    OUT.mkdir(exist_ok=True)
    d = load_california()
    rng = np.random.RandomState(0)
    n = 800
    idx = rng.choice(d.n, n, replace=False)
    X, y = d.Xtr[idx], d.ytr[idx]
    ybar = y.mean()
    yc = y - ybar
    lam = 1e-2

    depths = (2, 3, 4, 6, 8, 10)
    Hs = (1, 2, 4, 8, 16, 32)

    with threadpool_limits(N_THREADS):
        # --- free atoms: leaf count grows with depth ---
        free = []
        for depth in depths:
            m = GradientBoostingRegressor(n_estimators=150, max_depth=depth,
                                          learning_rate=LR, random_state=0).fit(X, y)
            lk = LeafKernel(m)
            K = lk.gram(X, X)
            a = np.linalg.solve(K + lam * np.eye(n), yc)
            tr = rmse(K @ a + ybar, y)
            te = rmse(lk.gram(d.Xte, X) @ a + ybar, d.yte)
            free.append(dict(K=int(lk.n_cols), depth=int(depth),
                             train=tr, test=te, gap=te - tr))
            print(f"  free: depth={depth:2d} K={lk.n_cols:6d}  train={tr:.3f} "
                  f"test={te:.3f} gap={te - tr:.3f}", flush=True)

        # --- banks: H RBF bandwidths in a uniform convex mixture ---
        banks = []
        for H in Hs:
            ells = np.logspace(-0.5, 1.0, H)
            w = np.ones(H) / H
            Kss = _mix([iso_rbf(X, X, e) for e in ells], w)
            Kts = _mix([iso_rbf(d.Xte, X, e) for e in ells], w)
            a = np.linalg.solve(Kss + lam * np.eye(n), yc)
            tr = rmse(Kss @ a + ybar, y)
            te = rmse(Kts @ a + ybar, d.yte)
            banks.append(dict(H=int(H), train=tr, test=te, gap=te - tr))
            print(f"  banks: H={H:2d}  train={tr:.3f} test={te:.3f} gap={te - tr:.3f}",
                  flush=True)

    # shape checks: free-atom gap rises over its range; bank gap (H>=2) roughly flat
    gap_free = [r["gap"] for r in free]
    banks_ge2 = [r for r in banks if r["H"] >= 2]
    gap_bank = [r["gap"] for r in banks_ge2]
    checks = {
        "free_gap_rises": bool(gap_free[-1] > gap_free[0]),
        "bank_gap_range_small": bool(max(gap_bank) - min(gap_bank)
                                     < 0.5 * (gap_free[-1] - gap_free[0])),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FLAG'}  {k}")

    report = dict(free=free, banks=banks, checks=checks)
    (OUT / "r018_capacity_map.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"results: {OUT / 'r018_capacity_map.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
