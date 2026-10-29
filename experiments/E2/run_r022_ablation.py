"""R022 — ablations for the two reviewer objections to tau-df.

1. Frozen-kernel bootstrap: holding K fixed makes the covariance penalty equal to
   tr S in expectation, so tau-df must be zero. This driver measures the frozen
   variant next to the paper's full-pipeline refit.
2. Noise-variance collapse: a residual variance RSS/(n - tr S) tends to zero for
   interpolating fits. This driver records it next to the GCV estimate used in the
   paper, and recomputes tau-df with the residual estimate as a sensitivity check.

Also reports tau-df over seeds 0-2 on both real support folds (the paper's seed-0
numbers plus the seed range quoted in the revision).

Writes experiments/E2/results/r022_ablation.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.E1.exp1 import (  # noqa: E402
    load_california, load_taiwan, support_query_split,
    fit_ard_ell, iso_rbf, ard_gram, LeafKernel, GradientBoostingRegressor,
    T_TREES, DEPTH, LR)
from experiments.E2.exp2 import smoother, _sigma2_hat, tau_df  # noqa: E402

OUT = Path(__file__).parent / "results"
LAM = 1e-2
N_BOOT = 15


def channel_builders(d, seed):
    """Same three channels as M3: RBF fixed, ARD fit on the query fold, leaf fit
    on the support labels."""
    Xs, ys, Xq, yq = support_query_split(d, seed=seed)
    ard_ell = fit_ard_ell(Xs, ys, Xq, yq)
    return Xs, ys, {
        "RBF (fixed)": lambda X, y: iso_rbf(X, X),
        "ARD (learned)": lambda X, y: ard_gram(X, X, ard_ell),
        "leaf (labeled)": lambda X, y: LeafKernel(
            GradientBoostingRegressor(
                n_estimators=T_TREES, max_depth=DEPTH, learning_rate=LR,
                random_state=seed).fit(X, y)).gram(X, X),
    }


def tau_df_with_sigma(build, X, x, lam, sigma2, n_boot=N_BOOT, seed=2):
    """tau-df with an externally supplied noise variance (sensitivity check)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    K0 = build(X, x)
    S0 = smoother(K0, lam)
    tr_S0 = float(np.trace(S0))
    fit0 = S0 @ x
    Ys, Yhats = [], []
    for _ in range(n_boot):
        y_b = fit0 + rng.randn(n) * np.sqrt(max(sigma2, 0.0))
        Yhats.append(smoother(build(X, y_b), lam) @ y_b)
        Ys.append(y_b)
    Y = np.asarray(Ys)
    Yhat = np.asarray(Yhats)
    Yc = Y - Y.mean(0, keepdims=True)
    Yhc = Yhat - Yhat.mean(0, keepdims=True)
    cov_sum = float((Yc * Yhc).sum() / (n_boot - 1))
    df_boot = cov_sum / max(sigma2, 1e-12)
    return float(df_boot - tr_S0), float(df_boot), tr_S0


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    for dname, loader in [("california", load_california), ("taiwan", load_taiwan)]:
        d = loader()
        rows = []
        for seed in (0, 1, 2):
            Xs, ys, builders = channel_builders(d, seed)
            for name, build in builders.items():
                K0 = build(Xs, ys)
                S0 = smoother(K0, LAM)
                tr_S0 = float(np.trace(S0))
                s2_gcv = _sigma2_hat(K0, ys)
                rss = float(np.sum((ys - S0 @ ys) ** 2))
                s2_res = rss / max(len(ys) - tr_S0, 1.0)

                # paper's statistic: full-pipeline refit
                tau, df_boot, tr_s = tau_df(build, Xs, ys, lam=LAM, n_boot=N_BOOT)
                # ablation 1: frozen kernel (build ignores y)
                tau_frozen, df_frozen, _ = tau_df(
                    lambda X, y: K0, Xs, ys, lam=LAM, n_boot=N_BOOT)
                # ablation 2: full refit, residual-variance normalization
                tau_res, df_res, _ = tau_df_with_sigma(
                    build, Xs, ys, LAM, s2_res)

                rows.append(dict(
                    dataset=dname, seed=seed, channel=name,
                    tau_df=tau, df_boot=df_boot, tr_S=tr_s,
                    tau_frozen=tau_frozen, df_boot_frozen=df_frozen,
                    sigma2_gcv=s2_gcv, sigma2_resid=s2_res, rss=rss,
                    tau_df_resid_norm=tau_res, df_boot_resid_norm=df_res))
                print(f"{dname:10s} s{seed} {name:15s} tau={tau:8.2f} "
                      f"frozen={tau_frozen:7.2f} resid_norm={tau_res:8.2f} "
                      f"trS={tr_s:7.1f} s2gcv={s2_gcv:.3e} s2res={s2_res:.3e}",
                      flush=True)
        report[dname] = rows
    (OUT / "r022_ablation.json").write_text(json.dumps(report, indent=2))
    print(f"\nresults: {OUT / 'r022_ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
