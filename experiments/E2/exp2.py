"""E2 — the SURE-failure predictability statistic τ (the paper's novelty, M2 gate).

Theory frame. For ANY prediction procedure ŷ(x), Efron's covariance penalty gives the
in-sample optimism exactly: optimism = (2/p) Σ_i cov(y_i, ŷ_i). For a *fixed linear
smoother* ŷ = S y, cov(y_i, ŷ_i) = σ² S_ii, so the penalty is (2σ²/p) tr S and SURE is
exact. A label-fitted kernel (GBDT leaf kernel) makes the full procedure ŷ = P(y) —
kernel construction THEN smoothing — nonlinear in y. SURE still charges only the
smoothing stage (tr S of the final smoother) and never the construction stage, so it
undercounts. The gap

    τ-df  :=  df̂_boot − tr S₀,      df̂_boot = (1/σ̂²) Σ_i cov̂_boot(y_i, ŷ_i)

estimates exactly the degrees of freedom the construction stage consumed — via a
parametric label-noise bootstrap (refit the whole procedure on x + ε_b, predict at the
same X, take the sample covariance against y_b). Two cheaper companions:

  τ-boot — mean relative change of the smoother ACTION under the same bootstrap:
           ‖S_b x − S₀ x‖ / ‖S₀ x‖. Fixed kernel → identically 0.
  τ-cov  — sign-flip variant of τ-boot (screened, expected to drop).

M2 gate: Spearman ρ(τ, |SURE − true risk|) ≥ 0.8 across settings spanning the fixed
regime (RBF sweep — SURE valid) and the label-fitted regime (GBDT leaf kernel — SURE
invalid), plus the crucial CONTROL: a leaf kernel fit on *other*, independent labels
is fixed w.r.t. the labels being denoised — SURE must be valid again, τ-df ≈ 0.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.E1.exp1 import LeafKernel, iso_rbf, N_THREADS  # noqa: E402

# --- config -------------------------------------------------------------------
SIGMA = 1.0            # synthetic noise sd (known — SURE may use it)
T_TREES, DEPTH, LR = 100, 4, 0.1   # lighter GBDT: 600-point synthetic, refit-heavy


# --- synthetic problem with known f --------------------------------------------
def make_synthetic(p=600, dim=6, sigma=SIGMA, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(p, dim)
    f = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] * X[:, 3]
         + 0.3 * np.cos(2 * X[:, 4]))
    f = f - f.mean()
    x = f + rng.randn(p) * sigma
    return X, f, x, sigma ** 2


# --- smoother ------------------------------------------------------------------
def smoother(K, lam):
    """Linear smoother S = K (K + lam I)^-1, eigendecomposition form."""
    th, V = np.linalg.eigh(K)
    th = np.clip(th, 0.0, None)
    return (V * (th / (th + lam))) @ V.T


# --- kernel builders: the fixed regime vs the label-fitted regime --------------
def rbf_builder(ell):
    def build(X, y):
        return iso_rbf(X, X, ell)
    return build


def leaf_builder(depth=DEPTH, n_trees=T_TREES, seed=0):
    """Label-fitted: the GBDT partition consumes the labels it is scored on.
    Depth is the hidden-df dose axis for the M2 dose-response design."""
    def build(X, y):
        m = GradientBoostingRegressor(n_estimators=n_trees, max_depth=depth,
                                      learning_rate=LR, random_state=seed).fit(X, y)
        return LeafKernel(m).gram(X, X)
    return build


def fixed_leaf_builder(y_other, seed=0):
    """CONTROL: leaf kernel fit on independent labels y_other — fixed w.r.t. the
    labels that will be denoised through it."""
    def build(X, y):
        m = GradientBoostingRegressor(n_estimators=T_TREES, max_depth=DEPTH,
                                      learning_rate=LR, random_state=seed).fit(X, y_other)
        return LeafKernel(m).gram(X, X)
    return build


# --- per-setting evaluation: true risk, SURE, in-sample ------------------------
def evaluate(build, X, f, x0, sigma2, lams, n_draws=10, seed=1):
    """Monte-Carlo the denoising risk of the full procedure across noise draws,
    alongside SURE (computed per draw with the draw's own smoother) and the in-sample
    residual. Averaged over draws; returns arrays over the lam grid."""
    rng = np.random.RandomState(seed)
    p = len(f)
    true = np.zeros((n_draws, len(lams)))
    sure = np.zeros_like(true)
    insample = np.zeros_like(true)
    for d in range(n_draws):
        y = f + rng.randn(p) * np.sqrt(sigma2)
        K = build(X, y)                      # the FULL procedure: construction + smoothing
        th, V = np.linalg.eigh(K)
        th = np.clip(th, 0.0, None)
        ybar = y.mean()
        Vty = V.T @ (y - ybar)
        for li, lam in enumerate(lams):
            g = th / (th + lam)
            pred = ybar + V @ (g * Vty)
            true[d, li] = float(np.mean((pred - f) ** 2))
            insample[d, li] = float(np.mean(((y - ybar) - V @ (g * Vty)) ** 2))
            df = float(np.sum(g))
            sure[d, li] = insample[d, li] + 2.0 * sigma2 * df / p - sigma2
    return dict(true=true.mean(0), sure=sure.mean(0), insample=insample.mean(0),
                lams=np.asarray(lams))


# --- the tau statistics ----------------------------------------------------------
def _sigma2_hat(K, y, lam=1e-2):
    """Leakage-free-ish noise estimate from a mild ridge on the given kernel (GCV)."""
    n = len(y)
    th, V = np.linalg.eigh(K)
    th = np.clip(th, 0.0, None)
    Vty = V.T @ (y - y.mean())
    best, best_val = np.inf, None
    for lam_g in np.logspace(-4, 1, 16):
        df = float(np.sum(th / (th + lam_g)))
        rss = float(np.sum((lam_g / (th + lam_g)) ** 2 * Vty ** 2))
        gcv = (rss / n) / ((1.0 - df / n) ** 2 + 1e-12)
        if gcv < best:
            best, best_val = gcv, rss / max(n - df, 1.0)
    return max(best_val, 1e-6)


def tau_df(build, X, x, lam, n_boot=20, seed=2):
    """τ-df: bootstrap covariance-penalty df minus nominal tr S. Both the bootstrap
    refits and tr S are evaluated at the SAME lam so the gap isolates the
    construction stage's hidden df. Bootstrap = *residual* bootstrap: refit labels
    are ŷ₀ + ε_b (smoothed fit + fresh noise), matching how the y-b variation enters
    each covariance term."""
    rng = np.random.RandomState(seed)
    n = len(x)
    K0 = build(X, x)
    sigma2h = _sigma2_hat(K0, x)
    S0 = smoother(K0, lam)
    tr_S0 = float(np.trace(S0))
    fit0 = S0 @ x
    Ys, Yhats = [], []
    for _ in range(n_boot):
        y_b = fit0 + rng.randn(n) * np.sqrt(sigma2h)
        K_b = build(X, y_b)
        Yhats.append(smoother(K_b, lam) @ y_b)
        Ys.append(y_b)
    Y = np.asarray(Ys)          # (B, n)
    Yhat = np.asarray(Yhats)
    Yc = Y - Y.mean(0, keepdims=True)
    Yhc = Yhat - Yhat.mean(0, keepdims=True)
    # df = (1/sigma^2) * SUM_i cov_b(y_i, yhat_i): sum over coordinates, not mean
    cov_sum = float((Yc * Yhc).sum() / (n_boot - 1))   # sum_i cov_b(y_i, yhat_i)
    df_boot = float(cov_sum / sigma2h)
    return float(df_boot - tr_S0), df_boot, tr_S0


def tau_boot(build, X, x, lam, n_boot=20, seed=2):
    """τ-boot: mean relative Frobenius change of the smoother ACTION under label-noise
    bootstrap refits (residual bootstrap around the original fit). A fixed kernel
    refits identically → exactly 0."""
    rng = np.random.RandomState(seed)
    n = len(x)
    K0 = build(X, x)
    S0 = smoother(K0, lam)
    sigma2h = _sigma2_hat(K0, x)
    fit0 = S0 @ x
    denom = np.linalg.norm(fit0)
    rel = []
    for _ in range(n_boot):
        y_b = fit0 + rng.randn(n) * np.sqrt(sigma2h)
        S_b = smoother(build(X, y_b), lam)
        rel.append(float(np.linalg.norm(S_b @ x - fit0) / max(denom, 1e-12)))
    return float(np.mean(rel)), float(np.std(rel))


# --- the M2 screen ---------------------------------------------------------------
def m2_screen(n_draws=10, n_boot=20):
    """Screen all settings × statistics; return the Spearman table and per-setting
    detail. Design (revised after gate run #1, see m2_gate_notes.md):
      - 4 fixed RBFs = SURE positive controls (constant SURE error floor),
      - 1 fixed-leaf control (construction on independent labels → fixed w.r.t. y),
      - 5 label-fitted leafs at depth 1..6 = the DOSE axis (monotone hidden-df).
    The gate criteria are separation, control-not-flagged, and monotone dose-response
    (Spearman over the DOSE settings only, where |SURE−true| actually varies); plain
    rank correlation over all settings is ill-posed when 5/6 tie at the error floor."""
    lams = np.logspace(-3, 2, 12)
    X, f, x0, sigma2 = make_synthetic(seed=0)
    rng = np.random.RandomState(7)
    y_other = f + rng.randn(len(f)) * np.sqrt(sigma2)   # independent labels for the control

    settings = [
        ("RBF ell=0.25 (fixed)", rbf_builder(0.25)),
        ("RBF ell=0.5 (fixed)", rbf_builder(0.5)),
        ("RBF ell=1.0 (fixed)", rbf_builder(1.0)),
        ("RBF ell=2.0 (fixed)", rbf_builder(2.0)),
        ("leaf CONTROL (other labels)", fixed_leaf_builder(y_other, seed=99)),
        ("leaf LABEL-FITTED depth=1", leaf_builder(depth=1, seed=0)),
        ("leaf LABEL-FITTED depth=2", leaf_builder(depth=2, seed=0)),
        ("leaf LABEL-FITTED depth=3", leaf_builder(depth=3, seed=0)),
        ("leaf LABEL-FITTED depth=4", leaf_builder(depth=4, seed=0)),
        ("leaf LABEL-FITTED depth=6", leaf_builder(depth=6, seed=0)),
    ]
    rows, curves = [], {}
    with threadpool_limits(N_THREADS):
        for name, build in settings:
            ev = evaluate(build, X, f, x0, sigma2, lams, n_draws=n_draws)
            err = np.abs(ev["sure"] - ev["true"])             # SURE's error over the sweep
            td, dfb, trs = tau_df(build, X, x0, lam=1e-2, n_boot=n_boot)
            tb, tbs = tau_boot(build, X, x0, lam=1e-2, n_boot=n_boot)
            rows.append(dict(setting=name, tau_df=td, df_boot=dfb, tr_S=trs,
                             tau_boot=tb, sure_err_mean=float(err.mean()),
                             sure_err_max=float(err.max()),
                             corr_sure=float(np.corrcoef(ev["sure"], ev["true"])[0, 1])))
            curves[name] = ev
            print(f"  {name:32s} τ-df={td:8.1f}  τ-boot={tb:.3f}  "
                  f"|SURE−true| mean={err.mean():7.3f}  corr={rows[-1]['corr_sure']:.3f}",
                  flush=True)

    out = {"rows": rows, "curves": {k: {kk: v.tolist() for kk, v in c.items()}
                                    for k, c in curves.items()},
           "lams": lams.tolist()}
    # plain rho over all settings (kept for the record; ill-posed under ties)
    for tau in ("tau_df", "tau_boot"):
        rho, pval = spearmanr([r[tau] for r in rows], [r["sure_err_mean"] for r in rows])
        out[f"spearman_{tau}_meanerr"] = {"rho": float(rho), "p": float(pval)}
        print(f"  Spearman ρ({tau}, |SURE−true| mean, all settings) = {rho:.3f}")
    # dose-response rho: over the label-fitted depths only, where error varies
    dose = [r for r in rows if "depth" in r["setting"]]
    for tau in ("tau_df", "tau_boot"):
        rho_d, p_d = spearmanr([r[tau] for r in dose], [r["sure_err_mean"] for r in dose])
        out[f"spearman_{tau}_dose"] = {"rho": float(rho_d), "p": float(p_d)}
        print(f"  Spearman ρ({tau}, |SURE−true| mean, dose axis) = {rho_d:.3f} (p={p_d:.2g})")
    return out
