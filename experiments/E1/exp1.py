"""E1 — independent re-implementation of the over-credit ledger and fusion selection.

Re-derives the Ch. 7 experiments of Sudjianto, *The Learned Kernel* (2026) from the
methods section of the paper: three candidate kernels (chosen isotropic RBF, learned
ARD, supervised GBDT leaf kernel), each fit on a support fold; support-vs-query R^2
ledger; and fusion-weight selection by in-sample SURE vs held-out query R^2.

Written from the paper's equations, then cross-checked against the author's reference
implementation (notes/upstream/Learned-Kernel, MIT). Any divergence > 1e-2 is a bug
or a version note, whichever the investigation turns up.

Config block at top per ADR-002. Run via run_r003_crosscheck.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# --- config -------------------------------------------------------------------
N_SUP, N_QRY = 800, 800          # support / query fold sizes (upstream convention)
LAM = 1e-2                       # ridge for the per-channel ledger
RBF_ELL = 2.0                    # hand-chosen isotropic bandwidth
T_TREES, DEPTH, LR = 200, 5, 0.1 # the near-interpolating leaf kernel
LAM_GRID = np.logspace(-4, 1, 16) # ridge grid for fusion selection
SIMPLEX_RES = 6                  # fusion simplex grid resolution
N_THREADS = 4

SEED = 0


# --- data ---------------------------------------------------------------------
@dataclass
class Dataset:
    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray

    @property
    def n(self):
        return len(self.Xtr)


def load_california() -> Dataset:
    """California housing: 80/20 split at seed 0, scaler fit on train (upstream convention)."""
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        d.data, d.target, test_size=0.2, random_state=SEED)
    sc = StandardScaler().fit(Xtr_raw)
    return Dataset(sc.transform(Xtr_raw), sc.transform(Xte_raw), ytr, yte)


def load_taiwan() -> Dataset:
    """Taiwan credit default, vendored npz from the reference repo (Yeh & Lien 2009, UCI 350)."""
    path = _taiwan_npz_path()
    z = np.load(path)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        np.asarray(z["X"], float), np.asarray(z["y"], float),
        test_size=0.2, random_state=SEED)
    sc = StandardScaler().fit(Xtr_raw)
    return Dataset(sc.transform(Xtr_raw), sc.transform(Xte_raw), ytr, yte)


def _taiwan_npz_path():
    """Prefer the vendored copy in this repository; fall back to lkbook."""
    local = Path(__file__).resolve().parents[2] / "data" / "taiwan_credit.npz"
    if local.exists():
        return local
    from importlib import resources
    return resources.files("lkbook").joinpath("_data/taiwan_credit.npz")


def support_query_split(d: Dataset, seed=SEED):
    """Draw n_sup + n_qry rows from the train fold; first n_sup are support, rest query."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, N_SUP + N_QRY, replace=False)
    return d.Xtr[idx[:N_SUP]], d.ytr[idx[:N_SUP]], d.Xtr[idx[N_SUP:]], d.ytr[idx[N_SUP:]]


# --- kernels (each built from the SUPPORT fold only) --------------------------
def iso_rbf(A, B, ell=RBF_ELL):
    """Isotropic RBF, unit diagonal: exp(-||a-b||^2 / (2 ell^2))."""
    d2 = np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)
    return np.exp(-d2 / (2.0 * ell * ell))


def ard_gram(A, B, ell_vec):
    """ARD RBF: exp(-1/2 * sum_j (a_j - b_j)^2 / ell_j^2). Unit diagonal by construction."""
    Aw, Bw = A / ell_vec, B / ell_vec
    d2 = np.maximum((Aw * Aw).sum(1)[:, None] + (Bw * Bw).sum(1)[None, :] - 2 * Aw @ Bw.T, 0.0)
    return np.exp(-0.5 * d2)


def fit_ard_ell(Xs, ys, Xq, yq, lam=LAM, maxiter=20):
    """Per-feature ARD length scales minimizing held-out QUERY MSE (the Ch. 3 leakage-free
    rule: the length scales are fit by scoring the query fold, so the resulting kernel is
    fixed relative to the support labels it will be graded on)."""
    ybar = ys.mean()
    n = len(ys)

    def objective(log_ell):
        K = ard_gram(Xs, Xs, np.exp(log_ell))
        a = np.linalg.solve(K + lam * np.eye(n), ys - ybar)
        return float(np.mean((ard_gram(Xq, Xs, np.exp(log_ell)) @ a + ybar - yq) ** 2))

    res = minimize(objective, np.zeros(Xs.shape[1]), method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return np.exp(res.x)


class LeafKernel:
    """Leaf co-membership kernel of a fitted GBDT ensemble:
    k(x, x') = (1/T) * sum_t 1{leaf_t(x) = leaf_t(x')} = (1/T) psi(x)^T psi(x'),
    psi the stacked one-hot leaf indicators. Unit diagonal, PSD by construction."""

    def __init__(self, model):
        self.model = model
        self.trees = [est[0].tree_ for est in model.estimators_]  # GBR: (T, 1)
        self.T = len(self.trees)
        self.col_of, off = [], 0
        for tr in self.trees:
            leaves = np.where(tr.children_left == -1)[0]
            self.col_of.append({int(l): off + i for i, l in enumerate(leaves)})
            off += len(leaves)
        self.n_cols = off

    def transform(self, X):
        L = self.model.apply(X)
        L = L[:, :, 0] if L.ndim == 3 else L
        n = L.shape[0]
        cols = np.array([[self.col_of[t][int(L[i, t])] for t in range(self.T)]
                         for i in range(n)])
        rows = np.repeat(np.arange(n), self.T)
        return sp.csr_matrix((np.ones(n * self.T), (rows, cols.ravel())),
                             shape=(n, self.n_cols))

    def gram(self, A, B):
        return (self.transform(A) @ self.transform(B).T).toarray() / self.T


def build_candidates(d: Dataset, seed=SEED):
    """Fit the three candidate kernels on the support fold; return their Gram blocks
    (support, query, test) plus the fold targets. Everything downstream consumes this."""
    Xs, ys, Xq, yq = support_query_split(d, seed=seed)
    ard_ell = fit_ard_ell(Xs, ys, Xq, yq)
    gbm = GradientBoostingRegressor(n_estimators=T_TREES, max_depth=DEPTH,
                                    learning_rate=LR, random_state=seed).fit(Xs, ys)
    lk = LeafKernel(gbm)

    kfns = [
        ("RBF (chosen)", lambda A, B: iso_rbf(A, B)),
        ("ARD (learned)", lambda A, B: ard_gram(A, B, ard_ell)),
        ("leaf / tree", lambda A, B: lk.gram(A, B)),
    ]
    return dict(
        names=[n for n, _ in kfns],
        Bss=[k(Xs, Xs) for _, k in kfns],
        Bqs=[k(Xq, Xs) for _, k in kfns],
        Bts=[k(d.Xte, Xs) for _, k in kfns],
        ys=ys, yq=yq, yte=d.yte, ybar=float(ys.mean()),
    )


# --- metrics ------------------------------------------------------------------
def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2(pred, y, ybar):
    return float(1.0 - np.sum((pred - y) ** 2) / np.sum((y - ybar) ** 2))


# --- E1a: per-channel support-vs-query credit ---------------------------------
def per_channel_credit(d: Dataset, seed=SEED, lam=LAM):
    """Ledger: for each candidate kernel alone, alpha = (Kss + lam I)^-1 y_S; report
    support R^2 (in-sample), query R^2 (held out), their gap, and test RMSE."""
    with threadpool_limits(N_THREADS):
        c = build_candidates(d, seed=seed)
        n, ybar = len(c["ys"]), c["ybar"]
        yc = c["ys"] - ybar
        rows = []
        for name, Kss, Kqs, Kts in zip(c["names"], c["Bss"], c["Bqs"], c["Bts"]):
            a = np.linalg.solve(Kss + lam * np.eye(n), yc)
            sup = r2(Kss @ a + ybar, c["ys"], ybar)
            qry = r2(Kqs @ a + ybar, c["yq"], ybar)
            rows.append(dict(name=name, support_r2=sup, query_r2=qry,
                             gap=sup - qry, test_rmse=rmse(Kts @ a + ybar, c["yte"])))
    return rows


# --- E1b: in-sample (SURE) vs query-fold fusion selection ---------------------
def _simplex_grid(C, res):
    def comps(total, parts):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for r in comps(total - i, parts - 1):
                yield (i,) + r
    for comp in comps(res, C):
        yield np.asarray(comp, dtype=float) / res


def _mix(blocks, w):
    return sum(wi * b for wi, b in zip(w, blocks))


def select_in_sample_vs_query(d: Dataset, seed=SEED):
    """Fuse the three channels (convex mixture) and select weights x ridge two ways:
    QUERY by held-out query R^2 (leakage-free); IN-SAMPLE by SURE on the support with
    sigma^2 estimated at the GCV optimum. SURE sees the smoother's nominal df only —
    it cannot count the degrees of freedom boosting spent finding the leaf partition,
    which is exactly the over-credit the query fold exposes."""
    with threadpool_limits(N_THREADS):
        c = build_candidates(d, seed=seed)
        names, Bss, Bqs, Bts = c["names"], c["Bss"], c["Bqs"], c["Bts"]
        ys, yq, yte, ybar = c["ys"], c["yq"], c["yte"], c["ybar"]
        n, yc, C = len(ys), ys - ybar, len(names)
        cands = list(_simplex_grid(C, SIMPLEX_RES))

        best_q = (-np.inf, None, None)
        best_gcv = (np.inf, None)
        cache = []
        for w in cands:
            Kss = _mix(Bss, w)
            th, V = np.linalg.eigh(Kss)
            th = np.clip(th, 0.0, None)
            Vtyc = V.T @ yc
            c2 = Vtyc ** 2
            M = _mix(Bqs, w) @ V
            cache.append((w, th, c2))
            for lam in LAM_GRID:
                coef = Vtyc / (th + lam)
                r2q = r2(M @ coef + ybar, yq, ybar)
                if r2q > best_q[0]:
                    best_q = (r2q, w, float(lam))
                df = float(np.sum(th / (th + lam)))
                rss = float(np.sum((lam / (th + lam)) ** 2 * c2))
                gcv = (rss / n) / ((1.0 - df / n) ** 2 + 1e-12)
                if gcv < best_gcv[0]:
                    best_gcv = (gcv, (rss, df))

        rss_g, df_g = best_gcv[1]
        sigma2 = max(rss_g / max(n - df_g, 1.0), 1e-6)
        best_s = (np.inf, None, None)
        for w, th, c2 in cache:
            for lam in LAM_GRID:
                df = float(np.sum(th / (th + lam)))
                rss = float(np.sum((lam / (th + lam)) ** 2 * c2))
                sure = rss / n + 2.0 * sigma2 * df / n - sigma2
                if sure < best_s[0]:
                    best_s = (sure, w, float(lam))

        def test_rmse(w, lam):
            Kss = _mix(Bss, w)
            th, V = np.linalg.eigh(Kss)
            th = np.clip(th, 0.0, None)
            alpha = V @ ((V.T @ yc) / (th + lam))
            return rmse(_mix(Bts, w) @ alpha + ybar, yte)

        wq, lamq = best_q[1], best_q[2]
        ws, lams = best_s[1], best_s[2]
    return dict(
        names=names, sigma2=sigma2,
        query={"weights": dict(zip(names, wq)), "lam": lamq,
               "test_rmse": test_rmse(wq, lamq)},
        in_sample={"weights": dict(zip(names, ws)), "lam": lams,
                   "test_rmse": test_rmse(ws, lams)},
    )
