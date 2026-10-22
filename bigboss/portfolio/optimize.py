"""Portfolio construction.

Core objective (horizon h, all quantities in h-day units):
    maximize   w' mu_excess  -  c' |w - w0|  -  lambda * Risk(w)
    subject to 0 <= w_i <= cap_i, sector caps, sum(w) <= 1 (rest in cash), turnover <= T
where mu_excess are forecast benchmark-relative returns, c are one-way trading costs, and Risk is
variance (w' Sigma_h w), CVaR, or a drawdown proxy. The benchmark itself can be a candidate asset with
mu = 0 so that "just own the index" competes directly with every active idea.

Other constructors: risk parity, hierarchical risk parity (HRP), Black-Litterman posterior returns,
CVaR (Rockafellar-Uryasev LP), and a robust variant that shrinks mu by its estimation uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform


@dataclass
class OptResult:
    weights: pd.Series
    method: str
    objective: float
    expected_excess: float
    risk: float
    est_cost: float
    turnover: float
    success: bool
    message: str = ""

    def as_dict(self):
        return {"weights": {k: float(v) for k, v in self.weights.items() if abs(v) > 1e-6}, "method": self.method,
                "objective": self.objective, "expected_excess": self.expected_excess, "risk": self.risk,
                "est_cost": self.est_cost, "turnover": self.turnover, "success": self.success, "message": self.message}


def _sector_matrix(assets: list[str], sectors: dict[str, str], exempt: set[str]) -> tuple[np.ndarray, list[str]]:
    secs = sorted({sectors.get(a, "Unknown") for a in assets if a not in exempt} - {"Unknown"})
    M = np.array([[1.0 if sectors.get(a) == s and a not in exempt else 0.0 for a in assets] for s in secs]) if secs else np.zeros((0, len(assets)))
    return M, secs


def mean_variance(mu: pd.Series, cov: pd.DataFrame, w0: pd.Series | None = None, cost: pd.Series | None = None,
                  risk_aversion: float = 4.0, max_weight: float | pd.Series = 0.1, sectors: dict[str, str] | None = None,
                  max_sector: float = 0.3, max_turnover: float | None = None, budget: float = 1.0,
                  exempt: set[str] | None = None, min_invested: float = 0.0, w_ref: pd.Series | None = None) -> OptResult:
    """Cost-aware long-only mean-variance with linear trading costs via buy/sell split variables.

    If `w_ref` (e.g. 100% in the benchmark fund) is given, risk is ACTIVE risk (w - w_ref)' S (w - w_ref),
    i.e. tracking-error variance, which is the risk consistent with a benchmark-relative objective.
    """
    assets = list(mu.index)
    n = len(assets)
    S = cov.reindex(index=assets, columns=assets).astype(float).fillna(0).to_numpy()
    S = (S + S.T) / 2 + 1e-10 * np.eye(n)
    m = mu.to_numpy(float)
    w0v = (w0.reindex(assets).astype(float).fillna(0) if w0 is not None else pd.Series(0.0, index=assets)).to_numpy(float)
    c = (cost.reindex(assets).astype(float).fillna(0) if cost is not None else pd.Series(0.0, index=assets)).to_numpy(float)
    exempt = exempt or set()
    caps = (max_weight.reindex(assets).astype(float).fillna(0).to_numpy(float) if isinstance(max_weight, pd.Series)
            else np.array([1.0 if a in exempt else max_weight for a in assets]))
    ref = w_ref.reindex(assets).astype(float).fillna(0).to_numpy(float) if w_ref is not None else np.zeros(n)

    # variables x = [b (buys), s (sells)], w = w0 + b - s
    def unpack(x):
        return w0v + x[:n] - x[n:]

    def obj(x):
        w = unpack(x)
        a = w - ref
        return -(m @ w - c @ (x[:n] + x[n:]) - risk_aversion * a @ S @ a)

    def grad(x):
        w = unpack(x)
        gw = -(m - 2 * risk_aversion * S @ (w - ref))
        return np.concatenate([gw + c, -gw + c])

    cons = [{"type": "ineq", "fun": lambda x: budget - unpack(x).sum(), "jac": lambda x: -np.concatenate([np.ones(n), -np.ones(n)])}]
    if min_invested > 0:
        cons.append({"type": "ineq", "fun": lambda x: unpack(x).sum() - min_invested,
                     "jac": lambda x: np.concatenate([np.ones(n), -np.ones(n)])})
    for i in range(n):
        e = np.zeros(2 * n); e[i], e[n + i] = 1, -1
        cons.append({"type": "ineq", "fun": (lambda x, i=i: unpack(x)[i]), "jac": (lambda x, e=e: e)})
        cons.append({"type": "ineq", "fun": (lambda x, i=i: caps[i] - unpack(x)[i]), "jac": (lambda x, e=e: -e)})
    if sectors:
        M, _ = _sector_matrix(assets, sectors, exempt)
        for row in M:
            J = np.concatenate([row, -row])
            cons.append({"type": "ineq", "fun": (lambda x, row=row: max_sector - row @ unpack(x)), "jac": (lambda x, J=J: -J)})
    if max_turnover is not None:
        cons.append({"type": "ineq", "fun": lambda x: max_turnover - x.sum(), "jac": lambda x: -np.ones(2 * n)})
    x0 = np.zeros(2 * n)
    res = minimize(obj, x0, jac=grad, bounds=[(0, None)] * (2 * n), constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(unpack(res.x), 0, None)
    w[w < 1e-5] = 0
    if w.sum() > budget:
        w = w / w.sum() * budget
    traded = np.abs(w - w0v)
    a = w - ref
    return OptResult(pd.Series(w, index=assets), "mean_variance", float(-res.fun), float(m @ w), float(a @ S @ a),
                     float(c @ traded), float(traded.sum()), bool(res.success), str(res.message))


def risk_parity(cov: pd.DataFrame, budget: float = 1.0, risk_budget: np.ndarray | None = None) -> pd.Series:
    """Equal risk contribution weights via the convex program of Spinu (2013):
    minimize 0.5 y'Sy - sum(b_i log y_i), y > 0; then w = y / sum(y). The optimum satisfies y_i (Sy)_i = b_i."""
    S = cov.to_numpy()
    n = S.shape[0]
    b = np.ones(n) / n if risk_budget is None else np.asarray(risk_budget) / np.sum(risk_budget)
    y0 = 1 / np.sqrt(np.diag(S))
    res = minimize(lambda y: 0.5 * y @ S @ y - b @ np.log(y), y0, jac=lambda y: S @ y - b / y,
                   bounds=[(1e-12, None)] * n, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-12})
    w = res.x / res.x.sum()
    return pd.Series(w * budget, index=cov.index)


def hrp(cov: pd.DataFrame, budget: float = 1.0) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado 2016)."""
    sd = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "single"))
    items = list(cov.index[order])
    w = pd.Series(1.0, index=items)
    clusters = [items]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            a = 1 - v0 / (v0 + v1)
            w[c0] *= a
            w[c1] *= 1 - a
    return (w / w.sum() * budget).reindex(cov.index)


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    c = cov.loc[items, items].to_numpy()
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def black_litterman(cov: pd.DataFrame, w_mkt: pd.Series, views: pd.Series, view_var: pd.Series,
                    risk_aversion: float = 2.5, tau: float = 0.05) -> pd.Series:
    """Posterior expected returns with absolute views on single assets.

    pi = delta * Sigma * w_mkt (implied equilibrium), views Q with uncertainty Omega = diag(view_var).
    Model forecasts enter as views; their calibrated interval width sets Omega, so uncertain forecasts move
    the posterior less.
    """
    assets = list(cov.index)
    S = cov.to_numpy()
    pi = risk_aversion * S @ w_mkt.reindex(assets).fillna(0).to_numpy()
    v = views.dropna()
    v = v[v.index.isin(assets)]
    if v.empty:
        return pd.Series(pi, index=assets)
    P = np.array([[1.0 if a == s else 0.0 for a in assets] for s in v.index])
    Q = v.to_numpy()
    Om = np.diag(np.maximum(view_var.reindex(v.index).fillna(view_var.max()).to_numpy(), 1e-8))
    tS = tau * S
    A = np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(Om) @ P)
    post = A @ (np.linalg.inv(tS) @ pi + P.T @ np.linalg.inv(Om) @ Q)
    return pd.Series(post, index=assets)


def min_cvar(scenarios: pd.DataFrame, mu: pd.Series | None = None, alpha: float = 0.05, max_weight: float = 0.1,
             target_return: float | None = None, budget: float = 1.0, risk_aversion: float = 1.0,
             fully_invested: bool = True) -> OptResult:
    """Rockafellar-Uryasev LP: minimize CVaR_alpha of scenario losses (- lambda^-1 * expected return)."""
    R = scenarios.fillna(0).to_numpy()
    T, n = R.shape
    assets = list(scenarios.columns)
    m = (mu.reindex(assets).fillna(0).to_numpy() if mu is not None else R.mean(axis=0))
    # variables: w (n), zeta (1), u (T)
    c = np.concatenate([-m / max(risk_aversion, 1e-9), [1.0], np.ones(T) / (alpha * T)])
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    b_ub = np.zeros(T)
    A_ub = np.vstack([A_ub, np.concatenate([np.ones(n), [0], np.zeros(T)])])
    b_ub = np.append(b_ub, budget)
    if target_return is not None:
        A_ub = np.vstack([A_ub, np.concatenate([-m, [0], np.zeros(T)])])
        b_ub = np.append(b_ub, -target_return)
    bounds = [(0, max_weight if isinstance(max_weight, float) else 1.0)] * n + [(None, None)] + [(0, None)] * T
    if isinstance(max_weight, dict):
        bounds[:n] = [(0, max_weight.get(a, 1.0)) for a in assets]
    A_eq = b_eq = None
    if fully_invested:
        A_eq = np.concatenate([np.ones(n), [0], np.zeros(T)])[None, :]
        b_eq = np.array([budget])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return OptResult(pd.Series(0.0, index=assets), "min_cvar", float("nan"), 0, float("nan"), 0, 0, False, res.message)
    w = pd.Series(res.x[:n], index=assets)
    port = R @ w.to_numpy()
    q = np.quantile(port, alpha)
    cvar = float(-port[port <= q].mean())
    return OptResult(w, "min_cvar", float(res.fun), float(m @ w.to_numpy()), cvar, 0.0, 0.0, True, "")


def robust_mu(mu: pd.Series, mu_se: pd.Series, kappa: float = 1.0) -> pd.Series:
    """Worst-case-in-box expected return: shrink each mu toward zero by kappa standard errors (never flips sign)."""
    shrunk = np.sign(mu) * np.maximum(mu.abs() - kappa * mu_se.reindex(mu.index).fillna(mu.abs()), 0)
    return shrunk


def optimize(method: str, mu: pd.Series, cov: pd.DataFrame, **kw) -> OptResult:
    method = method.lower()
    if method in ("mean_variance", "mv"):
        return mean_variance(mu, cov, **kw)
    if method == "robust":
        se = kw.pop("mu_se")
        r = mean_variance(robust_mu(mu, se, kw.pop("kappa", 1.0)), cov, **kw)
        r.method = "robust_mean_variance"
        return r
    if method == "black_litterman":
        post = black_litterman(cov, kw.pop("w_mkt"), mu, kw.pop("view_var"), tau=kw.pop("tau", 0.05))
        r = mean_variance(post, cov, **kw)
        r.method = "black_litterman"
        return r
    if method in ("risk_parity", "hrp"):
        w = risk_parity(cov) if method == "risk_parity" else hrp(cov)
        cap = kw.get("max_weight", 1.0)
        if not isinstance(cap, pd.Series):
            w = _cap_and_redistribute(w, cap)
        S = cov.to_numpy()
        return OptResult(w, method, float("nan"), float(mu.reindex(w.index).fillna(0) @ w), float(w @ S @ w), 0.0, 0.0, True)
    if method == "min_cvar":
        return min_cvar(kw.pop("scenarios"), mu, max_weight=kw.get("max_weight", 0.1), risk_aversion=kw.get("risk_aversion", 1.0))
    raise ValueError(f"unknown optimizer {method}")


def _cap_and_redistribute(w: pd.Series, cap: float, iters: int = 50) -> pd.Series:
    w = w.copy()
    for _ in range(iters):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over & (w < cap)
        if not under.any():
            break
        w[under] += excess * w[under] / w[under].sum()
    return w
