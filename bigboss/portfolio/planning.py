"""Goal-based planning: probability distributions of outcomes, never guarantees.

Monte Carlo uses a stationary block bootstrap of the proposed portfolio's HISTORICAL daily returns (so fat
tails and volatility clustering are preserved), optionally shifted to a user-specified expected return.
All personal inputs (capital, contributions, horizon, goal) must be supplied by the user; none are assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_paths(daily: np.ndarray, n_days: int, n_paths: int = 5000, block: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(daily)
    if n < block * 2:
        raise ValueError("not enough return history for bootstrap")
    nb = int(np.ceil(n_days / block))
    starts = rng.integers(0, n - block, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n_days]
    return daily[idx]


def goal_projection(weights: pd.Series, rets: pd.DataFrame, capital: float, years: float, goal: float | None = None,
                    monthly_contribution: float = 0.0, n_paths: int = 5000, mean_override_ann: float | None = None,
                    seed: int = 0) -> dict:
    w = weights.reindex(rets.columns).fillna(0)
    hist = (rets.fillna(0) @ w).dropna()
    hist = hist.iloc[-2520:]
    daily = hist.to_numpy()
    if mean_override_ann is not None:
        daily = daily - daily.mean() + (1 + mean_override_ann) ** (1 / 252) - 1
    n_days = int(round(years * 252))
    paths = block_bootstrap_paths(daily, n_days, n_paths, seed=seed)
    wealth = np.full(n_paths, float(capital))
    traj = np.empty((n_paths, n_days))
    contrib_every = 21
    for t in range(n_days):
        wealth = wealth * (1 + paths[:, t])
        if monthly_contribution and (t + 1) % contrib_every == 0:
            wealth = wealth + monthly_contribution
        traj[:, t] = wealth
    q = np.percentile(traj, [5, 25, 50, 75, 95], axis=0)
    step = max(1, n_days // 120)
    peak = np.maximum.accumulate(traj, axis=1)
    mdd = (traj / peak - 1).min(axis=1)
    out = {
        "capital": capital, "years": years, "monthly_contribution": monthly_contribution, "n_paths": n_paths,
        "history_days_used": int(len(daily)), "method": "block bootstrap (20d blocks) of historical portfolio returns",
        "terminal_percentiles": dict(zip(["p5", "p25", "p50", "p75", "p95"], np.percentile(wealth, [5, 25, 50, 75, 95]).tolist())),
        "prob_loss": float((wealth < capital + monthly_contribution * (n_days // contrib_every)).mean()),
        "max_drawdown_percentiles": dict(zip(["p50", "p95_worst"], [float(np.percentile(mdd, 50)), float(np.percentile(mdd, 5))])),
        "fan": {"day": list(range(0, n_days, step)), **{k: q[i, ::step].round(2).tolist() for i, k in enumerate(["p5", "p25", "p50", "p75", "p95"])}},
        "caveat": "Bootstrapped from past returns; the future can be worse than any historical path. Not a guarantee.",
    }
    if goal is not None:
        out["goal"] = goal
        out["prob_reach_goal"] = float((wealth >= goal).mean())
    return out
