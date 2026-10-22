"""Risk analytics: covariance estimation, VaR/ES, risk contributions, stress scenarios, and the
independent risk gate that every proposed portfolio must pass (not overridable by LLM output).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def returns_wide(prices: pd.DataFrame, field_: str = "adj_close") -> pd.DataFrame:
    px = prices.pivot(index="date", columns="symbol", values=field_).sort_index()
    return px.pct_change(fill_method=None)


def shrunk_cov(rets: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """Ledoit-Wolf shrunk daily covariance. Columns with too little data are dropped."""
    r = rets.dropna(axis=1, thresh=min_obs)
    r = r.fillna(0.0)
    if r.shape[0] < min_obs or r.shape[1] == 0:
        raise ValueError("not enough return history for covariance")
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def portfolio_var_es(weights: pd.Series, rets: pd.DataFrame, alpha: float = 0.05, horizon: int = 1) -> dict:
    """Historical-simulation VaR/ES of the portfolio (positive numbers = losses)."""
    w = weights.reindex(rets.columns).fillna(0)
    pr = rets.fillna(0) @ w
    if horizon > 1:
        pr = (1 + pr).rolling(horizon).apply(np.prod, raw=True).dropna() - 1
    q = pr.quantile(alpha)
    return {"var": float(-q), "es": float(-pr[pr <= q].mean()), "alpha": alpha, "horizon_days": horizon, "n_obs": int(len(pr))}


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    w = weights.reindex(cov.index).fillna(0).to_numpy()
    s = cov.to_numpy()
    port_var = w @ s @ w
    if port_var <= 0:
        return pd.Series(0.0, index=cov.index)
    return pd.Series(w * (s @ w) / port_var, index=cov.index)


STRESS_SCENARIOS = {
    # name -> (start, end) windows of historical crises; applied to current weights using each asset's realized return
    "GFC_2008": ("2008-09-01", "2009-03-09"),
    "Euro_crisis_2011": ("2011-07-22", "2011-10-03"),
    "Covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "Rate_shock_2022": ("2022-01-03", "2022-10-12"),
}


def historical_stress(weights: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    rows = []
    for name, (a, b) in STRESS_SCENARIOS.items():
        win = px.loc[a:b]
        if len(win) < 5:
            continue
        ret = win.iloc[-1] / win.iloc[0] - 1
        w = weights.reindex(ret.index).fillna(0)
        covered = float(w[ret.notna()].sum() / w.sum()) if w.sum() > 0 else 0.0  # share of invested weight
        rows.append({"scenario": name, "start": a, "end": b, "portfolio_return": float((w * ret.fillna(0)).sum()),
                     "weight_covered": covered, "note": "" if covered > 0.95 else "some holdings lack history for this window"})
    return pd.DataFrame(rows)


@dataclass
class RiskLimits:
    max_weight: float = 0.10
    max_sector_weight: float = 0.30
    max_gross: float = 1.0
    min_cash: float = 0.0
    max_ann_vol: float | None = None
    max_participation: float = 0.05
    long_only: bool = True


@dataclass
class RiskCheck:
    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def risk_gate(weights: pd.Series, limits: RiskLimits, sectors: dict[str, str], cov: pd.DataFrame | None = None,
              exempt: set[str] | None = None) -> RiskCheck:
    """Deterministic constraint check. `exempt` = diversified funds allowed above the single-name cap (e.g. the benchmark)."""
    exempt = exempt or set()
    v: list[str] = []
    w = weights[weights.abs() > 1e-9]
    if limits.long_only and (w < -1e-9).any():
        v.append(f"short positions not allowed: {list(w[w < 0].index)}")
    single = w.drop([s for s in w.index if s in exempt], errors="ignore")
    over = single[single > limits.max_weight + 1e-6]
    if len(over):
        v.append("position cap exceeded: " + ", ".join(f"{s}={x:.1%}" for s, x in over.items()))
    sec = single.groupby(lambda s: sectors.get(s, "Unknown")).sum()
    sec = sec.drop("Unknown", errors="ignore")
    over_s = sec[sec > limits.max_sector_weight + 1e-6]
    if len(over_s):
        v.append("sector cap exceeded: " + ", ".join(f"{s}={x:.1%}" for s, x in over_s.items()))
    gross = float(w.abs().sum())
    if gross > limits.max_gross + 1e-6:
        v.append(f"gross exposure {gross:.1%} > {limits.max_gross:.0%}")
    if 1 - float(w.sum()) < limits.min_cash - 1e-6:
        v.append(f"cash below minimum {limits.min_cash:.0%}")
    metrics = {"gross": gross, "n_positions": int(len(w)), "sector_weights": sec.round(4).to_dict()}
    if cov is not None:
        ww = w.reindex(cov.index).fillna(0).to_numpy()
        vol = float(np.sqrt(ww @ cov.to_numpy() @ ww * 252))
        metrics["ex_ante_ann_vol"] = vol
        if limits.max_ann_vol and vol > limits.max_ann_vol + 1e-6:
            v.append(f"ex-ante volatility {vol:.1%} > limit {limits.max_ann_vol:.0%}")
    return RiskCheck(not v, v, metrics)
