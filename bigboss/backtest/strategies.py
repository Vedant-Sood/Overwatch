"""Allocation rules evaluated by the backtester. Each returns target weights (sum <= 1; rest is cash).

Transparent baselines: benchmark buy-and-hold, equal weight, 12-1 momentum top-k, low-volatility top-k.
Model strategies: cost-aware active-risk mean-variance on a model's OOS forecasts, with the benchmark
fund as the default holding (so the strategy only deviates from the index when the net edge pays for it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bigboss.backtest.engine import MarketData
from bigboss.costs import CostModel
from bigboss.models.ensemble import abstain_mask
from bigboss.portfolio.optimize import mean_variance


def benchmark_strategy(bench: str):
    def fn(d, cur_w, ctx):
        return pd.Series({bench: 1.0})
    return fn


def equal_weight_strategy(md: MarketData, universe: list[str], candidates: pd.DataFrame | None = None):
    """Equal weight of the names available at d. `candidates` (the dataset frame, already restricted to the
    point-in-time universe) limits the choice to names that are members on d; without it every symbol in `universe` is
    eligible, which is only correct for a fixed universe (a name held before it joins an index is look-ahead)."""
    members = None
    if candidates is not None:
        members = candidates.index.to_frame(index=False).groupby("date")["symbol"].apply(set)

    def fn(d, cur_w, ctx):
        pool = universe
        if members is not None:
            if d not in members.index:
                return None
            pool = [s for s in universe if s in members[d]]
        avail = [s for s in pool if s in md.adj.columns and pd.notna(md.adj.at[d, s])
                 and md.adj[s].loc[:d].notna().sum() > 60]
        return pd.Series(1.0 / len(avail), index=avail) if avail else None
    return fn


def factor_topk_strategy(features: pd.DataFrame, column: str, k: int, ascending: bool = False):
    """Rank on a single PIT feature at date d (e.g. mom_12_1 descending, vol_60 ascending); equal-weight top k."""
    def fn(d, cur_w, ctx):
        try:
            x = features.xs(d, level="date")[column].dropna()
        except KeyError:
            return None
        if len(x) < k:
            return None
        pick = x.sort_values(ascending=ascending).index[:k]
        return pd.Series(1.0 / k, index=pick)
    return fn


def model_strategy(md: MarketData, preds: pd.DataFrame, cost_model: CostModel, horizon: int, *, risk_aversion: float,
                   max_weight: float, max_sector: float, max_turnover: float | None, top_k: int, sectors: dict[str, str],
                   gate: dict | None = None):
    """preds: OOS rows (date, symbol) with mean,q10,q50,q90 (+ disagreement, sign_agreement, skill for gating)."""
    bench = md.benchmark
    rets = md.adj.pct_change(fill_method=None)
    adv = md.adv_dollar()
    dvol = md.daily_vol()
    base = {s: cost_model.base_bps(s, str(md.meta.loc[s, "asset_class"]) if s in md.meta.index else "equity") for s in md.adj.columns}
    pdates = set(preds.index.get_level_values("date").unique())

    def fn(d, cur_w, ctx):
        if d not in pdates:
            return None
        p = preds.xs(d, level="date")
        p = p[p["mean"].notna()]
        if gate is not None:
            skill = float(p["skill_t"].iloc[0]) if "skill_t" in p and len(p) else 0.0
            ab = abstain_mask(p, gate["max_disagreement"], gate["min_sign_agreement"], skill >= gate["min_skill_tstat"],
                              p.get("feature_coverage"))
            ctx["last_decision"] = {"date": str(d.date()), "skill_t": skill, "n_candidates": int(len(p)), "n_abstained": int(ab.sum())}
            p = p[~ab]
        nav_proxy = 1e6
        one_way = {}
        for s in p.index:
            one_way[s] = cost_model.vectorized_one_way(np.array([max_weight * nav_proxy * 0.5]), np.array([adv.at[d, s] if s in adv else np.nan]),
                                                        np.array([dvol.at[d, s] if s in dvol else np.nan]), np.array([base.get(s, 10.0)]))[0]
        ow = pd.Series(one_way, dtype=float)
        mu = (p["mean"] - ow.reindex(p.index).fillna(0.0)) if len(p) else pd.Series(dtype=float)  # exit cost amortized into mu
        held = list(cur_w[cur_w > 1e-6].index)
        cand = list(mu.sort_values(ascending=False).index[: 3 * top_k])
        universe = list(dict.fromkeys(cand + [h for h in held if h != bench] + [bench]))
        mu_u = mu.reindex(universe).fillna(-0.0)
        mu_u[bench] = 0.0
        hist = rets.loc[:d, universe].iloc[-252:]
        hist = hist.loc[:, hist.notna().sum() > 60]
        universe = list(hist.columns)
        if bench not in universe:
            return pd.Series({bench: 1.0})
        from bigboss.risk import shrunk_cov
        cov = shrunk_cov(hist, min_obs=60) * horizon
        c = pd.Series({s: (ow.get(s, base.get(s, 1.0) / 1e4)) for s in universe})
        res = mean_variance(mu_u.reindex(universe).fillna(0.0), cov, w0=cur_w.reindex(universe).fillna(0.0), cost=c,
                            risk_aversion=risk_aversion, max_weight=max_weight, sectors=sectors, max_sector=max_sector,
                            max_turnover=max_turnover, exempt={bench}, min_invested=1.0, w_ref=pd.Series({bench: 1.0}))
        w = res.weights[res.weights > 1e-4]
        return w if len(w) else pd.Series({bench: 1.0})
    return fn
