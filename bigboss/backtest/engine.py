"""Daily portfolio simulator with execution delay, transaction costs, liquidity caps, and benchmark tracking.

Timeline for a rebalance decided at the close of day t:
  - the strategy sees only data/forecasts dated <= t,
  - orders execute at the close of t + delay (default 1 day) at that close price,
  - each trade pays a one-way cost (commission + half spread + slippage + sqrt market impact),
  - trades are capped at `max_participation` x 20-day average dollar volume (unfilled remainder is dropped).
Cash earns `cash_rate_daily` (0 by default; documented as a conservative simplification).
Delisting: if an asset's price series ends, the position is liquidated at its last price with full costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from bigboss.backtest.metrics import by_regime, regime_labels, summarize
from bigboss.costs import CostModel


@dataclass
class SimConfig:
    rebalance_every: int = 20
    delay: int = 1
    initial_capital: float = 1_000_000.0
    max_participation: float = 0.05
    cash_rate_daily: float = 0.0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


@dataclass
class SimResult:
    name: str
    returns: pd.Series
    benchmark: pd.Series
    nav: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame          # target weights at each decision
    trades: pd.DataFrame
    summary: dict = field(default_factory=dict)
    regimes: pd.DataFrame | None = None
    decisions: list[dict] = field(default_factory=list)


@dataclass
class MarketData:
    adj: pd.DataFrame        # adjusted close (returns)
    close: pd.DataFrame      # raw close (for dollar volume)
    volume: pd.DataFrame
    meta: pd.DataFrame       # symbol metadata: asset_class, sector
    benchmark: str

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, meta: pd.DataFrame, benchmark: str) -> "MarketData":
        p = lambda f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
        return cls(p("adj_close"), p("close"), p("volume"), meta.set_index("symbol") if "symbol" in meta else meta, benchmark)

    def adv_dollar(self) -> pd.DataFrame:
        # ADV known at the close of t-1 when trading at t's close (avoid using the execution day's own volume)
        return (self.close * self.volume).rolling(20, min_periods=5).mean().shift(1)

    def daily_vol(self) -> pd.DataFrame:
        return np.log(self.adj).diff().rolling(60, min_periods=20).std().shift(1)


TargetFn = Callable[[pd.Timestamp, pd.Series, dict], pd.Series]


def simulate(name: str, md: MarketData, target_fn: TargetFn, sim: SimConfig, cost_model: CostModel) -> SimResult:
    adj = md.adj
    dates = adj.index
    if sim.start is not None:
        dates = dates[dates >= sim.start]
    if sim.end is not None:
        dates = dates[dates <= sim.end]
    rets = adj.pct_change(fill_method=None)
    last_valid = adj.apply(lambda s: s.last_valid_index())
    adv = md.adv_dollar()
    dvol = md.daily_vol()
    assets = list(adj.columns)
    base_bps = pd.Series({s: cost_model.base_bps(s, str(md.meta.loc[s, "asset_class"]) if s in md.meta.index else "equity") for s in assets})

    hold = pd.Series(0.0, index=assets)
    cash = sim.initial_capital
    pending: dict[pd.Timestamp, pd.Series] = {}
    navs, turn, costs = {}, {}, {}
    weights_log, trades_log, decisions = {}, [], []
    ctx: dict = {}
    for i, d in enumerate(dates):
        # 1) mark to market with today's returns
        r = rets.loc[d].reindex(assets)
        held = hold != 0
        hold[held] = hold[held] * (1 + r[held].fillna(0.0))
        cash *= 1 + sim.cash_rate_daily
        day_cost = 0.0
        day_traded = 0.0
        # 2) forced liquidation of assets whose history has ended (delisted / data ends)
        for s in hold.index[held]:
            if last_valid.get(s) is not None and d >= last_valid[s] and d != dates[-1] and hold[s] != 0:
                c = abs(hold[s]) * cost_model.vectorized_one_way(np.array([abs(hold[s])]), np.array([adv.at[d, s] if d in adv.index else np.nan]),
                                                                 np.array([dvol.at[d, s] if d in dvol.index else np.nan]), np.array([base_bps[s]]))[0]
                cash += hold[s] - c
                day_cost += c
                day_traded += abs(hold[s])
                trades_log.append({"date": d, "symbol": s, "notional": -hold[s], "cost": c, "reason": "data_end_liquidation"})
                hold[s] = 0.0
        nav = hold.sum() + cash
        # 3) execute any order scheduled for today
        if d in pending:
            target = pending.pop(d).reindex(assets).fillna(0.0)
            desired = target * nav
            trade = desired - hold
            tradable = adj.loc[d].notna()
            trade[~tradable] = 0.0
            # Unknown liquidity means no fill (never an unlimited one). Sells of held positions are allowed so
            # that risk can always be reduced.
            cap = (sim.max_participation * adv.loc[d]).reindex(assets)
            cap = cap.where(cap.notna(), 0.0)
            trade = trade.clip(lower=-np.maximum(cap, np.where(trade < 0, hold.abs(), 0)), upper=cap)
            nz = trade.abs() > 1e-6
            if nz.any():
                c_vec = cost_model.vectorized_one_way(trade[nz].abs().to_numpy(), adv.loc[d].reindex(trade[nz].index).to_numpy(),
                                                      dvol.loc[d].reindex(trade[nz].index).to_numpy(), base_bps[trade[nz].index].to_numpy())
                tc = trade[nz].abs().to_numpy() * c_vec
                # No implicit leverage: buys plus all costs must be funded by cash plus sale proceeds.
                buys = trade[nz].clip(lower=0)
                funds = cash - trade[nz].clip(upper=0).sum()  # cash + sale proceeds
                need = buys.sum() + tc.sum()
                if need > funds + 1e-9 and buys.sum() > 0:
                    scale = max(0.0, (funds - tc[(trade[nz] < 0).to_numpy()].sum()) / (buys.sum() * (1 + c_vec[(trade[nz] > 0).to_numpy()].max())))
                    trade[nz] = np.where(trade[nz] > 0, trade[nz] * min(scale, 1.0), trade[nz])
                    tc = trade[nz].abs().to_numpy() * c_vec
                hold[nz] += trade[nz]
                cash -= trade[nz].sum() + tc.sum()
                day_cost += tc.sum()
                day_traded += trade[nz].abs().sum()
                for s, n_, c_ in zip(trade[nz].index, trade[nz].to_numpy(), tc):
                    trades_log.append({"date": d, "symbol": s, "notional": float(n_), "cost": float(c_), "reason": "rebalance"})
        nav = hold.sum() + cash
        navs[d] = nav
        costs[d] = day_cost / max(nav, 1e-9)
        turn[d] = day_traded / max(nav, 1e-9) / 2  # one-way turnover
        # 4) decide at today's close; execute after delay
        if i % sim.rebalance_every == 0 and i + sim.delay < len(dates):
            cur_w = hold / nav if nav > 0 else hold * 0
            tgt = target_fn(d, cur_w, ctx)
            if tgt is not None:
                tgt = tgt[tgt.abs() > 1e-8]
                if tgt.sum() > 1 + 1e-6:
                    tgt = tgt / tgt.sum()
                pending[dates[i + sim.delay]] = tgt
                weights_log[d] = tgt
                if "last_decision" in ctx:
                    decisions.append(ctx.pop("last_decision"))
    nav_s = pd.Series(navs)
    ret = nav_s.pct_change().fillna(0.0)
    bench = rets[md.benchmark].reindex(nav_s.index).fillna(0.0)
    res = SimResult(name, ret, bench, nav_s, pd.Series(turn), pd.Series(costs),
                    pd.DataFrame(weights_log).T.fillna(0.0) if weights_log else pd.DataFrame(),
                    pd.DataFrame(trades_log), decisions=decisions)
    return res


def finalize(res: SimResult, md: MarketData, n_trials: int = 1, skip_days: int = 0) -> SimResult:
    """Compute summary metrics, skipping the initial `skip_days` (before the first position is established)."""
    r = res.returns.iloc[skip_days:]
    b = res.benchmark.iloc[skip_days:]
    res.summary = summarize(r, b, res.turnover, res.costs, n_trials=n_trials)
    res.summary["total_costs_paid_pct"] = float(res.costs.sum())
    res.regimes = by_regime(r, b, regime_labels(md.adj[md.benchmark].dropna()))
    return res
