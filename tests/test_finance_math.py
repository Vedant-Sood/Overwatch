"""Verification of financial calculations: costs, metrics, optimizers, backtest accounting."""

import numpy as np
import pandas as pd
import pytest

from bigboss.backtest.engine import MarketData, SimConfig, simulate
from bigboss.backtest.metrics import alpha_beta, ann_return, max_drawdown, sharpe, summarize
from bigboss.costs import CostModel
from bigboss.portfolio.optimize import black_litterman, hrp, mean_variance, min_cvar, risk_parity
from bigboss.risk import RiskLimits, risk_contributions, risk_gate
from tests.conftest import make_prices


# ---------------- costs ----------------
def test_square_root_impact_and_round_trip(cfg):
    cm = CostModel(cfg)
    c = cfg.section("costs")
    e = cm.estimate("AAPL", "equity", order_notional=1e6, adv_dollar=1e8, daily_vol=0.02)
    exp_impact = c["impact_coefficient"] * 0.02 * np.sqrt(0.01) * 1e4
    assert np.isclose(e.impact_bps, exp_impact)
    assert np.isclose(e.total_one_way_bps, c["commission_bps"]["US"] + c["half_spread_bps"]["large_cap"] + c["slippage_bps"] + exp_impact)
    assert np.isclose(e.total_round_trip_bps, 2 * e.total_one_way_bps)
    assert e.liquidity_ok
    big = cm.estimate("AAPL", "equity", order_notional=1e7, adv_dollar=1e8, daily_vol=0.02)
    assert not big.liquidity_ok and big.impact_bps > e.impact_bps


def test_india_tax_and_cost_multiplier(cfg):
    e = CostModel(cfg).estimate("TCS.NS", "equity", 1e5, 1e9, 0.015)
    assert e.tax_bps == cfg.get("costs.india_stt_bps")
    e2 = CostModel(cfg, multiplier=2.0).estimate("TCS.NS", "equity", 1e5, 1e9, 0.015)
    assert np.isclose(e2.total_one_way_bps, 2 * e.total_one_way_bps)


def test_missing_volume_is_flagged_not_assumed_liquid(cfg):
    e = CostModel(cfg).estimate("XYZ", "equity", 1e5, float("nan"), 0.02)
    assert not e.liquidity_ok and "no volume" in e.note


# ---------------- metrics ----------------
def test_metrics_on_known_series():
    r = pd.Series([0.1, -0.5, 0.2], index=pd.bdate_range("2020-01-01", periods=3))
    assert np.isclose(max_drawdown(r), -0.5)
    flat = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    assert np.isclose(ann_return(flat), 1.001 ** 252 - 1)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
    p = 0.0002 + 1.5 * b
    a, beta = alpha_beta(p, b)
    assert np.isclose(beta, 1.5) and np.isclose(a, 0.0002 * 252)
    s = summarize(b, b)
    assert s["ann_excess_return"] == 0 and s["information_ratio"] is None  # zero tracking error
    x = pd.Series(rng.normal(0.001, 0.01, 2000))
    assert np.isclose(sharpe(x), x.mean() / x.std(ddof=1) * np.sqrt(252))


# ---------------- optimizers ----------------
@pytest.fixture
def cov4():
    rng = np.random.default_rng(2)
    A = rng.normal(0, 0.01, (500, 4))
    A[:, 1] += A[:, 0]
    return pd.DataFrame(np.cov(A.T), index=list("ABCD"), columns=list("ABCD"))


def test_mean_variance_obeys_constraints(cov4):
    mu = pd.Series([0.02, 0.015, 0.01, -0.01], index=cov4.index)
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
    r = mean_variance(mu, cov4, risk_aversion=1.0, max_weight=0.3, sectors=sectors, max_sector=0.4)
    w = r.weights
    assert (w >= -1e-8).all() and (w <= 0.3 + 1e-6).all() and w.sum() <= 1 + 1e-6
    assert w["A"] + w["B"] <= 0.4 + 1e-6
    assert w["D"] < 1e-4  # negative expected return is not bought


def test_costs_reduce_trading(cov4):
    mu = pd.Series([0.010, 0.011, 0.0, 0.0], index=cov4.index)
    w0 = pd.Series([0.3, 0.0, 0.0, 0.0], index=cov4.index)
    free = mean_variance(mu, cov4, w0=w0, risk_aversion=1.0, max_weight=0.3)
    costly = mean_variance(mu, cov4, w0=w0, cost=pd.Series(0.01, index=cov4.index), risk_aversion=1.0, max_weight=0.3)
    assert costly.turnover < free.turnover


def test_active_risk_reference_keeps_benchmark(cov4):
    mu = pd.Series(0.0, index=cov4.index)
    r = mean_variance(mu, cov4, risk_aversion=10, max_weight=0.3, exempt={"C"}, min_invested=1.0, w_ref=pd.Series({"C": 1.0}))
    assert r.weights["C"] > 0.95  # with no edge, the optimum is to hold the benchmark


def test_risk_parity_equalizes_contributions(cov4):
    w = risk_parity(cov4)
    rc = risk_contributions(w, cov4)
    assert np.isclose(w.sum(), 1) and np.allclose(rc, 0.25, atol=1e-3)


def test_hrp_and_cvar_and_black_litterman(cov4):
    w = hrp(cov4)
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    rng = np.random.default_rng(3)
    sc = pd.DataFrame(rng.normal(0.001, 0.02, (400, 4)), columns=cov4.index)
    sc["D"] = rng.normal(0.001, 0.06, 400)
    r = min_cvar(sc, max_weight=0.5)
    assert r.success and np.isclose(r.weights.sum(), 1) and r.weights["D"] < 0.2
    wm = pd.Series(0.25, index=cov4.index)
    views = pd.Series({"A": 0.05})
    tight = black_litterman(cov4, wm, views, pd.Series({"A": 1e-6}))
    loose = black_litterman(cov4, wm, views, pd.Series({"A": 1.0}))
    prior = black_litterman(cov4, wm, pd.Series(dtype=float), pd.Series(dtype=float))
    assert abs(tight["A"] - 0.05) < abs(loose["A"] - 0.05)
    assert abs(loose["A"] - prior["A"]) < 1e-3


def test_risk_gate_detects_violations(cov4):
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    chk = risk_gate(w, RiskLimits(max_weight=0.4, max_sector_weight=0.6, max_ann_vol=0.05), {"A": "T", "B": "T", "C": "E"}, cov4)
    assert not chk.passed
    joined = " ".join(chk.violations)
    assert "position cap" in joined and "sector cap" in joined and "volatility" in joined
    ok = risk_gate(pd.Series({"C": 1.0}), RiskLimits(max_weight=0.1), {}, exempt={"C"})
    assert ok.passed


# ---------------- backtest accounting ----------------
def _md(n=300):
    p = make_prices(n)
    meta = pd.DataFrame({"symbol": ["A", "B", "M"], "asset_class": ["equity", "equity", "equity_etf"], "sector": ["X", "Y", "Z"]})
    return MarketData.from_prices(p, meta, "M")


def test_buy_and_hold_benchmark_matches_index_minus_entry_cost(cfg):
    md = _md()
    start = md.adj.index[30]
    res = simulate("b", md, lambda d, w, c: pd.Series({"M": 1.0}), SimConfig(rebalance_every=10_000, delay=1, start=start, initial_capital=1e5), CostModel(cfg))
    r_m = md.adj["M"].pct_change().loc[start:]
    # day 1: entry costs are paid (no leverage: cash never negative) and no market return is earned yet
    assert res.costs.iloc[1] > 0 and res.returns.iloc[1] < 0
    # afterwards the strategy holds slightly under 100% (costs were funded from capital) and tracks the index
    assert np.allclose(res.returns.iloc[3:], r_m.iloc[3:], atol=2e-6)
    assert (res.returns.iloc[3:].abs() <= r_m.iloc[3:].abs() + 1e-12).all()


def test_zero_cost_and_execution_delay(cfg):
    md = _md()
    zero = CostModel(cfg, multiplier=0.0)
    start = md.adj.index[30]
    res = simulate("a", md, lambda d, w, c: pd.Series({"A": 1.0}), SimConfig(rebalance_every=10_000, delay=2, start=start, initial_capital=1e5), zero)
    r_a = md.adj["A"].pct_change().loc[start:]
    assert np.allclose(res.returns.iloc[:3], 0)            # nothing held until the close of t+2
    assert np.isclose(res.returns.iloc[3], r_a.iloc[3])    # first return earned on t+3
    assert res.costs.sum() == 0


def test_liquidity_cap_limits_fills(cfg):
    md = _md()
    md.volume[:] = 10.0  # tiny volume => tiny allowed participation
    res = simulate("a", md, lambda d, w, c: pd.Series({"A": 1.0}), SimConfig(rebalance_every=10_000, delay=1, initial_capital=1e6, start=md.adj.index[30]), CostModel(cfg))
    traded = res.trades["notional"].abs().sum()
    assert traded < 1e5  # far less than the 1m requested


def test_unknown_liquidity_means_no_fill(cfg):
    md = _md()
    res = simulate("a", md, lambda d, w, c: pd.Series({"A": 1.0}), SimConfig(rebalance_every=10_000, delay=1), CostModel(cfg))
    assert res.trades.empty  # day-1 execution has no ADV history yet


def test_delisted_asset_is_liquidated(cfg):
    p = make_prices(200)
    p = p[~((p["symbol"] == "A") & (p["date"] > p["date"].iloc[100]))]
    meta = pd.DataFrame({"symbol": ["A", "B", "M"], "asset_class": "equity", "sector": "X"})
    md = MarketData.from_prices(p, meta, "M")
    res = simulate("a", md, lambda d, w, c: pd.Series({"A": 0.5, "B": 0.5}),
                   SimConfig(rebalance_every=10_000, delay=1, start=md.adj.index[30], initial_capital=1e5), CostModel(cfg))
    assert (res.trades["reason"] == "data_end_liquidation").any()
