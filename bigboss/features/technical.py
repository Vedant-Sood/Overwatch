"""Price/volume features computed in wide (date x symbol) form.

Every feature at date t uses only rows dated <= t (rolling windows, shifts >= 0). This is enforced by the
truncation test in tests/test_features.py (bigboss.data.validation.assert_no_lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS: dict[str, list[str]] = {}


def _reg(group: str, name: str) -> str:
    FEATURE_GROUPS.setdefault(group, [])
    if name not in FEATURE_GROUPS[group]:
        FEATURE_GROUPS[group].append(name)
    return name


def wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="symbol", values=field).sort_index()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Corwin & Schultz (2012) high-low bid-ask spread estimator, averaged over `window` days (negative -> 0)."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    h2 = np.maximum(high, high.shift(1))
    l2 = np.minimum(low, low.shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0).rolling(window, min_periods=window // 2).mean()


def technical_features(prices: pd.DataFrame, benchmark: str | None = None) -> pd.DataFrame:
    """Return a long frame indexed by (date, symbol) with technical, risk and liquidity features."""
    adj = wide(prices, "adj_close")
    close = wide(prices, "close")
    high = wide(prices, "high")
    low = wide(prices, "low")
    vol = wide(prices, "volume")
    # Scale OHLC to the adjusted basis so ranges are consistent across splits.
    factor = adj / close
    high_a, low_a = high * factor, low * factor
    r = adj.pct_change(fill_method=None)
    lr = np.log(adj).diff()
    f: dict[str, pd.DataFrame] = {}

    for n in (1, 5, 20, 60, 120, 252):
        f[_reg("momentum", f"ret_{n}")] = adj / adj.shift(n) - 1
    f[_reg("momentum", "mom_12_1")] = adj.shift(21) / adj.shift(252) - 1
    f[_reg("momentum", "px_ma50")] = adj / adj.rolling(50, min_periods=40).mean() - 1
    f[_reg("momentum", "px_ma200")] = adj / adj.rolling(200, min_periods=150).mean() - 1
    f[_reg("momentum", "dist_52w_high")] = adj / adj.rolling(252, min_periods=200).max() - 1
    f[_reg("reversal", "rsi_14")] = rsi(adj)
    f[_reg("reversal", "rev_5_z")] = f["ret_5"] / (lr.rolling(60, min_periods=40).std() * np.sqrt(5))

    f[_reg("volatility", "vol_20")] = lr.rolling(20, min_periods=15).std() * np.sqrt(252)
    f[_reg("volatility", "vol_60")] = lr.rolling(60, min_periods=40).std() * np.sqrt(252)
    f[_reg("volatility", "vol_ratio")] = f["vol_20"] / f["vol_60"]
    f[_reg("volatility", "downside_vol_60")] = lr.where(lr < 0, 0).rolling(60, min_periods=40).std() * np.sqrt(252)
    f[_reg("volatility", "skew_60")] = lr.rolling(60, min_periods=40).skew()
    f[_reg("volatility", "parkinson_20")] = np.sqrt((np.log(high_a / low_a) ** 2).rolling(20, min_periods=15).mean() / (4 * np.log(2)) * 252)
    roll_max = adj.rolling(252, min_periods=60).max()
    f[_reg("volatility", "drawdown_252")] = adj / roll_max - 1

    dollar_vol = close * vol
    f[_reg("liquidity", "log_dollar_vol_20")] = np.log1p(dollar_vol.rolling(20, min_periods=10).mean())
    f[_reg("liquidity", "amihud_20")] = (r.abs() / dollar_vol.replace(0, np.nan)).rolling(20, min_periods=10).mean() * 1e9
    f[_reg("liquidity", "volume_ratio_5_60")] = vol.rolling(5).mean() / vol.rolling(60, min_periods=40).mean()
    f[_reg("liquidity", "cs_spread_20")] = corwin_schultz_spread(high_a, low_a)

    if benchmark and benchmark in adj:
        br = lr[benchmark]
        cov = lr.rolling(252, min_periods=120).cov(br)
        var = br.rolling(252, min_periods=120).var()
        beta = cov.div(var, axis=0)
        f[_reg("risk", "beta_252")] = beta
        resid = lr - beta.mul(br, axis=0)
        f[_reg("risk", "idio_vol_60")] = resid.rolling(60, min_periods=40).std() * np.sqrt(252)
        f[_reg("risk", "corr_bench_60")] = lr.rolling(60, min_periods=40).corr(br)
        for n in (20, 60, 252):
            f[_reg("momentum", f"rel_ret_{n}")] = f[f"ret_{n}"].sub(f[f"ret_{n}"][benchmark], axis=0)

    long = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    # Cross-sectional percentile ranks of the main signals (robust to scale shifts over time).
    for c in ["ret_20", "ret_60", "mom_12_1", "vol_60", "log_dollar_vol_20", "rev_5_z"]:
        if c in long:
            long[_reg("cross_section", f"{c}_xrank")] = long[c].groupby(level="date").rank(pct=True)
    return long.replace([np.inf, -np.inf], np.nan)


def market_features(prices: pd.DataFrame, benchmark: str | None) -> pd.DataFrame:
    """Market-wide state features (identical for every symbol on a date)."""
    adj = wide(prices, "adj_close")
    lr = np.log(adj).diff()
    out = pd.DataFrame(index=adj.index)
    if benchmark and benchmark in adj:
        b = adj[benchmark]
        out[_reg("market", "mkt_ret_20")] = b / b.shift(20) - 1
        out[_reg("market", "mkt_ret_60")] = b / b.shift(60) - 1
        out[_reg("market", "mkt_vol_20")] = lr[benchmark].rolling(20, min_periods=15).std() * np.sqrt(252)
        out[_reg("market", "mkt_drawdown")] = b / b.rolling(252, min_periods=60).max() - 1
        out[_reg("market", "mkt_above_ma200")] = (b > b.rolling(200, min_periods=150).mean()).astype(float)
    members = [c for c in adj.columns if c != benchmark]
    if len(members) >= 3:
        out[_reg("market", "xs_dispersion_20")] = (adj[members] / adj[members].shift(20) - 1).std(axis=1)
        out[_reg("market", "avg_corr_60")] = average_pairwise_corr(lr[members], 60)
    return out


def average_pairwise_corr(lr: pd.DataFrame, window: int) -> pd.Series:
    """Average pairwise correlation via the variance identity (fast, O(N) per date)."""
    z = (lr - lr.rolling(window, min_periods=window // 2).mean()) / lr.rolling(window, min_periods=window // 2).std()
    n = z.notna().sum(axis=1)
    port = z.mean(axis=1)
    var_port = port.rolling(window, min_periods=window // 2).var()
    # var(mean z) = (1 + (n-1) * rho_bar) / n  ->  rho_bar = (n * var - 1) / (n - 1)
    rho = ((n * var_port - 1) / (n - 1)).clip(-1, 1)
    return rho.where(n >= 3)  # undefined with fewer than 3 assets trading
