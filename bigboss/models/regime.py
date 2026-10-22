"""Model C: market regime detection.

Methods (all computed using information available at each date):
- Rule-based state: trend (vs 200d MA) x volatility (vs expanding median).
- 2-state Markov-switching model on benchmark returns (statsmodels), FILTERED probabilities only
  (smoothed probabilities use future data and would leak). Parameters are fit on data up to `as_of`.
- CUSUM change-point detection on volatility (Page 1954), flagging recent structural shifts.
- Correlation regime: average pairwise correlation vs. its own history (diversification breakdown).
- Liquidity stress: cross-sectional median Amihud illiquidity vs. its history.
- Momentum vs. mean-reversion: sign of the cross-sectional rank IC between past 20d return and next 20d return,
  evaluated only on completed windows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bigboss.backtest.metrics import regime_labels
from bigboss.features.technical import average_pairwise_corr


def markov_switching_filtered(r: pd.Series, k: int = 2, max_obs: int = 2520) -> pd.DataFrame | None:
    """Fit a k-regime switching-variance model; return filtered P(high-vol regime) per date."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    x = r.dropna().iloc[-max_obs:] * 100
    if len(x) < 300:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = MarkovRegression(x, k_regimes=k, trend="c", switching_variance=True).fit(disp=False, maxiter=200)
        except Exception:  # numerical failures are reported as "unavailable", not hidden
            return None
    sig = [res.params.get(f"sigma2[{i}]", np.nan) for i in range(k)]
    hi = int(np.nanargmax(sig))
    fp = res.filtered_marginal_probabilities
    fp = fp[hi] if isinstance(fp, pd.DataFrame) else pd.Series(fp[:, hi], index=x.index)
    return pd.DataFrame({"p_high_vol": fp, "sigma_high_ann": np.sqrt(sig[hi]) / 100 * np.sqrt(252),
                         "sigma_low_ann": np.sqrt(np.nanmin(sig)) / 100 * np.sqrt(252)})


def cusum_changepoints(x: pd.Series, threshold: float = 5.0, drift: float = 0.5) -> list[pd.Timestamp]:
    """Two-sided CUSUM on a standardized series using an expanding mean/std (no look-ahead)."""
    z = (x - x.expanding(20).mean().shift(1)) / x.expanding(20).std().shift(1)
    gp = gn = 0.0
    out = []
    for d, v in z.dropna().items():
        gp = max(0.0, gp + v - drift)
        gn = max(0.0, gn - v - drift)
        if gp > threshold or gn > threshold:
            out.append(d)
            gp = gn = 0.0
    return out


def momentum_regime(adj: pd.DataFrame, lookback: int = 20, window: int = 120) -> pd.Series:
    """Rolling mean of cross-sectional rank correlation between past and NEXT 20d returns.
    The series at date t only uses pairs whose forward window ended by t (shifted by `lookback`)."""
    past = adj / adj.shift(lookback) - 1
    fwd = adj.shift(-lookback) / adj - 1
    ic = past.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)
    return ic.shift(lookback).rolling(window, min_periods=window // 2).mean()


def detect_regimes(prices: pd.DataFrame, benchmark: str, as_of: pd.Timestamp | None = None) -> dict:
    adj = prices.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    if as_of is not None:
        adj = adj.loc[:as_of]
    if benchmark not in adj:
        return {"available": False, "reason": f"benchmark {benchmark} has no data"}
    b = adj[benchmark].dropna()
    r = np.log(b).diff()
    labels = regime_labels(b)
    vol20 = r.rolling(20).std() * np.sqrt(252)
    members = [c for c in adj.columns if c != benchmark and not c.startswith("^")]
    lr = np.log(adj[members]).diff() if members else pd.DataFrame()
    corr = average_pairwise_corr(lr, 60) if len(members) >= 3 else pd.Series(dtype=float)
    out = {
        "available": True, "as_of": str(b.index[-1].date()), "benchmark": benchmark,
        "label": labels.iloc[-1] if len(labels.dropna()) else None,
        "trend": "bull" if b.iloc[-1] > b.rolling(200).mean().iloc[-1] else "bear",
        "vol_20d_ann": float(vol20.iloc[-1]),
        "vol_percentile": float((vol20.dropna() <= vol20.iloc[-1]).mean()),
        "drawdown": float(b.iloc[-1] / b.cummax().iloc[-1] - 1),
    }
    ms = markov_switching_filtered(r)
    if ms is not None:
        out["markov_p_high_vol"] = float(ms["p_high_vol"].iloc[-1])
        out["markov_sigma_high_ann"] = float(ms["sigma_high_ann"].iloc[-1])
        out["markov_sigma_low_ann"] = float(ms["sigma_low_ann"].iloc[-1])
        out["markov_history"] = {"dates": [str(d.date()) for d in ms.index[-500:]], "p_high_vol": ms["p_high_vol"].iloc[-500:].round(4).tolist()}
    # CUSUM on non-overlapping 20-day log-volatility blocks (daily vol20 is too autocorrelated for CUSUM)
    block_vol = np.log(r.dropna().groupby(np.arange(len(r.dropna())) // 20).std().replace(0, np.nan))
    block_vol.index = r.dropna().index[::20][: len(block_vol)]
    cps = cusum_changepoints(block_vol.dropna(), threshold=4.0, drift=0.5)
    out["vol_changepoints_recent"] = [str(d.date()) for d in cps if d >= b.index[-1] - pd.Timedelta(days=365)]
    if len(corr.dropna()):
        out["avg_corr_60d"] = float(corr.iloc[-1])
        out["avg_corr_percentile"] = float((corr.dropna() <= corr.iloc[-1]).mean())
        out["correlation_breakdown"] = bool(out["avg_corr_percentile"] > 0.9)
    if members and "volume" in prices:
        close = prices.pivot(index="date", columns="symbol", values="close").sort_index()[members]
        vol = prices.pivot(index="date", columns="symbol", values="volume").sort_index()[members]
        if as_of is not None:
            close, vol = close.loc[:as_of], vol.loc[:as_of]
        amihud = (np.log(adj[members]).diff().abs() / (close * vol).replace(0, np.nan)).rolling(20).mean().median(axis=1)
        if amihud.notna().sum() > 100:
            out["liquidity_stress_percentile"] = float((amihud.dropna() <= amihud.iloc[-1]).mean())
            out["liquidity_stress"] = bool(out["liquidity_stress_percentile"] > 0.9)
    if len(members) >= 5:
        mr = momentum_regime(adj[members])
        if mr.notna().any():
            out["xs_momentum_ic_120d"] = float(mr.dropna().iloc[-1])
            out["momentum_or_reversal"] = "momentum" if mr.dropna().iloc[-1] > 0 else "mean_reversion"
    hist = pd.DataFrame({"label": labels, "vol20": vol20}).dropna().iloc[-750:]
    out["history"] = {"dates": [str(d.date()) for d in hist.index], "label": hist["label"].tolist(),
                      "vol20": hist["vol20"].round(4).tolist()}
    return out


def macro_instability(macro: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Simple macro stress dashboard: VIX, HY spread, curve inversion, each vs. own history (PIT by available_at)."""
    out = {}
    for sid, label in (("VIXCLS", "vix"), ("BAMLH0A0HYM2", "hy_spread"), ("T10Y2Y", "term_spread_10y2y")):
        s = macro[macro["series_id"] == sid]
        if as_of is not None:
            s = s[s["available_at"] <= as_of]
        if s.empty:
            continue
        v = s.set_index("date")["value"]
        out[label] = {"value": float(v.iloc[-1]), "date": str(v.index[-1].date()), "percentile_10y": float((v.iloc[-2520:] <= v.iloc[-1]).mean())}
    if "term_spread_10y2y" in out:
        out["curve_inverted"] = out["term_spread_10y2y"]["value"] < 0
    return out
