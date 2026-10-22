"""Performance metrics. Inputs are DAILY simple returns (pd.Series indexed by date)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS = 252


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def sharpe(r: pd.Series, rf_daily: float | pd.Series = 0.0) -> float:
    x = (r - rf_daily).dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS)) if x.std(ddof=1) > 0 else float("nan")


def sortino(r: pd.Series, rf_daily: float = 0.0) -> float:
    x = (r - rf_daily).dropna()
    dd = np.sqrt((np.minimum(x, 0) ** 2).mean())
    return float(x.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


def expected_shortfall(r: pd.Series, alpha: float = 0.05) -> float:
    x = r.dropna()
    if x.empty:
        return float("nan")
    q = x.quantile(alpha)
    return float(x[x <= q].mean())


def alpha_beta(r: pd.Series, b: pd.Series) -> tuple[float, float]:
    d = pd.concat([r, b], axis=1).dropna()
    if len(d) < 30:
        return float("nan"), float("nan")
    x, y = d.iloc[:, 1].to_numpy(), d.iloc[:, 0].to_numpy()
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = (y.mean() - beta * x.mean()) * TRADING_DAYS
    return float(alpha), float(beta)


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado): P(true SR > sr_benchmark) given sample length, skew, kurtosis. Per-period SR."""
    x = r.dropna()
    n = len(x)
    if n < 30 or x.std() == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int, sr_trials_std: float | None = None) -> float:
    """DSR: PSR against the expected max Sharpe of `n_trials` unskilled strategies (multiple-testing haircut)."""
    x = r.dropna()
    if len(x) < 30 or n_trials < 1:
        return float("nan")
    sd = sr_trials_std if sr_trials_std is not None else 1 / np.sqrt(len(x))
    emc = 0.5772156649
    n = max(n_trials, 2)
    sr0 = sd * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))
    return probabilistic_sharpe(x, sr0)


def block_bootstrap_ci(x: pd.Series, stat, block: int = 20, n_boot: int = 500, seed: int = 0, ci: float = 0.9) -> tuple[float, float]:
    """Stationary-ish moving-block bootstrap CI for a statistic of a daily return series."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < block * 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    nb = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        sample = np.concatenate([v[s:s + block] for s in starts])[:n]
        stats.append(stat(pd.Series(sample)))
    lo, hi = np.nanquantile(stats, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def summarize(r: pd.Series, bench: pd.Series, turnover: pd.Series | None = None, costs: pd.Series | None = None,
              rf_daily: float = 0.0, n_trials: int = 1) -> dict:
    r = r.dropna()
    bench = bench.reindex(r.index)
    ex = (r - bench).dropna()
    te = ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
    a, b = alpha_beta(r, bench)
    gains, losses = ex[ex > 0].sum(), -ex[ex < 0].sum()
    mdd = max_drawdown(r)
    ar = ann_return(r)
    out = {
        "start": str(r.index[0].date()) if len(r) else None,
        "end": str(r.index[-1].date()) if len(r) else None,
        "n_days": int(len(r)),
        "total_return": float((1 + r).prod() - 1),
        "ann_return": ar,
        "ann_benchmark_return": ann_return(bench),
        "ann_excess_return": float(ar - ann_return(bench)),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(r, rf_daily),
        "sortino": sortino(r, rf_daily),
        "max_drawdown": mdd,
        "calmar": float(ar / abs(mdd)) if mdd < 0 else float("nan"),
        "tracking_error": float(te),
        "information_ratio": float(ex.mean() * TRADING_DAYS / te) if te > 0 else float("nan"),
        "alpha_ann": a,
        "beta": b,
        "es_5pct_daily": expected_shortfall(r, 0.05),
        "profit_factor_excess": float(gains / losses) if losses > 0 else float("nan"),
        "hit_rate_excess": float((ex > 0).mean()) if len(ex) else float("nan"),
        "psr_excess": probabilistic_sharpe(ex),
        "deflated_sharpe_excess": deflated_sharpe(ex, n_trials),
    }
    lo, hi = block_bootstrap_ci(ex, lambda s: s.mean() * TRADING_DAYS / (s.std(ddof=1) * np.sqrt(TRADING_DAYS) + 1e-12))
    out["ir_ci90"] = [lo, hi]
    if turnover is not None:
        out["ann_turnover"] = float(turnover.reindex(r.index).fillna(0).sum() / max(len(r), 1) * TRADING_DAYS)
    if costs is not None:
        out["ann_cost_drag"] = float(costs.reindex(r.index).fillna(0).sum() / max(len(r), 1) * TRADING_DAYS)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in out.items()}


def regime_labels(bench_px: pd.Series) -> pd.Series:
    """Ex-ante regime label per day from benchmark history only: trend (vs 200d MA) x volatility (vs expanding median)."""
    r = np.log(bench_px).diff()
    vol = r.rolling(20, min_periods=15).std()
    hi_vol = vol > vol.expanding(min_periods=120).median()
    up = bench_px > bench_px.rolling(200, min_periods=150).mean()
    lab = np.where(up, "bull", "bear")
    lab = pd.Series(lab, index=bench_px.index) + np.where(hi_vol, "_highvol", "_lowvol")
    return lab.where(vol.notna() & bench_px.rolling(200, min_periods=150).mean().notna())


def by_regime(r: pd.Series, bench: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    rows = []
    reg = regimes.shift(1).reindex(r.index)  # regime known at the start of the day
    for g in sorted(reg.dropna().unique()):
        m = reg == g
        if m.sum() < 20:
            continue
        rr, bb = r[m], bench.reindex(r.index)[m]
        ex = rr - bb
        rows.append({"regime": g, "days": int(m.sum()), "ann_return": float(rr.mean() * TRADING_DAYS),
                     "ann_benchmark": float(bb.mean() * TRADING_DAYS), "ann_excess": float(ex.mean() * TRADING_DAYS),
                     "information_ratio": float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else None})
    return pd.DataFrame(rows)


def forecast_quality(oos: pd.DataFrame, model: str, horizon: int) -> dict:
    """Forecast diagnostics on OOS rows: IC, calibration of intervals and probabilities."""
    from bigboss.models.distribution import calibration_table, interval_coverage
    from bigboss.models.walkforward import ic_tstat, rank_ic_series

    d = oos.dropna(subset=["y", f"mean_{model}"])
    ics = rank_ic_series(d[f"mean_{model}"], d["y"])
    out = {"model": model, "n": int(len(d)), "rank_ic_mean": float(ics.mean()) if len(ics) else None,
           "rank_ic_t": ic_tstat(ics, horizon), "ic_hit_rate": float((ics > 0).mean()) if len(ics) else None,
           "mae": float((d[f"mean_{model}"] - d["y"]).abs().mean()),
           "direction_accuracy": float((np.sign(d[f"mean_{model}"]) == np.sign(d["y"])).mean())}
    if f"q10_{model}" in d:
        out["interval80_coverage"] = interval_coverage(d["y"], d[f"q10_{model}"], d[f"q90_{model}"])
    if f"p_pos_{model}" in d:
        p = d[f"p_pos_{model}"].to_numpy()
        yb = (d["y"] > 0).to_numpy().astype(float)
        out["brier"] = float(np.mean((p - yb) ** 2))
        out["brier_climatology"] = float(np.mean((yb.mean() - yb) ** 2))
        out["calibration"] = calibration_table(p, yb).to_dict("records")
    return out
