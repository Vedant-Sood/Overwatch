"""SYNTHETIC market generator, for testing the machinery offline. It says nothing about real markets.

Daily simple returns:  r_it = beta_i * m_t + s_{k(i),t} + alpha_it + e_it
  m_t       market factor with two volatility regimes (calm / stressed, persistent)
  s_kt      sector factors (5 sectors)
  alpha_it  the PLANTED signal: a slowly varying, cross-sectionally demeaned AR(1) per asset (half-life ~250 sessions).
            Because it persists, trailing returns (momentum) partially reveal it; `alpha_scale = 0` removes it entirely.
  e_it      idiosyncratic noise with per-asset volatility (fat-tailed, Student t with 5 dof)

SYNMKT is the investable benchmark: an equal-weight index of all assets, rebalanced daily (the mean of simple returns).
Everything this module produces is labelled source="synthetic" and lives only in the SYNTHETIC store.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

ALPHA_DAILY_SD = 0.00025  # cross-sectional sd of the planted daily alpha at alpha_scale = 1 (~6%/yr)
BENCHMARK = "SYNMKT"
ADJUSTMENT = "SYNTHETIC (simulated market, not real data)"


def generate_market(n_assets: int = 30, n_days: int = 2520, seed: int = 7, alpha_scale: float = 2.0,
                    start: str = "2014-01-02") -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    n_sec = 5

    # market factor with persistent calm/stressed volatility regimes
    stressed = np.zeros(n_days, dtype=bool)
    state = False
    for t in range(n_days):
        if rng.random() < (0.02 if state else 0.004):
            state = not state
        stressed[t] = state
    m_vol = np.where(stressed, 0.018, 0.008)
    m = rng.normal(0.0004, 1.0, n_days) * m_vol
    sec = rng.normal(0.0, 0.004, (n_days, n_sec))

    beta = rng.uniform(0.8, 1.2, n_assets)
    sector = np.arange(n_assets) % n_sec
    idio_vol = rng.uniform(0.008, 0.016, n_assets)
    e = rng.standard_t(5, (n_days, n_assets)) / np.sqrt(5 / 3) * idio_vol

    phi = 0.5 ** (1 / 250)
    sd = ALPHA_DAILY_SD * alpha_scale
    alpha = np.zeros((n_days, n_assets))
    a = rng.normal(0.0, sd, n_assets)
    for t in range(n_days):
        a = phi * a + np.sqrt(1 - phi ** 2) * rng.normal(0.0, sd, n_assets)
        alpha[t] = a - a.mean()  # relative signal only: the benchmark's return is unaffected

    r = beta * m[:, None] + sec[:, sector] + alpha + e
    r = np.clip(r, -0.5, 0.5)
    bench_r = r.mean(axis=1)

    rows = []
    syms = [f"SYN{i:03d}" for i in range(n_assets)]
    p0 = rng.uniform(20, 200, n_assets)
    adv = rng.uniform(5e7, 5e8, n_assets)
    for i, s in enumerate(syms):
        rows.append(_ohlcv(s, dates, r[:, i], p0[i], idio_vol[i], adv[i], rng))
    rows.append(_ohlcv(BENCHMARK, dates, bench_r, 100.0, 0.004, 5e9, rng))
    prices = pd.concat(rows, ignore_index=True)

    meta = pd.DataFrame({
        "symbol": syms + [BENCHMARK],
        "name": [f"Synthetic asset {i:03d}" for i in range(n_assets)] + ["Synthetic equal-weight market index"],
        "asset_class": ["equity"] * n_assets + ["etf"],
        "sector": [f"Synthetic sector {k}" for k in sector] + ["Benchmark"],
        "benchmark": BENCHMARK, "market": "US", "currency": "USD", "exchange": "SYNTHETIC", "source": "synthetic",
        "note": "SYNTHETIC - simulated data for testing the platform only",
    })
    truth = pd.DataFrame(alpha, index=dates, columns=syms)
    return {"prices": prices, "symbols": meta, "alpha": truth,
            "params": {"n_assets": n_assets, "n_days": n_days, "seed": seed, "alpha_scale": alpha_scale,
                       "alpha_daily_sd": sd, "alpha_half_life_days": 250}}


def _ohlcv(symbol: str, dates: pd.DatetimeIndex, r: np.ndarray, p0: float, vol: float, adv: float,
           rng: np.random.Generator) -> pd.DataFrame:
    close = p0 * np.cumprod(1 + r)
    prev = np.r_[p0, close[:-1]]
    op = prev * np.exp(rng.normal(0, 0.3 * vol, len(r)))
    hi = np.maximum(op, close) * np.exp(np.abs(rng.normal(0, 0.5 * vol, len(r))))
    lo = np.minimum(op, close) * np.exp(-np.abs(rng.normal(0, 0.5 * vol, len(r))))
    dollar = adv * np.exp(rng.normal(0, 0.35, len(r)) + 8 * np.abs(r))
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": op, "high": hi, "low": lo, "close": close, "adj_close": close,
                         "volume": dollar / close, "source": "synthetic", "adjustment": ADJUSTMENT})


def load_into_store(store, n_assets: int = 30, n_days: int = 2520, seed: int = 7, alpha_scale: float = 2.0) -> dict:
    """Replace the SYNTHETIC store's market with a freshly generated one."""
    if store.mode != "SYNTHETIC":
        raise ValueError("synthetic data can only be loaded into the SYNTHETIC store")
    mk = generate_market(n_assets=n_assets, n_days=n_days, seed=seed, alpha_scale=alpha_scale)
    store.clear(["prices", "symbols", "universe_membership", "validation_issues", "macro", "fundamentals", "events", "news"])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    store.upsert("prices", mk["prices"].assign(fetched_at=now), ["symbol", "date"])
    store.upsert("symbols", mk["symbols"].assign(updated_at=now), ["symbol"])
    run_id = now.strftime("%Y%m%dT%H%M%S") + "-synthetic"
    store.append("ingestion_log", pd.DataFrame([{"run_id": run_id, "ts": now, "kind": "prices", "symbol": "*", "provider": "synthetic",
                                                 "status": "OK", "n_rows": len(mk["prices"]), "detail": str(mk["params"])}]))
    return mk["params"]
