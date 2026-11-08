"""Data validation and the look-ahead (truncation) test.

ERROR-level issues exclude a symbol from modelling until it is re-ingested clean (see Store.error_symbols);
WARN-level issues are shown in `bigboss status` and the Data page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Price ratios (new / old) that a split, reverse split or bonus issue produces.
_SPLIT_FACTORS = [2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100, 1.5, 1.25, 4 / 3]
SPLIT_RATIOS = sorted({r for f in _SPLIT_FACTORS for r in (1 / f, f)})


def looks_like_split(ratio: float, tol: float = 0.03) -> bool:
    """True when a close-to-close price ratio is within `tol` (relative) of a common split ratio."""
    if not np.isfinite(ratio) or ratio <= 0:
        return False
    return any(abs(ratio / r - 1) <= tol for r in SPLIT_RATIOS)


@dataclass
class Issue:
    symbol: str
    check_name: str
    severity: str  # ERROR | WARN | INFO
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def validate_prices(df: pd.DataFrame, symbol: str, today: pd.Timestamp | None = None, max_staleness_days: int = 5,
                    max_gap_days: int = 10, extreme_move: float = 0.4) -> list[Issue]:
    """Checks: unique dates, positive prices, OHLC consistency, extreme moves (unadjusted splits), gaps, freshness."""
    out: list[Issue] = []
    if df.empty:
        return [Issue(symbol, "non_empty", "ERROR", "no rows")]
    d = df.sort_values("date")
    dup = d["date"].duplicated().sum()
    if dup:
        out.append(Issue(symbol, "unique_dates", "ERROR", f"{dup} duplicated dates"))
    d = d.drop_duplicates("date", keep="last")
    px_cols = [c for c in ("open", "high", "low", "close", "adj_close") if c in d]
    bad = (d[px_cols] <= 0).any(axis=1)
    if bad.any():
        out.append(Issue(symbol, "positive_prices", "ERROR",
                         f"{int(bad.sum())} rows with zero/negative prices (first {d.loc[bad, 'date'].iloc[0].date()})"))
    if {"high", "low", "close"} <= set(d) and d[["high", "low"]].notna().all(axis=None):
        tol = 1e-6 * d["close"].abs()
        incons = (d["high"] < d["low"] - tol) | (d["close"] > d["high"] + tol) | (d["close"] < d["low"] - tol)
        if incons.any():
            out.append(Issue(symbol, "ohlc_consistency", "WARN", f"{int(incons.sum())} rows with close outside [low, high]"))
    s = d.set_index("date")["adj_close" if "adj_close" in d else "close"]
    s = s[s > 0]
    ratio = s / s.shift(1)
    ext = ratio[(ratio - 1).abs() >= extreme_move]
    if len(ext):
        splitlike = [(t, r) for t, r in ext.items() if looks_like_split(r)]
        if splitlike:
            t, r = splitlike[0]
            out.append(Issue(symbol, "extreme_moves", "ERROR",
                             f"{len(splitlike)} split-like jumps in the adjusted series (first {t.date()}: x{r:.3f}) - unadjusted split?"))
        others = [(t, r) for t, r in ext.items() if not looks_like_split(r)]
        if others:
            t, r = others[0]
            out.append(Issue(symbol, "extreme_moves", "WARN", f"{len(others)} daily moves beyond ±{extreme_move:.0%} (first {t.date()}: {r - 1:+.1%})"))
    gaps = s.index.to_series().diff().dt.days
    big = gaps[gaps > max_gap_days * 7 / 5]
    if len(big):
        out.append(Issue(symbol, "gaps", "WARN", f"{len(big)} gaps longer than {max_gap_days} sessions (largest {int(big.max())} days)"))
    if today is not None:
        age = (pd.Timestamp(today).normalize() - s.index.max()).days
        if age > max_staleness_days + 2:  # allow for weekends
            out.append(Issue(symbol, "freshness", "WARN", f"last bar {s.index.max().date()} is {age} days old"))
    return out


def assert_no_lookahead(feature_fn: Callable[[pd.DataFrame], pd.DataFrame], prices: pd.DataFrame, probes: list[pd.Timestamp],
                        atol: float = 1e-9) -> None:
    """Features at t must be identical whether or not rows after t exist.

    Both directions count as a leak: a value that changes, and a value that is NaN without future rows but present with them.
    """
    full = feature_fn(prices)
    for t in probes:
        t = pd.Timestamp(t)
        trunc = feature_fn(prices[prices["date"] <= t])
        a = full.xs(t, level="date") if isinstance(full.index, pd.MultiIndex) else full.loc[[t]]
        b = trunc.xs(t, level="date") if isinstance(trunc.index, pd.MultiIndex) else trunc.loc[[t]]
        a, b = a.align(b, join="outer")
        for c in a.columns:
            x = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float)
            same = np.isclose(x, y, atol=atol, rtol=1e-7, equal_nan=True)
            if not same.all():
                bad = a.index[~same][:5].tolist()
                raise AssertionError(f"look-ahead in feature '{c}' at {t.date()}: differs for {bad} when future rows are removed")
