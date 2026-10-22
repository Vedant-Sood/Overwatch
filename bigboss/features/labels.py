"""Forward return labels relative to a benchmark, with explicit execution delay.

Decision at close of day t. Execution at close of t + delay. The label for horizon h is
    R = P[t+delay+h] / P[t+delay] - 1
and the excess label subtracts the benchmark's return over the same window. `label_end` is the date
the label becomes known; purging in the walk-forward splitter uses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_labels(prices: pd.DataFrame, horizons: list[int], delay: int, benchmark_map: dict[str, str]) -> pd.DataFrame:
    adj = prices.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    dates = adj.index
    out = {}
    for h in horizons:
        entry = adj.shift(-delay)
        exit_ = adj.shift(-(delay + h))
        fwd = exit_ / entry - 1
        out[f"fwd_ret_{h}"] = fwd
        bench = pd.DataFrame({s: fwd[benchmark_map.get(s, s)] if benchmark_map.get(s, s) in fwd else np.nan for s in adj.columns},
                             index=dates)
        out[f"fwd_bench_{h}"] = bench
        out[f"fwd_excess_{h}"] = fwd - bench
        end_idx = np.arange(len(dates)) + delay + h
        end_dates = pd.Series([dates[i] if i < len(dates) else pd.NaT for i in end_idx], index=dates)
        out[f"label_end_{h}"] = pd.DataFrame({s: end_dates for s in adj.columns}, index=dates)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in out.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    return long
