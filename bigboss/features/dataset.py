"""Assemble the modelling dataset: features (PIT) + forward labels, indexed by (date, symbol)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bigboss.config import Config
from bigboss.data.store import Store
from bigboss.features.context import event_features, macro_features
from bigboss.features.fundamental import fundamental_features
from bigboss.features.labels import forward_labels
from bigboss.features.technical import FEATURE_GROUPS, market_features, technical_features


@dataclass
class Dataset:
    frame: pd.DataFrame                 # (date, symbol) -> features + labels
    feature_cols: list[str]
    groups: dict[str, list[str]]
    benchmark_map: dict[str, str]
    tradeable: list[str]
    data_mode: str
    sources: dict[str, str] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)  # symbol -> validation ERROR that excluded it

    def label(self, h: int) -> str:
        return f"fwd_excess_{h}"


def build_dataset(store: Store, cfg: Config, symbols: list[str] | None = None, min_history: int = 260) -> Dataset:
    prices = store.prices(symbols)
    if prices.empty:
        raise ValueError(f"no price data in the {store.mode} store - run ingestion first")
    meta = store.symbols().set_index("symbol")
    counts = prices.groupby("symbol").size()
    try:
        has_membership = int(store.query("SELECT COUNT(*) AS n FROM universe_membership")["n"].iloc[0]) > 0
    except Exception:  # older stores without the table
        has_membership = False
    if has_membership:
        # The point-in-time universe already requires >= 40 sessions of history at each date. A global
        # history filter would silently drop names that listed late or delisted early (survivorship bias).
        min_history = min(min_history, 40)
    keep = counts[counts >= min_history].index
    bad = store.error_symbols()  # ERROR-level validation issues exclude a symbol until data is fixed/re-ingested
    keep = [s for s in keep if s not in bad]
    prices = prices[prices["symbol"].isin(keep)]
    syms = sorted(prices["symbol"].unique())

    bmap = {s: (meta.loc[s, "benchmark"] if s in meta.index and pd.notna(meta.loc[s, "benchmark"]) else cfg.benchmark_for(s)) for s in syms}
    broken = {b for b in bmap.values() if b in bad}
    if broken:
        raise ValueError(f"benchmark(s) failed data validation and cannot be used: "
                         + "; ".join(f"{b} ({bad[b][:160]})" for b in sorted(broken)))
    benches = sorted({b for b in bmap.values() if b in syms})
    primary_bench = benches[0] if benches else None
    if "SPY" in benches:
        primary_bench = "SPY"
    tradeable = [s for s in syms if not s.startswith("^") and s not in benches]

    tech = technical_features(prices, primary_bench)
    # Assets benchmarked to a different market (e.g. NSE vs SPY): recompute relative features against their own benchmark.
    mkt = market_features(prices, primary_bench)
    calendar = tech.index.get_level_values("date").unique().sort_values()
    macro = macro_features(store.macro(), calendar)
    ctx = mkt.join(macro, how="left")
    frame = tech.join(ctx, on="date")

    fund = fundamental_features(store.fundamentals(), prices)
    if not fund.empty:
        frame = frame.join(fund, how="left")
    ev = event_features(store.events(limit=1_000_000), calendar, tradeable)
    if not ev.empty:
        frame = frame.join(ev, how="left")

    horizons = cfg.get("forecast.horizons")
    labels = forward_labels(prices, horizons, cfg.get("forecast.execution_delay_days", 1), bmap)
    frame = frame.join(labels, how="left")
    frame = frame[frame.index.get_level_values("symbol").isin(tradeable)]
    # Point-in-time universe: a managed symbol is a candidate on date t only if it was a member at t.
    try:
        mem = store.membership()
    except Exception:  # older stores without the table
        mem = pd.DataFrame()
    if not mem.empty:
        managed = set(mem["symbol"])
        idx = pd.MultiIndex.from_frame(mem[["date", "symbol"]])
        sym = frame.index.get_level_values("symbol")
        frame = frame[~sym.isin(managed) | frame.index.isin(idx)]

    label_like = [c for c in frame.columns if c.startswith(("fwd_", "label_end_"))]
    feature_cols = [c for c in frame.columns if c not in label_like and frame[c].notna().mean() > 0.05]
    groups = {g: [c for c in cols if c in feature_cols] for g, cols in FEATURE_GROUPS.items()}
    groups = {g: c for g, c in groups.items() if c}
    sources = prices.groupby("symbol")["source"].first().to_dict()
    return Dataset(frame.sort_index(), feature_cols, groups, bmap, tradeable, store.mode, sources, bad)


def train_matrix(ds: Dataset, h: int, dates: pd.DatetimeIndex | None = None, require_label: bool = True):
    df = ds.frame
    if dates is not None:
        df = df[df.index.get_level_values("date").isin(dates)]
    y = df[ds.label(h)]
    if require_label:
        df = df[y.notna()]
        y = y[y.notna()]
    X = df[ds.feature_cols].astype(np.float32)
    return X, y, df
