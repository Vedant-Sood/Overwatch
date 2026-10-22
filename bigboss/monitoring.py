"""Prediction/outcome logging, drift detection, and live performance monitoring."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bigboss.data.store import Store

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions_log (
    version VARCHAR, horizon INTEGER, as_of DATE, symbol VARCHAR, mean DOUBLE, q10 DOUBLE, q50 DOUBLE, q90 DOUBLE,
    p_pos DOUBLE, action VARCHAR, expected_net_excess DOUBLE, suggested_weight DOUBLE, logged_at TIMESTAMP,
    realized_excess DOUBLE, resolved_at TIMESTAMP,
    PRIMARY KEY (version, as_of, symbol)
);
"""


def ensure_tables(store: Store) -> None:
    with store.connect() as con:
        con.execute(LOG_SCHEMA)


def log_predictions(store: Store, decisions: pd.DataFrame, version: str, horizon: int, as_of: pd.Timestamp) -> int:
    ensure_tables(store)
    df = pd.DataFrame({
        "version": version, "horizon": horizon, "as_of": as_of, "symbol": decisions.index,
        "mean": decisions["mean"].to_numpy(), "q10": decisions["q10"].to_numpy(), "q50": decisions["q50"].to_numpy(),
        "q90": decisions["q90"].to_numpy(), "p_pos": decisions["p_pos"].to_numpy(), "action": decisions["action"].to_numpy(),
        "expected_net_excess": decisions["expected_net_excess"].to_numpy(), "suggested_weight": decisions["suggested_weight"].to_numpy(),
        "logged_at": datetime.now(timezone.utc).replace(tzinfo=None), "realized_excess": np.nan, "resolved_at": pd.NaT,
    })
    return store.upsert("predictions_log", df, ["version", "as_of", "symbol"])


def resolve_outcomes(store: Store, delay: int = 1) -> int:
    """Fill realized excess returns for logged predictions whose horizon has elapsed (Evaluation Agent's tool)."""
    ensure_tables(store)
    pending = store.query("SELECT * FROM predictions_log WHERE realized_excess IS NULL")
    if pending.empty:
        return 0
    adj = store.panel("adj_close")
    meta = store.symbols().set_index("symbol")
    cal = adj.index
    done = []
    for r in pending.itertuples():
        t = pd.Timestamp(r.as_of)
        i = cal.searchsorted(t)
        if i >= len(cal) or cal[i] != t or i + delay + r.horizon >= len(cal):
            continue
        b = meta.loc[r.symbol, "benchmark"] if r.symbol in meta.index else None
        if r.symbol not in adj or b not in adj:
            continue
        e, x = i + delay, i + delay + r.horizon
        rs = adj[r.symbol].iloc[x] / adj[r.symbol].iloc[e] - 1
        rb = adj[b].iloc[x] / adj[b].iloc[e] - 1
        if np.isfinite(rs) and np.isfinite(rb):
            done.append({"version": r.version, "as_of": t, "symbol": r.symbol, "realized_excess": rs - rb,
                         "resolved_at": datetime.now(timezone.utc).replace(tzinfo=None)})
    if done:
        upd = pd.DataFrame(done)
        with store.connect() as con:
            con.register("upd", upd)
            con.execute("""UPDATE predictions_log p SET realized_excess = upd.realized_excess, resolved_at = upd.resolved_at
                           FROM upd WHERE p.version = upd.version AND p.as_of = upd.as_of AND p.symbol = upd.symbol""")
            con.unregister("upd")
    return len(done)


def live_performance(store: Store) -> dict:
    ensure_tables(store)
    d = store.query("SELECT * FROM predictions_log WHERE realized_excess IS NOT NULL")
    total = store.query("SELECT COUNT(*) AS n FROM predictions_log")["n"].iloc[0]
    if d.empty:
        return {"n_logged": int(total), "n_resolved": 0, "note": "no resolved predictions yet"}
    out = {"n_logged": int(total), "n_resolved": int(len(d))}
    ic = d.groupby("as_of").apply(lambda g: g["mean"].rank().corr(g["realized_excess"].rank()) if len(g) > 4 else np.nan).dropna()
    out["rank_ic_mean"] = float(ic.mean()) if len(ic) else None
    out["coverage80"] = float(((d["realized_excess"] >= d["q10"]) & (d["realized_excess"] <= d["q90"])).mean())
    out["brier"] = float(((d["p_pos"] - (d["realized_excess"] > 0)) ** 2).mean())
    by_action = d.groupby("action")["realized_excess"].agg(["mean", "count"]).reset_index()
    out["realized_by_action"] = by_action.to_dict("records")
    return out


def psi(reference_deciles: pd.Series, current: pd.Series) -> float:
    """Population Stability Index of `current` against bins defined by training deciles."""
    edges = np.unique(reference_deciles.dropna().to_numpy())
    cur = current.dropna().to_numpy()
    if len(edges) < 3 or len(cur) < 20:
        return float("nan")
    inner = edges[1:-1]
    exp_frac = np.full(len(inner) + 1, 1 / (len(inner) + 1))
    counts = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1)
    act = np.maximum(counts / counts.sum(), 1e-4)
    return float(np.sum((act - exp_frac) * np.log(act / exp_frac)))


def feature_drift(reference: pd.DataFrame, frame: pd.DataFrame, feature_cols: list[str], lookback_days: int = 126,
                  warn: float = 0.1, alert: float = 0.25) -> pd.DataFrame:
    """PSI per feature over the last `lookback_days`.

    kind = 'asset' for cross-sectional features, 'market_state' for features identical across assets on a date
    (market/macro). Market-state drift usually means a regime change, not a broken input pipeline.
    """
    dates = frame.index.get_level_values("date").unique().sort_values()[-lookback_days:]
    recent = frame.loc[frame.index.get_level_values("date").isin(dates)]
    rows = []
    for c in feature_cols:
        if c not in reference.index or c not in recent:
            continue
        per_date_nuniq = recent[c].groupby(level="date").nunique().median()
        kind = "market_state" if per_date_nuniq <= 1 else "asset"
        v = psi(reference.loc[c], recent[c])
        rows.append({"feature": c, "kind": kind, "psi": v, "status": "alert" if v >= alert else "warn" if v >= warn else "ok"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
