"""Model D: news and event intelligence.

Primary source: SEC filings (8-K items, 10-K/10-Q, Form 4, 13D) with exact acceptance timestamps.
Market effect is estimated by a point-in-time EVENT STUDY: for each event type, the average benchmark-relative
return over the 5 trading days after the event became usable, computed only from events that occurred
before `as_of`. Importance = prior x empirical |effect| significance. Credibility is set by source type.

Leakage controls: events are usable on the next session if accepted after 16:00 ET; anything accepted after
`as_of` is excluded; stale events (> max_age_days) are flagged and not counted as catalysts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CREDIBILITY = {"primary_regulatory_filing": 0.95, "company_press_release": 0.8, "major_newswire": 0.7,
               "aggregator": 0.5, "social_media": 0.2, "user_supplied": 0.5}


def usable_date(accepted_at: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    acc = pd.to_datetime(accepted_at)
    base = acc.dt.normalize() + pd.to_timedelta((acc.dt.hour >= 16).astype(int), unit="D")
    idx = calendar.searchsorted(base.values)
    return pd.Series([calendar[i] if i < len(calendar) else pd.NaT for i in idx], index=accepted_at.index)


def event_study(events: pd.DataFrame, adj: pd.DataFrame, bench_map: dict[str, str], as_of: pd.Timestamp,
                window: int = 5, min_events: int = 8) -> pd.DataFrame:
    """Average excess return over [usable, usable+window) by event type, from events strictly before as_of - window."""
    if events.empty:
        return pd.DataFrame()
    cal = adj.index
    ev = events.copy()
    ev["usable"] = usable_date(ev["accepted_at"], cal)
    ev = ev.dropna(subset=["usable"])
    cutoff_pos = cal.searchsorted(as_of) - window - 1
    if cutoff_pos <= 0:
        return pd.DataFrame()
    ev = ev[ev["usable"] <= cal[cutoff_pos]]
    rows = []
    for r in ev.itertuples():
        s, b = r.symbol, bench_map.get(r.symbol)
        if s not in adj or b not in adj:
            continue
        i = cal.get_loc(r.usable)
        if i + window >= len(cal) or i == 0:
            continue
        # entry at the close BEFORE the usable session (event priced during usable day), exit after window
        rs = adj[s].iloc[i + window - 1] / adj[s].iloc[i - 1] - 1
        rb = adj[b].iloc[i + window - 1] / adj[b].iloc[i - 1] - 1
        if np.isfinite(rs) and np.isfinite(rb):
            rows.append({"event_type": r.event_type, "excess": rs - rb, "abs_excess": abs(rs - rb)})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    g = d.groupby("event_type").agg(n=("excess", "size"), mean_excess=("excess", "mean"), std_excess=("excess", "std"),
                                    mean_abs_excess=("abs_excess", "mean"))
    g["t_stat"] = g["mean_excess"] / (g["std_excess"] / np.sqrt(g["n"]))
    g["reliable"] = g["n"] >= min_events
    return g.reset_index()


def score_events(events: pd.DataFrame, study: pd.DataFrame, as_of: pd.Timestamp, max_age_days: int = 30) -> pd.DataFrame:
    """Score recent events: importance, credibility, staleness, and historically estimated effect."""
    if events.empty:
        return events
    ev = events.copy()
    ev["accepted_at"] = pd.to_datetime(ev["accepted_at"])
    ev = ev[ev["accepted_at"] <= as_of]  # leakage guard
    ev["age_days"] = (as_of - ev["accepted_at"]).dt.days
    ev["stale"] = ev["age_days"] > max_age_days
    ev["credibility_score"] = ev["credibility"].map(CREDIBILITY).fillna(0.5)
    if not study.empty:
        ev = ev.merge(study[["event_type", "n", "mean_excess", "mean_abs_excess", "t_stat", "reliable"]], on="event_type", how="left")
    else:
        for c in ("n", "mean_excess", "mean_abs_excess", "t_stat"):
            ev[c] = np.nan
        ev["reliable"] = False
    emp = np.clip(ev["mean_abs_excess"].fillna(0) / 0.05, 0, 1)
    ev["importance"] = np.where(ev["reliable"].fillna(False).astype(bool), 0.5 * ev["importance_prior"] + 0.5 * emp, ev["importance_prior"])
    ev["importance"] = ev["importance"] * ev["credibility_score"] * np.where(ev["stale"], 0.3, 1.0)
    return ev.sort_values(["stale", "importance"], ascending=[True, False])
