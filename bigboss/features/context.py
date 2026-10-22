"""Macro and event features, as-of joined on availability timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bigboss.features.technical import _reg

MACRO_TRANSFORMS = {
    # series -> list of (feature_name, transform)
    "DGS10": [("macro_dgs10", "level"), ("macro_dgs10_chg_60", "diff60")],
    "T10Y2Y": [("macro_term_spread", "level")],
    "VIXCLS": [("macro_vix", "level"), ("macro_vix_z_252", "z252")],
    "BAMLH0A0HYM2": [("macro_hy_spread", "level"), ("macro_hy_spread_chg_60", "diff60")],
    "DTB3": [("macro_tbill", "level")],
    "DCOILWTICO": [("macro_oil_ret_60", "ret60")],
}


def macro_features(macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro features on the trading calendar, using only values whose available_at <= date."""
    out = pd.DataFrame(index=calendar)
    if macro.empty:
        return out
    for sid, specs in MACRO_TRANSFORMS.items():
        s = macro[macro["series_id"] == sid].sort_values("date")
        if s.empty:
            continue
        # Transform on the observation timeline, then shift to availability time.
        v = s.set_index("date")["value"]
        for name, how in specs:
            if how == "level":
                x = v
            elif how == "diff60":
                x = v - v.shift(60)
            elif how == "ret60":
                x = v / v.shift(60) - 1
            elif how == "z252":
                x = (v - v.rolling(252, min_periods=120).mean()) / v.rolling(252, min_periods=120).std()
            avail = pd.DataFrame({"available_at": s["available_at"].values, name: x.values}).dropna().sort_values("available_at")
            cal = pd.DataFrame({"date": calendar})
            j = pd.merge_asof(cal, avail, left_on="date", right_on="available_at", direction="backward")
            out[_reg("macro", name)] = j[name].values
    return out


def event_features(events: pd.DataFrame, calendar: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Filing-event features. A filing accepted after 16:00 ET is only usable on the next trading day."""
    names = [_reg("events", n) for n in ("days_since_earnings", "filings_30d", "max_importance_5d", "restatement_flag_365d")]
    if events.empty:
        return pd.DataFrame(columns=names)
    ev = events.copy()
    ev["accepted_at"] = pd.to_datetime(ev["accepted_at"])
    after_close = ev["accepted_at"].dt.hour >= 16
    ev["usable"] = ev["accepted_at"].dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    cal = pd.Series(calendar, index=calendar)
    out = []
    for sym in symbols:
        e = ev[ev["symbol"] == sym]
        if e.empty:
            continue
        # snap usable dates to the next trading day on the calendar
        idx = calendar.searchsorted(e["usable"].values)
        ok = idx < len(calendar)
        e = e[ok].assign(td=calendar[idx[ok]])
        df = pd.DataFrame(index=calendar)
        earn = e[e["event_type"] == "earnings_release"]["td"].drop_duplicates().sort_values()
        last_earn = pd.merge_asof(pd.DataFrame({"date": calendar}), pd.DataFrame({"date": earn.values, "e": earn.values}),
                                  on="date", direction="backward")["e"]
        df["days_since_earnings"] = (cal.values - last_earn.values).astype("timedelta64[D]").astype(float)
        counts = e.groupby("td").size().reindex(calendar, fill_value=0)
        df["filings_30d"] = counts.rolling(21, min_periods=1).sum().values
        imp = e.groupby("td")["importance_prior"].max().reindex(calendar, fill_value=0)
        df["max_importance_5d"] = imp.rolling(5, min_periods=1).max().values
        rs = e[e["event_type"] == "non_reliance_restatement"].groupby("td").size().reindex(calendar, fill_value=0)
        df["restatement_flag_365d"] = (rs.rolling(252, min_periods=1).sum() > 0).astype(float).values
        df["symbol"] = sym
        df.index.name = "date"
        out.append(df.reset_index().set_index(["date", "symbol"]))
    return pd.concat(out) if out else pd.DataFrame(columns=names)
