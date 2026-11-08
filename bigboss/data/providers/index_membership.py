"""Historical index membership -> daily point-in-time membership, and per-day price-source selection.

Sources:
* S&P 500: `Vedant-Sood/sp500dataset` (= fja05680/sp500): one row per composition change, `date,tickers` with the
  full comma-separated member list.
* NIFTY 50: "Historical Nifty 50 Constituent Weights" (Bera 2025): wide `DATE,<ticker>...` with index weights in %.

Membership on a day is the latest snapshot dated on or before it (carried forward until the next change).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bigboss.data.providers.local_files import kaggle_file, load_ticker_file


def sp500_snapshots(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rows = [(pd.Timestamp(d), t.strip()) for d, ts in zip(df["date"], df["tickers"]) for t in str(ts).split(",") if t.strip()]
    return pd.DataFrame(rows, columns=["date", "ticker"]).drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)


def nifty_snapshots(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    dcol = next(c for c in df.columns if c.strip().lower() == "date")
    long = df.melt(id_vars=[dcol], var_name="ticker", value_name="weight")
    long["weight"] = pd.to_numeric(long["weight"], errors="coerce")
    long = long[long["weight"] > 0]
    long["date"] = pd.to_datetime(long[dcol])
    return long[["date", "ticker", "weight"]].sort_values(["date", "ticker"]).reset_index(drop=True)


def daily_membership(snap: pd.DataFrame, days) -> pd.DataFrame:
    """(date, ticker[, weight]) for every day on or after the first snapshot."""
    days = pd.DatetimeIndex(days)
    sd = pd.DatetimeIndex(sorted(snap["date"].unique()))
    days = days[days >= sd[0]]
    if len(days) == 0:
        return pd.DataFrame(columns=list(snap.columns))
    which = sd[np.searchsorted(sd, days, side="right") - 1]
    m = pd.DataFrame({"date": days, "snap_date": which})
    out = m.merge(snap.rename(columns={"date": "snap_date"}), on="snap_date").drop(columns="snap_date")
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def spans(snap: pd.DataFrame) -> pd.DataFrame:
    """Contiguous membership spans: ticker, start, end (the first snapshot without it; NaT while still a member)."""
    sd = sorted(snap["date"].unique())
    present = {d: set(g["ticker"]) for d, g in snap.groupby("date")}
    rows, open_ = [], {}
    for d in sd:
        cur = present[d]
        for t in cur - set(open_):
            open_[t] = d
        for t in [t for t in open_ if t not in cur]:
            rows.append((t, open_.pop(t), d))
    rows += [(t, s, pd.NaT) for t, s in open_.items()]
    return pd.DataFrame(rows, columns=["ticker", "start", "end"]).sort_values(["ticker", "start"]).reset_index(drop=True)


def coverage_by_year(md: pd.DataFrame, priced) -> pd.Series:
    """Share of index weight (or of members, without weights) that has prices, averaged over days within each year."""
    d = md.assign(priced=np.asarray(priced, dtype=bool))
    w = d["weight"] if "weight" in d and d["weight"].notna().any() else pd.Series(1.0, index=d.index)
    d = d.assign(w=w, wp=w * d["priced"])
    per_day = d.groupby("date")[["wp", "w"]].sum()
    cov = per_day["wp"] / per_day["w"]
    return cov.groupby(cov.index.year).mean()


def candidate_prices(tickers: list[str], dolthub_px: pd.DataFrame, kaggle_dir: str | Path | None) -> pd.DataFrame:
    """All price candidates per member ticker: DoltHub (symbol = ticker or ticker@YYYYMM) and Kaggle 2017 files (ticker@K)."""
    parts = []
    if dolthub_px is not None and not dolthub_px.empty:
        d = dolthub_px.assign(ticker=dolthub_px["symbol"].str.split("@").str[0])
        d = d[d["ticker"].isin(tickers)].assign(source="dolthub")
        parts.append(d)
    if kaggle_dir:
        for t in tickers:
            f = kaggle_file(kaggle_dir, t)
            if f.exists():
                try:
                    k = load_ticker_file(f)
                except Exception:  # unreadable/empty Kaggle files exist in the dataset
                    continue
                parts.append(k.assign(symbol=f"{t}@K", ticker=t, source="kaggle_2017"))
    if not parts:
        return pd.DataFrame(columns=["symbol", "ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    return pd.concat(parts, ignore_index=True)


def assign_symbols(md: pd.DataFrame, px: pd.DataFrame) -> pd.Series:
    """Per (date, ticker) membership row, the price series to use: DoltHub where it has a bar that day, else Kaggle, else NaN."""
    if px.empty:
        return pd.Series(np.nan, index=md.index, dtype=object)
    have = px[["ticker", "date", "symbol", "source"]].copy()
    have["prio"] = np.where(have["source"] == "dolthub", 0, 1)
    have = have.sort_values("prio").drop_duplicates(["ticker", "date"], keep="first")
    j = md[["date", "ticker"]].merge(have[["ticker", "date", "symbol"]], on=["ticker", "date"], how="left")
    return pd.Series(j["symbol"].to_numpy(dtype=object), index=md.index)
