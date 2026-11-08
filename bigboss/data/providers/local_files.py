"""Local bulk files (e.g. the Vedant-Sood/Training-Data repository): the ETF prices CSV and per-ticker files.

Per-ticker formats understood: epoch `timestamp` + lowercase OHLCV (the zipped datasets), `Date` + capitalized OHLCV
(+ optional Symbol / Adj Close), and the Kaggle "Huge Stock Market Dataset" `.txt` files (Date, Open, ..., OpenInt).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bigboss.data.providers.base import ProviderError

ETF_COLUMNS = ["fund_symbol", "price_date", "open", "high", "low", "close", "adj_close", "volume"]


def load_ticker_file(path: str | Path) -> pd.DataFrame:
    """One ticker's daily bars -> date, open, high, low, close, adj_close, volume (US session dates)."""
    df = pd.read_csv(path)
    if df.empty:
        raise ProviderError(f"{path}: empty file")
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    if "timestamp" in df:
        t = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        df["date"] = t.dt.normalize()
    elif "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    else:
        raise ProviderError(f"{path}: no date or timestamp column")
    if "adj_close" not in df:
        df["adj_close"] = df["close"]
    out = df[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def kaggle_file(kaggle_dir: str | Path, ticker: str) -> Path:
    return Path(kaggle_dir) / f"{ticker.lower().replace('.', '-')}.us.txt"


def load_etf_file(path: str | Path, start=None, symbols: list[str] | None = None) -> pd.DataFrame:
    """The bulk ETF prices CSV -> symbol, date, open, high, low, close (raw), adj_close (dividend-adjusted), volume."""
    df = pd.read_csv(path, usecols=lambda c: c in ETF_COLUMNS)
    missing = set(ETF_COLUMNS) - set(df.columns)
    if missing:
        raise ProviderError(f"{path}: missing columns {sorted(missing)}")
    df = df.rename(columns={"fund_symbol": "symbol", "price_date": "date"})
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.dropna(subset=["close"]).drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)


def load_etf_meta(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["symbol", "category", "name", "family"])
    m = pd.read_csv(path)
    cols = {"fund_symbol": "symbol", "fund_category": "category", "fund_long_name": "name", "fund_family": "family"}
    m = m.rename(columns=cols)
    for c in cols.values():
        if c not in m:
            m[c] = None
    return m[list(cols.values())].drop_duplicates("symbol")
