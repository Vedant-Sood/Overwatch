"""Whole-US-market daily bars from Massive's grouped-daily endpoint, and the point-in-time liquidity universe.

One request returns every US stock for one session, including names that later delisted, so a universe built from it
is free of survivorship bias within the plan's history window (free plan: ~2 years, 5 requests/minute).
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from bigboss.data.providers.base import HttpClient, ProviderError

BASE = "https://api.massive.com"


def point_in_time_universe(bars: pd.DataFrame, top_n: int = 500, lookback: int = 60, min_history: int = 40,
                           min_price: float = 5.0, start=None) -> pd.DataFrame:
    """Top-N symbols per date by trailing median dollar volume, using only information before that date.

    bars: symbol, date, close (raw), volume (raw). A symbol is eligible on date t if it trades on t, had >= min_history
    sessions before t, and its previous close was >= min_price. Liquidity is the median of close x volume over the
    `lookback` sessions ending t-1. Returns (date, symbol, rank, metric).
    """
    b = bars.dropna(subset=["close"])
    dv = (b.assign(dv=b["close"] * b["volume"]).pivot_table(index="date", columns="symbol", values="dv", aggfunc="last")
          .sort_index())
    close = b.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").reindex(index=dv.index, columns=dv.columns)
    trades = close.notna()
    med = dv.rolling(lookback, min_periods=min(lookback, min_history)).median().shift(1)
    hist = trades.cumsum().shift(1).fillna(0) >= min_history
    price_ok = close.ffill().shift(1) >= min_price
    ok = trades & hist & price_ok & med.notna() & (med > 0)
    metric = med.where(ok)
    if start is not None:
        metric = metric[metric.index >= pd.Timestamp(start)]
    rank = metric.rank(axis=1, ascending=False, method="first")
    st = rank.stack()
    st = st[st <= top_n]
    if st.empty:
        return pd.DataFrame(columns=["date", "symbol", "rank", "metric"])
    out = st.rename("rank").reset_index()
    out.columns = ["date", "symbol", "rank"]
    out["rank"] = out["rank"].astype(int)
    m = metric.stack()
    out["metric"] = m.reindex(pd.MultiIndex.from_frame(out[["date", "symbol"]])).to_numpy()
    return out.sort_values(["date", "rank"]).reset_index(drop=True)


class MassiveMarket:
    def __init__(self, http: HttpClient, api_key: str | None = None):
        self.http = http
        self.api_key = api_key

    def _p(self, **kw) -> dict:
        return {**kw, **({"apiKey": self.api_key} if self.api_key else {})}

    @staticmethod
    def parse_grouped(body: str, date: pd.Timestamp) -> pd.DataFrame:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ProviderError(f"massive grouped: not JSON {body[:120]!r}") from e
        if data.get("status") not in ("OK", "DELAYED"):
            raise ProviderError(f"massive grouped {pd.Timestamp(date).date()}: {data.get('message') or data.get('error') or data.get('status')}")
        df = pd.DataFrame(data.get("results") or [])
        if df.empty:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "n_trades"])
        df = df.rename(columns={"T": "symbol", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap",
                                "n": "n_trades"})
        for c in ("vwap", "n_trades"):
            if c not in df:
                df[c] = np.nan
        df["date"] = pd.Timestamp(date).normalize()
        return df[["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "n_trades"]].reset_index(drop=True)

    def grouped(self, date: pd.Timestamp) -> pd.DataFrame:
        d = pd.Timestamp(date).strftime("%Y-%m-%d")
        # Unadjusted bars: splits are applied afterwards from the per-ticker adjusted series where needed.
        body, _ = self.http.get(f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d}", self._p(adjusted="false"),
                                max_age_s=365 * 86400 if pd.Timestamp(date) < pd.Timestamp.today().normalize() - pd.Timedelta(days=3) else 3600)
        return self.parse_grouped(body, date)

    def market_history(self, start, end=None, progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
        days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end or pd.Timestamp.today().normalize() - pd.Timedelta(days=1)))
        parts, failed = [], 0
        for i, d in enumerate(days):
            try:
                g = self.grouped(d)
                if not g.empty:
                    parts.append(g)
            except ProviderError as e:
                failed += 1
                if "NOT_AUTHORIZED" in str(e) or "plan" in str(e).lower():
                    continue  # outside the plan's history window
                if failed > 20 and not parts:
                    raise
            if progress and i % 20 == 0:
                progress(i / max(len(days), 1), f"grouped daily {d.date()}")
        if not parts:
            raise ProviderError("massive: no grouped-daily data in the requested window")
        return pd.concat(parts, ignore_index=True)

    def common_stocks(self) -> list[str]:
        """Common-stock tickers (active and delisted). Each record's type is checked: pagination can drop the filter."""
        out: set[str] = set()
        for active in ("true", "false"):
            url, params = f"{BASE}/v3/reference/tickers", self._p(market="stocks", type="CS", active=active, limit=1000)
            pages = 0
            while url and pages < 200:
                body, _ = self.http.get(url, params, max_age_s=7 * 86400)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    raise ProviderError(f"massive tickers: not JSON {body[:120]!r}") from e
                out.update(r["ticker"] for r in data.get("results", []) or [] if r.get("type") == "CS")
                url, params = data.get("next_url"), self._p()
                pages += 1
        return sorted(out)
