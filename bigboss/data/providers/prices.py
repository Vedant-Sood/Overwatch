"""Keyless fallbacks: Stooq CSV and Yahoo's chart endpoint (used only where reachable)."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd

from bigboss.data.providers.base import HttpClient, PriceProvider, ProviderError, frame
from bigboss.data.providers.yfinance_provider import yahoo_symbol


class StooqProvider(PriceProvider):
    """https://stooq.com daily CSV. Split-adjusted (and for US stocks largely dividend-adjusted, undocumented)."""

    name = "stooq"
    URL = "https://stooq.com/q/d/l/"

    def __init__(self, http: HttpClient):
        self.http = http

    def supports(self, symbol: str) -> bool:
        s = symbol.upper()
        return self.disabled_reason is None and not s.startswith(("XA:", "SYN")) and not s.endswith((".NS", ".BO"))

    @staticmethod
    def stooq_symbol(symbol: str) -> str:
        s = symbol.lower()
        if s.startswith("^"):
            return {"^spx": "^spx", "^ndx": "^ndx", "^dji": "^dji", "^n225": "^nkx", "^ftse": "^ftm", "^gdaxi": "^dax",
                    "^hsi": "^hsi"}.get(s, s)
        return s.replace(".", "-") + ".us"

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        body, _ = self.http.get(self.URL, {"s": self.stooq_symbol(symbol), "i": "d", "d1": pd.Timestamp(start).strftime("%Y%m%d")},
                                max_age_s=6 * 3600)
        if not body.startswith("Date"):
            raise ProviderError(f"stooq {symbol}: {body[:120]!r}")
        df = pd.read_csv(io.StringIO(body)).rename(columns=str.lower)
        if df.empty:
            raise ProviderError(f"stooq {symbol}: no rows")
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start)]
        return frame(df.assign(adj_close=df["close"]), "stooq", "split_adjusted_provider")


class YahooChartProvider(PriceProvider):
    """Yahoo's v8 chart endpoint (needed for NSE/BSE when yfinance is unavailable). Personal research use only."""

    name = "yahoo"
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

    def __init__(self, http: HttpClient):
        self.http = http

    def supports(self, symbol: str) -> bool:
        return self.disabled_reason is None and not symbol.upper().startswith(("XA:", "SYN"))

    @staticmethod
    def parse(body: str) -> pd.DataFrame:
        try:
            res = json.loads(body)["chart"]
        except (json.JSONDecodeError, KeyError) as e:
            raise ProviderError(f"yahoo chart: unexpected response {body[:120]!r}") from e
        if res.get("error") or not res.get("result"):
            raise ProviderError(f"yahoo chart: {res.get('error')}")
        r = res["result"][0]
        tz = r.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
        q = r["indicators"]["quote"][0]
        adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
        dates = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
        df = pd.DataFrame({"date": dates, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                           "adj_close": adj, "volume": q["volume"]}).replace({None: np.nan})
        return df

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        p1 = int(pd.Timestamp(start).timestamp())
        p2 = int(pd.Timestamp.now(tz="UTC").timestamp())
        body, _ = self.http.get(self.URL.format(sym=yahoo_symbol(symbol)),
                                {"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"}, max_age_s=6 * 3600)
        return frame(self.parse(body), "yahoo", "adj_close_split_and_dividend")
