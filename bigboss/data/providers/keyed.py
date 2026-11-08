"""Providers that need a (free) API key: Financial Modeling Prep, Massive (formerly Polygon.io), Alpha Vantage.

Keys come only from environment variables. When no key is visible the adapter still tries: in some environments an
egress proxy injects the credential. A provider that answers "no credentials" is disabled for the rest of the run.
"""

from __future__ import annotations

import io
import json
import os

import pandas as pd

from bigboss.data.providers.base import HttpClient, PriceProvider, ProviderError, frame, to_ny_naive

_AUTH_HINTS = ("api key", "apikey", "not authorized", "unauthorized", "invalid key", "credentials", "forbidden", "401", "403")


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _maybe_disable(provider: PriceProvider, err: Exception) -> None:
    msg = str(err).lower()
    if any(h in msg for h in _AUTH_HINTS):
        provider.disabled_reason = f"no credentials / not authorized: {str(err)[:160]}"


class FmpProvider(PriceProvider):
    """FMP end-of-day prices; the dividend-adjusted endpoint gives a total-return `adj_close`. Free plan: 250 calls/day."""

    name = "fmp"
    BASE = "https://financialmodelingprep.com/stable"
    WINDOW_YEARS = 5  # long histories are fetched in windows (the API caps rows per request)

    def __init__(self, http: HttpClient | None, api_key: str | None = None):
        self.http = http
        self.api_key = api_key

    @classmethod
    def from_env(cls, http: HttpClient) -> "FmpProvider":
        return cls(http, _env("FMP_API_KEY", "FMP_API"))

    def supports(self, symbol: str) -> bool:
        return self.disabled_reason is None and not symbol.upper().startswith(("XA:", "SYN"))

    def _params(self, **kw) -> dict:
        return {**kw, **({"apikey": self.api_key} if self.api_key else {})}

    @staticmethod
    def _load(body: str):
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ProviderError(f"fmp: not JSON: {body[:120]!r}") from e
        if isinstance(data, dict):
            if "Error Message" in data or "error" in data:
                raise ProviderError(f"fmp: {data.get('Error Message') or data.get('error')}")
            data = data.get("historical", [])
        return data

    @classmethod
    def parse(cls, body: str) -> pd.DataFrame:
        data = cls._load(body)
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    @classmethod
    def parse_dividend_adjusted(cls, body: str) -> pd.DataFrame:
        df = pd.DataFrame(cls._load(body))
        if df.empty:
            return pd.DataFrame(columns=["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _windows(self, start: str) -> list[dict]:
        s = pd.Timestamp(start)
        end = pd.Timestamp.today().normalize()
        out = []
        while s <= end:
            e = min(s + pd.DateOffset(years=self.WINDOW_YEARS) - pd.Timedelta(days=1), end)
            out.append({"from": s.strftime("%Y-%m-%d"), "to": e.strftime("%Y-%m-%d")})
            s = e + pd.Timedelta(days=1)
        return out

    def _get_all(self, path: str, symbol: str, start: str, parser) -> pd.DataFrame:
        parts = []
        for w in self._windows(start):
            try:
                body, _ = self.http.get(f"{self.BASE}/{path}", self._params(symbol=symbol, **w), max_age_s=6 * 3600)
            except ProviderError as e:
                _maybe_disable(self, e)
                raise
            parts.append(parser(body))
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
        if df.empty:
            raise ProviderError(f"fmp {symbol}: no rows")
        return df.drop_duplicates("date").sort_values("date")

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        raw = self._get_all("historical-price-eod/full", symbol, start, self.parse)
        try:
            adj = self._get_all("historical-price-eod/dividend-adjusted", symbol, start, self.parse_dividend_adjusted)
            df = raw.merge(adj[["date", "adjClose"]], on="date", how="left").rename(columns={"adjClose": "adj_close"})
            adjustment = "adj_close_split_and_dividend"
        except ProviderError:
            df, adjustment = raw.assign(adj_close=raw["close"]), "split_adjusted_only"
        df["adj_close"] = df["adj_close"].fillna(df["close"]) if "adj_close" in df else df["close"]
        return frame(df, "fmp", adjustment)


class MassiveProvider(PriceProvider):
    """Massive (formerly Polygon.io) daily aggregates for US stocks/ETFs. Free plan: ~2 years, 5 calls/min. Split-adjusted only."""

    name = "massive"
    BASE = "https://api.massive.com"

    def __init__(self, http: HttpClient | None, api_key: str | None = None):
        self.http = http
        self.api_key = api_key

    @classmethod
    def from_env(cls, http: HttpClient) -> "MassiveProvider":
        return cls(http, _env("MASSIVE_API_KEY", "POLYGON_API_KEY"))

    def supports(self, symbol: str) -> bool:
        s = symbol.upper()
        return self.disabled_reason is None and not s.startswith(("XA:", "^", "SYN")) and not s.endswith((".NS", ".BO"))

    def _params(self, **kw) -> dict:
        return {**kw, **({"apiKey": self.api_key} if self.api_key else {})}

    @staticmethod
    def parse(body: str) -> pd.DataFrame:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ProviderError(f"massive: not JSON: {body[:120]!r}") from e
        if data.get("status") not in ("OK", "DELAYED"):
            raise ProviderError(f"massive: {data.get('error') or data.get('message') or data.get('status')}")
        df = pd.DataFrame(data.get("results") or [])
        if df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = to_ny_naive(pd.to_datetime(df["t"], unit="ms", utc=True)).dt.normalize()
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
            ["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        url = f"{self.BASE}/v2/aggs/ticker/{symbol.upper()}/range/1/day/{pd.Timestamp(start).strftime('%Y-%m-%d')}/{end}"
        try:
            body, _ = self.http.get(url, self._params(adjusted="true", sort="asc", limit=50000), max_age_s=6 * 3600)
            df = self.parse(body)
        except ProviderError as e:
            _maybe_disable(self, e)
            raise
        if df.empty:
            raise ProviderError(f"massive {symbol}: no rows")
        return frame(df.assign(adj_close=df["close"]), "massive", "split_adjusted_only")


class AlphaVantageProvider(PriceProvider):
    """Alpha Vantage TIME_SERIES_DAILY: free data is UNADJUSTED (flagged), so it is a last-resort fallback."""

    name = "alphavantage"
    URL = "https://www.alphavantage.co/query"

    def __init__(self, http: HttpClient | None, api_key: str | None = None):
        self.http = http
        self.api_key = api_key

    @classmethod
    def from_env(cls, http: HttpClient) -> "AlphaVantageProvider":
        return cls(http, _env("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY"))

    def supports(self, symbol: str) -> bool:
        s = symbol.upper()
        return self.disabled_reason is None and bool(self.api_key) and not s.startswith(("XA:", "^", "SYN"))

    @staticmethod
    def parse(body: str) -> pd.DataFrame:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ProviderError(f"alphavantage: not JSON: {body[:120]!r}") from e
        ts = data.get("Time Series (Daily)")
        if not ts:
            raise ProviderError(f"alphavantage: {data.get('Note') or data.get('Information') or data.get('Error Message') or 'no data'}")
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.columns = [c.split(". ", 1)[-1] for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.rename_axis("date").reset_index().sort_values("date")
        return df[["date", "open", "high", "low", "close", "volume"]]

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        body, _ = self.http.get(self.URL, {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full",
                                           "apikey": self.api_key}, max_age_s=6 * 3600)
        df = self.parse(body)
        df = df[df["date"] >= pd.Timestamp(start)]
        return frame(df.assign(adj_close=df["close"]), "alphavantage", "UNADJUSTED (Alpha Vantage free tier)")

    def listing_status(self, state: str = "delisted") -> pd.DataFrame:
        """LISTING_STATUS CSV (symbol, name, exchange, assetType, ipoDate, delistingDate, status)."""
        body, _ = self.http.get(self.URL, {"function": "LISTING_STATUS", "state": state, "apikey": self.api_key}, max_age_s=7 * 86400)
        if not body.startswith("symbol"):
            raise ProviderError(f"alphavantage listing: {body[:120]!r}")
        return pd.read_csv(io.StringIO(body))
