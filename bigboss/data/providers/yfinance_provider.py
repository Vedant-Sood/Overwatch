"""yfinance (Yahoo Finance via the yfinance library): the default keyless daily-price provider.

Unofficial and not affiliated with Yahoo; Yahoo's terms allow personal, non-commercial use only. No delisted tickers.
`adj_close` is split- and dividend-adjusted; `close` is split-adjusted only.
"""

from __future__ import annotations

import pandas as pd

from bigboss.data.providers.base import PriceProvider, ProviderError

ADJUSTMENT = "adj_close_split_and_dividend"
# neutral symbol -> Yahoo symbol
ALIASES = {"^SPX": "^GSPC", "^NDX": "^NDX", "^DJI": "^DJI", "^NSEI": "^NSEI", "^N225": "^N225", "^FTSE": "^FTSE",
           "^GDAXI": "^GDAXI", "^HSI": "^HSI"}


def yahoo_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s in ALIASES:
        return ALIASES[s]
    if not s.startswith("^") and "." in s and not s.endswith((".NS", ".BO")):
        return s.replace(".", "-")  # BRK.B -> BRK-B
    return s


def frame_from_history(h: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """yfinance `history(auto_adjust=False)` output -> store frame. Dates are exchange-local session dates."""
    if h is None or h.empty or "Close" not in h:
        raise ProviderError("yfinance returned no rows")
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = pd.DataFrame({"date": idx.normalize(), "open": h["Open"].to_numpy(), "high": h["High"].to_numpy(),
                       "low": h["Low"].to_numpy(), "close": h["Close"].to_numpy(),
                       "adj_close": (h["Adj Close"] if "Adj Close" in h else h["Close"]).to_numpy(),
                       "volume": h["Volume"].to_numpy() if "Volume" in h else float("nan")})
    df = df.dropna(subset=["close"]).drop_duplicates("date", keep="last")
    if today is not None:
        df = df[df["date"] <= pd.Timestamp(today).normalize()]
    df = df.astype({c: float for c in ("open", "high", "low", "close", "adj_close", "volume")})
    df["date"] = df["date"].astype("datetime64[ns]")
    df["source"] = "yfinance"
    df["adjustment"] = ADJUSTMENT
    return df.sort_values("date").reset_index(drop=True)


class YFinanceProvider(PriceProvider):
    name = "yfinance"

    def supports(self, symbol: str) -> bool:
        return self.disabled_reason is None and not symbol.upper().startswith(("XA:", "SYN"))

    def history(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover - it is a core dependency
            raise ProviderError("yfinance is not installed") from e
        try:
            return yf.Ticker(yahoo_symbol(symbol)).history(start=start, auto_adjust=False, actions=True, interval="1d")
        except Exception as e:  # yfinance raises many exception types on network/format problems
            raise ProviderError(f"yfinance {symbol}: {e}") from e

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        return frame_from_history(self.history(symbol, start), pd.Timestamp.today())

    def adjusted_close(self, symbol: str, start: str) -> pd.Series:
        """Total-return closes, used to verify suspicious moves in other sources."""
        df = self.fetch_daily(symbol, start)
        return df.set_index("date")["adj_close"]
