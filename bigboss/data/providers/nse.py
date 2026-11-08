"""NSE exchange data via the optional `jugaad-data` package (`pip install -e .[india]`).

NSE's archive still serves delisted names (Satyam, Ranbaxy, ...). Prices are RAW exchange closes (no corporate-action
adjustment), so they are used for cross-checks and for names other sources lack, with adjustments applied separately.
"""

from __future__ import annotations

import pandas as pd

from bigboss.data.providers.base import PriceProvider, ProviderError

ADJUSTMENT = "UNADJUSTED_EXCHANGE_RAW"


def normalize_nse_frame(raw: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """jugaad-data `stock_df` output -> store frame. Keeps the EQ series only; dates are IST session dates."""
    if raw is None or raw.empty:
        raise ProviderError("NSE returned no rows")
    df = raw.copy()
    df.columns = [c.upper() for c in df.columns]
    if "SERIES" in df:
        df = df[df["SERIES"].astype(str).str.upper() == "EQ"]
    d = pd.to_datetime(df["DATE"])
    if d.dt.tz is not None:
        d = d.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    elif (d.dt.hour != 0).any():  # timestamps serialized as UTC: 18:30 UTC is midnight IST of the next day
        d = d + pd.Timedelta(hours=5, minutes=30)
    out = pd.DataFrame({"date": d.dt.normalize().astype("datetime64[ns]"), "open": df["OPEN"].astype(float),
                        "high": df["HIGH"].astype(float), "low": df["LOW"].astype(float), "close": df["CLOSE"].astype(float),
                        "volume": df["VOLUME"].astype(float)})
    out["adj_close"] = out["close"]
    if today is not None:
        out = out[out["date"] <= pd.Timestamp(today).normalize()]
    out = out.drop_duplicates("date", keep="last").sort_values("date")
    out["source"] = "nse"
    out["adjustment"] = ADJUSTMENT
    return out[["date", "open", "high", "low", "close", "adj_close", "volume", "source", "adjustment"]].reset_index(drop=True)


class NseProvider(PriceProvider):
    name = "nse"

    def supports(self, symbol: str) -> bool:
        return self.disabled_reason is None and symbol.upper().endswith(".NS")

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        try:
            from jugaad_data.nse import stock_df
        except ImportError as e:
            raise ProviderError("jugaad-data is not installed (pip install -e .[india])") from e
        try:
            raw = stock_df(symbol=symbol.upper().removesuffix(".NS"), from_date=pd.Timestamp(start).date(),
                           to_date=pd.Timestamp.today().date(), series="EQ")
        except Exception as e:  # network / NSE format problems
            raise ProviderError(f"NSE {symbol}: {e}") from e
        return normalize_nse_frame(raw, pd.Timestamp.today())
