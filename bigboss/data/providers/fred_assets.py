"""Keyless cross-asset proxies (XA:*) built from FRED series.

They are PROXIES, and every limitation is stored in the `adjustment` column: equity indices exclude dividends, Brent is
spot (no futures roll), Treasury total returns are re-priced from constant-maturity par yields, FX excludes interest
carry, and there is no volume (costs treat them as liquid ETF/futures exposures).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from bigboss.data.providers.base import HttpClient, PriceProvider, ProviderError

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

MEMBERS = ["XA:NDX", "XA:NIKKEI_USD", "XA:BRENT", "XA:UST10_TR", "XA:UST2_TR", "XA:EUR", "XA:GBP", "XA:JPY", "XA:INR"]
SPEC = {
    # symbol: (name, asset_class, limitation)
    "XA:NDX": ("Nasdaq-100 index (FRED NASDAQ100)", "proxy_equity", "price index, excludes dividends"),
    "XA:NIKKEI_USD": ("Nikkei 225 in USD (NIKKEI225 / DEXJPUS)", "proxy_equity", "price index in USD, excludes dividends"),
    "XA:BRENT": ("Brent crude spot (DCOILBRENTEU)", "proxy_commodity", "spot price, no futures roll or storage costs"),
    "XA:UST10_TR": ("10y Treasury total return (from DGS10)", "proxy_rates", "re-priced constant-maturity par bond, carry included"),
    "XA:UST2_TR": ("2y Treasury total return (from DGS2)", "proxy_rates", "re-priced constant-maturity par bond, carry included"),
    "XA:EUR": ("EUR in USD (DEXUSEU)", "proxy_fx", "spot FX, excludes interest carry"),
    "XA:GBP": ("GBP in USD (DEXUSUK)", "proxy_fx", "spot FX, excludes interest carry"),
    "XA:JPY": ("JPY in USD (1 / DEXJPUS)", "proxy_fx", "spot FX, excludes interest carry"),
    "XA:INR": ("INR in USD (1 / DEXINUS)", "proxy_fx", "spot FX, excludes interest carry"),
    "XA:EW": ("Equal-weight cross-asset basket (daily rebalanced)", "proxy_benchmark", "equal weight of the nine XA proxies"),
}


def par_bond_returns(yields_pct: pd.Series, maturity_years: float, freq: int = 2) -> pd.Series:
    """Daily total returns of a constant-maturity par bond re-priced from yields (in percent).

    Each day holds yesterday's par bond (coupon = yesterday's yield): return = carry (coupon x elapsed days / 365)
    plus the price change from re-pricing at today's yield with the same maturity.
    """
    y = yields_pct.dropna().astype(float) / 100
    if len(y) < 2:
        return pd.Series(dtype=float)
    y0, y1 = y.shift(1).iloc[1:], y.iloc[1:]
    dt = y.index.to_series().diff().dt.days.iloc[1:].astype(float)
    n = maturity_years * freq
    c, r = y0 / freq, y1 / freq
    disc = (1 + r) ** (-n)
    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.where(np.abs(r) > 1e-12, c * (1 - disc) / r + disc, c * n + 1)
    return pd.Series(price - 1 + y0.to_numpy() * dt.to_numpy() / 365, index=y1.index)


class FredCrossAssetProvider(PriceProvider):
    name = "fred_xa"

    def __init__(self, http: HttpClient):
        self.http = http
        self._cache: dict[str, pd.Series] = {}

    def supports(self, symbol: str) -> bool:
        return self.disabled_reason is None and symbol.upper() in SPEC

    def series(self, sid: str) -> pd.Series:
        if sid not in self._cache:
            body, _ = self.http.get(FRED_CSV, {"id": sid}, max_age_s=6 * 3600)
            try:
                df = pd.read_csv(io.StringIO(body), na_values=["."])
            except Exception as e:
                raise ProviderError(f"FRED {sid}: unreadable CSV ({e})") from e
            if df.shape[1] < 2:
                raise ProviderError(f"FRED {sid}: unexpected CSV {body[:120]!r}")
            df.columns = ["date", "value"]
            s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"])).dropna()
            self._cache[sid] = s.sort_index()
        return self._cache[sid]

    def _level(self, symbol: str) -> pd.Series:
        s = symbol.upper()
        if s == "XA:NDX":
            return self.series("NASDAQ100")
        if s == "XA:NIKKEI_USD":
            nk, jpy = self.series("NIKKEI225"), self.series("DEXJPUS")
            return (nk / jpy).dropna()
        if s == "XA:BRENT":
            return self.series("DCOILBRENTEU")
        if s in ("XA:UST10_TR", "XA:UST2_TR"):
            y = self.series("DGS10" if s == "XA:UST10_TR" else "DGS2")
            lvl = 100 * (1 + par_bond_returns(y, 10 if s == "XA:UST10_TR" else 2)).cumprod()
            return pd.concat([pd.Series([100.0], index=[y.index[0]]), lvl])
        if s == "XA:EUR":
            return self.series("DEXUSEU")
        if s == "XA:GBP":
            return self.series("DEXUSUK")
        if s == "XA:JPY":
            return 1 / self.series("DEXJPUS")
        if s == "XA:INR":
            return 1 / self.series("DEXINUS")
        if s == "XA:EW":
            rets = pd.concat({m: self._level(m).pct_change() for m in MEMBERS}, axis=1).sort_index()
            r = rets.mean(axis=1, skipna=True).fillna(0.0)
            return 100 * (1 + r).cumprod()
        raise ProviderError(f"unknown cross-asset proxy {symbol}")

    def fetch_daily(self, symbol: str, start: str = "1990-01-01") -> pd.DataFrame:
        lvl = self._level(symbol).dropna()
        lvl = lvl[lvl.index >= pd.Timestamp(start)]
        if lvl.empty:
            raise ProviderError(f"{symbol}: no FRED data")
        v = lvl.to_numpy(dtype=float)
        return pd.DataFrame({"date": pd.DatetimeIndex(lvl.index).astype("datetime64[ns]"), "open": v, "high": v, "low": v,
                             "close": v, "adj_close": v, "volume": np.nan, "source": "fred_xa",
                             "adjustment": f"PROXY: {SPEC[symbol.upper()][2]}"}).reset_index(drop=True)
