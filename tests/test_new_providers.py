"""Cross-asset proxies, keyed price adapters, and news adapters (parsers tested against documented response shapes)."""

import json

import numpy as np
import pandas as pd
import pytest

from bigboss.data.providers.base import ProviderError
from bigboss.data.providers.fred_assets import FredCrossAssetProvider, par_bond_returns
from bigboss.data.providers.keyed import FmpProvider, MassiveProvider
from bigboss.data.providers.news import FinnhubNews, GdeltNews, credibility_for, lexicon_sentiment


def test_par_bond_return_math():
    d = pd.bdate_range("2024-01-01", periods=4)
    flat = par_bond_returns(pd.Series([4.0, 4.0, 4.0, 4.0], index=d), 10)
    # unchanged yield: return = carry only (yield x elapsed calendar days / 365)
    assert np.allclose(flat.to_numpy(), 0.04 * d.to_series().diff().dt.days.iloc[1:].to_numpy() / 365)
    up = par_bond_returns(pd.Series([4.0, 4.10], index=d[:2]), 10)
    # a 10bp rise on a 10y par bond (modified duration ~8.2) loses ~0.8%
    assert -0.0090 < up.iloc[0] - 0.04 / 365 < -0.0075
    short = par_bond_returns(pd.Series([4.0, 4.10], index=d[:2]), 2)
    assert abs(short.iloc[0]) < abs(up.iloc[0])  # shorter duration -> smaller price move


class _FakeHttp:
    def __init__(self, bodies):
        self.bodies = bodies

    def get(self, url, params=None, max_age_s=0):
        key = params.get("id") if params and "id" in params else url
        return self.bodies[key], pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime()


def _csv(sid, values, dates):
    return "observation_date," + sid + "\n" + "\n".join(f"{d.date()},{v}" for d, v in zip(dates, values))


def test_cross_asset_proxies_and_equal_weight_benchmark():
    d = pd.bdate_range("2024-01-01", periods=5)
    bodies = {
        "NASDAQ100": _csv("NASDAQ100", [100, 101, 102, 101, 103], d), "NIKKEI225": _csv("NIKKEI225", [30000, 30300, 30000, 30100, 30200], d),
        "DEXJPUS": _csv("DEXJPUS", [150, 150, 151, 151, 150], d), "DCOILBRENTEU": _csv("DCOILBRENTEU", [80, 81, ".", 82, 83], d),
        "DGS10": _csv("DGS10", [4, 4.1, 4.0, 4.0, 3.9], d), "DGS2": _csv("DGS2", [4.5, 4.5, 4.4, 4.4, 4.4], d),
        "DEXUSEU": _csv("DEXUSEU", [1.1, 1.1, 1.11, 1.1, 1.1], d), "DEXUSUK": _csv("DEXUSUK", [1.3] * 5, d),
        "DEXINUS": _csv("DEXINUS", [83] * 5, d),
    }
    p = FredCrossAssetProvider(_FakeHttp(bodies))
    nk = p.fetch_daily("XA:NIKKEI_USD")
    assert np.isclose(nk["close"].iloc[0], 30000 / 150)
    jpy = p.fetch_daily("XA:JPY")
    assert np.isclose(jpy["close"].iloc[2] / jpy["close"].iloc[1] - 1, 150 / 151 - 1)  # yen weakened -> loss in USD
    assert len(p.fetch_daily("XA:BRENT")) == 4  # missing '.' observation dropped, not filled
    ew = p.fetch_daily("XA:EW")
    r = ew.set_index("date")["close"].pct_change().loc[d[2]]
    members = [p._level(m).pct_change().get(d[2], np.nan) for m in ["XA:NDX", "XA:NIKKEI_USD", "XA:BRENT", "XA:UST10_TR", "XA:UST2_TR",
                                                              "XA:EUR", "XA:GBP", "XA:JPY", "XA:INR"]]
    assert np.isclose(r, np.nanmean(members))
    assert (ew["volume"].isna()).all() and (ew["high"] == ew["close"]).all()


def test_massive_parser():
    body = json.dumps({"status": "OK", "resultsCount": 2, "results": [
        {"t": 1704171600000, "o": 187.15, "h": 188.44, "l": 183.89, "c": 185.64, "v": 82488674},
        {"t": 1704258000000, "o": 184.22, "h": 185.88, "l": 183.43, "c": 184.25, "v": 58414460}]})
    df = MassiveProvider.parse(body)
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert df["close"].iloc[0] == 185.64
    with pytest.raises(ProviderError):
        MassiveProvider.parse(json.dumps({"status": "ERROR", "error": "API Key was not provided"}))
    mp = MassiveProvider(None, api_key=None)
    assert mp.supports("AAPL")  # a proxy-injected network secret may authenticate requests without a visible key
    mp.disabled_reason = "no credentials"
    assert not mp.supports("AAPL") and not MassiveProvider(None, api_key="k").supports("XA:NDX")


def test_fmp_parser():
    body = json.dumps([{"symbol": "AAPL", "date": "2024-01-03", "open": 184.2, "high": 185.9, "low": 183.4, "close": 184.25, "volume": 58414460},
                       {"symbol": "AAPL", "date": "2024-01-02", "open": 187.2, "high": 188.4, "low": 183.9, "close": 185.64, "volume": 82488674}])
    assert len(FmpProvider.parse(body)) == 2
    with pytest.raises(ProviderError):
        FmpProvider.parse(json.dumps({"Error Message": "Invalid API KEY."}))
    adj = FmpProvider.parse_dividend_adjusted(json.dumps([{"symbol": "AAPL", "date": "2016-01-06", "adjOpen": 22.6, "adjHigh": 23.0,
                                                            "adjLow": 22.4, "adjClose": 22.64, "volume": 273829600}]))
    assert adj["adjClose"].iloc[0] == 22.64
    fp = FmpProvider(None, api_key=None)
    assert fp.supports("AAPL")  # may be authenticated by a proxy-injected header
    fp.disabled_reason = "no credentials"
    assert not fp.supports("AAPL")


def test_news_parsers_and_sentiment():
    g = GdeltNews.parse(json.dumps({"articles": [{"url": "https://www.reuters.com/x", "title": "Oil prices surge to record as supply falls",
                                                   "seendate": "20261021T143000Z", "domain": "reuters.com", "language": "English"}]}))
    assert g[0]["published_at"] == pd.Timestamp("2026-10-21 10:30")  # UTC -> New York
    with pytest.raises(ProviderError):
        GdeltNews.parse("Please limit requests to one every 5 seconds")
    f = FinnhubNews.parse(json.dumps([{"category": "company", "datetime": 1761000000, "headline": "Apple shares slump on probe",
                                       "id": 1, "related": "AAPL", "source": "Yahoo", "summary": "", "url": "https://finance.yahoo.com/n/1"}]))
    assert f[0]["source_domain"] == "finance.yahoo.com"
    assert lexicon_sentiment("Apple shares slump on probe") == -1.0
    assert lexicon_sentiment("Profits beat estimates despite weak guidance") > 0
    assert lexicon_sentiment("Company holds annual meeting") == 0.0
    assert credibility_for("www.reuters.com") == "major_newswire" and credibility_for("randomblog.net") == "aggregator"


def test_proxy_costs_assume_liquidity(cfg):
    from bigboss.costs import CostModel

    e = CostModel(cfg).estimate("XA:NDX", "proxy_equity", 1e6, float("nan"), 0.012)
    assert e.liquidity_ok and e.half_spread_bps == cfg.get("costs.half_spread_bps")["proxy_equity"] and "proxy" in e.note
    e2 = CostModel(cfg).estimate("ABC", "equity", 1e6, float("nan"), 0.012)
    assert not e2.liquidity_ok  # ordinary securities with unknown volume stay unverified


def test_point_in_time_universe_uses_only_past_liquidity():
    from bigboss.data.providers.massive_market import point_in_time_universe

    dates = pd.bdate_range("2025-01-01", periods=120)
    rows = []
    for i, d in enumerate(dates):
        for s, dv in (("BIG", 1e9), ("MID", 1e8), ("RISER", 1e5 if i < 100 else 1e10)):
            rows.append({"symbol": s, "date": d, "close": 50.0, "volume": dv / 50})
    bars = pd.DataFrame(rows)
    m = point_in_time_universe(bars, top_n=2, lookback=20, min_history=10)
    full = m.set_index(["date", "symbol"])
    # Membership at a date must not change when future bars are removed (no look-ahead).
    t = dates[90]
    trunc = point_in_time_universe(bars[bars["date"] <= t], top_n=2, lookback=20, min_history=10).set_index(["date", "symbol"])
    assert set(full.xs(t, level="date").index) == set(trunc.xs(t, level="date").index) == {"BIG", "MID"}
    # RISER only enters after its liquidity shows up in the trailing window (shifted by one day).
    assert "RISER" not in set(full.xs(dates[100], level="date").index)
    assert "RISER" in set(full.xs(dates[115], level="date").index)
    # Penny stocks are excluded.
    cheap = bars.assign(close=np.where(bars["symbol"] == "MID", 2.0, bars["close"]))
    assert "MID" not in set(point_in_time_universe(cheap, top_n=3, lookback=20, min_history=10)["symbol"])


def test_massive_parsers():
    from bigboss.data.providers.massive_market import MassiveMarket
    from bigboss.data.providers.news import MassiveNews

    g = MassiveMarket.parse_grouped(json.dumps({"status": "OK", "resultsCount": 1, "results": [
        {"T": "AAPL", "v": 5e7, "vw": 201.1, "o": 200, "c": 202, "h": 203, "l": 199, "t": 1748894400000, "n": 9e5}]}), pd.Timestamp("2025-06-02"))
    assert g.loc[0, "symbol"] == "AAPL" and g.loc[0, "close"] == 202 and g.loc[0, "date"] == pd.Timestamp("2025-06-02")
    with pytest.raises(ProviderError):
        MassiveMarket.parse_grouped(json.dumps({"status": "NOT_AUTHORIZED", "message": "plan"}), pd.Timestamp("2016-01-04"))
    n = MassiveNews.parse(json.dumps({"status": "OK", "results": [{
        "title": "Apple beats estimates", "published_utc": "2026-10-21T14:30:00Z", "article_url": "https://www.reuters.com/a",
        "description": "", "insights": [{"ticker": "AAPL", "sentiment": "negative", "sentiment_reasoning": "x"}]}]}), "AAPL")
    assert n[0]["provider_sentiment"] == -1.0 and n[0]["source_domain"] == "reuters.com"


def test_credentials_are_redacted_from_errors():
    from bigboss.data.providers.base import redact

    msg = "Client error '402' for url 'https://financialmodelingprep.com/x?symbol=QQQ&apikey=S3CRET&from=2005-01-01'"
    assert "S3CRET" not in redact(msg) and "apikey=***" in redact(msg)
    assert "tok" not in redact("https://finnhub.io/api?token=tok123&x=1").split("token=")[1][:3]


def test_fmp_fetches_long_history_in_windows():
    p = FmpProvider(None, api_key="k")
    w = p._windows("2005-01-01")
    assert w[0]["from"] == "2005-01-01" and len(w) >= 2
    assert all(pd.Timestamp(a["to"]) < pd.Timestamp(b["from"]) for a, b in zip(w, w[1:]))  # contiguous, no overlap
