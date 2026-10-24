"""Historical index membership: snapshots -> daily point-in-time membership, spans, source choice, coverage."""

import pandas as pd

from bigboss.data.providers.index_membership import (choose_sources, coverage_by_year, daily_membership, nifty_snapshots,
                                                     sp500_snapshots, spans)


def test_sp500_snapshots_daily_membership_and_spans(tmp_path):
    f = tmp_path / "h.csv"
    pd.DataFrame({"date": ["2015-01-02", "2015-01-07", "2015-01-12"],
                  "tickers": ["AAA,BBB,CCC", "AAA,BBB,DDD", "AAA,DDD,CCC"]}).to_csv(f, index=False)
    snap = sp500_snapshots(f)
    days = pd.bdate_range("2015-01-01", "2015-01-13")
    m = daily_membership(snap, days)
    assert m["date"].min() == pd.Timestamp("2015-01-02")                       # nothing before the first snapshot
    assert set(m[m["date"] == "2015-01-06"]["ticker"]) == {"AAA", "BBB", "CCC"}  # carried forward until the next change
    assert set(m[m["date"] == "2015-01-07"]["ticker"]) == {"AAA", "BBB", "DDD"}
    sp = spans(snap).set_index("ticker")
    assert sp.loc["BBB", "end"] == pd.Timestamp("2015-01-12")
    ccc = spans(snap)[lambda x: x["ticker"] == "CCC"]
    assert len(ccc) == 2                                                       # left and re-joined


def test_nifty_snapshots_and_weighted_coverage(tmp_path):
    f = tmp_path / "w.csv"
    pd.DataFrame({"DATE": ["2008-01-31", "2008-02-29"], "X": [60.0, 0.0], "Y": [40.0, 70.0], "Z": [0.0, 30.0]}).to_csv(f, index=False)
    snap = nifty_snapshots(f)
    assert len(snap) == 4 and set(snap[snap["date"] == "2008-02-29"]["ticker"]) == {"Y", "Z"}
    md = daily_membership(snap, pd.bdate_range("2008-02-01", "2008-03-05"))
    cov = coverage_by_year(md, md["ticker"] != "Z")   # Z (30%) has no prices
    assert abs(cov[2008] - (20 * 1.0 + 4 * 0.7) / 24) < 1e-9


def test_choose_sources_prefers_best_coverage(tmp_path):
    d = pd.bdate_range("2014-01-01", "2016-12-30")
    dh = pd.DataFrame({"symbol": "AAA", "date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "adj_close": 1.0, "volume": 1.0})
    kd = tmp_path / "Stocks"
    kd.mkdir()
    pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Open": 2.0, "High": 2.0, "Low": 2.0, "Close": 2.0, "Volume": 5,
                  "OpenInt": 0}).to_csv(kd / "bbb.us.txt", index=False)
    sp = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "start": pd.Timestamp("2014-01-01"), "end": pd.Timestamp("2016-06-01")})
    px, m = choose_sources(sp, dh, kd, pd.Timestamp("2014-01-01"))
    m = m.set_index("ticker")
    assert m.at["AAA", "source"] == "dolthub" and m.at["BBB", "source"] == "kaggle" and m.at["BBB", "symbol"] == "BBB@K"
    assert pd.isna(m.at["CCC", "symbol"]) and set(px["symbol"]) == {"AAA", "BBB@K"}
