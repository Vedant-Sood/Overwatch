"""Historical index membership: snapshots -> daily point-in-time membership, spans, source choice, coverage."""

import pandas as pd

from bigboss.data.providers.index_membership import (assign_symbols, candidate_prices, coverage_by_year, daily_membership, nifty_snapshots,
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


def test_per_day_source_assignment_prefers_dolthub_and_falls_back_to_kaggle(tmp_path):
    k_days = pd.bdate_range("2014-01-01", "2017-11-10")
    d_days = pd.bdate_range("2018-02-26", "2019-11-20")        # e.g. Celgene: DoltHub history starts 2018
    dh = pd.DataFrame({"symbol": "CELG", "date": d_days, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "adj_close": 1.0,
                       "volume": 1.0})
    kd = tmp_path / "Stocks"
    kd.mkdir()
    pd.DataFrame({"Date": k_days.strftime("%Y-%m-%d"), "Open": 2.0, "High": 2.0, "Low": 2.0, "Close": 2.0, "Volume": 5,
                  "OpenInt": 0}).to_csv(kd / "celg.us.txt", index=False)
    px = candidate_prices(["CELG", "ENRNQ"], dh, kd)
    md = pd.DataFrame({"date": pd.to_datetime(["2016-06-01", "2018-01-02", "2018-06-01", "2016-06-01"]),
                       "ticker": ["CELG", "CELG", "CELG", "ENRNQ"]})
    sym = assign_symbols(md, px)
    assert list(sym.fillna("-")) == ["CELG@K", "-", "CELG", "-"]   # Kaggle, gap (unpriced), DoltHub, no data
