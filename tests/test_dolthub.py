"""DoltHub provider: common-stock filter, ticker-reuse segmentation, split/dividend adjustment, unrecorded splits."""

import numpy as np
import pandas as pd

from bigboss.data.providers.dolthub import adjust_market, common_stocks, segment_symbols


def test_common_stock_filter_keeps_delisted_operating_companies_only():
    sym = pd.DataFrame([
        ("AAPL", "Apple Inc. - Common Stock", "0", "0"),
        ("SIVB", "SVB Financial Group - Common Stock", "0", "0"),                    # delisted (bankrupt): kept
        ("BRK.B", "Berkshire Hathaway Inc. New Common Stock", "0", "0"),
        ("GOOGL", "Alphabet Inc. - Class A Common Stock", "0", "0"),
        ("SPY", "SPDR S&P 500 ETF Trust", "1", "0"),
        ("BABA", "Alibaba Group Holding Limited American Depositary Shares", "0", "0"),
        ("XYZW", "XYZ Corp - Warrant", "0", "0"),
        ("PFX", "Foo Inc 6.5% Series A Preferred Stock", "0", "0"),
        ("ZVZZT", "NASDAQ TEST STOCK Common Stock", "0", "1"),
        ("ABCU", "ABC Acquisition Corp - Units", "0", "0"),
    ], columns=["act_symbol", "security_name", "is_etf", "is_test_issue"])
    assert set(common_stocks(sym)["act_symbol"]) == {"AAPL", "SIVB", "BRK.B", "GOOGL"}


def test_reused_ticker_is_split_into_separate_securities():
    d1 = pd.bdate_range("2015-01-01", periods=50)
    d2 = pd.bdate_range("2019-01-01", periods=50)  # same ticker, years later: a different company
    bars = pd.DataFrame({"act_symbol": "ABC", "date": d1.append(d2), "close": 10.0})
    seg = segment_symbols(bars)
    assert set(seg["symbol"]) == {"ABC@201501", "ABC"}
    assert (seg.loc[seg["date"] >= "2019-01-01", "symbol"] == "ABC").all()


def test_split_and_dividend_adjustment():
    d = pd.bdate_range("2020-01-01", periods=10)
    raw = np.array([100, 101, 102, 103, 104, 26.5, 27, 27.5, 27.0, 28.0])  # 4:1 split on d[5]; $0.27 dividend ex d[8]
    bars = segment_symbols(pd.DataFrame({"act_symbol": "AAA", "date": d, "open": raw, "high": raw, "low": raw,
                                         "close": raw, "volume": 1000.0}))
    splits = pd.DataFrame({"act_symbol": ["AAA"], "ex_date": [d[5]], "to_factor": [4.0], "for_factor": [1.0]})
    divs = pd.DataFrame({"act_symbol": ["AAA", "AAA"], "ex_date": [d[8], d[3]], "amount": [0.275, 80.0]})  # 2nd: implausible
    px, st = adjust_market(bars, splits, divs)
    px = px.set_index("date")
    assert np.isclose(px["close"].iloc[0], 25.0) and np.isclose(px["volume"].iloc[0], 4000.0)  # split-adjusted
    assert np.isclose(px["raw_close"].iloc[0], 100.0)
    r_close = px["close"].pct_change()
    assert abs(r_close.iloc[5] - (26.5 / 26.0 - 1)) < 1e-12       # no fake -75% on the split day
    yld = 0.275 / 27.5
    r_tr = px["adj_close"].pct_change()
    # multiplicative (Yahoo/CRSP-style) adjustment: P_t / (P_{t-1} * (1 - D / P_{t-1})) - 1
    assert abs(r_tr.iloc[8] - (27.0 / (27.5 - 0.275) - 1)) < 1e-12 and r_tr.iloc[8] > r_close.iloc[8]
    assert np.isclose(px["adj_close"].iloc[-1], 28.0) and np.isclose(px["adj_close"].iloc[0], 25.0 * (1 - yld))
    assert st["splits_used"] == 1 and st["dividends_used"] == 1 and st["dividends_dropped_implausible"] == 1
    assert st["unrecorded_split_candidates"] == []


def test_unrecorded_split_is_flagged_but_crash_is_not():
    d = pd.bdate_range("2020-01-01", periods=6)
    a = segment_symbols(pd.DataFrame({"act_symbol": "AAA", "date": d, "close": [100, 101, 50.4, 51, 52, 53.0]}))   # 2:1, missing
    b = segment_symbols(pd.DataFrame({"act_symbol": "BBB", "date": d, "close": [100, 101, 37, 36, 35, 34.0]}))     # -63% crash
    bars = pd.concat([a, b]).assign(open=lambda x: x["close"], high=lambda x: x["close"], low=lambda x: x["close"], volume=1.0)
    empty_s = pd.DataFrame(columns=["act_symbol", "ex_date", "to_factor", "for_factor"])
    empty_d = pd.DataFrame(columns=["act_symbol", "ex_date", "amount"])
    _, st = adjust_market(bars, empty_s, empty_d)
    flagged = {c[0]: c[4] for c in st["unrecorded_split_candidates"]}
    assert flagged == {"AAA": True, "BBB": False}  # both are verification candidates; only AAA looks like a split


def test_recorded_split_is_checked_against_prices():
    d = pd.bdate_range("2022-06-27", periods=20)
    raw = np.r_[np.full(15, 2200.0), np.full(5, 110.0)]  # real 20:1 at d[15]
    bars = segment_symbols(pd.DataFrame({"act_symbol": "G", "date": d, "open": raw, "high": raw, "low": raw, "close": raw,
                                         "volume": 1.0}))
    splits = pd.DataFrame({"act_symbol": ["G", "G"], "ex_date": [d[3], d[14]], "to_factor": [20.0, 20.0], "for_factor": [1.0, 1.0]})
    px, st = adjust_market(bars, splits, pd.DataFrame(columns=["act_symbol", "ex_date", "amount"]))
    assert st["splits_dropped_unconfirmed"] == 1 and st["splits_moved"] == 1 and st["splits_used"] == 1
    assert np.allclose(px["close"], 110.0)


def test_alpha_vantage_listing_classifies_names_missing_from_dolthub_directory():
    from bigboss.data.providers.dolthub import classify_with_listing

    sym = pd.DataFrame([("AAPL", "Apple Inc. - Common Stock", "0", "0")],
                       columns=["act_symbol", "security_name", "is_etf", "is_test_issue"])
    lst = pd.DataFrame([("LNKD", "LinkedIn Corp", "NYSE", "Stock"), ("EMC", "EMC Corp", "NYSE", "Stock"),
                        ("XIV", "VelocityShares Daily Inverse VIX ETN", "NASDAQ", "ETF"),
                        ("ABCW", "ABC Corp Warrant", "NASDAQ", "Stock"), ("AAPL", "Apple Inc", "NASDAQ", "Stock")],
                       columns=["symbol", "name", "exchange", "assetType"])
    out = classify_with_listing(sym, lst)
    assert set(out["act_symbol"]) == {"AAPL", "LNKD", "EMC"}
    assert out.set_index("act_symbol").at["AAPL", "class_source"] == "dolthub_symbol"


def test_split_confirmation_uses_same_span_and_corrects_only_disagreements():
    from bigboss.data.providers.dolthub import confirm_unrecorded_splits

    idx = pd.to_datetime(["2021-02-01", "2021-02-02", "2021-02-03", "2018-11-06", "2018-11-07"])
    yf = {"GME": pd.Series([225.0, 90.0, 92.41], index=idx[:3]),   # DoltHub misses 02-02: its 2-day move is real
          "TJX": pd.Series([50.0, 50.6], index=idx[3:])}           # real 2:1 split DoltHub did not record
    cands = [("GME", "2021-02-03", 0.4107, "2021-02-01"), ("TJX", "2018-11-07", 0.5025, "2018-11-06"),
             ("SIVB", "2023-03-09", 0.3959, "2023-03-08")]

    def fetch(t, start):
        if t not in yf:
            raise KeyError(t)
        return yf[t]

    fixes, rep = confirm_unrecorded_splits(cands, fetch)
    assert list(fixes["act_symbol"]) == ["TJX"]
    assert abs(fixes["to_factor"].iloc[0] - (50.6 / 50.0) / 0.5025) < 1e-9
    assert [r["action"].split(" ")[0] for r in rep] == ["kept", "corrected", "kept"]
