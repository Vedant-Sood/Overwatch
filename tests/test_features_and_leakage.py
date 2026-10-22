"""Point-in-time integrity: features must not change when future data is appended; labels must be exact."""

import numpy as np
import pandas as pd

from bigboss.data.validation import assert_no_lookahead, validate_prices
from bigboss.features.context import event_features, macro_features
from bigboss.features.labels import forward_labels
from bigboss.features.technical import corwin_schultz_spread, market_features, technical_features
from tests.conftest import make_prices


def test_technical_features_have_no_lookahead(synth_market):
    p = synth_market["prices"]
    p = p[p["symbol"].isin(["SYN000", "SYN001", "SYN002", "SYN003", "SYNMKT"])]
    dates = sorted(p["date"].unique())
    probes = [pd.Timestamp(dates[i]) for i in (300, 700, 1100)]
    assert_no_lookahead(lambda x: technical_features(x, "SYNMKT"), p, probes)


def test_market_features_have_no_lookahead(synth_market):
    p = synth_market["prices"]
    dates = sorted(p["date"].unique())
    full = market_features(p, "SYNMKT")
    for i in (400, 900):
        t = pd.Timestamp(dates[i])
        trunc = market_features(p[p["date"] <= t], "SYNMKT")
        pd.testing.assert_series_equal(full.loc[t], trunc.loc[t], check_names=False)


def test_lookahead_detector_catches_a_leaky_feature(synth_market):
    p = synth_market["prices"][lambda d: d["symbol"].isin(["SYN000", "SYN001"])]

    def leaky(x):
        w = x.pivot(index="date", columns="symbol", values="adj_close")
        f = (w.shift(-5) / w - 1).stack(future_stack=True).to_frame("future_ret")
        f.index.names = ["date", "symbol"]
        return f
    t = pd.Timestamp(sorted(p["date"].unique())[500])
    try:
        assert_no_lookahead(leaky, p, [t])
    except AssertionError:
        return
    raise AssertionError("leaky feature was not detected")


def test_forward_labels_exact():
    p = make_prices(50)
    lab = forward_labels(p, [5], delay=1, benchmark_map={"A": "M", "B": "M", "M": "M"})
    w = p.pivot(index="date", columns="symbol", values="adj_close")
    t = w.index[10]
    exp_a = w["A"].iloc[16] / w["A"].iloc[11] - 1
    exp_m = w["M"].iloc[16] / w["M"].iloc[11] - 1
    assert np.isclose(lab.loc[(t, "A"), "fwd_ret_5"], exp_a)
    assert np.isclose(lab.loc[(t, "A"), "fwd_excess_5"], exp_a - exp_m)
    assert lab.loc[(t, "A"), "label_end_5"] == w.index[16]
    # labels at the end of the sample are unknown, not zero
    assert np.isnan(lab.loc[(w.index[-3], "A"), "fwd_ret_5"])


def test_macro_features_respect_publication_lag():
    cal = pd.bdate_range("2024-01-01", "2024-03-29")
    macro = pd.DataFrame({"series_id": "DGS10", "date": pd.to_datetime(["2024-01-31", "2024-02-29"]), "value": [4.0, 5.0],
                          "available_at": pd.to_datetime(["2024-02-15", "2024-03-15"])})
    f = macro_features(macro, cal)
    assert np.isnan(f.loc["2024-02-14", "macro_dgs10"])
    assert f.loc["2024-02-15", "macro_dgs10"] == 4.0
    assert f.loc["2024-03-14", "macro_dgs10"] == 4.0
    assert f.loc["2024-03-15", "macro_dgs10"] == 5.0


def test_after_hours_filing_is_usable_next_session():
    cal = pd.bdate_range("2024-05-01", periods=10)
    ev = pd.DataFrame({"symbol": ["A"], "accepted_at": [pd.Timestamp("2024-05-02 16:30")], "event_type": ["earnings_release"],
                       "importance_prior": [0.8], "form": ["8-K"]})
    f = event_features(ev, cal, ["A"])
    assert f.loc[(pd.Timestamp("2024-05-02"), "A"), "max_importance_5d"] == 0
    assert f.loc[(pd.Timestamp("2024-05-03"), "A"), "max_importance_5d"] == 0.8
    assert f.loc[(pd.Timestamp("2024-05-03"), "A"), "days_since_earnings"] == 0


def test_corwin_schultz_non_negative():
    p = make_prices(200)
    w = lambda f: p.pivot(index="date", columns="symbol", values=f)
    s = corwin_schultz_spread(w("high"), w("low"))
    assert (s.dropna() >= 0).all().all()


def test_validation_flags_bad_data():
    p = make_prices(100, symbols=("A",))
    bad = pd.concat([p, p.iloc[[5]]])
    bad.loc[bad.index[10], "close"] = -1
    checks = {i.check_name for i in validate_prices(bad, "A", today=p["date"].max())}
    assert {"unique_dates", "positive_prices"} <= checks
    split = p.copy()
    split.loc[split.index[50:], ["open", "high", "low", "close", "adj_close"]] /= 4
    assert any(i.check_name == "extreme_moves" for i in validate_prices(split, "A", today=p["date"].max()))
    stale = validate_prices(p, "A", today=p["date"].max() + pd.Timedelta(days=30))
    assert any(i.check_name == "freshness" for i in stale)


def test_pit_universe_keeps_short_history_names(cfg, synth_market):
    """A stock that delists after 120 sessions must stay in a point-in-time universe (no survivorship filter)."""
    from bigboss.data.store import Store
    from bigboss.features.dataset import build_dataset

    c = cfg.with_overrides({"benchmarks": {"default": "SYNMKT", "US": "SYNMKT"}})
    st = Store.for_mode(c.data_dir, "SYNTHETIC")
    p = synth_market["prices"]
    dates = sorted(p["date"].unique())
    p = p[~((p["symbol"] == "SYN001") & (p["date"] > dates[119]))]  # SYN001 "delists"
    st.upsert("prices", p, ["symbol", "date"])
    st.upsert("symbols", synth_market["symbols"], ["symbol"])
    assert "SYN001" not in build_dataset(st, c).tradeable  # without a PIT universe, the 260-day filter applies
    mem = pd.DataFrame([{"date": d, "symbol": s, "rank": 1} for d in dates[50:120] for s in ("SYN001", "SYN002")])
    st.upsert("universe_membership", mem, ["date", "symbol"])
    ds = build_dataset(st, c)
    assert "SYN001" in ds.tradeable
    assert ds.frame.xs("SYN001", level="symbol").index.max() <= dates[119]
