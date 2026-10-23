"""Training-Data bulk files: ETF file -> point-in-time ETF universe; per-ticker file formats."""

import numpy as np
import pandas as pd

from bigboss.config import get_config
from bigboss.data.ingest import Ingestor
from bigboss.data.providers.local_files import load_ticker_file
from bigboss.data.store import Store


def test_etf_file_builds_point_in_time_universe(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBOSS_DATA_DIR", str(tmp_path / "data"))
    d = pd.bdate_range("2010-01-01", periods=120)
    rows = []
    for sym, px, vol, start in [("SPY", 100, 1e8, 0), ("BIG", 50, 1e7, 0), ("SMALL", 20, 1e3, 0), ("LATE", 30, 5e7, 60),
                                ("PENNY", 2, 1e9, 0), ("LEV3X", 40, 1e10, 0), ("INV", 40, 1e10, 0)]:
        for i, day in enumerate(d[start:]):
            rows.append((sym, day.date(), px, px, px, px * (1 + 0.001 * i), px * (1 + 0.001 * i), vol))
    # adjusted paths driven by one market factor: LEV3X has beta 3, INV beta -1
    spy_r = np.random.default_rng(0).normal(0, 0.01, len(d))
    betas = {"SPY": 1.0, "BIG": 1.0, "SMALL": 0.8, "LATE": 1.0, "PENNY": 1.0, "LEV3X": 3.0, "INV": -1.0}
    adj = {k: np.cumprod(1 + b * spy_r) for k, b in betas.items()}
    idx = {day.date(): i for i, day in enumerate(d)}
    rows = [(s, dt, o, h, lo, c, adj[s][idx[dt]] * c, v) for s, dt, o, h, lo, c, _, v in rows]
    csv = tmp_path / "etf.csv"
    pd.DataFrame(rows, columns=["fund_symbol", "price_date", "open", "high", "low", "close", "adj_close", "volume"]).to_csv(csv, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"fund_symbol": ["BIG", "LEV3X"], "fund_long_name": ["Big Fund", "x"],
                  "fund_category": ["Large Blend", "Trading--Leveraged Equity"], "fund_family": ["X", "Y"]}).to_csv(meta, index=False)
    cfg = get_config()
    store = Store.for_mode(cfg.data_dir, "OBSERVED")
    r = Ingestor(cfg, store).ingest_etf_file(csv, meta, top_n=2)
    m = store.membership()
    assert "SPY" not in set(m["symbol"]) and "PENNY" not in set(m["symbol"])      # benchmark held separately; < $5
    first = m[m["date"] == m["date"].min()]
    assert set(first["symbol"]) == {"BIG", "SMALL"}                                # LATE has no history yet
    assert set(m[m["date"] == m["date"].max()]["symbol"]) == {"BIG", "LATE"}       # LATE outranks SMALL once known
    syms = store.symbols().set_index("symbol")
    assert syms.at["BIG", "sector"] == "Large Blend" and syms.at["BIG", "asset_class"] == "etf"
    assert r["etfs_ever_in_universe"] == 3 and r["excluded_trading_products"] == 2   # LEV3X by category, INV by beta


def test_ticker_file_formats(tmp_path):
    a = tmp_path / "a.csv"
    pd.DataFrame({"high": [2.0], "volume": [10], "low": [1.0], "open": [1.5], "close": [1.8],
                  "timestamp": [1262615400]}).to_csv(a, index=False)
    b = tmp_path / "b.csv"
    pd.DataFrame({"Date": ["2010-01-04"], "Close": [6.4], "High": [6.5], "Low": [6.3], "Open": [6.35], "Volume": [5],
                  "Symbol": ["AAPL"]}).to_csv(b, index=False)
    fa, fb = load_ticker_file(a), load_ticker_file(b)
    assert str(fa["date"].iloc[0].date()) == "2010-01-04" and np.isclose(fa["close"].iloc[0], 1.8)
    assert str(fb["date"].iloc[0].date()) == "2010-01-04" and np.isclose(fb["close"].iloc[0], 6.4)
