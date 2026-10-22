import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session", autouse=True)
def _quiet_logs():
    from bigboss.logging_utils import get_logger

    get_logger("tests")
    logging.getLogger("bigboss").setLevel(logging.WARNING)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBOSS_DATA_DIR", str(tmp_path))
    from bigboss.config import load_config

    return load_config()


@pytest.fixture(scope="session")
def synth_market():
    from bigboss.data.synthetic import generate_market

    return generate_market(n_assets=20, n_days=1500, seed=3, alpha_scale=3.0)


@pytest.fixture(scope="session")
def synth_service(tmp_path_factory):
    """A SYNTHETIC-mode service with a trained 20d champion (shared across integration tests)."""
    d = tmp_path_factory.mktemp("bb")
    os.environ["BIGBOSS_DATA_DIR"] = str(d)
    from bigboss.config import load_config
    from bigboss.data.synthetic import load_into_store
    from bigboss.service import BigBoss

    cfg = load_config().with_overrides({"monitoring": {"min_eval_periods_for_promotion": 8}})
    bb = BigBoss(cfg, "SYNTHETIC")
    load_into_store(bb.store, n_assets=20, n_days=1500, seed=3, alpha_scale=3.0)
    vid = bb.registry.train(bb.dataset(), 20, ["zero_excess", "hist_mean", "momentum_rule", "linear_ridge", "lightgbm"], min_train=500, step=126)
    bb.registry.promote(vid, force_reason="test fixture")
    return bb


def make_prices(n=400, symbols=("A", "B", "M"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for s in symbols:
        c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        rows.append(pd.DataFrame({"symbol": s, "date": dates, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                                  "adj_close": c, "volume": rng.uniform(1e5, 2e5, n)}))
    return pd.concat(rows, ignore_index=True)
