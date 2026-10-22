import numpy as np
import pandas as pd

from bigboss.backtest.splits import holdout_boundary, purged_kfold, walk_forward_folds
from bigboss.features.dataset import build_dataset
from bigboss.models.distribution import ConformalInterval, expected_shortfall, interval_coverage, prob_above
from bigboss.models.ensemble import StackingEnsemble, abstain_mask
from bigboss.models.walkforward import ic_tstat, rank_ic_series, run_walk_forward


def _label_end(dates, h, delay=1):
    return pd.Series([dates[min(i + h + delay, len(dates) - 1)] for i in range(len(dates))], index=dates)


def test_walk_forward_purges_overlapping_labels_and_locks_holdout():
    dates = pd.bdate_range("2015-01-01", periods=1500)
    le = _label_end(dates, 20)
    folds = walk_forward_folds(dates, le, min_train=500, step=100, horizon=20, embargo=5, holdout_fraction=0.15)
    hb = holdout_boundary(dates, 0.15)
    assert folds
    for f in folds:
        assert le.reindex(f.train_dates).max() < f.test_start  # no train label overlaps the test period
        assert le.reindex(f.calib_dates).max() < f.test_start
        assert f.train_dates.max() < f.calib_dates.min()
        assert le.reindex(f.train_dates).max() < f.calib_dates.min()  # fit slice purged from calibration slice
        assert f.test_dates.max() < hb
    with_h = walk_forward_folds(dates, le, 500, 100, 20, 5, holdout_fraction=0.15, include_holdout=True)
    assert with_h[-1].test_dates.max() >= hb


def test_purged_kfold_has_no_overlap():
    dates = pd.bdate_range("2015-01-01", periods=600)
    le = _label_end(dates, 10)
    for train, test in purged_kfold(dates, le, k=5, embargo=5):
        before = train[train < test[0]]
        assert le.reindex(before).max() < test[0] if len(before) else True
        assert not set(train) & set(test)


def test_prob_above_and_shortfall():
    q10, q50, q90 = np.array([-0.1]), np.array([0.0]), np.array([0.1])
    assert np.isclose(prob_above(0.0, q10, q50, q90), 0.5)
    assert np.isclose(prob_above(-0.1, q10, q50, q90), 0.9, atol=1e-3)
    assert prob_above(0.05, q10, q50, q90) < 0.5
    assert expected_shortfall(q10, q50)[0] < -0.1


def test_conformal_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y = rng.standard_t(4, 4000) * 0.05
    lo, hi = np.full(4000, -0.01), np.full(4000, 0.01)  # badly under-dispersed raw interval
    c = ConformalInterval(0.8).fit(y[:2000], lo[:2000], hi[:2000])
    a, b = c.apply(lo[2000:], hi[2000:])
    assert abs(interval_coverage(y[2000:], a, b) - 0.8) < 0.03


def test_ensemble_weights_and_abstention():
    rng = np.random.default_rng(1)
    n = 5000
    good = rng.normal(0, 1, n)
    y = 0.5 * good + rng.normal(0, 1, n)
    oos = pd.DataFrame({"mean_good": good, "mean_noise": rng.normal(0, 1, n), "y": y})
    e = StackingEnsemble(["good", "noise"], shrink=0.0, min_oos_rows=100).fit(oos)
    assert e.weights["good"] > 0.8
    pred = pd.DataFrame({"mean": [0.01, 0.01, 0.01], "q10": [-0.05] * 3, "q50": [0.0] * 3, "q90": [0.05] * 3,
                         "disagreement": [0.001, 0.2, 0.001], "sign_agreement": [1.0, 1.0, 0.2]})
    m = abstain_mask(pred, max_disagreement=0.03, min_sign_agreement=0.6)
    assert list(m) == [False, True, True]
    assert abstain_mask(pred, 0.03, 0.6, skill_ok=False).all()


def test_models_recover_planted_signal_and_stay_calibrated(synth_service):
    bb = synth_service
    vid = bb.registry.champion(20)
    meta = bb.registry.meta(vid)
    ens = next(f for f in meta["forecast_quality"] if f["model"] == "ensemble")
    # The synthetic market has a real (weak) persistent alpha; the ensemble should find it out of sample...
    assert ens["rank_ic_mean"] > 0.02
    # ...but a realistic IC is small. A very high IC on this data would indicate leakage.
    assert ens["rank_ic_mean"] < 0.35
    assert 0.7 <= ens["interval80_coverage"] <= 0.9


def test_zero_signal_market_shows_no_skill(cfg):
    """On a market with no planted alpha, OOS skill must be statistically indistinguishable from zero."""
    from bigboss.data.store import Store
    from bigboss.data.synthetic import load_into_store

    c = cfg.with_overrides({"benchmarks": {"default": "SYNMKT", "US": "SYNMKT"}})
    st = Store.for_mode(c.data_dir, "SYNTHETIC")
    load_into_store(st, n_assets=20, n_days=1300, seed=11, alpha_scale=0.0)
    ds = build_dataset(st, c)
    run = run_walk_forward(ds, 20, ["linear_ridge", "lightgbm"], min_train=500, step=126)
    ics = rank_ic_series(run.oos["mean_ensemble"], run.oos["y"])
    assert abs(ic_tstat(ics, 20)) < 2.5
