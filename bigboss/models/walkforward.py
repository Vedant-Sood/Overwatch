"""Walk-forward training and out-of-sample forecasting for Model A (+ ensemble E).

For each fold:
  1. fit every base model on the purged training slice (dates subsampled to reduce label overlap),
  2. conformally calibrate each model's [q10, q90] on the calibration slice (tail of the train window),
  3. predict the test block,
  4. fit ensemble weights on PRIOR folds' OOS predictions whose labels were known before this fold began.
The output is a long table of genuinely out-of-sample forecasts used by the backtester and diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bigboss.backtest.splits import Fold, walk_forward_folds
from bigboss.features.dataset import Dataset
from bigboss.logging_utils import get_logger, log
from bigboss.models.distribution import ConformalInterval, ProbabilityCalibrator, prob_above
from bigboss.models.ensemble import StackingEnsemble
from bigboss.models.zoo import make_model

logger = get_logger("models.walkforward")


@dataclass
class ForecastRun:
    horizon: int
    models: list[str]
    oos: pd.DataFrame                      # index (date, symbol); columns per model + ensemble + y + label_end
    folds: list[dict]
    feature_importance: pd.DataFrame
    ensemble_weights: list[dict] = field(default_factory=list)

    def model_frame(self, model: str) -> pd.DataFrame:
        cols = {f"{k}_{model}": k for k in ("mean", "q10", "q50", "q90", "p_pos")}
        out = self.oos[[c for c in cols if c in self.oos]].rename(columns=cols)
        out["y"] = self.oos["y"]
        return out


def rank_ic_series(pred: pd.Series, y: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    def _ic(g):
        if len(g) < 5 or g["p"].nunique() < 2:
            return np.nan
        return spearmanr(g["p"], g["y"]).statistic
    return df.groupby(level="date").apply(_ic).dropna()


def ic_tstat(ics: pd.Series, horizon: int) -> float:
    """t-stat of mean rank IC with an overlap correction (effective sample = n / horizon)."""
    if len(ics) < 10:
        return 0.0
    n_eff = max(len(ics) / max(horizon, 1), 2)
    return float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n_eff) + 1e-12))


def _subsample(dates: pd.DatetimeIndex, every: int) -> pd.DatetimeIndex:
    return dates[::every] if every > 1 else dates


def run_walk_forward(
    ds: Dataset,
    horizon: int,
    models: list[str],
    min_train: int = 756,
    step: int = 126,
    embargo: int = 5,
    train_window: int | None = None,
    holdout_fraction: float = 0.15,
    include_holdout: bool = False,
    subsample_every: int | None = None,
    model_params: dict | None = None,
    feature_cols: list[str] | None = None,
    progress=None,
) -> ForecastRun:
    frame = ds.frame
    feats = feature_cols or ds.feature_cols
    ycol = ds.label(horizon)
    le = frame[f"label_end_{horizon}"].groupby(level="date").max()
    dates = frame.index.get_level_values("date").unique().sort_values()
    folds = walk_forward_folds(dates, le, min_train, step, horizon, embargo, train_window, holdout_fraction, include_holdout)
    if not folds:
        raise ValueError("not enough history for a single walk-forward fold")
    every = subsample_every or max(1, min(horizon, 5))
    model_params = model_params or {}
    rows, fold_meta, imp_rows, ens_w = [], [], [], []
    oos_so_far: pd.DataFrame | None = None

    for k, fold in enumerate(folds):
        t0 = time.time()
        tr = frame.loc[frame.index.get_level_values("date").isin(_subsample(fold.train_dates, every))]
        tr = tr[tr[ycol].notna()]
        cal = frame.loc[frame.index.get_level_values("date").isin(fold.calib_dates)]
        cal = cal[cal[ycol].notna()]
        te = frame.loc[frame.index.get_level_values("date").isin(fold.test_dates)]
        if len(tr) < 500 or te.empty:
            continue
        out = pd.DataFrame(index=te.index)
        out["y"] = te[ycol]
        out["label_end"] = te[f"label_end_{horizon}"]
        out["fold"] = k
        Xtr, ytr = tr[feats].astype(np.float32), tr[ycol]
        for name in models:
            m = make_model(name, **model_params.get(name, {})).fit(Xtr, ytr)
            pc = m.predict(cal[feats].astype(np.float32)) if len(cal) else None
            pt = m.predict(te[feats].astype(np.float32))
            conf = ConformalInterval(0.8)
            prob = ProbabilityCalibrator()
            if pc is not None and len(cal) >= 30:
                conf.fit(cal[ycol].to_numpy(), pc["q10"].to_numpy(), pc["q90"].to_numpy())
                lo_c, hi_c = conf.apply(pc["q10"], pc["q90"])
                prob.fit(prob_above(0.0, lo_c, pc["q50"], hi_c), (cal[ycol] > 0).to_numpy())
            lo, hi = conf.apply(pt["q10"], pt["q90"])
            out[f"mean_{name}"] = pt["mean"].to_numpy()
            out[f"q10_{name}"] = lo
            out[f"q50_{name}"] = pt["q50"].to_numpy()
            out[f"q90_{name}"] = hi
            out[f"p_pos_{name}"] = prob.apply(prob_above(0.0, lo, pt["q50"].to_numpy(), hi))
            for feat, v in m.feature_importance().items():
                imp_rows.append({"fold": k, "model": name, "feature": feat, "importance": v})

        # ---- ensemble over non-trivial members, using only prior OOS rows with labels known by now ----
        members = [m for m in models if m != "zero_excess"]
        ens = StackingEnsemble(members, min_oos_rows=max(500, 20 * len(set(te.index.get_level_values("symbol")))))
        known = None
        if oos_so_far is not None:
            known = oos_so_far[(oos_so_far["label_end"] < fold.test_start) & oos_so_far["y"].notna()]
            ens.fit(known)
        preds = {m: pd.DataFrame({c: out[f"{c}_{m}"].to_numpy() for c in ("mean", "q10", "q50", "q90")}) for m in members}
        comb = ens.combine(preds)
        conf = ConformalInterval(0.8)
        prob = ProbabilityCalibrator()
        skill_t = 0.0
        if known is not None and len(known) >= 500 and "mean_ensemble" in known:
            conf.fit(known["y"], known["q10_ensemble_raw"], known["q90_ensemble_raw"])
            ics = rank_ic_series(known["mean_ensemble"], known["y"])
            skill_t = ic_tstat(ics, horizon)
        lo, hi = conf.apply(comb["q10"], comb["q90"])
        out["mean_ensemble"] = comb["mean"].to_numpy()
        out["q10_ensemble_raw"] = comb["q10"].to_numpy()
        out["q90_ensemble_raw"] = comb["q90"].to_numpy()
        out["q10_ensemble"], out["q50_ensemble"], out["q90_ensemble"] = lo, comb["q50"].to_numpy(), hi
        p_raw = prob_above(0.0, lo, comb["q50"].to_numpy(), hi)
        if known is not None and len(known) >= 500 and "p_raw_ensemble" in known:
            prob.fit(known["p_raw_ensemble"], known["y"] > 0)
        out["p_raw_ensemble"] = p_raw
        out["p_pos_ensemble"] = prob.apply(p_raw)
        out["disagreement"] = comb["disagreement"].to_numpy()
        out["sign_agreement"] = comb["sign_agreement"].to_numpy()
        out["ensemble_skill_t"] = skill_t
        out["feature_coverage"] = te[feats].notna().mean(axis=1).to_numpy()
        ens_w.append({"fold": k, "test_start": str(fold.test_start.date()), "fitted_on_rows": ens.fitted_on, **ens.weights})
        rows.append(out)
        oos_so_far = pd.concat(rows)
        meta = {"fold": k, "train_start": str(fold.train_dates[0].date()), "train_end": str(fold.train_dates[-1].date()),
                "calib_start": str(fold.calib_dates[0].date()), "test_start": str(fold.test_dates[0].date()),
                "test_end": str(fold.test_dates[-1].date()), "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                "ensemble_skill_t": round(skill_t, 3), "seconds": round(time.time() - t0, 2)}
        fold_meta.append(meta)
        log(logger, 20, "fold done", horizon=horizon, **meta)
        if progress:
            progress((k + 1) / len(folds), f"fold {k + 1}/{len(folds)}")

    oos = pd.concat(rows).sort_index()
    return ForecastRun(horizon, models, oos, fold_meta, pd.DataFrame(imp_rows), ens_w)
