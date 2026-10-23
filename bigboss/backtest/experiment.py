"""End-to-end research experiment: walk-forward forecasts -> strategies -> backtests -> diagnostics.

Outputs (saved under data/experiments/<id>/):
  summary.json      metrics for every strategy, cost sensitivity, ablations, forecast quality, folds, config
  returns.parquet   daily returns of every strategy + benchmark
  oos.parquet       every out-of-sample forecast (for the Model Intelligence page and audits)
Negative results are kept and reported exactly like positive ones.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bigboss.backtest.engine import MarketData, SimConfig, finalize, simulate
from bigboss.backtest.metrics import drawdown_series, forecast_quality
from bigboss.backtest.splits import holdout_boundary, record_holdout_access
from bigboss.backtest.strategies import benchmark_strategy, equal_weight_strategy, factor_topk_strategy, model_strategy
from bigboss.config import Config
from bigboss.costs import CostModel
from bigboss.data.store import Store
from bigboss.features.dataset import Dataset, build_dataset
from bigboss.logging_utils import get_logger, log
from bigboss.models.walkforward import ForecastRun, run_walk_forward
from bigboss.models.zoo import available_models

logger = get_logger("backtest.experiment")


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("records"))
    return o


def model_preds(run: ForecastRun, model: str) -> pd.DataFrame:
    cols = {f"{k}_{model}": k for k in ("mean", "q10", "q50", "q90")}
    p = run.oos[list(cols)].rename(columns=cols)
    if model == "ensemble":
        p["disagreement"] = run.oos["disagreement"]
        p["sign_agreement"] = run.oos["sign_agreement"]
        p["skill_t"] = run.oos["ensemble_skill_t"]
        p["feature_coverage"] = run.oos["feature_coverage"]
    return p


def run_experiment(store: Store, cfg: Config, horizon: int | None = None, models: list[str] | None = None,
                   include_holdout: bool = False, ablation: bool = False, cost_sensitivity: bool = True,
                   min_train: int | None = None, step: int | None = None, progress=None, name: str | None = None,
                   ds: Dataset | None = None) -> dict:
    t_start = time.time()
    h = horizon or cfg.get("forecast.primary_horizon", 20)
    bt = cfg.section("backtest")
    pf = cfg.section("portfolio")
    dec = cfg.section("decision")
    exp_id = name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4])
    out_dir = cfg.data_dir / "experiments" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda f, m: None)

    ds = ds or build_dataset(store, cfg)
    models = models or available_models()
    say(0.05, "dataset built")
    if include_holdout:
        n = record_holdout_access(cfg.data_dir, f"experiment {exp_id} h={h}")
        log(logger, 30, "FINAL HOLDOUT UNLOCKED", experiment=exp_id, total_accesses=n)
    run = run_walk_forward(ds, h, models, min_train=min_train or 756, step=step or bt["retrain_every"],
                           embargo=bt["embargo_days"], train_window=bt.get("train_window"),
                           holdout_fraction=bt["final_holdout_fraction"], include_holdout=include_holdout,
                           progress=lambda f, m: say(0.05 + 0.6 * f, m))
    prices = store.prices()
    meta = store.symbols()
    bench = _primary_benchmark(ds)
    md = MarketData.from_prices(prices[prices["symbol"].isin(ds.tradeable + [bench])], meta, bench)
    sectors = meta.set_index("symbol")["sector"].to_dict()
    oos_dates = run.oos.index.get_level_values("date").unique().sort_values()
    sim = SimConfig(rebalance_every=bt["rebalance_every"], delay=cfg.get("forecast.execution_delay_days", 1),
                    initial_capital=bt["initial_capital"], max_participation=cfg.get("costs.max_participation", 0.05),
                    start=oos_dates[0], end=oos_dates[-1])
    cm = CostModel(cfg)
    k = min(bt["top_k"], max(3, len(ds.tradeable) // 3))
    feats = ds.frame
    gate = {"max_disagreement": dec["max_model_disagreement"], "min_sign_agreement": dec.get("min_sign_agreement", 0.6),
            "min_skill_tstat": dec.get("min_skill_tstat", 1.0)}
    strat_kw = dict(risk_aversion=pf["risk_aversion"], max_weight=pf["max_weight"], max_sector=pf["max_sector_weight"],
                    max_turnover=pf.get("max_turnover"), top_k=k, sectors=sectors)
    strategies = {
        "benchmark": benchmark_strategy(bench),
        "equal_weight": equal_weight_strategy(md, ds.tradeable),
        "momentum_12_1_topk": factor_topk_strategy(feats, "mom_12_1", k),
        "low_vol_topk": factor_topk_strategy(feats, "vol_60", k, ascending=True),
    }
    for m in [m for m in models if m != "zero_excess"] + ["ensemble"]:
        strategies[f"model_{m}"] = model_strategy(md, model_preds(run, m), cm, h, **strat_kw)
    strategies["ensemble_gated"] = model_strategy(md, model_preds(run, "ensemble"), cm, h, gate=gate, **strat_kw)

    results = {}
    n_trials = len(strategies)
    for i, (nm, fn) in enumerate(strategies.items()):
        res = finalize(simulate(nm, md, fn, sim, cm), md, n_trials=n_trials, skip_days=sim.delay)
        results[nm] = res
        say(0.65 + 0.2 * (i + 1) / len(strategies), f"backtest {nm}")

    cost_sens = []
    if cost_sensitivity:
        for mult in (0.0, 0.5, 1.0, 2.0, 3.0):
            cmm = CostModel(cfg, multiplier=mult)
            for nm in ("ensemble_gated", "model_ensemble", "equal_weight"):
                base_fn = (model_strategy(md, model_preds(run, "ensemble"), cmm, h, gate=gate if nm == "ensemble_gated" else None, **strat_kw)
                           if nm != "equal_weight" else equal_weight_strategy(md, ds.tradeable))
                r = finalize(simulate(nm, md, base_fn, sim, cmm), md, n_trials=n_trials, skip_days=sim.delay)
                cost_sens.append({"strategy": nm, "cost_multiplier": mult, "ann_excess_return": r.summary["ann_excess_return"],
                                  "information_ratio": r.summary["information_ratio"], "ann_turnover": r.summary.get("ann_turnover")})
        say(0.9, "cost sensitivity")

    ablations = []
    if ablation:
        ablations = run_ablation(ds, h, cfg, md, cm, sim, strat_kw, min_train or 756, step or bt["retrain_every"])
        say(0.97, "ablation")

    fq = [forecast_quality(run.oos, m, h) for m in models + ["ensemble"]]
    summary = {
        "id": exp_id, "created_at": datetime.now(timezone.utc).isoformat(), "data_mode": store.mode, "horizon": h,
        "models": models, "benchmark": bench, "universe_size": len(ds.tradeable), "universe": ds.tradeable,
        "include_holdout": include_holdout,
        "holdout_starts": str(holdout_boundary(ds.frame.index.get_level_values("date"), bt["final_holdout_fraction"]).date()),
        "oos_start": str(oos_dates[0].date()), "oos_end": str(oos_dates[-1].date()),
        "n_strategies_tested": n_trials,
        "strategies": {nm: r.summary for nm, r in results.items()},
        "regimes": {nm: _jsonable(r.regimes) for nm, r in results.items()},
        "cost_sensitivity": cost_sens, "ablation": ablations, "forecast_quality": fq, "folds": run.folds,
        "ensemble_weights": run.ensemble_weights,
        "gated_decisions": results["ensemble_gated"].decisions[-50:],
        "feature_importance": (run.feature_importance.query("model == 'lightgbm'").groupby("feature")["importance"].mean()
                               .sort_values(ascending=False).head(25).to_dict() if not run.feature_importance.empty else {}),
        "assumptions": {
            "execution": f"decision at close t, fill at close t+{sim.delay}", "rebalance_every_days": sim.rebalance_every,
            "cash_rate": "0% (conservative)", "costs": cfg.section("costs"), "portfolio": pf,
            "survivorship": "universe = configured symbols that exist today unless delisted names are added; results are biased upward by survivorship",
            "benchmark": f"{bench} total-return proxy (adjusted close)",
        },
        "runtime_seconds": round(time.time() - t_start, 1),
        "sources": sorted(set(ds.sources.values())),
    }
    summary = _jsonable(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    rets = pd.DataFrame({nm: r.returns for nm, r in results.items()})
    rets["benchmark_index"] = results["benchmark"].benchmark
    rets.to_parquet(out_dir / "returns.parquet")
    run.oos.reset_index().to_parquet(out_dir / "oos.parquet")
    # Target weights at every decision date for every strategy (used by audits and the QuantConnect exporter).
    for nm, r in results.items():
        if not r.weights.empty:
            r.weights.to_parquet(out_dir / f"weights_{nm}.parquet")
    say(1.0, "done")
    log(logger, 20, "experiment complete", id=exp_id, seconds=summary["runtime_seconds"])
    return summary


def _primary_benchmark(ds: Dataset) -> str:
    vals = pd.Series(list(ds.benchmark_map.values()))
    return str(vals.mode().iloc[0])


def run_ablation(ds: Dataset, h: int, cfg: Config, md: MarketData, cm: CostModel, sim: SimConfig, strat_kw: dict,
                 min_train: int, step: int) -> list[dict]:
    """Drop one feature group at a time from LightGBM and measure the change in OOS IC and net IR."""
    bt = cfg.section("backtest")
    rows = []
    variants = {"all_features": ds.feature_cols}
    for g, cols in ds.groups.items():
        variants[f"without_{g}"] = [c for c in ds.feature_cols if c not in cols]
    for nm, cols in variants.items():
        if not cols:
            continue
        run = run_walk_forward(ds, h, ["lightgbm"], min_train=min_train, step=step, embargo=bt["embargo_days"],
                               train_window=bt.get("train_window"), holdout_fraction=bt["final_holdout_fraction"], feature_cols=cols)
        fq = forecast_quality(run.oos, "lightgbm", h)
        res = finalize(simulate(nm, md, model_strategy(md, model_preds(run, "lightgbm"), cm, h, **strat_kw), sim, cm), md, skip_days=sim.delay)
        rows.append({"variant": nm, "n_features": len(cols), "rank_ic_mean": fq["rank_ic_mean"], "rank_ic_t": fq["rank_ic_t"],
                     "information_ratio": res.summary["information_ratio"], "ann_excess_return": res.summary["ann_excess_return"]})
    base = next((r for r in rows if r["variant"] == "all_features"), None)
    if base:
        for r in rows:
            r["delta_ic_vs_all"] = (r["rank_ic_mean"] or 0) - (base["rank_ic_mean"] or 0)
            r["delta_ir_vs_all"] = (r["information_ratio"] or 0) - (base["information_ratio"] or 0)
    return rows


def list_experiments(cfg: Config) -> list[dict]:
    root = cfg.data_dir / "experiments"
    out = []
    for p in sorted(root.glob("*/summary.json"), reverse=True):
        s = json.loads(p.read_text())
        g = s["strategies"].get("ensemble_gated", {})
        out.append({"id": s["id"], "created_at": s["created_at"], "data_mode": s["data_mode"], "horizon": s["horizon"],
                    "oos_start": s["oos_start"], "oos_end": s["oos_end"], "universe_size": s["universe_size"],
                    "ensemble_gated_ir": g.get("information_ratio"), "ensemble_gated_excess": g.get("ann_excess_return"),
                    "include_holdout": s.get("include_holdout", False)})
    return out


def load_experiment(cfg: Config, exp_id: str) -> dict:
    d = cfg.data_dir / "experiments" / exp_id
    s = json.loads((d / "summary.json").read_text())
    rets = pd.read_parquet(d / "returns.parquet")
    eq = (1 + rets.fillna(0)).cumprod()
    s["equity_curves"] = {c: {"dates": [str(x.date()) for x in eq.index], "values": eq[c].round(5).tolist()} for c in eq.columns}
    s["drawdowns"] = {c: drawdown_series(rets[c]).round(5).tolist() for c in ("ensemble_gated", "benchmark", "equal_weight") if c in rets}
    return s
