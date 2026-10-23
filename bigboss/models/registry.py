"""Model versioning, champion/challenger evaluation, promotion gates, and rollback.

A model version = walk-forward OOS record + final models fitted on all labelled data + calibrators fitted on
the OOS record + ensemble weights + training-feature reference distribution (for drift) + metadata.
Promotion requires passing PREDETERMINED gates on the full OOS record (never a recent winning streak).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bigboss.backtest.metrics import forecast_quality
from bigboss.config import Config
from bigboss.features.dataset import Dataset
from bigboss.logging_utils import get_logger, log
from bigboss.models.distribution import ConformalInterval, ProbabilityCalibrator, prob_above
from bigboss.models.ensemble import StackingEnsemble
from bigboss.models.walkforward import ForecastRun, ic_tstat, rank_ic_series, run_walk_forward
from bigboss.models.zoo import make_model

logger = get_logger("models.registry")

DEFAULT_GATES = {
    "min_rank_ic_t": 1.0,            # ensemble OOS rank-IC t-stat (overlap-corrected)
    "coverage_band": [0.72, 0.88],   # 80% interval empirical coverage must land in this band
    "max_brier_excess": 0.005,       # Brier score may not exceed climatology by more than this
    "min_eval_periods": 24,          # independent horizon-length periods in the OOS record
    "challenger_min_ic_diff_t": 0.0, # paired t-stat of (challenger IC - champion IC) on common dates
    "min_net_ir": 0.0,               # OOS net-of-cost information ratio of the gated strategy (the actual objective)...
    "min_excess_psr": 0.90,          # ...AND >= 90% probability (PSR) that its true net excess Sharpe is positive
}


class Registry:
    def __init__(self, cfg: Config, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.root = cfg.data_dir / "models" / mode.lower()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "registry.json"

    # ---- state ----
    def state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"champions": {}, "history": []}

    def _save_state(self, st: dict) -> None:
        self.state_path.write_text(json.dumps(st, indent=1))

    def versions(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/meta.json"), reverse=True):
            out.append(json.loads(p.read_text()))
        return out

    def meta(self, vid: str) -> dict:
        return json.loads((self.root / vid / "meta.json").read_text())

    def champion(self, horizon: int) -> str | None:
        return self.state()["champions"].get(str(horizon))

    # ---- training ----
    def train(self, ds: Dataset, horizon: int, models: list[str], min_train: int = 756, step: int | None = None,
              progress=None) -> str:
        bt = self.cfg.section("backtest")
        run = run_walk_forward(ds, horizon, models, min_train=min_train, step=step or bt["retrain_every"],
                               embargo=bt["embargo_days"], train_window=bt.get("train_window"),
                               holdout_fraction=bt["final_holdout_fraction"], progress=progress)
        vid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-h{horizon}-" + uuid.uuid4().hex[:4]
        d = self.root / vid
        d.mkdir(parents=True)
        # Final models: all labelled data EXCEPT the locked holdout, subsampled like the walk-forward.
        ycol = ds.label(horizon)
        frame = ds.frame
        dates = frame.index.get_level_values("date").unique().sort_values()
        holdout_start = dates[int(len(dates) * (1 - bt["final_holdout_fraction"]))]
        lab = frame[(frame[ycol].notna()) & (frame[f"label_end_{horizon}"] < holdout_start)]
        ld = lab.index.get_level_values("date").unique().sort_values()
        every = max(1, min(horizon, 5))
        tr = lab.loc[lab.index.get_level_values("date").isin(ld[::every])]
        X, y = tr[ds.feature_cols].astype(np.float32), tr[ycol]
        fitted = {m: make_model(m).fit(X, y) for m in models}
        members = [m for m in models if m != "zero_excess"]
        ens = StackingEnsemble(members).fit(run.oos)
        oos = run.oos.dropna(subset=["y"])
        conf = ConformalInterval(0.8).fit(oos["y"], oos["q10_ensemble_raw"], oos["q90_ensemble_raw"])
        prob = ProbabilityCalibrator().fit(oos["p_raw_ensemble"], oos["y"] > 0)
        ics = rank_ic_series(oos["mean_ensemble"], oos["y"])
        skill_t = ic_tstat(ics, horizon)
        with open(d / "models.pkl", "wb") as f:
            pickle.dump({"models": fitted, "ensemble": ens, "conformal": conf, "prob": prob}, f)
        run.oos.reset_index().to_parquet(d / "oos.parquet")
        ref = X.quantile(np.linspace(0, 1, 11)).T  # deciles of training features (drift reference)
        ref.columns = [f"q{int(c * 100)}" for c in ref.columns]
        ref.to_parquet(d / "feature_reference.parquet")
        fq = [forecast_quality(run.oos, m, horizon) for m in models + ["ensemble"]]
        meta = {
            "version": vid, "horizon": horizon, "models": models, "mode": self.mode,
            "created_at": datetime.now(timezone.utc).isoformat(), "feature_cols": ds.feature_cols,
            "n_train_rows": int(len(tr)), "train_end_label_date": str(holdout_start.date()),
            "data_fingerprint": _fingerprint(ds), "ensemble_weights": ens.weights, "ensemble_skill_t": skill_t,
            "oos_start": str(run.oos.index.get_level_values("date").min().date()),
            "oos_end": str(run.oos.index.get_level_values("date").max().date()),
            "n_eval_periods": int(len(ics) / max(horizon, 1)), "forecast_quality": fq, "folds": run.folds,
            "conformal_adjust": conf.adjust,
        }
        meta["economic"] = self.economic_check(run, ds, horizon)
        meta["gates"] = self.evaluate_gates(meta)
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default))
        st = self.state()
        st["history"].append({"ts": meta["created_at"], "event": "trained", "version": vid, "horizon": horizon,
                              "gates_passed": meta["gates"]["passed"]})
        if self.champion(horizon) is None and meta["gates"]["passed"]:
            st["champions"][str(horizon)] = vid
            st["history"].append({"ts": meta["created_at"], "event": "promoted (first passing version)", "version": vid, "horizon": horizon})
        self._save_state(st)
        log(logger, 20, "model version trained", version=vid, skill_t=skill_t, gates=meta["gates"]["passed"])
        return vid

    # ---- gates ----
    def evaluate_gates(self, meta: dict, champion_meta: dict | None = None) -> dict:
        g = {**DEFAULT_GATES, **self.cfg.section("model_gates")}
        g["min_eval_periods"] = self.cfg.get("monitoring.min_eval_periods_for_promotion", g["min_eval_periods"])
        ens = next(f for f in meta["forecast_quality"] if f["model"] == "ensemble")
        checks = {
            "skill": {"value": meta["ensemble_skill_t"], "threshold": g["min_rank_ic_t"], "pass": meta["ensemble_skill_t"] >= g["min_rank_ic_t"]},
            "coverage": {"value": ens.get("interval80_coverage"), "band": g["coverage_band"],
                         "pass": ens.get("interval80_coverage") is not None and g["coverage_band"][0] <= ens["interval80_coverage"] <= g["coverage_band"][1]},
            "brier": {"value": ens.get("brier"), "climatology": ens.get("brier_climatology"),
                      "pass": ens.get("brier") is not None and ens["brier"] <= ens["brier_climatology"] + g["max_brier_excess"]},
            "sample_size": {"value": meta["n_eval_periods"], "threshold": g["min_eval_periods"], "pass": meta["n_eval_periods"] >= g["min_eval_periods"]},
        }
        econ = meta.get("economic") or {}
        ir, psr = econ.get("information_ratio"), econ.get("psr_excess")
        checks["economic_value"] = {"value": ir, "threshold": g["min_net_ir"], "psr_excess": psr, "min_psr": g["min_excess_psr"],
                                    "ann_excess_return": econ.get("ann_excess_return"), "ir_ci90": econ.get("ir_ci90"),
                                    "pass": ir is not None and psr is not None and ir > g["min_net_ir"] and psr >= g["min_excess_psr"]}
        if champion_meta is not None:
            checks["beats_champion"] = self._paired_ic_test(meta, champion_meta, g["challenger_min_ic_diff_t"])
        return {"passed": all(c["pass"] for c in checks.values()), "checks": checks}

    def economic_check(self, run: ForecastRun, ds: Dataset, horizon: int) -> dict:
        """Backtest the gated ensemble on the walk-forward OOS forecasts, net of costs (the platform's real objective)."""
        from bigboss.backtest.engine import MarketData, SimConfig, finalize, simulate
        from bigboss.backtest.experiment import _primary_benchmark, model_preds
        from bigboss.backtest.strategies import model_strategy
        from bigboss.costs import CostModel
        from bigboss.data.store import Store

        store = Store.for_mode(self.cfg.data_dir, self.mode)
        bt, pf, dec = self.cfg.section("backtest"), self.cfg.section("portfolio"), self.cfg.section("decision")
        bench = _primary_benchmark(ds)
        meta = store.symbols()
        prices = store.prices(ds.tradeable + [bench])
        md = MarketData.from_prices(prices, meta, bench)
        dates = run.oos.index.get_level_values("date").unique().sort_values()
        sim = SimConfig(rebalance_every=bt["rebalance_every"], delay=self.cfg.get("forecast.execution_delay_days", 1),
                        initial_capital=bt["initial_capital"], max_participation=self.cfg.get("costs.max_participation", 0.05),
                        start=dates[0], end=dates[-1])
        gate = {"max_disagreement": dec["max_model_disagreement"], "min_sign_agreement": dec.get("min_sign_agreement", 0.6),
                "min_skill_tstat": dec.get("min_skill_tstat", 1.0)}
        k = min(bt["top_k"], max(3, len(ds.tradeable) // 3))
        fn = model_strategy(md, model_preds(run, "ensemble"), CostModel(self.cfg), horizon, gate=gate,
                            risk_aversion=pf["risk_aversion"], max_weight=pf["max_weight"], max_sector=pf["max_sector_weight"],
                            max_turnover=pf.get("max_turnover"), top_k=k, sectors=meta.set_index("symbol")["sector"].to_dict())
        res = finalize(simulate("ensemble_gated", md, fn, sim, CostModel(self.cfg)), md, skip_days=sim.delay)
        keys = ("information_ratio", "ir_ci90", "ann_excess_return", "sharpe", "max_drawdown", "ann_turnover", "ann_cost_drag", "psr_excess")
        return {k_: res.summary.get(k_) for k_ in keys} | {"strategy": "ensemble_gated", "start": str(dates[0].date()), "end": str(dates[-1].date())}

    def reevaluate(self, vid: str) -> dict:
        """Re-run all gates (e.g. after gate definitions change) on a version's stored OOS record.

        A champion that no longer passes is demoted, and the demotion is recorded in the registry history.
        """
        from types import SimpleNamespace

        from bigboss.data.store import Store
        from bigboss.features.dataset import build_dataset

        meta = self.meta(vid)
        ds = build_dataset(Store.for_mode(self.cfg.data_dir, self.mode), self.cfg)
        meta["economic"] = self.economic_check(SimpleNamespace(oos=self.oos(vid)), ds, meta["horizon"])
        meta["gates"] = self.evaluate_gates(meta)
        (self.root / vid / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default))
        st = self.state()
        h = str(meta["horizon"])
        now = datetime.now(timezone.utc).isoformat()
        st["history"].append({"ts": now, "event": "re-evaluated", "version": vid, "horizon": meta["horizon"], "gates_passed": meta["gates"]["passed"]})
        demoted = False
        if st["champions"].get(h) == vid and not meta["gates"]["passed"]:
            del st["champions"][h]
            demoted = True
            failed = [k for k, c in meta["gates"]["checks"].items() if not c["pass"]]
            st["history"].append({"ts": now, "event": f"demoted: failed re-evaluation ({', '.join(failed)})", "version": vid, "horizon": meta["horizon"]})
        self._save_state(st)
        return {"version": vid, "gates": meta["gates"], "economic": meta["economic"], "demoted": demoted}

    def latest(self, horizon: int) -> str | None:
        vs = [v for v in self.versions() if v["horizon"] == horizon]
        return vs[0]["version"] if vs else None

    def _paired_ic_test(self, meta: dict, champ: dict, thresh: float) -> dict:
        a = pd.read_parquet(self.root / meta["version"] / "oos.parquet").set_index(["date", "symbol"])
        b = pd.read_parquet(self.root / champ["version"] / "oos.parquet").set_index(["date", "symbol"])
        ia = rank_ic_series(a["mean_ensemble"], a["y"])
        ib = rank_ic_series(b["mean_ensemble"], b["y"])
        diff = (ia - ib).dropna()
        t = ic_tstat(diff, meta["horizon"]) if len(diff) > 10 else 0.0
        return {"value": t, "threshold": thresh, "common_dates": int(len(diff)), "pass": len(diff) > 10 and t > thresh}

    def challenge(self, vid: str) -> dict:
        meta = self.meta(vid)
        champ = self.champion(meta["horizon"])
        res = self.evaluate_gates(meta, self.meta(champ) if champ and champ != vid else None)
        return {"version": vid, "champion": champ, "result": res}

    def promote(self, vid: str, force_reason: str | None = None) -> dict:
        res = self.challenge(vid)
        if not res["result"]["passed"] and not force_reason:
            return {"promoted": False, **res}
        st = self.state()
        h = str(self.meta(vid)["horizon"])
        prev = st["champions"].get(h)
        st["champions"][h] = vid
        st["history"].append({"ts": datetime.now(timezone.utc).isoformat(), "event": "promoted" if not force_reason else f"FORCED: {force_reason}",
                              "version": vid, "previous": prev, "horizon": int(h)})
        self._save_state(st)
        return {"promoted": True, "previous": prev, **res}

    def rollback(self, horizon: int) -> dict:
        st = self.state()
        promos = [e for e in st["history"] if e.get("horizon") == horizon and str(e.get("event", "")).startswith(("promoted", "FORCED"))]
        if len(promos) < 2:
            return {"rolled_back": False, "reason": "no previous champion"}
        target = promos[-2]["version"]
        cur = st["champions"].get(str(horizon))
        st["champions"][str(horizon)] = target
        st["history"].append({"ts": datetime.now(timezone.utc).isoformat(), "event": "promoted (rollback)", "version": target,
                              "previous": cur, "horizon": horizon})
        self._save_state(st)
        return {"rolled_back": True, "from": cur, "to": target}

    def delete(self, vid: str) -> None:
        if vid in self.state()["champions"].values():
            raise ValueError("cannot delete the current champion")
        shutil.rmtree(self.root / vid)

    # ---- inference ----
    def predict(self, vid: str, ds: Dataset, date: pd.Timestamp | None = None) -> pd.DataFrame:
        meta = self.meta(vid)
        with open(self.root / vid / "models.pkl", "rb") as f:
            art = pickle.load(f)
        frame = ds.frame
        date = date or latest_complete_date(ds)
        rows = frame.loc[frame.index.get_level_values("date") == date]
        missing = [c for c in meta["feature_cols"] if c not in rows.columns]
        for c in missing:
            rows = rows.assign(**{c: np.nan})
        X = rows[meta["feature_cols"]].astype(np.float32)
        preds = {m: mdl.predict(X) for m, mdl in art["models"].items()}
        ens: StackingEnsemble = art["ensemble"]
        comb = ens.combine({m: preds[m] for m in ens.members})
        lo, hi = art["conformal"].apply(comb["q10"], comb["q90"])
        out = pd.DataFrame(index=rows.index)
        for m, p in preds.items():
            out[f"mean_{m}"] = p["mean"].to_numpy()
        out["mean"], out["q10"], out["q50"], out["q90"] = comb["mean"].to_numpy(), lo, comb["q50"].to_numpy(), hi
        out["p_pos"] = art["prob"].apply(prob_above(0.0, lo, comb["q50"].to_numpy(), hi))
        out["disagreement"] = comb["disagreement"].to_numpy()
        out["sign_agreement"] = comb["sign_agreement"].to_numpy()
        out["n_models"] = len(ens.members)
        out["feature_coverage"] = X.notna().mean(axis=1).to_numpy()
        out["missing_features"] = len(missing)
        lgb = art["models"].get("lightgbm")
        if lgb is not None:
            contrib = lgb.mean_m.booster_.predict(X, pred_contrib=True)[:, :-1]
            names = np.array(meta["feature_cols"])
            out["top_drivers"] = [[(str(names[j]), float(c[j])) for j in np.argsort(-np.abs(c))[:5]] for c in contrib]
        out.attrs.update({"version": vid, "horizon": meta["horizon"], "skill_t": meta["ensemble_skill_t"], "as_of": str(date.date())})
        return out

    def feature_reference(self, vid: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / vid / "feature_reference.parquet")

    def oos(self, vid: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / vid / "oos.parquet").set_index(["date", "symbol"])


def latest_complete_date(ds: Dataset, min_fraction: float = 0.8) -> pd.Timestamp:
    """Latest date on which at least `min_fraction` of the universe has a fresh observation.

    Sources publish on different lags (e.g. Fed FX rates arrive ~weekly), so the very last calendar date can hold a
    nearly empty cross-section. Forecasting "as of" such a date would rank one asset against stale or missing ones.
    """
    obs = ds.frame["ret_1"].notna().groupby(level="date").sum() if "ret_1" in ds.frame else None
    dates = ds.frame.index.get_level_values("date").unique().sort_values()
    if obs is None or obs.empty:
        return dates.max()
    ok = obs[obs >= min_fraction * len(ds.tradeable)]
    return ok.index.max() if len(ok) else dates.max()


def _fingerprint(ds: Dataset) -> str:
    idx = ds.frame.index
    s = f"{ds.data_mode}|{idx.get_level_values('date').min()}|{idx.get_level_values('date').max()}|{len(idx)}|{','.join(sorted(ds.tradeable))}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
