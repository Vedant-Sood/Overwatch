"""Application service: one façade over data, models, decisions, portfolio, paper trading, and monitoring.

Everything the API, CLI, and agents do goes through here, so the same deterministic code path produces
every number a user sees.
"""

from __future__ import annotations

import threading
from functools import cached_property

import numpy as np
import pandas as pd

from bigboss.backtest.metrics import regime_labels
from bigboss.config import Config, get_config
from bigboss.costs import CostModel
from bigboss.data.store import Store
from bigboss.decision import DecisionContext, evaluate
from bigboss.features.dataset import Dataset, build_dataset
from bigboss.models import events as ev_model
from bigboss.models.regime import detect_regimes, macro_instability
from bigboss.models.registry import Registry
from bigboss.models.valuation import valuation_profile
from bigboss.monitoring import feature_drift, live_performance, log_predictions, resolve_outcomes
from bigboss.paper.broker import PaperBroker
from bigboss.portfolio.optimize import optimize
from bigboss.portfolio.planning import goal_projection
from bigboss.risk import RiskLimits, historical_stress, portfolio_var_es, risk_contributions, risk_gate, shrunk_cov


class NoModelError(RuntimeError):
    pass


class NoDataError(RuntimeError):
    pass


def _clean(o):
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return None if pd.isna(o) else str(o.date()) if o == o.normalize() else o.isoformat()
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict("records"))
    if isinstance(o, pd.Series):
        return _clean(o.to_dict())
    return o


class BigBoss:
    _ds_lock = threading.Lock()

    def __init__(self, cfg: Config | None = None, mode: str = "OBSERVED"):
        self.cfg = cfg or get_config()
        self.mode = mode.upper()
        self.store = Store.for_mode(self.cfg.data_dir, self.mode)
        if self.mode == "SYNTHETIC":
            self.cfg = self.cfg.with_overrides({"benchmarks": {"default": "SYNMKT", "US": "SYNMKT"}})
        self.registry = Registry(self.cfg, self.mode)
        self._ds: Dataset | None = None
        self.costs = CostModel(self.cfg)

    @cached_property
    def broker(self) -> PaperBroker:
        return PaperBroker(self.cfg, self.store)

    # ---------------- data ----------------
    def has_data(self) -> bool:
        return int(self.store.query("SELECT COUNT(*) AS n FROM prices")["n"].iloc[0]) > 0

    def dataset(self, refresh: bool = False) -> Dataset:
        with self._ds_lock:
            if self._ds is None or refresh:
                if not self.has_data():
                    raise NoDataError(f"the {self.mode} store has no price data")
                self._ds = build_dataset(self.store, self.cfg)
            return self._ds

    def data_status(self) -> dict:
        fr = self.store.freshness()
        today = pd.Timestamp.today().normalize()
        max_stale = self.cfg.get("data.max_staleness_days", 5)
        if not fr.empty:
            fr["last_date"] = pd.to_datetime(fr["last_date"])
            fr["age_days"] = (today - fr["last_date"]).dt.days
            fr["stale"] = fr["age_days"] > max_stale
        issues = self.store.query("""SELECT symbol, check_name, severity, detail, ts FROM validation_issues
                                     WHERE run_id = (SELECT MAX(run_id) FROM validation_issues) ORDER BY severity, symbol""")
        log = self.store.query("SELECT * FROM ingestion_log ORDER BY ts DESC LIMIT 200")
        macro = self.store.query("SELECT series_id, MAX(date) AS last_date, COUNT(*) AS n, ANY_VALUE(source) AS source FROM macro GROUP BY series_id ORDER BY series_id")
        fund = self.store.query("SELECT symbol, COUNT(DISTINCT metric) AS metrics, MAX(available) AS last_filing FROM fundamentals GROUP BY symbol ORDER BY symbol")
        n_news = int(self.store.query("SELECT COUNT(*) AS n FROM news")["n"].iloc[0])
        return _clean({"data_mode": self.mode, "prices": fr, "validation_issues": issues, "ingestion_log": log, "n_news": n_news,
                       "macro": macro, "fundamentals": fund, "n_events": int(self.store.query("SELECT COUNT(*) AS n FROM events")["n"].iloc[0])})

    # ---------------- market intelligence ----------------
    def market_overview(self) -> dict:
        p = self.store.prices()
        if p.empty:
            raise NoDataError("no price data")
        adj = p.pivot(index="date", columns="symbol", values="adj_close").sort_index()
        meta = self.store.symbols().set_index("symbol")
        rows = []
        for s in adj.columns:
            x = adj[s].dropna()
            if len(x) < 2:
                continue
            last = x.index[-1]
            def ret(n):
                return float(x.iloc[-1] / x.iloc[-1 - n] - 1) if len(x) > n else None
            ytd_base = x[x.index < pd.Timestamp(year=last.year, month=1, day=1)]
            rows.append({"symbol": s, "name": meta.loc[s, "name"] if s in meta.index else None,
                         "asset_class": meta.loc[s, "asset_class"] if s in meta.index else None,
                         "sector": meta.loc[s, "sector"] if s in meta.index else None, "last_date": last, "last": float(x.iloc[-1]),
                         "ret_1d": ret(1), "ret_5d": ret(5), "ret_20d": ret(20), "ret_252d": ret(252),
                         "ret_ytd": float(x.iloc[-1] / ytd_base.iloc[-1] - 1) if len(ytd_base) else None,
                         "vol_20d": float(np.log(x).diff().tail(20).std() * np.sqrt(252))})
        tab = pd.DataFrame(rows)
        bench = self._primary_benchmark()
        regime = detect_regimes(p, bench) if bench in adj else {"available": False}
        macro = self.store.macro()
        return _clean({"data_mode": self.mode, "as_of": adj.index[-1], "assets": tab, "benchmark": bench,
                       "regime": regime, "macro": macro_instability(macro) if not macro.empty else {},
                       "macro_series": self._macro_series(macro)})

    def _macro_series(self, macro: pd.DataFrame) -> dict:
        out = {}
        for sid in ("DGS10", "DGS2", "VIXCLS", "BAMLH0A0HYM2", "T10Y2Y"):
            s = macro[macro["series_id"] == sid].tail(750)
            if len(s):
                out[sid] = {"dates": s["date"].dt.strftime("%Y-%m-%d").tolist(), "values": s["value"].round(4).tolist()}
        return out

    def _primary_benchmark(self) -> str:
        if self.mode == "SYNTHETIC":
            return "SYNMKT"
        return self.cfg.get("benchmarks.US", "SPY")

    def bench_expected(self, horizon: int, bench: str | None = None) -> float:
        """Long-run expected benchmark return over the horizon: trailing 10y geometric mean (an assumption, disclosed)."""
        bench = bench or self._primary_benchmark()
        x = self.store.panel("adj_close", [bench])
        if x.empty:
            return 0.0
        s = x[bench].dropna().iloc[-2520:]
        g = (s.iloc[-1] / s.iloc[0]) ** (1 / max(len(s) - 1, 1)) - 1
        return float((1 + g) ** horizon - 1)

    # ---------------- forecasting & decisions ----------------
    def forecast(self, horizon: int | None = None, date: pd.Timestamp | None = None) -> pd.DataFrame:
        h = horizon or self.cfg.get("forecast.primary_horizon", 20)
        vid = self.registry.champion(h)
        if vid is None:
            raise NoModelError(f"no champion model for horizon {h} in {self.mode} mode - run `bigboss train`")
        return self.registry.predict(vid, self.dataset(), date)

    def _liquidity(self, symbols: list[str]) -> pd.DataFrame:
        p = self.store.prices(symbols)
        p = p.sort_values("date").groupby("symbol").tail(60)
        rows = []
        for s, g in p.groupby("symbol"):
            dv = (g["close"] * g["volume"]).tail(20).mean()
            lr = np.log(g["adj_close"]).diff()
            rows.append({"symbol": s, "price": float(g["close"].iloc[-1]), "price_date": g["date"].iloc[-1],
                         "adv_dollar": float(dv) if np.isfinite(dv) else np.nan, "daily_vol": float(lr.std())})
        liq = pd.DataFrame(rows).set_index("symbol")
        ds = self.dataset()
        last = ds.frame.index.get_level_values("date").max()
        cs = ds.frame.xs(last, level="date")["cs_spread_20"] if "cs_spread_20" in ds.frame else pd.Series(dtype=float)
        liq["cs_spread"] = cs.reindex(liq.index)
        return liq

    def scan(self, horizon: int | None = None, account_id: str | None = None, log: bool = False) -> dict:
        h = horizon or self.cfg.get("forecast.primary_horizon", 20)
        f = self.forecast(h)
        as_of = pd.Timestamp(f.attrs["as_of"])
        cands = f.droplevel("date")
        meta = self.store.symbols().set_index("symbol")
        cands = cands.join(meta[["asset_class", "sector", "name", "market", "currency"]], how="left")
        cands = cands.join(self._liquidity(list(cands.index)), how="left")
        cur_w, pv = None, float(self.cfg.get("paper.initial_cash", 100000))
        if account_id:
            s = self.broker.summary(account_id)
            cur_w, pv = pd.Series(s["weights"], dtype=float), s["equity"]
        regime = self._regime_now()
        bench = self._primary_benchmark()
        cov = None
        try:
            rets = self.store.panel("adj_close", list(cands.index) + [bench]).pct_change(fill_method=None).iloc[-252:]
            cov = shrunk_cov(rets)
        except ValueError:
            pass
        ctx = DecisionContext(h, pv, self.bench_expected(h), f.attrs["skill_t"], regime, str(as_of.date()), self.mode)
        dec = evaluate(cands, self.cfg, ctx, self.costs, cur_w, cov, bench)
        if log:
            log_predictions(self.store, dec, f.attrs["version"], h, as_of)
        cols = ["name", "sector", "asset_class", "market", "currency", "action", "price", "price_date", "mean", "q10", "q50", "q90",
                "expected_gross", "expected_excess", "cost_commission_bps", "cost_half_spread_bps", "cost_slippage_bps",
                "cost_impact_bps", "cost_tax_bps", "cost_total_round_trip_bps", "expected_net_excess", "sigma_h", "es10",
                "expected_downside", "p_outperform", "p_net_positive", "disagreement", "sign_agreement", "feature_coverage",
                "cost_liquidity_ok", "cost_participation", "adv_dollar", "daily_vol", "te_per_1pct", "current_weight", "suggested_weight",
                "risk_adjusted_score", "abstain", "thesis", "top_drivers"] + [c for c in dec.columns if c.startswith("mean_")]
        out = dec[[c for c in cols if c in dec.columns]].reset_index().rename(columns={"index": "symbol"})
        return _clean({"data_mode": self.mode, "as_of": as_of, "horizon": h, "model_version": f.attrs["version"],
                       "skill_t": f.attrs["skill_t"], "skill_gate": self.cfg.get("decision.min_skill_tstat"),
                       "bench_expected_h": ctx.bench_expected_h, "benchmark": bench, "regime": regime,
                       "ensemble_weights": self.registry.meta(f.attrs["version"])["ensemble_weights"],
                       "counts": dec["action"].value_counts().to_dict(), "candidates": out})

    def _regime_now(self) -> dict:
        p = self.store.prices()
        r = detect_regimes(p, self._primary_benchmark())
        return {k: v for k, v in r.items() if k not in ("history", "markov_history")}

    # ---------------- single-asset research ----------------
    def research(self, symbol: str) -> dict:
        symbol = symbol.upper()
        p = self.store.prices([symbol])
        if p.empty:
            raise NoDataError(f"no price data for {symbol}")
        meta = self.store.symbols().set_index("symbol")
        info = meta.loc[symbol].to_dict() if symbol in meta.index else {}
        bench = info.get("benchmark") or self._primary_benchmark()
        bp = self.store.prices([bench])
        hist = p.tail(1500)
        out = {"data_mode": self.mode, "symbol": symbol, "info": info, "benchmark": bench,
               "prices": {"dates": hist["date"].dt.strftime("%Y-%m-%d").tolist(), "open": hist["open"].round(4).tolist(),
                          "high": hist["high"].round(4).tolist(), "low": hist["low"].round(4).tolist(),
                          "close": hist["close"].round(4).tolist(), "adj_close": hist["adj_close"].round(4).tolist(),
                          "volume": hist["volume"].tolist()},
               "benchmark_prices": {"dates": bp.tail(1500)["date"].dt.strftime("%Y-%m-%d").tolist(),
                                    "adj_close": bp.tail(1500)["adj_close"].round(4).tolist()},
               "source": p["source"].iloc[-1], "adjustment": p["adjustment"].iloc[-1], "last_date": p["date"].iloc[-1]}
        try:
            ds = self.dataset()
            if symbol in ds.tradeable:
                last = ds.frame.index.get_level_values("date").max()
                row = ds.frame.loc[(last, symbol), ds.feature_cols]
                out["features"] = {g: {c: row[c] for c in cols} for g, cols in ds.groups.items()}
        except NoDataError:
            pass
        forecasts = {}
        for h in self.cfg.get("forecast.horizons"):
            vid = self.registry.champion(h)
            if not vid:
                continue
            try:
                f = self.registry.predict(vid, self.dataset())
                if symbol in f.index.get_level_values("symbol"):
                    r = f.xs(symbol, level="symbol").iloc[0]
                    forecasts[str(h)] = {**{k: r[k] for k in ("mean", "q10", "q50", "q90", "p_pos", "disagreement", "sign_agreement")},
                                         "model_means": {c[5:]: r[c] for c in f.columns if c.startswith("mean_")},
                                         "top_drivers": r.get("top_drivers"), "version": vid, "skill_t": f.attrs["skill_t"]}
            except Exception as e:  # surface model problems instead of hiding them
                forecasts[str(h)] = {"error": str(e)}
        out["forecasts"] = forecasts
        primary = str(self.cfg.get("forecast.primary_horizon", 20))
        if primary in forecasts and "error" not in forecasts[primary]:
            try:
                sc = self.scan(int(primary))
                row = next((c for c in sc["candidates"] if c["symbol"] == symbol), None)
                out["decision"] = row
            except (NoModelError, NoDataError) as e:
                out["decision_error"] = str(e)
        fund = self.store.fundamentals(symbol)
        out["valuation"] = valuation_profile(fund, float(p["close"].iloc[-1]), p["date"].iloc[-1]) if not fund.empty else {"available": False, "reason": "no SEC fundamentals (non-US, ETF, or not ingested)"}
        out["events"] = self.events(symbol)
        out["news"] = self.news(symbol)
        return _clean(out)

    def events(self, symbol: str | None = None, limit: int = 50) -> dict:
        evs = self.store.events(limit=100000)
        if evs.empty:
            return {"events": [], "event_study": [], "note": "no filing events ingested"}
        adj = self.store.panel("adj_close")
        bmap = self.store.symbols().set_index("symbol")["benchmark"].to_dict()
        as_of = adj.index[-1]
        study = ev_model.event_study(evs, adj, bmap, as_of)
        scored = ev_model.score_events(evs if symbol is None else evs[evs["symbol"] == symbol], study, as_of)
        return _clean({"as_of": as_of, "events": scored.head(limit), "event_study": study})

    def news(self, symbol: str | None = None, limit: int = 60) -> dict:
        n = self.store.news(symbol, limit)
        if n.empty:
            return {"items": [], "note": "no news ingested (run ingestion with news enabled)"}
        recent = n[pd.to_datetime(n["published_at"]) >= pd.Timestamp.now() - pd.Timedelta(days=30)]
        return _clean({"items": n, "n_30d": int(len(recent)),
                       "mean_sentiment_30d": float(recent["sentiment"].mean()) if len(recent) else None,
                       "share_major_wires_30d": float((recent["credibility"] == "major_newswire").mean()) if len(recent) else None,
                       "method": "headline lexicon sentiment in [-1, 1]; weak signal, not used by forecasting models"})

    # ---------------- portfolio ----------------
    def propose_portfolio(self, profile: str = "balanced", capital: float | None = None, holdings: dict[str, float] | None = None,
                          method: str = "mean_variance", horizon: int | None = None, account_id: str | None = None,
                          goal: float | None = None, years: float | None = None, monthly_contribution: float = 0.0) -> dict:
        prof = self.cfg.section("profiles").get(profile)
        if prof is None:
            raise ValueError(f"unknown profile {profile}")
        h = horizon or self.cfg.get("forecast.primary_horizon", 20)
        if account_id:
            s = self.broker.summary(account_id)
            capital = s["equity"]
            holdings = {k: v * s["equity"] for k, v in s["weights"].items()}
        if not capital or capital <= 0:
            raise ValueError("capital must be provided (BigBoss does not assume personal financial figures)")
        holdings = holdings or {}
        scan = self.scan(h, account_id)
        c = pd.DataFrame(scan["candidates"]).set_index("symbol")
        bench = scan["benchmark"]
        usable = c[(~c["abstain"].astype(bool))]
        # alpha forecasts net of exit costs; abstained names get zero alpha (they can still be held at index-like risk)
        mu = (usable["mean"] - usable["cost_total_round_trip_bps"] / 2e4)
        cand = list(mu.sort_values(ascending=False).index[:30]) + [s for s in holdings if s in c.index]
        universe = list(dict.fromkeys(cand + [bench]))
        rets = self.store.panel("adj_close", universe).pct_change(fill_method=None).iloc[-504:]
        cov_d = shrunk_cov(rets)
        universe = list(cov_d.index)
        cov_h = cov_d * h
        mu_u = mu.reindex(universe).fillna(0.0)
        mu_u[bench] = 0.0
        w0 = pd.Series({k: v / capital for k, v in holdings.items()}).reindex(universe).fillna(0.0)
        cost = (c["cost_total_round_trip_bps"] / 2e4).reindex(universe).fillna(0.0005)
        sectors = self.store.symbols().set_index("symbol")["sector"].to_dict()
        kw = dict(w0=w0, cost=cost, risk_aversion=prof["risk_aversion"], max_weight=prof["max_weight"], sectors=sectors,
                  max_sector=self.cfg.get("portfolio.max_sector_weight", 0.3), max_turnover=None, exempt={bench},
                  min_invested=1.0, w_ref=pd.Series({bench: 1.0}))
        if method == "robust":
            sig = ((c["q90"] - c["q10"]) / 2.563).reindex(universe).fillna(0.05)
            kw["mu_se"] = sig / np.sqrt(max(1, self.registry.meta(scan["model_version"])["n_eval_periods"]))
        if method == "black_litterman":
            kw["w_mkt"] = pd.Series({bench: 1.0}).reindex(universe).fillna(0.0)
            kw["view_var"] = (((c["q90"] - c["q10"]) / 2.563) ** 2).reindex(universe)
        if method == "min_cvar":
            caps = {s: (1.0 if s == bench else float(prof["max_weight"])) for s in universe}
            kw = {"scenarios": rets[universe].fillna(0).rolling(h).sum().dropna().iloc[::max(1, h // 2)], "max_weight": caps,
                  "risk_aversion": prof["risk_aversion"]}
        if method in ("risk_parity", "hrp"):
            kw = {"max_weight": prof["max_weight"]}
            sub = [s for s in universe if s in cand[:15] or s == bench]
            cov_h = cov_h.loc[sub, sub]
            mu_u = mu_u.reindex(sub)
        res = optimize(method, mu_u, cov_h, **kw)
        w = res.weights[res.weights > 1e-4]
        limits = RiskLimits(max_weight=prof["max_weight"], max_sector_weight=self.cfg.get("portfolio.max_sector_weight", 0.3),
                            max_ann_vol=prof.get("max_vol"))
        check = risk_gate(w, limits, sectors, cov_d, exempt={bench})
        notes = []
        vol = check.metrics.get("ex_ante_ann_vol")
        if limits.max_ann_vol and vol and vol > limits.max_ann_vol:
            # De-risk by holding cash rather than silently breaching the profile's volatility limit.
            scale = limits.max_ann_vol / vol * 0.995
            w = w * scale
            notes.append(f"scaled risky weights by {scale:.2f} (rest in cash) to meet the {limits.max_ann_vol:.0%} volatility limit")
            check = risk_gate(w, limits, sectors, cov_d, exempt={bench})
        trades = []
        trade_syms = sorted(set(w.index) | set(k for k, v in holdings.items() if v))
        extra = [x for x in trade_syms if x not in c.index]
        if extra:  # e.g. the benchmark fund: look up its real liquidity rather than assume none
            meta_x = self.store.symbols().set_index("symbol")
            liq_x = self._liquidity(extra).join(meta_x[["asset_class"]], how="left")
            c = pd.concat([c, liq_x])
        for s in trade_syms:
            tgt = float(w.get(s, 0.0)) * capital
            cur = float(holdings.get(s, 0.0))
            delta = tgt - cur
            if abs(delta) < 1:
                continue
            row = c.loc[s] if s in c.index else None
            est = self.costs.estimate(s, str(row["asset_class"]) if row is not None else "equity_etf", abs(delta),
                                      float(row["adv_dollar"]) if row is not None and row["adv_dollar"] is not None else np.nan,
                                      float(row["daily_vol"]) if row is not None and row["daily_vol"] is not None else np.nan)
            trades.append({"symbol": s, "side": "BUY" if delta > 0 else "SELL", "notional": abs(delta), "target_weight": float(w.get(s, 0.0)),
                           "current_weight": cur / capital, "est_cost": abs(delta) * est.total_one_way_bps / 1e4,
                           "est_cost_bps": est.total_one_way_bps})
        port_rets = self.store.panel("adj_close", list(w.index)).pct_change(fill_method=None)
        out = {"data_mode": self.mode, "profile": profile, "profile_params": prof, "method": res.method, "capital": capital,
               "horizon": h, "benchmark": bench, "weights": w.round(5).to_dict(), "optimizer": res.as_dict(),
               "risk_check": {"passed": check.passed, "violations": check.violations, **check.metrics},
               "risk_contributions": risk_contributions(w, cov_d).loc[lambda s: s.abs() > 1e-6].round(5).to_dict(),
               "var_es_1d": portfolio_var_es(w, port_rets.iloc[-1260:]),
               "var_es_h": portfolio_var_es(w, port_rets.iloc[-1260:], horizon=h),
               "stress": historical_stress(w, self.store.prices(list(w.index))),
               "trades": trades, "total_est_cost": float(sum(t["est_cost"] for t in trades)),
               "expected_active_return_h": float((mu_u.reindex(w.index).fillna(0) * w).sum()),
               "expected_tracking_error_h": float(np.sqrt(max(res.risk, 0))) if np.isfinite(res.risk) else None,
               "cash_weight": float(1 - w.sum()), "notes": notes,
               "model_version": scan["model_version"], "skill_t": scan["skill_t"]}
        if prof.get("simulated_only"):
            out["notice"] = "Experimental profile: simulated funds only."
        if years:
            out["projection"] = goal_projection(w, port_rets, capital, years, goal, monthly_contribution)
        if not check.passed:
            out["status"] = "REJECTED_BY_RISK_GATE"
        return _clean(out)

    # ---------------- model intelligence ----------------
    def model_intelligence(self) -> dict:
        st = self.registry.state()
        versions = self.registry.versions()
        out = {"data_mode": self.mode, "champions": st["champions"], "history": st["history"][-50:],
               "versions": [{k: v for k, v in m.items() if k not in ("feature_cols", "folds")} for m in versions[:30]]}
        resolve_outcomes(self.store, self.cfg.get("forecast.execution_delay_days", 1))
        out["live_performance"] = live_performance(self.store)
        h = str(self.cfg.get("forecast.primary_horizon", 20))
        vid = st["champions"].get(h)
        if vid:
            try:
                ds = self.dataset()
                drift = feature_drift(self.registry.feature_reference(vid), ds.frame, self.registry.meta(vid)["feature_cols"],
                                      warn=self.cfg.get("monitoring.psi_warn", 0.1), alert=self.cfg.get("monitoring.psi_alert", 0.25))
                out["drift"] = drift
                oos = self.registry.oos(vid)
                from bigboss.models.walkforward import rank_ic_series
                ics = rank_ic_series(oos["mean_ensemble"], oos["y"]).rolling(63, min_periods=20).mean().dropna()
                out["rolling_ic"] = {"dates": [str(d.date()) for d in ics.index], "values": ics.round(4).tolist()}
            except NoDataError:
                pass
        return _clean(out)
