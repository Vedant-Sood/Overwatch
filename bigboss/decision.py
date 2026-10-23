"""Economically optimal decision engine.

For each candidate we compute (horizon h, decimals):
  1 expected gross return          = expected excess + expected benchmark return (long-run assumption)
  2 expected benchmark-relative    = ensemble mean excess
  3-6 commissions, spread, slippage, market impact (from CostModel at the proposed size)
  7 expected net excess            = excess - round-trip cost (new position) or excess (already held)
  8 volatility / downside          = split-normal sigma, ES10, expected downside
  9 liquidity                      = participation vs. ADV
  10 portfolio impact              = marginal tracking-error contribution (if a covariance is supplied)
  11 P(net excess > 0)
  12 forecast uncertainty          = calibrated 80% interval + cross-model disagreement
and choose BUY / HOLD / REDUCE / SELL / AVOID / INSUFFICIENT_EVIDENCE. Doing nothing is always an option:
a trade must beat simply owning the benchmark fund after costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bigboss.config import Config
from bigboss.costs import CostModel
from bigboss.models.distribution import expected_downside, expected_shortfall, prob_above, sigma_from_quantiles
from bigboss.models.ensemble import abstain_mask

ACTIONS = ("BUY", "HOLD", "REDUCE", "SELL", "AVOID", "INSUFFICIENT_EVIDENCE")


@dataclass
class DecisionContext:
    horizon: int
    portfolio_value: float
    bench_expected_h: float          # expected benchmark return over horizon (assumption, shown to user)
    skill_t: float                   # ensemble OOS rank-IC t-stat
    regime: dict | None = None
    as_of: str | None = None
    data_mode: str = "OBSERVED"
    research_only: bool = False      # model failed promotion gates: show forecasts, never actionable decisions


def evaluate(cands: pd.DataFrame, cfg: Config, ctx: DecisionContext, cost_model: CostModel | None = None,
             current_weights: pd.Series | None = None, cov: pd.DataFrame | None = None, bench_symbol: str | None = None) -> pd.DataFrame:
    """cands: index symbol; columns mean,q10,q50,q90,p_pos,disagreement,sign_agreement,feature_coverage,
    asset_class,sector,price,adv_dollar,daily_vol,cs_spread (+ optional model means mean_<m>)."""
    d = cfg.section("decision")
    pcfg = cfg.section("portfolio")
    cm = cost_model or CostModel(cfg)
    cw = (current_weights if current_weights is not None else pd.Series(dtype=float)).reindex(cands.index).fillna(0.0)
    out = cands.copy()
    out["current_weight"] = cw
    sig = sigma_from_quantiles(out["q10"], out["q90"])
    out["sigma_h"] = sig
    out["expected_excess"] = out["mean"]
    out["expected_gross"] = out["mean"] + ctx.bench_expected_h

    # Cost at a provisional size (half the max weight) - sizes are refined below.
    prov = pcfg.get("max_weight", 0.1) * 0.5 * ctx.portfolio_value
    cost_rows = []
    for s, r in out.iterrows():
        ce = cm.estimate(s, str(r.get("asset_class", "equity")), prov, float(r.get("adv_dollar", np.nan)),
                         float(r.get("daily_vol", np.nan)), r.get("cs_spread"))
        cost_rows.append(ce.as_dict())
    costs = pd.DataFrame(cost_rows, index=out.index)
    out = out.join(costs.add_prefix("cost_"))
    rt = out["cost_total_round_trip_bps"] / 1e4
    held = out["current_weight"] > 1e-6
    # Entry costs of an existing position are sunk, so holding is compared to selling on its gross edge.
    out["expected_net_excess"] = np.where(held, out["mean"], out["mean"] - rt)
    hurdle = np.where(held, 0.0, rt)
    out["p_outperform"] = out["p_pos"]
    out["p_net_positive"] = prob_above(hurdle, out["q10"], out["q50"], out["q90"])
    # Re-scale the raw split-normal probability by the calibration applied to P(excess>0).
    raw0 = prob_above(0.0, out["q10"], out["q50"], out["q90"])
    out["p_net_positive"] = np.clip(out["p_net_positive"] * np.where(raw0 > 0, out["p_pos"] / np.maximum(raw0, 1e-6), 1), 0, 1)
    out["es10"] = expected_shortfall(out["q10"], out["q50"])
    out["expected_downside"] = expected_downside(out["q10"], out["q50"], out["q90"])
    out["cost_share_of_edge"] = np.where(out["mean"] > 0, rt / np.maximum(out["mean"], 1e-9), np.nan)

    if cov is not None and bench_symbol is not None:
        mte = []
        for s in out.index:
            if s in cov.index and bench_symbol in cov.index:
                # marginal active variance of moving 1% from the benchmark into s (per horizon)
                v = cov.loc[s, s] + cov.loc[bench_symbol, bench_symbol] - 2 * cov.loc[s, bench_symbol]
                mte.append(float(np.sqrt(max(v, 0) * ctx.horizon) * 0.01))
            else:
                mte.append(np.nan)
        out["te_per_1pct"] = mte

    skill_ok = ctx.skill_t >= d.get("min_skill_tstat", 1.0) and not ctx.research_only
    out["abstain"] = abstain_mask(out, d["max_model_disagreement"], d.get("min_sign_agreement", 0.6), skill_ok,
                                  out.get("feature_coverage"))
    # sizing: fractional Kelly on net edge, capped by max weight and by liquidity
    var_h = np.maximum(out["sigma_h"] ** 2, 1e-8)
    kelly = d.get("kelly_fraction", 0.25) * np.maximum(out["expected_net_excess"], 0) / var_h
    liq_cap = pcfg.get("max_weight", 0.1)
    adv = out.get("adv_dollar", pd.Series(np.nan, index=out.index))
    liq = np.where(adv > 0, cfg.get("costs.max_participation", 0.05) * adv * 3 / max(ctx.portfolio_value, 1), liq_cap)
    out["suggested_weight"] = np.minimum(np.minimum(kelly, liq_cap), liq)

    actions = []
    for s, r in out.iterrows():
        if r["abstain"]:
            a = "INSUFFICIENT_EVIDENCE"
        elif r["current_weight"] > 1e-6:
            if r["mean"] <= -(r["cost_total_one_way_bps"] / 1e4 + d["min_net_excess"]) and r["p_outperform"] <= 1 - d["min_prob_positive"]:
                a = "SELL"
            elif r["mean"] < 0 or r["suggested_weight"] < r["current_weight"] * 0.5:
                a = "REDUCE"
            else:
                a = "HOLD"
        else:
            ok = (r["expected_net_excess"] >= d["min_net_excess"] and r["p_net_positive"] >= d["min_prob_positive"]
                  and bool(r.get("cost_liquidity_ok", True)) and r["suggested_weight"] > 0.0025)
            a = "BUY" if ok else "AVOID"
        actions.append(a)
    out["action"] = actions
    out.loc[out["action"].isin(["AVOID", "INSUFFICIENT_EVIDENCE"]), "suggested_weight"] = 0.0
    out.loc[out["action"] == "SELL", "suggested_weight"] = 0.0
    out["skill_t"] = ctx.skill_t
    out["thesis"] = [build_thesis(s, r, ctx, cfg) for s, r in out.iterrows()]
    score = out["expected_net_excess"] / np.maximum(out["sigma_h"], 1e-6)
    out["risk_adjusted_score"] = score
    return out.sort_values("risk_adjusted_score", ascending=False)


def _pct(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{nd}f}%"


def build_thesis(symbol: str, r: pd.Series, ctx: DecisionContext, cfg: Config) -> dict:
    h = ctx.horizon
    summary = (f"{symbol}: expected {h}-day excess return vs benchmark {_pct(r['mean'])} "
               f"(80% interval {_pct(r['q10'])} to {_pct(r['q90'])}); estimated round-trip cost "
               f"{r['cost_total_round_trip_bps']:.1f} bps; expected net excess {_pct(r['expected_net_excess'])}; "
               f"P(net excess > 0) = {r['p_net_positive']:.0%}.")
    evidence = [f"Ensemble of {int(r.get('n_models', 0)) or 'several'} models; cross-model forecast std {_pct(r.get('disagreement', np.nan))}, "
                f"{r.get('sign_agreement', np.nan):.0%} of models agree on direction."]
    model_cols = [c for c in r.index if c.startswith("mean_") and c != "mean_ensemble"]
    if model_cols:
        evidence.append("Model forecasts: " + ", ".join(f"{c[5:]} {_pct(r[c])}" for c in model_cols))
    if "top_drivers" in r and isinstance(r["top_drivers"], list):
        evidence.append("Largest LightGBM feature contributions: " + ", ".join(f"{k} ({v:+.4f})" for k, v in r["top_drivers"]))
    evidence.append(f"Out-of-sample ensemble rank-IC t-stat: {ctx.skill_t:.2f} (gate: {cfg.get('decision.min_skill_tstat', 1.0)}).")
    counter = []
    if ctx.research_only:
        counter.append("This model version has NOT passed the promotion gates (including the net-of-cost economic test); research only.")
    if r["q10"] < 0 < r["q90"]:
        counter.append(f"The 80% interval spans both gains and losses; 10th percentile excess is {_pct(r['q10'])}.")
    if np.isfinite(r.get("cost_share_of_edge", np.nan)) and r.get("cost_share_of_edge", 0) > 0.3:
        counter.append(f"Trading costs consume {r['cost_share_of_edge']:.0%} of the expected gross edge.")
    if r.get("disagreement", 0) > 0.5 * cfg.get("decision.max_model_disagreement", 0.03):
        counter.append("Models disagree materially on magnitude.")
    if ctx.skill_t < 2:
        counter.append(f"Demonstrated forecast skill is weak (IC t-stat {ctx.skill_t:.2f} < 2); treat the edge as uncertain.")
    if ctx.regime and "bear" in str(ctx.regime.get("label", "")):
        counter.append(f"Current market regime is {ctx.regime.get('label')}; drawdown risk is elevated.")
    if r.get("cost_note"):
        counter.append(str(r["cost_note"]))
    if not counter:
        counter.append("No specific counterargument flagged; residual model risk and unknown events remain.")
    price = r.get("price", np.nan)
    invalidation = []
    if np.isfinite(price):
        stop = price * (1 + ctx.bench_expected_h + r["q10"])
        invalidation.append(f"Price closes below {stop:.2f} (the forecast's 10th-percentile path) before the horizon ends.")
    invalidation += [
        "The next forecast update shows expected net excess return below zero.",
        "Ensemble out-of-sample skill falls below the gate, or models start to disagree beyond the abstention threshold.",
        "A material filing (8-K items 1.03, 4.02, 5.01) or other high-importance event occurs.",
    ]
    risk = {"sigma_h": r["sigma_h"], "es10_h": r["es10"], "expected_downside_h": r["expected_downside"],
            "liquidity_ok": bool(r.get("cost_liquidity_ok", True)), "participation_at_size": r.get("cost_participation")}
    return {"summary": summary, "evidence": evidence, "counterarguments": counter, "invalidation": invalidation, "risk": risk,
            "disclaimer": "Model-based estimate for research; not a guarantee of future returns." +
                          (" SYNTHETIC DATA - not a real market." if ctx.data_mode == "SYNTHETIC" else "")}
