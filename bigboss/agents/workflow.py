"""Autonomous research agent harness.

Agents are deterministic programs that call typed tools (the same service functions the dashboard uses) and
emit typed, timestamped findings with sources. An optional local LLM only turns the evidence bundle into
prose, and is rejected if it introduces numbers not present in the evidence.

Committee rule: the Investment Committee can make the quantitative decision MORE conservative (downgrade to
AVOID / REDUCE / INSUFFICIENT_EVIDENCE) when other agents raise objections, but can never upgrade it. All
disagreements are recorded explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

from bigboss import __version__
from bigboss.agents.llm import get_llm, grounded_narrative
from bigboss.monitoring import live_performance, resolve_outcomes
from bigboss.service import BigBoss, NoDataError, NoModelError, _clean


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    started_at: str
    duration_ms: float
    ok: bool
    output_sha256: str | None = None
    error: str | None = None


class Finding(BaseModel):
    statement: str
    value: Any = None
    source: str
    as_of: str | None = None


class AgentReport(BaseModel):
    agent: str
    status: str = "ok"                      # ok | degraded | failed
    findings: list[Finding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class CommitteeDecision(BaseModel):
    symbol: str
    quant_action: str | None
    final_action: str
    suggested_weight: float = 0.0
    thesis: dict[str, Any] = Field(default_factory=dict)
    disagreements: list[str] = Field(default_factory=list)
    vetoes: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    report_id: str
    created_at: str
    question: str
    data_mode: str
    symbols: list[str]
    agents: list[AgentReport]
    decisions: list[CommitteeDecision]
    narrative: dict[str, Any]
    reproducibility: dict[str, Any]


class Toolbox:
    """Typed tool registry with call auditing."""

    def __init__(self, bb: BigBoss):
        self.bb = bb
        self.tools: dict[str, Callable] = {
            "market_overview": bb.market_overview,
            "scan": bb.scan,
            "research": bb.research,
            "events": bb.events,
            "propose_portfolio": bb.propose_portfolio,
            "model_intelligence": bb.model_intelligence,
            "data_status": bb.data_status,
            "resolve_outcomes": lambda: resolve_outcomes(bb.store, bb.cfg.get("forecast.execution_delay_days", 1)),
            "live_performance": lambda: live_performance(bb.store),
        }

    def call(self, report: AgentReport, name: str, **args):
        t0 = time.time()
        started = datetime.now(timezone.utc).isoformat()
        try:
            out = self.tools[name](**args)
            digest = hashlib.sha256(json.dumps(_clean(out), default=str, sort_keys=True).encode()).hexdigest()
            report.tool_calls.append(ToolCall(tool=name, args=args, started_at=started, duration_ms=(time.time() - t0) * 1000,
                                              ok=True, output_sha256=digest))
            return out
        except (NoDataError, NoModelError, ValueError, KeyError) as e:
            report.tool_calls.append(ToolCall(tool=name, args=args, started_at=started, duration_ms=(time.time() - t0) * 1000,
                                              ok=False, error=str(e)))
            report.status = "degraded"
            report.warnings.append(f"{name} failed: {e}")
            return None


# ------------------------- agents -------------------------

def market_analyst(tb: Toolbox) -> AgentReport:
    r = AgentReport(agent="market_analyst")
    mo = tb.call(r, "market_overview")
    if not mo:
        return r
    reg = mo.get("regime", {})
    src = f"{tb.bb.mode} price store"
    r.findings.append(Finding(statement=f"Market regime ({mo['benchmark']}): {reg.get('label')}", value=reg.get("label"), source=src, as_of=mo["as_of"]))
    if reg.get("vol_20d_ann") is not None:
        r.findings.append(Finding(statement=f"Benchmark 20d realized volatility {reg['vol_20d_ann']:.1%} (percentile {reg.get('vol_percentile', 0):.0%})",
                                  value=reg["vol_20d_ann"], source=src, as_of=mo["as_of"]))
    if reg.get("markov_p_high_vol") is not None:
        r.findings.append(Finding(statement=f"Markov-switching filtered P(high-vol regime) = {reg['markov_p_high_vol']:.0%}",
                                  value=reg["markov_p_high_vol"], source="statsmodels MarkovRegression on benchmark returns", as_of=mo["as_of"]))
    if reg.get("correlation_breakdown"):
        r.warnings.append("Average pairwise correlation is in the top decile of its history - diversification is weaker than usual.")
    if reg.get("liquidity_stress"):
        r.warnings.append("Cross-sectional illiquidity is in the top decile of its history.")
    assets = [a for a in mo["assets"] if a.get("ret_20d") is not None]
    if assets:
        best = sorted(assets, key=lambda a: a["ret_20d"], reverse=True)[:3]
        worst = sorted(assets, key=lambda a: a["ret_20d"])[:3]
        r.findings.append(Finding(statement="Strongest 20d: " + ", ".join(f"{a['symbol']} {a['ret_20d']:+.1%}" for a in best), source=src, as_of=mo["as_of"]))
        r.findings.append(Finding(statement="Weakest 20d: " + ", ".join(f"{a['symbol']} {a['ret_20d']:+.1%}" for a in worst), source=src, as_of=mo["as_of"]))
    for k, v in (mo.get("macro") or {}).items():
        if isinstance(v, dict) and "value" in v:
            r.findings.append(Finding(statement=f"{k} = {v['value']:.2f} ({v['percentile_10y']:.0%} 10y percentile)", value=v["value"], source="FRED", as_of=v["date"]))
    r.data = {"regime": {k: v for k, v in reg.items() if k not in ("history", "markov_history")}}
    return r


def fundamental_analyst(tb: Toolbox, symbol: str, research: dict | None) -> AgentReport:
    r = AgentReport(agent="fundamental_analyst")
    val = (research or {}).get("valuation") or {}
    if not val.get("available"):
        r.status = "degraded"
        r.warnings.append(f"No fundamentals for {symbol}: {val.get('reason', 'unavailable')}")
        return r
    src = f"SEC EDGAR XBRL (filing available {val['filing_based_on']})"
    for k, label in (("pe", "P/E"), ("fcf_yield", "FCF yield"), ("revenue_growth_yoy", "Revenue growth YoY"), ("roic", "ROIC"),
                     ("operating_margin", "Operating margin"), ("debt_to_equity", "Debt/Equity")):
        v = val.get(k)
        if v is not None:
            txt = f"{v:.1f}" if k in ("pe", "debt_to_equity") else f"{v:.1%}"
            r.findings.append(Finding(statement=f"{label}: {txt}", value=v, source=src, as_of=val["as_of"]))
    q = val.get("quality_score", {})
    if q.get("score") is not None:
        r.findings.append(Finding(statement=f"Quality score {q['score']:.0f}/100 from {q['n_components']} components", value=q["score"], source=src, as_of=val["as_of"]))
    for name, sc in val.get("scenarios", {}).items():
        if sc.get("implied_return") is not None:
            r.findings.append(Finding(statement=f"DCF {name} case: value/share {sc['value_per_share']:.2f}, implied return {sc['implied_return']:+.0%} "
                                                f"(growth {sc['growth']:.1%}, discount {sc['discount_rate']:.1%})", value=sc["implied_return"], source="BigBoss DCF on SEC data", as_of=val["as_of"]))
    r.data = {"quality_score": q.get("score"), "base_implied_return": val.get("scenarios", {}).get("base", {}).get("implied_return"),
              "bear_implied_return": val.get("scenarios", {}).get("bear", {}).get("implied_return")}
    return r


def quant_researcher(tb: Toolbox, symbol: str, research: dict | None) -> AgentReport:
    r = AgentReport(agent="quantitative_research")
    fc = (research or {}).get("forecasts") or {}
    if not fc:
        r.status = "degraded"
        r.warnings.append("No champion forecast models - run `bigboss train`.")
        return r
    for h, f in sorted(fc.items(), key=lambda x: int(x[0])):
        if "error" in f:
            r.warnings.append(f"h={h}: {f['error']}")
            continue
        r.findings.append(Finding(statement=f"{h}d excess-return forecast {f['mean']:+.2%} (80% interval {f['q10']:+.2%} to {f['q90']:+.2%}), "
                                            f"P(outperform) {f['p_pos']:.0%}; model {f['version']}, OOS IC t-stat {f['skill_t']:.2f}",
                                  value=f["mean"], source=f"model {f['version']}", as_of=research.get("last_date")))
        if f.get("top_drivers"):
            r.findings.append(Finding(statement=f"{h}d top LightGBM drivers: " + ", ".join(f"{d[0]} {d[1]:+.4f}" for d in f["top_drivers"]),
                                      source=f"model {f['version']} (TreeSHAP contributions)", as_of=research.get("last_date")))
    r.data = {"forecasts": fc}
    return r


def news_intelligence(tb: Toolbox, symbol: str, research: dict | None) -> AgentReport:
    r = AgentReport(agent="news_intelligence")
    ev = (research or {}).get("events") or {}
    rows = ev.get("events") or []
    if not rows:
        r.status = "degraded"
        r.warnings.append(f"No filing events for {symbol} (SEC events cover US registrants only).")
        return r
    fresh = [e for e in rows if not e.get("stale")]
    for e in fresh[:5]:
        eff = (f"; historical 5d excess after this event type {e['mean_excess']:+.2%} (n={int(e['n'])}, t={e['t_stat']:.1f})"
               if e.get("n") and e.get("reliable") else "; too few past events to estimate an effect")
        r.findings.append(Finding(statement=f"{e['form']} {e['event_type']} accepted {e['accepted_at']} importance {e['importance']:.2f}{eff}",
                                  value=e["importance"], source=e["url"], as_of=str(e["accepted_at"])))
    if not fresh:
        r.findings.append(Finding(statement=f"No filings in the last 30 days; latest: {rows[0]['form']} {rows[0]['accepted_at']}", source="SEC EDGAR", as_of=str(rows[0]["accepted_at"])))
    hi = [e for e in fresh if (e.get("importance") or 0) >= 0.7]
    r.data = {"high_importance_recent": [{"form": e["form"], "event_type": e["event_type"], "accepted_at": e["accepted_at"], "url": e["url"]} for e in hi]}
    return r


def risk_manager(tb: Toolbox, symbol: str, research: dict | None, decision: dict | None, regime: dict | None) -> AgentReport:
    r = AgentReport(agent="risk_management")
    if not decision:
        r.status = "degraded"
        r.warnings.append("No decision row - risk review limited.")
        return r
    th = decision.get("thesis") or {}
    risk = th.get("risk", {})
    r.findings.append(Finding(statement=f"Horizon sigma {risk.get('sigma_h', 0):.2%}; 10% expected shortfall {risk.get('es10_h', 0):.2%}",
                              value=risk.get("es10_h"), source="calibrated forecast distribution"))
    r.findings.append(Finding(statement=f"Liquidity OK at proposed size: {risk.get('liquidity_ok')}", value=risk.get("liquidity_ok"), source="cost model / 20d ADV"))
    vetoes = []
    if not risk.get("liquidity_ok", True):
        vetoes.append("liquidity: proposed size exceeds the participation limit")
    if regime and regime.get("liquidity_stress"):
        vetoes.append("market-wide liquidity stress")
    if decision.get("suggested_weight", 0) > tb.bb.cfg.get("portfolio.max_weight", 0.1) + 1e-9:
        vetoes.append("suggested weight above max_weight")
    r.data = {"vetoes": vetoes}
    return r


def evaluation_agent(tb: Toolbox, symbol: str | None) -> AgentReport:
    r = AgentReport(agent="evaluation")
    n = tb.call(r, "resolve_outcomes")
    perf = tb.call(r, "live_performance") or {}
    r.findings.append(Finding(statement=f"Resolved {n or 0} matured predictions; {perf.get('n_resolved', 0)} of {perf.get('n_logged', 0)} logged predictions have outcomes",
                              value=perf.get("n_resolved"), source="predictions_log"))
    if perf.get("rank_ic_mean") is not None:
        r.findings.append(Finding(statement=f"Live (post-deployment) rank IC {perf['rank_ic_mean']:+.3f}; interval coverage {perf['coverage80']:.0%}",
                                  value=perf["rank_ic_mean"], source="predictions_log"))
    r.data = perf
    return r


def investment_committee(symbol: str, decision: dict | None, fa: AgentReport, ni: AgentReport, rm: AgentReport, mk: AgentReport) -> CommitteeDecision:
    if not decision:
        return CommitteeDecision(symbol=symbol, quant_action=None, final_action="INSUFFICIENT_EVIDENCE",
                                 disagreements=["No quantitative decision available (missing model or data)."])
    qa = decision["action"]
    final = qa
    dis, vetoes = [], list(rm.data.get("vetoes", []))
    bear = fa.data.get("bear_implied_return")
    qs = fa.data.get("quality_score")
    if qa == "BUY" and bear is not None and bear < -0.4:
        dis.append(f"Fundamental analyst: bear-case DCF implies {bear:+.0%}; valuation risk conflicts with the quantitative BUY.")
    if qa == "BUY" and qs is not None and qs < 30:
        dis.append(f"Fundamental analyst: low quality score ({qs:.0f}/100).")
    if ni.data.get("high_importance_recent"):
        e = ni.data["high_importance_recent"][0]
        dis.append(f"News agent: recent high-importance filing ({e['form']} {e['event_type']} at {e['accepted_at']}) may not be reflected in price-based features.")
    reg = mk.data.get("regime", {})
    if qa == "BUY" and "bear_highvol" == reg.get("label"):
        dis.append("Market analyst: bear/high-volatility regime.")
    if vetoes and qa in ("BUY", "HOLD"):
        final = "AVOID" if qa == "BUY" else "REDUCE"
    elif qa == "BUY" and len(dis) >= 2:
        final = "INSUFFICIENT_EVIDENCE"
        dis.append("Committee: two or more independent objections - downgraded from BUY to INSUFFICIENT_EVIDENCE.")
    w = decision.get("suggested_weight") or 0.0
    if final != qa:
        w = 0.0 if final in ("AVOID", "INSUFFICIENT_EVIDENCE", "SELL") else w * 0.5
    return CommitteeDecision(symbol=symbol, quant_action=qa, final_action=final, suggested_weight=w, thesis=decision.get("thesis") or {},
                             disagreements=dis, vetoes=vetoes)


# ------------------------- orchestration -------------------------

TICKER = re.compile(r"\b[\^]?[A-Z][A-Z0-9]{0,9}(?:\.(?:NS|BO))?\b")


def parse_question(question: str, known: set[str]) -> tuple[list[str], set[str]]:
    syms = [t for t in TICKER.findall(question.upper()) if t in known]
    q = question.lower()
    intents = set()
    for word, intent in (("portfolio", "portfolio"), ("allocat", "portfolio"), ("rebalanc", "portfolio"), ("risk", "risk"),
                         ("news", "news"), ("filing", "news"), ("event", "news"), ("valuation", "fundamental"), ("fundamental", "fundamental"),
                         ("market", "market"), ("regime", "market"), ("opportunit", "scan"), ("scan", "scan"), ("best", "scan"),
                         ("track record", "evaluation"), ("performance", "evaluation")):
        if word in q:
            intents.add(intent)
    if syms:
        intents.add("symbol")
    if not intents:
        intents = {"market", "scan"}
    return list(dict.fromkeys(syms)), intents


def run_research(bb: BigBoss, question: str, symbols: list[str] | None = None, account_id: str | None = None,
                 profile: str = "balanced") -> dict:
    tb = Toolbox(bb)
    known = set(bb.store.symbols()["symbol"]) if not bb.store.symbols().empty else set()
    parsed, intents = parse_question(question, known)
    syms = list(dict.fromkeys((symbols or []) + parsed))[:5]
    if syms:
        intents.add("symbol")
    agents: list[AgentReport] = []
    decisions: list[CommitteeDecision] = []
    mk = market_analyst(tb)
    agents.append(mk)
    regime = mk.data.get("regime")
    for s in syms:
        holder = AgentReport(agent=f"data_collection[{s}]")
        res = tb.call(holder, "research", symbol=s)
        agents.append(holder)
        fa = fundamental_analyst(tb, s, res)
        qr = quant_researcher(tb, s, res)
        ni = news_intelligence(tb, s, res)
        dec = (res or {}).get("decision")
        rm = risk_manager(tb, s, res, dec, regime)
        for a in (qr, fa, ni, rm):
            a.agent = f"{a.agent}[{s}]"
            agents.append(a)
        decisions.append(investment_committee(s, dec, fa, ni, rm, mk))
    if "scan" in intents and not syms:
        a = AgentReport(agent="quantitative_research[scan]")
        sc = tb.call(a, "scan")
        if sc:
            buys = [c for c in sc["candidates"] if c["action"] == "BUY"][:10]
            a.findings.append(Finding(statement=f"Scanner ({sc['horizon']}d, model {sc['model_version']}, OOS IC t {sc['skill_t']:.2f}): {sc['counts']}",
                                      value=sc["counts"], source=f"model {sc['model_version']}", as_of=str(sc["as_of"])))
            for c in buys:
                a.findings.append(Finding(statement=f"{c['symbol']}: net excess {c['expected_net_excess']:+.2%}, P(net>0) {c['p_net_positive']:.0%}, size {c['suggested_weight']:.1%}",
                                          value=c["expected_net_excess"], source=f"model {sc['model_version']}", as_of=str(sc["as_of"])))
        agents.append(a)
    if "portfolio" in intents:
        a = AgentReport(agent="portfolio_optimization")
        kwargs = {"profile": profile}
        if account_id:
            kwargs["account_id"] = account_id
        else:
            kwargs["capital"] = float(bb.cfg.get("paper.initial_cash", 100000))
            a.warnings.append("No account given: proposal sized on the paper-trading default capital, not your finances.")
        pp = tb.call(a, "propose_portfolio", **kwargs)
        if pp:
            a.findings.append(Finding(statement=f"{pp['profile']} {pp['method']} proposal: {len(pp['weights'])} positions, cash {pp['cash_weight']:.0%}, "
                                                f"risk gate {'PASSED' if pp['risk_check']['passed'] else 'FAILED'}; est. cost {pp['total_est_cost']:.2f}",
                                      value=pp["weights"], source=f"model {pp['model_version']}"))
            a.data = {"weights": pp["weights"], "risk_check": pp["risk_check"], "trades": pp["trades"]}
        agents.append(a)
    if "evaluation" in intents:
        agents.append(evaluation_agent(tb, None))

    evidence = {"question": question, "data_mode": bb.mode,
                "findings": [{"agent": a.agent, "statement": f.statement} for a in agents for f in a.findings],
                "warnings": [w for a in agents for w in a.warnings],
                "decisions": [d.model_dump(include={"symbol", "quant_action", "final_action", "suggested_weight", "disagreements", "vetoes"}) for d in decisions]}
    fallback = _template_narrative(evidence)
    narrative = grounded_narrative(get_llm(bb.cfg), "Answer the question using the evidence; list key risks.", evidence, fallback)
    report = ResearchReport(
        report_id="RR-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4],
        created_at=datetime.now(timezone.utc).isoformat(), question=question, data_mode=bb.mode, symbols=syms,
        agents=agents, decisions=decisions, narrative=narrative,
        reproducibility={"bigboss_version": __version__, "champions": bb.registry.state()["champions"],
                         "config_sha256": hashlib.sha256(json.dumps(bb.cfg.raw, sort_keys=True, default=str).encode()).hexdigest()[:16],
                         "llm": narrative["generator"]})
    out = _clean(report.model_dump())
    d = bb.cfg.data_dir / "reports" / bb.mode.lower()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{report.report_id}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def _template_narrative(ev: dict) -> str:
    lines = [f"Question: {ev['question']}", f"Data mode: {ev['data_mode']}."]
    for d in ev["decisions"]:
        s = f"{d['symbol']}: committee decision {d['final_action']} (quant model: {d['quant_action']})"
        if d["disagreements"]:
            s += "; objections: " + " ".join(d["disagreements"])
        if d["vetoes"]:
            s += "; risk vetoes: " + ", ".join(d["vetoes"])
        lines.append(s)
    lines += [f"- [{f['agent']}] {f['statement']}" for f in ev["findings"][:25]]
    if ev["warnings"]:
        lines.append("Warnings: " + " | ".join(ev["warnings"][:8]))
    lines.append("All figures are model estimates computed by deterministic tools; they are not guarantees.")
    return "\n".join(lines)


def list_reports(bb: BigBoss, limit: int = 30) -> list[dict]:
    d = bb.cfg.data_dir / "reports" / bb.mode.lower()
    out = []
    for p in sorted(d.glob("RR-*.json"), reverse=True)[:limit]:
        r = json.loads(p.read_text())
        out.append({"report_id": r["report_id"], "created_at": r["created_at"], "question": r["question"], "symbols": r["symbols"]})
    return out
