"""End-to-end: data -> features -> forecasts -> costs -> decisions -> portfolio -> paper trading -> agents -> API."""

import numpy as np
import pandas as pd
import pytest

from bigboss.paper.broker import OrderRequest


def test_scan_produces_complete_recommendations(synth_service):
    s = synth_service.scan(20, log=True)
    assert s["data_mode"] == "SYNTHETIC"
    c = pd.DataFrame(s["candidates"])
    assert set(c["action"]) <= {"BUY", "HOLD", "REDUCE", "SELL", "AVOID", "INSUFFICIENT_EVIDENCE"}
    for col in ("expected_gross", "expected_excess", "cost_commission_bps", "cost_half_spread_bps", "cost_slippage_bps",
                "cost_impact_bps", "expected_net_excess", "sigma_h", "es10", "cost_liquidity_ok", "p_net_positive", "q10", "q90"):
        assert c[col].notna().all(), col
    assert (c["q10"] <= c["q90"]).all()
    assert "SYNTHETIC DATA" in c["thesis"].iloc[0]["disclaimer"]


def test_portfolio_proposal_respects_risk_limits(synth_service):
    cfg = synth_service.cfg
    for method in ("mean_variance", "robust", "black_litterman", "min_cvar"):
        p = synth_service.propose_portfolio("balanced", capital=100000, method=method, years=2, goal=120000)
        w = pd.Series(p["weights"])
        prof = cfg.section("profiles")["balanced"]
        single = w.drop("SYNMKT", errors="ignore")
        assert (single <= prof["max_weight"] + 1e-6).all(), method
        assert w.sum() <= 1 + 1e-6
        if p["risk_check"]["passed"]:
            assert p["risk_check"]["ex_ante_ann_vol"] <= prof["max_vol"] + 1e-6
        else:
            assert p["status"] == "REJECTED_BY_RISK_GATE"
        assert 0 <= p["projection"]["prob_reach_goal"] <= 1


def test_portfolio_requires_user_capital(synth_service):
    with pytest.raises(ValueError):
        synth_service.propose_portfolio("balanced")


def test_paper_broker_lifecycle(synth_service):
    b = synth_service.broker
    a = b.create_account("t", 100000)
    r = b.submit(a, OrderRequest("SYN001", "BUY", 50))
    assert r["status"] == "FILLED" and r["fill_price"] > r["ref_price"]  # adverse fill: pays spread/slippage
    s = b.summary(a)
    assert s["cash"] < 100000 - 50 * r["ref_price"]
    assert b.submit(a, OrderRequest("SYN001", "SELL", 500))["status"] == "REJECTED"  # no shorting
    assert b.submit(a, OrderRequest("SYN002", "BUY", 1e7))["status"] == "REJECTED"   # notional limit
    r2 = b.submit(a, OrderRequest("SYN001", "SELL", 50))
    assert r2["status"] == "FILLED"
    s2 = b.summary(a)
    assert s2["n_positions"] == 0 and s2["realized_pnl"] < 0 + 1e-6 or s2["realized_pnl"] == pytest.approx(s2["equity"] - 100000, abs=1e-6)
    assert np.isclose(s2["equity"], 100000 + s2["realized_pnl"], atol=1e-6)
    b.set_kill_switch(a, True)
    assert "kill switch" in b.submit(a, OrderRequest("SYN001", "BUY", 1))["reason"]
    actions = [x["action"] for x in b.audit_log(a)]
    assert "kill_switch_on" in actions and "order_filled" in actions and "order_rejected" in actions


def test_agent_research_report_is_auditable(synth_service):
    from bigboss.agents.workflow import run_research

    r = run_research(synth_service, "Should I buy SYN001? What are the risks? Suggest a portfolio.")
    assert r["symbols"] == ["SYN001"]
    names = [a["agent"] for a in r["agents"]]
    assert "market_analyst" in names and "portfolio_optimization" in names and any(n.startswith("risk_management") for n in names)
    d = r["decisions"][0]
    assert d["final_action"] in ("BUY", "HOLD", "REDUCE", "SELL", "AVOID", "INSUFFICIENT_EVIDENCE")
    for a in r["agents"]:
        for t in a["tool_calls"]:
            assert t["started_at"] and (t["ok"] is False or t["output_sha256"])
    assert r["narrative"]["generator"] == "deterministic_template"
    assert r["reproducibility"]["champions"]


def test_committee_never_upgrades(synth_service):
    from bigboss.agents.workflow import AgentReport, investment_committee

    empty = AgentReport(agent="x")
    for qa in ("AVOID", "INSUFFICIENT_EVIDENCE", "SELL"):
        out = investment_committee("S", {"action": qa, "suggested_weight": 0.0, "thesis": {}}, empty, empty, empty, empty)
        assert out.final_action == qa
    rm = AgentReport(agent="risk", data={"vetoes": ["liquidity"]})
    out = investment_committee("S", {"action": "BUY", "suggested_weight": 0.05, "thesis": {}}, empty, empty, rm, empty)
    assert out.final_action == "AVOID" and out.suggested_weight == 0


def test_prediction_logging_and_monitoring(synth_service):
    from bigboss.monitoring import live_performance

    mi = synth_service.model_intelligence()
    assert mi["champions"]["20"]
    assert mi["live_performance"]["n_logged"] > 0
    assert mi["drift"] and {"feature", "psi", "status", "kind"} <= set(mi["drift"][0])
    assert live_performance(synth_service.store)["n_logged"] > 0


def test_registry_gates_and_rollback(synth_service):
    reg = synth_service.registry
    h = 20
    first = reg.champion(h)
    vid = reg.train(synth_service.dataset(), h, ["hist_mean", "linear_ridge"], min_train=500, step=252)
    res = reg.challenge(vid)
    assert "beats_champion" in res["result"]["checks"]
    reg.promote(vid, force_reason="test")
    assert reg.champion(h) == vid
    assert reg.rollback(h)["to"] == first and reg.champion(h) == first


def test_api_end_to_end(synth_service):
    from fastapi.testclient import TestClient

    import bigboss.api.app as appmod
    appmod._services["SYNTHETIC"] = synth_service
    c = TestClient(appmod.app)
    assert c.get("/api/health").json()["live_trading_enabled"] is False
    assert c.get("/api/scanner?mode=synthetic").json()["data_mode"] == "SYNTHETIC"
    assert c.get("/api/market/overview?mode=synthetic").status_code == 200
    assert c.get("/api/research/SYN001?mode=synthetic").json()["symbol"] == "SYN001"
    assert c.get("/api/models?mode=synthetic").status_code == 200
    acc = c.post("/api/paper/accounts", json={"mode": "synthetic", "name": "api", "initial_cash": 50000}).json()["account_id"]
    o = c.post(f"/api/paper/accounts/{acc}/orders", json={"mode": "synthetic", "symbol": "SYN002", "side": "BUY", "quantity": 10}).json()
    assert o["status"] == "FILLED" and o["simulated"] is True
    assert c.post(f"/api/paper/accounts/{acc}/rebalance", json={"mode": "synthetic", "trades": []}).status_code == 400
    assert c.post("/api/portfolio/propose", json={"mode": "synthetic", "profile": "balanced"}).status_code == 400
    assert c.get("/api/research/NOPE?mode=synthetic").status_code == 409
    assert c.get("/api/backtests/..%2F..%2Fetc").status_code in (400, 404)
