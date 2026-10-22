"""FastAPI backend for the BigBoss dashboard.

Every response that contains market-derived numbers carries `data_mode` (OBSERVED | SYNTHETIC) so the UI can
label simulated data. Long tasks run as background jobs (/api/jobs/*).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bigboss import __version__
from bigboss.agents.llm import get_llm
from bigboss.agents.workflow import list_reports, run_research
from bigboss.backtest.experiment import list_experiments, load_experiment, run_experiment
from bigboss.config import REPO_ROOT, get_config
from bigboss.data.ingest import Ingestor
from bigboss.data.synthetic import load_into_store
from bigboss.jobs import JobManager
from bigboss.models.zoo import available_models
from bigboss.paper.broker import OrderRequest
from bigboss.service import BigBoss, NoDataError, NoModelError, _clean

cfg = get_config()
app = FastAPI(title="BigBoss API", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
jobs = JobManager(cfg.data_dir / "jobs.jsonl")
_services: dict[str, BigBoss] = {}

Mode = Literal["observed", "synthetic"]


def svc(mode: str) -> BigBoss:
    m = mode.upper()
    if m not in _services:
        _services[m] = BigBoss(cfg, m)
    return _services[m]


@app.exception_handler(NoDataError)
async def _nodata(_, exc):
    return JSONResponse(status_code=409, content={"error": "no_data", "detail": str(exc)})


@app.exception_handler(NoModelError)
async def _nomodel(_, exc):
    return JSONResponse(status_code=409, content={"error": "no_model", "detail": str(exc)})


@app.exception_handler(ValueError)
async def _valueerr(_, exc):
    return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})


# ---------------- system ----------------
@app.get("/api/health")
def health():
    llm = get_llm(cfg)
    return {"version": __version__, "modes": {m.lower(): svc(m).has_data() for m in ("OBSERVED", "SYNTHETIC")},
            "champions": {m.lower(): svc(m).registry.state()["champions"] for m in ("OBSERVED", "SYNTHETIC")},
            "llm": {"provider": cfg.get("llm.provider"), "available": llm.available()},
            "live_trading_enabled": bool(cfg.get("live_trading.enabled", False)), "models_available": available_models(),
            "horizons": cfg.get("forecast.horizons"), "primary_horizon": cfg.get("forecast.primary_horizon"),
            "profiles": list(cfg.section("profiles"))}


@app.get("/api/config")
def config_view():
    safe = {k: v for k, v in cfg.raw.items() if k not in ("paths",)}
    return _clean(safe)


# ---------------- data ----------------
@app.get("/api/data/status")
def data_status(mode: Mode = "observed"):
    return svc(mode).data_status()


class IngestReq(BaseModel):
    groups: list[str] | None = None
    symbols: list[str] | None = None
    prices: bool = True
    macro: bool = True
    sec: bool = True
    news: bool = False


@app.post("/api/jobs/ingest")
def job_ingest(req: IngestReq):
    bb = svc("observed")
    symbols = req.symbols or cfg.universe(req.groups)

    def work(progress):
        ing = Ingestor(cfg, bb.store)
        out = {}
        if req.prices:
            progress(0.05, f"prices for {len(symbols)} symbols")
            out["prices"] = ing.ingest_prices(symbols)
        if req.macro:
            progress(0.6, "macro series")
            out["macro"] = ing.ingest_macro()
        if req.sec:
            progress(0.75, "SEC fundamentals and filings")
            out["sec"] = ing.ingest_sec(symbols)
        if req.news:
            progress(0.9, "news headlines")
            out["news"] = ing.ingest_news(symbols)
        if bb.has_data():
            bb.dataset(refresh=True)
        return out
    return {"job_id": jobs.submit("ingest", work, req.model_dump())}


class DemoReq(BaseModel):
    n_assets: int = Field(30, ge=5, le=200)
    n_days: int = Field(2520, ge=600, le=7560)
    seed: int = 7
    alpha_scale: float = 2.0


@app.post("/api/jobs/demo-data")
def job_demo(req: DemoReq):
    bb = svc("synthetic")

    def work(progress):
        load_into_store(bb.store, **req.model_dump())
        bb.dataset(refresh=True)
        return {"loaded": True, "note": "SYNTHETIC data - for testing the platform only"}
    return {"job_id": jobs.submit("demo-data", work, req.model_dump())}


class TrainReq(BaseModel):
    mode: Mode = "observed"
    horizon: int = 20
    models: list[str] | None = None
    min_train: int = 756


@app.post("/api/jobs/train")
def job_train(req: TrainReq):
    bb = svc(req.mode)

    def work(progress):
        vid = bb.registry.train(bb.dataset(), req.horizon, req.models or available_models(), min_train=req.min_train, progress=progress)
        return {"version": vid, "gates": bb.registry.meta(vid)["gates"], "champion": bb.registry.champion(req.horizon)}
    return {"job_id": jobs.submit("train", work, req.model_dump())}


class ExperimentReq(BaseModel):
    mode: Mode = "observed"
    horizon: int = 20
    models: list[str] | None = None
    ablation: bool = False
    include_holdout: bool = False
    cost_sensitivity: bool = True


@app.post("/api/jobs/experiment")
def job_experiment(req: ExperimentReq):
    bb = svc(req.mode)

    def work(progress):
        s = run_experiment(bb.store, bb.cfg, req.horizon, req.models, include_holdout=req.include_holdout, ablation=req.ablation,
                           cost_sensitivity=req.cost_sensitivity, progress=progress, ds=bb.dataset())
        return {"id": s["id"]}
    return {"job_id": jobs.submit("experiment", work, req.model_dump())}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return _clean({k: v for k, v in j.items()})


# ---------------- research ----------------
@app.get("/api/market/overview")
def market_overview(mode: Mode = "observed"):
    return svc(mode).market_overview()


@app.get("/api/scanner")
def scanner(mode: Mode = "observed", horizon: int | None = None, account_id: str | None = None, log: bool = False):
    return svc(mode).scan(horizon, account_id, log=log)


@app.get("/api/research/{symbol}")
def research(symbol: str, mode: Mode = "observed"):
    return svc(mode).research(symbol)


@app.get("/api/events")
def events(mode: Mode = "observed", symbol: str | None = None, limit: int = 100):
    return svc(mode).events(symbol, limit)


@app.get("/api/news")
def news(mode: Mode = "observed", symbol: str | None = None, limit: int = 60):
    return svc(mode).news(symbol, limit)


@app.get("/api/symbols")
def symbols(mode: Mode = "observed"):
    return _clean(svc(mode).store.symbols())


class PortfolioReq(BaseModel):
    mode: Mode = "observed"
    profile: str = "balanced"
    capital: float | None = None
    holdings: dict[str, float] | None = None
    method: str = "mean_variance"
    horizon: int | None = None
    account_id: str | None = None
    goal: float | None = None
    years: float | None = None
    monthly_contribution: float = 0.0


@app.post("/api/portfolio/propose")
def propose(req: PortfolioReq):
    d = req.model_dump()
    return svc(d.pop("mode")).propose_portfolio(**d)


# ---------------- backtests & models ----------------
@app.get("/api/backtests")
def backtests(mode: Mode = "observed"):
    return [e for e in list_experiments(cfg) if e["data_mode"] == mode.upper()]


@app.get("/api/backtests/{exp_id}")
def backtest(exp_id: str):
    if "/" in exp_id or ".." in exp_id:
        raise HTTPException(400, "bad id")
    try:
        return load_experiment(cfg, exp_id)
    except FileNotFoundError:
        raise HTTPException(404, "experiment not found")


@app.get("/api/models")
def models(mode: Mode = "observed"):
    return svc(mode).model_intelligence()


class PromoteReq(BaseModel):
    mode: Mode = "observed"
    force_reason: str | None = None


@app.post("/api/models/{vid}/promote")
def promote(vid: str, req: PromoteReq):
    return _clean(svc(req.mode).registry.promote(vid, req.force_reason))


@app.get("/api/models/{vid}/challenge")
def challenge(vid: str, mode: Mode = "observed"):
    return _clean(svc(mode).registry.challenge(vid))


class RollbackReq(BaseModel):
    mode: Mode = "observed"
    horizon: int = 20


@app.post("/api/models/rollback")
def rollback(req: RollbackReq):
    return svc(req.mode).registry.rollback(req.horizon)


# ---------------- agents ----------------
class ResearchReq(BaseModel):
    mode: Mode = "observed"
    question: str = Field(..., min_length=3, max_length=2000)
    symbols: list[str] | None = None
    account_id: str | None = None
    profile: str = "balanced"


@app.post("/api/agents/research")
def agent_research(req: ResearchReq):
    return run_research(svc(req.mode), req.question, req.symbols, req.account_id, req.profile)


@app.get("/api/agents/reports")
def agent_reports(mode: Mode = "observed"):
    return list_reports(svc(mode))


@app.get("/api/agents/reports/{rid}")
def agent_report(rid: str, mode: Mode = "observed"):
    if "/" in rid or ".." in rid:
        raise HTTPException(400, "bad id")
    p = cfg.data_dir / "reports" / mode / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "report not found")
    import json
    return json.loads(p.read_text())


# ---------------- paper trading ----------------
@app.get("/api/paper/accounts")
def paper_accounts(mode: Mode = "observed"):
    return svc(mode).broker.accounts()


class AccountReq(BaseModel):
    mode: Mode = "observed"
    name: str = Field(..., min_length=1, max_length=80)
    initial_cash: float = Field(100000, gt=0, le=1e9)
    profile: str = "balanced"


@app.post("/api/paper/accounts")
def paper_create(req: AccountReq):
    return {"account_id": svc(req.mode).broker.create_account(req.name, req.initial_cash, req.profile)}


@app.get("/api/paper/accounts/{aid}")
def paper_account(aid: str, mode: Mode = "observed"):
    b = svc(mode).broker
    try:
        summ = b.summary(aid)
    except KeyError:
        raise HTTPException(404, "account not found")
    return _clean({"summary": summ, "positions": b.positions(aid), "orders": b.orders(aid), "fills": b.fills(aid),
                   "equity_history": b.equity_history(aid), "attribution": b.attribution(aid), "simulated": True})


class OrderReq(BaseModel):
    mode: Mode = "observed"
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float = Field(..., gt=0)
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    strategy: str = "manual"
    recommendation_id: str | None = None


@app.post("/api/paper/accounts/{aid}/orders")
def paper_order(aid: str, req: OrderReq):
    d = req.model_dump()
    mode = d.pop("mode")
    return _clean(svc(mode).broker.submit(aid, OrderRequest(**d)))


class KillReq(BaseModel):
    mode: Mode = "observed"
    on: bool


@app.post("/api/paper/accounts/{aid}/kill-switch")
def paper_kill(aid: str, req: KillReq):
    svc(req.mode).broker.set_kill_switch(aid, req.on)
    return {"account_id": aid, "kill_switch": req.on}


@app.get("/api/paper/accounts/{aid}/audit")
def paper_audit(aid: str, mode: Mode = "observed"):
    return svc(mode).broker.audit_log(aid)


class RebalanceReq(BaseModel):
    mode: Mode = "observed"
    trades: list[dict]
    confirm: bool = False


@app.post("/api/paper/accounts/{aid}/rebalance")
def paper_rebalance(aid: str, req: RebalanceReq):
    """Execute a proposal's trades in the PAPER account. Requires confirm=true (user clicked confirm)."""
    if not req.confirm:
        raise HTTPException(400, "confirm must be true - the user must explicitly approve these simulated trades")
    bb = svc(req.mode)
    results = []
    for t in sorted(req.trades, key=lambda t: t["side"] != "SELL"):  # sells first to free cash
        px = bb.store.prices([t["symbol"]])
        if px.empty:
            results.append({"symbol": t["symbol"], "status": "REJECTED", "reason": "no price"})
            continue
        qty = round(float(t["notional"]) / float(px["close"].iloc[-1]), 4)
        if qty <= 0:
            continue
        results.append(bb.broker.submit(aid, OrderRequest(t["symbol"], t["side"], qty, strategy=t.get("strategy", "bigboss_rebalance"))))
    return _clean({"results": results, "simulated": True})


# ---------------- static frontend ----------------
DIST = Path(os.environ.get("BIGBOSS_FRONTEND_DIST", REPO_ROOT / "frontend" / "dist"))
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        root = DIST.resolve()
        f = (root / path).resolve()
        if path and f.is_file() and f.is_relative_to(root):
            return FileResponse(f)
        return FileResponse(root / "index.html")
