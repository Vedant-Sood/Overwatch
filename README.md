# BigBoss: Local-First AI Investment Research Platform

BigBoss is a research and paper-trading platform. It forecasts **probability distributions of benchmark-relative
returns**, prices in **realistic trading costs**, turns forecasts into **economically justified decisions**
(BUY / HOLD / REDUCE / SELL / AVOID / INSUFFICIENT EVIDENCE), builds **risk-constrained portfolios**, and tests
everything with **walk-forward backtests**. It also reports whether any of this beats simple baselines.

> **What it is not.** It does not guarantee profits, does not trade real money (no live broker adapter ships, and
> live trading is disabled by default), and an impressive backtest is never treated as proof of future returns.
> Nothing here is investment advice.

---

## Contents
1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [The objective and how decisions are made](#the-objective-and-how-decisions-are-made)
4. [Data sources, licensing and point-in-time integrity](#data-sources-licensing-and-point-in-time-integrity)
5. [Models](#models)
6. [Backtesting methodology](#backtesting-methodology)
7. [Portfolio construction & planning](#portfolio-construction--planning)
8. [Research agents & LLM safety](#research-agents--llm-safety)
9. [Paper trading & live-trading safeguards](#paper-trading--live-trading-safeguards)
10. [Continuous improvement: monitoring, gates, rollback](#continuous-improvement)
11. [Dashboard](#dashboard)
12. [CLI & API reference](#cli--api-reference)
13. [Testing](#testing)
14. [Results, negative results and known limitations](#results-negative-results-and-known-limitations)

---

## Quick start

Requirements: Python 3.11+ (developed on 3.13), Node.js 20+ for the dashboard. Built for Windows 11 / WSL2 Ubuntu,
an i7 CPU, 32 GB RAM, and an RTX 4060 (8 GB). The GPU is optional: XGBoost and the PyTorch MLP use CUDA when
available, and everything also runs on CPU.

```bash
# Ubuntu / WSL2
bash scripts/setup.sh            # venv + pip install -e ".[dev,xgb,tune]" + dashboard build
source .venv/bin/activate
# Windows PowerShell:  .\scripts\setup.ps1

# Optional: GPU PyTorch for the MLP model
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Set a contact e-mail for SEC EDGAR (required by SEC fair-access policy)
cp configs/bigboss.toml configs/local.toml   # then edit [data].sec_user_agent

bigboss ingest --groups cross_asset --news     # keyless real data: FRED cross-asset proxies + macro + GDELT headlines
bigboss ingest --groups us_etfs us_equities   # stock-level data: FMP_API_KEY (dividend-adjusted, long history) or Massive
bigboss ingest --no-prices --massive-market 500 --news   # whole US market (Massive) -> point-in-time top-500 universe incl. delisted
bigboss backtest --horizon 20 --min-train 200 --step 21 --ablation   # settings for a ~2-year history
bigboss status                                # coverage, freshness, validation issues
bigboss train --horizon 20                    # walk-forward training, registers a model version, applies promotion gates
bigboss backtest --horizon 20 --ablation      # baselines vs models, net of costs; cost sensitivity; ablation
bigboss scan --horizon 20                     # ranked recommendations from the champion model
bigboss research "Is MSFT attractive over the next month? What are the risks?"
bigboss export-qc --experiment ID --out qc/   # replay an experiment in QuantConnect LEAN (docs/QUANTCONNECT.md)
bigboss qc-compare --experiment ID --qc-results backtest.json   # compare the two engines
bigboss serve                                 # API + dashboard at http://127.0.0.1:8000
```

**Offline test mode.** `bigboss demo-data` generates a labelled SYNTHETIC market (see
`bigboss/data/synthetic.py`) in a separate database. Add `--synthetic` to `train`, `backtest`, `scan` and `research`, or pick
"Synthetic" in the dashboard. Every synthetic screen carries a striped **SYNTHETIC DATA MODE** banner. Synthetic data
is never written to, or mixed with, the observed-data store.

Frontend development: `cd frontend && npm install && npm run dev` (Vite on :5173 proxies `/api` to :8000).

Optional keys (environment variables, never stored in config): `MASSIVE_API_KEY`, `FMP_API_KEY`, `FINNHUB_API_KEY`,
`ALPHAVANTAGE_API_KEY`, `FRED_API_KEY`. All have free tiers.
Optional local LLM: install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, and set `[llm] provider = "ollama"`.

---

## Architecture

```
providers (Stooq · Yahoo · Alpha Vantage · FRED · World Bank · SEC EDGAR)
   │  rate limit · retries · raw-response cache · source attribution · fetched_at
   ▼
validation (duplicates, OHLC consistency, unadjusted splits, gaps, staleness)  ──►  DuckDB store (OBSERVED | SYNTHETIC)
   ▼
features (technical, liquidity, risk, cross-sectional ranks, market state, macro by available_at,
          PIT fundamentals by filing date, filing events by acceptance time)   +   forward excess labels (with execution delay)
   ▼
walk-forward (purge + embargo + locked holdout) ──► Model A family ──► conformal calibration ──► Model E stacking ensemble (OOS-only)
   ▼                                                                                        ▲  Model C regimes · Model B valuation · Model D events
decision engine (12 quantities, costs, abstention)  ──► portfolio optimizers ──► independent risk gate
   ▼                                                        ▼
backtester (delay, costs, liquidity, delisting)       paper broker (audit, limits, kill switch)  ·  research agents  ·  FastAPI  ·  React dashboard
   ▼
registry (versions, gates, champion/challenger, rollback) · prediction log · outcome resolution · drift (PSI)
```

| Package | Responsibility |
|---|---|
| `bigboss/data` | provider adapters, DuckDB store, validation, synthetic generator, ingestion |
| `bigboss/features` | PIT features and labels; `dataset.py` assembles the modelling frame |
| `bigboss/models` | `zoo` (A), `valuation` (B), `regime` (C), `events` (D), `ensemble` (E), `distribution` (calibration), `walkforward`, `registry` |
| `bigboss/costs.py` | ex-ante trading-cost model |
| `bigboss/decision.py` | economic decision engine and investment theses |
| `bigboss/portfolio` | optimizers (MV, robust, Black–Litterman, CVaR, risk parity, HRP) and Monte Carlo planning |
| `bigboss/risk.py` | covariance, VaR/ES, stress tests, the deterministic risk gate |
| `bigboss/backtest` | splitter, simulator, strategies, metrics, experiment runner |
| `bigboss/agents` | typed agents, tool registry, LLM clients, injection/grounding guards |
| `bigboss/paper` | paper broker and the guarded live-order gateway |
| `bigboss/monitoring.py` | prediction/outcome logging, live performance, drift |
| `bigboss/api` | FastAPI app (also serves the built dashboard) |
| `frontend/` | React + TypeScript + Plotly dashboard |

---

## The objective and how decisions are made

For a horizon *h*, BigBoss targets

```
maximize  E[R_p − R_b − C] − λ · Risk − constraint penalties
```

* **Labels** are benchmark-relative: `R = P[t+d+h]/P[t+d] − 1` minus the benchmark's return over the same window,
  where *d* is the execution delay (decision at close *t*, fill at close *t+d*, default *d* = 1).
* **Costs** (`costs.py`): commission + half spread + slippage + square-root market impact
  (`k · σ_daily · √(order/ADV)`) + taxes (e.g. an approximate India STT). Unknown liquidity is flagged and never
  treated as infinite.
* **Risk** is *active* risk, `(w − w_b)ᵀ Σ_h (w − w_b)`, consistent with the benchmark-relative objective. Owning
  the benchmark fund is always a candidate with zero expected excess, so an active position must pay for its own
  costs and tracking risk. CVaR and drawdown-limited profiles are available as alternatives. The choice of
  objective is compared empirically in the Strategy Lab, not assumed.
* **Annualization**: horizon quantities are kept in *h*-day units (Σ_h = Σ_daily·h); reported metrics are annualized
  with 252 trading days.

For every candidate the decision engine computes: (1) expected gross return, (2) expected excess, (3–6) commission,
spread, slippage and impact, (7) expected net excess, (8) σ, ES10 and expected downside, (9) liquidity/participation,
(10) marginal tracking error, (11) P(net excess > 0), and (12) a calibrated 80% interval plus cross-model
disagreement. Rules:

* **INSUFFICIENT EVIDENCE** when base models disagree (σ of forecasts > threshold, or < 60% agree on sign), when the
  ensemble's out-of-sample rank-IC t-stat is below the gate, or when features are mostly missing. Wide outcome
  intervals alone do not trigger abstention, because that noise is priced into the decision.
* **BUY** only if expected net excess ≥ threshold, calibrated P(net > 0) ≥ threshold, and liquidity is OK.
* Held positions: **SELL** if the edge is negative beyond exit costs, **REDUCE** if it is negative or oversized, otherwise **HOLD**.
* **AVOID** otherwise. Doing nothing is a first-class outcome.

Each recommendation carries a quantitative thesis, supporting evidence (model forecasts, TreeSHAP drivers, OOS skill),
counterarguments, a risk block, a suggested size (fractional Kelly on net edge, capped by weight and liquidity),
and explicit invalidation conditions.

---

## Data sources, licensing and point-in-time integrity

Recommendations and verification results per source: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

| Source | Key? | Used for | Notes |
|---|---|---|---|
| yfinance (Yahoo Finance) | no | **default daily prices**: long history, dividend-adjusted, ETFs, indices, NSE/BSE | unofficial; Yahoo terms = personal use; no delisted tickers; Indian corporate-action errors are caught by validation |
| FRED (`fred.stlouisfed.org`) | no (CSV) / `FRED_API_KEY` (API) | macro: rates, VIX, spreads, oil, FX, CPI, unemployment; **cross-asset proxies** (`XA:*`) | some series carry third-party copyright (ICE, S&P, Nikkei); ICE indices on FRED now cover only ~3 years |
| Treasury Fiscal Data | no | average interest rates on marketable Treasury debt (monthly) | US public domain |
| World Bank | no | annual CPI inflation and GDP growth per country | CC BY 4.0 |
| SEC EDGAR | no (contact UA) | XBRL fundamentals and filing events | public domain; ≤10 req/s |
| GDELT DOC 2.0 | no | recent news headlines (research/agents only, not model input) | ~1 request per 5 s; free/open, cite GDELT |
| Massive (formerly Polygon.io) | `MASSIVE_API_KEY` | daily OHLCV for US stocks/ETFs | free tier ≈ 2 years history, 5 calls/min; split-adjusted only |
| Financial Modeling Prep | `FMP_API` or `FMP_API_KEY` | daily OHLCV (US, many international) | free tier 250 calls/day; split-adjusted |
| Finnhub | `FINNHUB_API_KEY` | company news (preferred over GDELT for US tickers when set) | free tier for personal use |
| Alpha Vantage | `ALPHAVANTAGE_API_KEY` | fallback daily prices | free `TIME_SERIES_DAILY` is **unadjusted**; flagged |
| Stooq / Yahoo chart | no | daily OHLCV incl. NSE/BSE (`.NS`/`.BO`) via Yahoo | used only where reachable; Yahoo terms restrict to personal use |

Not integrated (documented decision): **BLS** (needs a key from shared IPs; FRED mirrors the BLS series used),
**RBI DBIE** (JavaScript-only portal, no documented data API) and **SEBI** (no data API). Scraping them is out of scope.

**Cross-asset proxies (`XA:*`, keyless).** `bigboss ingest --groups cross_asset` builds nine return series from FRED
(Nasdaq-100, Nikkei 225 in USD, Brent, 10y and 2y Treasury total return re-priced from constant-maturity yields, and
EUR/GBP/JPY/INR vs USD) plus `XA:EW`, an equal-weight daily-rebalanced benchmark basket. They are **proxies**: equity
indices exclude dividends, Brent is spot (no futures roll), FX excludes interest carry, and there is no volume (they are
costed as liquid ETF/futures exposures). Each limitation is stored in the `adjustment` column and shown in the dashboard.

* **Licensing.** Raw data and caches live in `data/` (git-ignored) and are never redistributed by this repo. Public API
  access does not automatically grant machine-learning or redistribution rights. Check each provider's terms for your use.
* **No silent substitution.** If every provider fails for a symbol, it is logged `FAILED` and excluded.
  Synthetic data exists only in its own store and is always labelled.
* **Point-in-time rules.** Macro values are usable from `available_at` (observation date + a conservative publication
  lag). XBRL facts are usable the day *after* their `filed` date, and for each period the **first-filed** value is kept,
  so restatements cannot leak backwards. Filings accepted after 16:00 ET are usable the next session. A truncation
  test (`assert_no_lookahead`) proves that features at *t* do not change when future rows are appended.
* **Freshness.** `bigboss status`, `/api/data/status` and the Data page show last date, age, and STALE flags per symbol.

---

## Models

| Model | What | Implementation |
|---|---|---|
| **A** Net excess return | mean + q10/q50/q90 of *h*-day excess return at 1/5/20/60/252 days | baselines: zero-excess (efficient market), shrunk historical mean, transparent 12-1 momentum rule, Ridge linear factor model; ML: LightGBM (Huber mean + 3 quantile models), XGBoost (pseudo-Huber + native multi-quantile, CUDA if available), small PyTorch MLP (pinball + Huber, CUDA if available) |
| Calibration | 80% intervals and P(excess > 0) | conformalized quantile regression on a purged calibration slice; isotonic probability calibration; split-normal for P(R > cost), ES and expected downside |
| **B** Valuation | PIT quality/value profile + bear/base/bull DCF | TTM from discrete quarters (YTD differencing), growth anchored 50/50 to a long-run rate, transparent 0–100 quality score |
| **C** Regimes | bull/bear × vol, Markov switching, change points, correlation & liquidity stress, momentum vs reversal | statsmodels `MarkovRegression` (**filtered** probabilities only), CUSUM on non-overlapping 20-day log vol |
| **D** Events | SEC filing events, importance, credibility, staleness, estimated effect | 8-K item taxonomy; PIT event study of 5-day excess returns per event type; optional LLM summaries with injection guards |
| **E** Ensemble | non-negative stacking of base models | weights fit by NNLS **only on prior folds' out-of-sample rows whose labels were already known**, shrunk 30% to equal weight; second-stage conformal + isotonic calibration; abstention |

Models were kept deliberately small (LightGBM with 15 leaves, an MLP of 64 units). On daily cross-sectional return
data, larger models mostly memorize noise. Hyperparameter search (Optuna, purged K-fold in `backtest/splits.py`) is
available but **not used by default**, because every extra configuration tried raises the multiple-testing bar
(see Deflated Sharpe).

---

## Backtesting methodology

* **Walk-forward** with expanding (or rolling, `train_window`) training windows retrained every `retrain_every` days.
* **Purging**: a training row is dropped unless its label window ends before the test block starts (minus an
  **embargo**). The calibration slice is itself purged from the fitting slice. Training dates are subsampled
  every min(h, 5) days to reduce label overlap.
* **Locked final holdout**: the last 15% of dates are excluded from every split and experiment. Unlocking requires
  `bigboss backtest --final-holdout`, a typed confirmation, and every access is appended to `data/holdout_access.log`.
* **Simulator** (`backtest/engine.py`): decisions at close *t*, fills at close *t+delay*, per-trade costs from the cost
  model, trades capped at `max_participation` × 20-day ADV (computed only from data before the fill day), **no implicit
  leverage** (buys and costs must be funded by cash plus sale proceeds), forced liquidation with costs when an asset's
  history ends (delisting), and cash earning 0% (conservative).
* **Strategies compared**: benchmark buy-and-hold, equal weight, 12-1 momentum top-k, low-volatility top-k, each
  model on its own, the ungated ensemble, and the **gated ensemble** (abstains without demonstrated skill).
  Model strategies use cost-aware active-risk mean-variance with the benchmark fund as the default holding.
* **Metrics**: annualized return and excess return, volatility, Sharpe, Sortino, max drawdown, Calmar, tracking error,
  information ratio with a block-bootstrap 90% CI, alpha/beta, daily ES5, profit factor and hit rate of daily excess,
  turnover, cost drag, Probabilistic Sharpe and **Deflated Sharpe** (adjusted for the number of strategies tried).
  Forecasts are scored on rank IC with an overlap-corrected t-stat, interval coverage, Brier vs climatology, and a
  calibration table. Results are also broken down by regime (ex-ante labels), by cost multiplier (0×–3×), and by
  ablation (LightGBM with each feature group removed).
* **Survivorship bias**: the default universe is today's names. Stooq/Yahoo do not provide delisted symbols, so
  backtests on it are biased upward. Every experiment states this. Delisted names you have data for can be added to the
  universe, and the simulator handles their exit.

---

## Portfolio construction & planning

Methods: cost-aware mean-variance (active or total risk), **robust** MV (μ shrunk by k·SE toward zero), **Black–Litterman**
(model forecasts as views, with Ω from calibrated interval width), **min-CVaR** (Rockafellar–Uryasev LP), **risk parity**
(Spinu convex program) and **HRP** (López de Prado).

Profiles (`[profiles.*]`): conservative, balanced, aggressive, experimental (simulated funds only). Each sets risk aversion,
max weight and max volatility. The **independent risk gate** (`risk.py`) checks single-name caps, sector caps, gross
exposure, and ex-ante volatility. If volatility is too high, the proposal is de-risked into cash and the change is
disclosed. Other violations mark the proposal `REJECTED_BY_RISK_GATE`. No LLM output can change limits.

Planning: Monte Carlo goal projections by block-bootstrapping the proposal's historical daily returns (5,000 paths),
reporting percentile fans, P(goal), P(loss), and drawdown percentiles. **Capital, goals, contributions and holdings
must be supplied by the user**, and the API refuses to invent them.

---

## Research agents & LLM safety

Agents (`agents/workflow.py`): Market Analyst, Fundamental Analyst, Quantitative Research, News Intelligence, Risk
Management, Portfolio Optimization, **Investment Committee**, and Evaluation. Each is a deterministic program that calls
typed tools (the same service functions the dashboard uses). Every tool call is logged with arguments, timing and a
SHA-256 of its output. Findings carry sources and timestamps, and each report is saved to `data/reports/` with a
reproducibility block (code version, config hash, champion model versions).

* The **committee can only make decisions more conservative**: risk vetoes or two or more independent objections
  downgrade a BUY. It never upgrades, and all disagreements are written into the report.
* The optional **local LLM** (Ollama or llama.cpp) only writes the narrative from a JSON evidence bundle. Any number in
  its output that is not in the evidence causes the draft to be discarded in favour of a deterministic template.
* Untrusted text (filings, news) is stripped of markup, truncated, fenced as data, and scanned for prompt-injection
  patterns (`agents/safety.py`).

---

## Paper trading & live-trading safeguards

`paper/broker.py`: virtual accounts, market/limit orders filled at the last stored close plus estimated spread,
slippage, impact and commission (always adverse), positions, realized/unrealized P&L, equity history, strategy
attribution, and a full audit log. Risk checks block shorting, oversize orders, position caps (diversified
benchmark funds exempt), insufficient cash, and new orders after a daily-loss breach. A **kill switch** cancels pending
orders and blocks new ones. Applying a portfolio proposal requires an explicit `confirm=true` (a confirmation dialog in the UI).

`paper/live.py`: **no live broker adapter ships.** The gateway any future adapter must use requires
`live_trading.enabled = true`, the absence of a global `data/KILL_SWITCH` file, and a single-use, 5-minute HMAC
approval token bound to the exact order. The token is issued only after the user types `APPROVE BUY 10 AAPL`.
Credentials may only come from environment variables.

---

## Continuous improvement

* `bigboss update` (daily): ingest → rebuild features → champion forecasts for every horizon → **log predictions** →
  **resolve matured outcomes**. Schedule it with `bigboss schedule --every-hours 24`, cron, or Windows Task Scheduler.
* **Registry** (`data/models/<mode>/`): each version stores fitted models, calibrators, ensemble weights, its full OOS
  record, a training-feature reference (deciles), and metadata (data fingerprint, folds, metrics).
* **Promotion gates** (predetermined, on the *full* OOS record, never a recent streak): ensemble rank-IC t ≥ 1.0,
  80% interval coverage within [0.72, 0.88], Brier ≤ climatology + 0.005, ≥ 24 independent evaluation periods,
  **an economic test: the gated ensemble's walk-forward backtest must have net-of-cost IR > 0 and ≥ 90% probability
  (PSR) that its true net edge is positive**, and, for a challenger, a positive paired IC difference against the champion
  on common dates. Forecast skill alone (a significant IC) is not enough; on real data it was not. Gate changes can be
  applied retroactively with `bigboss models reevaluate`, which demotes failing champions and logs why. Without a
  champion, the scanner shows the latest model's forecasts as **research only**, with every action INSUFFICIENT EVIDENCE. Forced promotions
  require a written reason and are recorded. `bigboss models rollback --horizon 20` restores the previous champion.
* **Drift**: PSI of each feature over the last 126 days against training deciles. Market-state features are labelled
  separately, since their drift usually means a regime change rather than a broken pipeline.
* **Live performance**: rank IC, interval coverage, Brier and realized excess by action, computed on logged predictions.

---

## Dashboard

`bigboss serve`, then open http://127.0.0.1:8000. Modules: Market Intelligence, AI Opportunity Scanner, Stock Research,
Portfolio Command Center, Strategy Laboratory, AI Research Console, Paper Trading Terminal, Model Intelligence, and Data & Setup.
Dark and light themes are supported. Every panel is badged **● OBSERVED**, **◇ MODEL PREDICTION** (dashed), or
**▲ SIMULATED** (backtests, projections, paper results, synthetic data). Empty states explain what to run and never
show placeholder numbers.

---

## CLI & API reference

CLI: `bigboss {ingest, status, demo-data, train, backtest, scan, research, models, update, schedule, serve} --help`.

Main API routes (all accept `mode=observed|synthetic`; interactive docs at `/docs`):
`GET /api/health`, `/api/data/status`, `/api/market/overview`, `/api/scanner`, `/api/research/{symbol}`, `/api/events`,
`/api/backtests[/{id}]`, `/api/models`; `POST /api/jobs/{ingest,demo-data,train,experiment}`, `GET /api/jobs/{id}`;
`POST /api/portfolio/propose`; `POST /api/agents/research`; `/api/paper/accounts[/{id}[/orders|/kill-switch|/rebalance|/audit]]`;
`POST /api/models/{vid}/promote`, `POST /api/models/rollback`.

---

## Testing

```bash
pytest -m "not network"     # 74 tests: leakage, labels, splits, calibration, costs, metrics, optimizers,
                            # simulator accounting, SEC parsing, decisions, safety, paper broker, agents, registry, API
pytest -m network           # live provider checks (skip cleanly when a provider is unreachable)
cd frontend && npm run typecheck && npm run build
```

Notable tests: features are invariant to appending future data, and the detector catches a deliberately leaky feature.
Labels are checked by hand computation. Purged folds never overlap the test period, and the holdout stays locked.
Conformal calibration reaches its target coverage. The ensemble finds a planted signal but reports **no** skill on
a zero-signal market. The simulator charges entry costs without leverage, respects execution delay and liquidity caps,
and liquidates delisted assets. The XBRL parser ignores restatements. The committee never upgrades a decision.
Live trading is refused without config, a matching single-use token, and an inactive kill switch.

---

## Results, negative results and known limitations

See [`docs/RESULTS.md`](docs/RESULTS.md) for the experiments run during development, including the failures.

Known limitations (honest list):

* **Real-market evidence (see `docs/RESULTS.md`).** The strongest test - a survivorship-free US top-500 from DoltHub,
  2020-2025 out of sample, 1,330 stocks incl. 302 that stopped trading - found **no forecast skill** (every model's IC
  negative); ungated models trailed SPY by 3-13%/yr and the gated system abstained and matched SPY (−0.3%/yr).
  A point-in-time top-100 ETF universe (2006-2019, 13 years OOS) showed skill only before 2009 (gated +0.2%/yr).
  Models trained on both datasets failed the promotion gates; no champion exists, so forecasts are research-only.
  On the **historical S&P 500** (actual membership, 2015-2024) no model was reliable (gated −2.4%/yr; the one positive
  model was an era bet). On the **historical NIFTY 50** (actual constituents incl. delisted, 2011-2024) the models
  showed significant ranking skill (ensemble IC t 2.3) but it barely survived Indian costs (gated +1.3%/yr).
  Across five other real datasets, no model showed economic value after
  costs that survives proper benchmarking and multiple-testing correction: FRED cross-asset proxies, 1992-2021
  (significant IC, gated IR −0.12); 20 US large caps, 2008-2023 (no model beat equal weight of the same survivor names);
  a point-in-time top-500 US stock universe including delisted names, Aug 2025 - Jul 2026 (IC ≈ 0; the gated system
  stayed near SPY); 24 US ETFs, 2002-2022 (significant IC t 2.2, but best IR 0.32 with Deflated Sharpe 0.42; gated
  0.0%); and 10 Indian large caps, 2010-2023 (no skill; equal weight's +7.4%/yr is survivorship plus a price-index benchmark). The point-in-time history is
  only ~2 years (Massive free plan), far too short for confident conclusions.
* Survivorship bias: the hand-picked default universe is biased (measured at ~2.5%/yr in 2025-26). `bigboss ingest
  --massive-market 500` builds a point-in-time universe with delisted names, limited to the data plan's history window.
  FMP's free plan serves only a subset of symbols (most ETFs and NSE/BSE tickers need a paid plan).
* Macro values use the latest vintage with conservative lags, not ALFRED real-time vintages (revisions can leak slightly).
* Corporate actions rely on provider-adjusted prices. Alpha Vantage free data is unadjusted and is flagged.
* Bid-ask spreads are configured per asset class (real quotes are not freely available). Corwin–Schultz estimates are
  used only for thin names. Market-impact coefficients are literature defaults, not fitted to your fills.
* Multi-currency: each market is measured against its own benchmark in local currency. FX translation of a mixed
  USD/INR portfolio is not yet modelled in the simulator.
* News: only SEC filings are ingested automatically. General financial news has no licensed free feed, so it is not scraped.
* Event study effects need ≥ 8 past events per type to be marked reliable.
* Paper fills use daily closes (no intraday simulation). The scheduler is a simple loop, not a durable job queue.
* The MLP and XGBoost paths are tested on CPU here. CUDA selection is automatic but was not exercised in development.
