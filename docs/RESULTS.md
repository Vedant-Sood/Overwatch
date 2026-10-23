# Experiment log: results, negative results, failed approaches

Experiments reported:

1. **Part 1 - real cross-asset proxies** (FRED, keyless), 1992-2021 out of sample.
2. **Part 1b - real US large caps** (FMP, dividend-adjusted), 2008-2023 out of sample - survivorship-biased universe.
3. **Part 1c - real point-in-time US universe** (Massive whole-market bars, top-500 by trailing liquidity, delisted names
   included), Aug 2025 - Jul 2026 out of sample.
4. **Part 2 - synthetic market** with a *known* planted signal, used to test the machinery (leakage, calibration, costs).

---

# Part 1 - Real data: FRED cross-asset proxies (experiment `20261022T060054-c2c1`)

| item | value |
|---|---|
| universe | XA:NDX (Nasdaq-100), XA:NIKKEI_USD, XA:BRENT, XA:UST10_TR, XA:UST2_TR, XA:EUR, XA:GBP, XA:JPY, XA:INR |
| benchmark | XA:EW, equal-weight daily-rebalanced basket of the members (4.7%/yr, 7.3% vol over the OOS period) |
| history | 1990-01-02 → 2026-10-21 (EUR from 1999) |
| OOS period | **1992-12-04 → 2021-04-26**, 58 walk-forward folds; final holdout from 2021-04-27 **not touched** |
| horizon / execution | 20 trading days, decision at close *t*, fill at close *t+1* |
| models | zero-excess, shrunk hist. mean, momentum rule, ridge, LightGBM, XGBoost, MLP, stacking ensemble |
| costs | proxy half spreads: equity 1bp, rates 1bp, FX 0.5bp, Brent 3bp, + commission, slippage, sqrt impact |
| runtime | 81 CPU-minutes (4 cores, no GPU) |

**Caveats.** These are proxies, not tradable instruments. Equity indices exclude dividends (understating equity versus
the basket by roughly 1%/yr). Brent is spot (no futures roll yield). FX is spot without interest carry, which is a large
omission for INR and JPY. Treasury returns are re-priced from constant-maturity yields. Nine assets is a very small cross-section.

### Forecast quality (OOS)

| model | rank IC | IC t | 80% coverage | Brier | Brier (climatology) |
|---|---|---|---|---|---|
| hist. mean | 0.074 | 3.31 | 79.5% | 0.2513 | 0.2492 |
| momentum rule | 0.044 | 1.73 | 79.3% | 0.2529 | 0.2492 |
| ridge | 0.051 | 2.21 | 79.2% | 0.2568 | 0.2492 |
| LightGBM | 0.040 | 1.76 | 80.1% | 0.2540 | 0.2492 |
| XGBoost | 0.040 | 1.75 | 79.6% | 0.2544 | 0.2492 |
| **MLP** | **0.080** | **3.31** | 79.6% | 0.2566 | 0.2492 |
| ensemble | 0.064 | 2.74 | 79.2% | 0.2502 | 0.2492 |

### Strategies (net of costs, OOS)

| strategy | ann. return | ann. excess | Sharpe | IR | IR 90% CI | max DD | turnover | Deflated SR |
|---|---|---|---|---|---|---|---|---|
| benchmark (XA:EW) | 4.7% | 0.0% | 0.67 | – | – | −24.9% | 0.02 | – |
| equal weight (20-day rebalance) | 4.0% | −0.7% | 0.54 | −0.14 | [−0.53, 0.13] | −26.1% | 0.77 | 0.01 |
| 12-1 momentum top-3 | 5.3% | +0.6% | 0.54 | 0.10 | [−0.20, 0.37] | −29.2% | 2.78 | 0.13 |
| low-vol top-3 | 1.3% | −3.4% | 0.40 | −0.49 | [−0.85, −0.17] | −20.6% | 1.08 | 0.00 |
| model: hist. mean | 5.2% | +0.5% | 0.70 | **0.28** | [0.02, 0.55] | −27.3% | 0.11 | 0.44 |
| model: momentum rule | 5.0% | +0.2% | 0.72 | 0.10 | [−0.19, 0.36] | −25.2% | 0.78 | 0.13 |
| model: ridge | 4.1% | −0.6% | 0.53 | −0.19 | [−0.46, 0.08] | −33.3% | 1.75 | 0.00 |
| model: LightGBM | 4.6% | −0.2% | 0.58 | −0.03 | [−0.36, 0.26] | −33.8% | 1.95 | 0.03 |
| model: XGBoost | 4.7% | −0.1% | 0.59 | 0.00 | [−0.32, 0.28] | −33.7% | 1.99 | 0.05 |
| model: MLP | 4.5% | −0.2% | 0.58 | −0.07 | [−0.38, 0.23] | −35.0% | 0.96 | 0.02 |
| ensemble (ungated) | 4.7% | −0.0% | 0.62 | 0.02 | [−0.30, 0.32] | −33.4% | 0.99 | 0.06 |
| **ensemble (gated, production rule)** | 4.4% | **−0.3%** | 0.59 | **−0.12** | [−0.42, 0.19] | −33.2% | 0.84 | 0.01 |

### Findings (real data)

1. **Statistical skill without economic value.** The ensemble's rank IC is significant (t = 2.7) and the MLP's even more so
   (t = 3.3), yet no model-driven portfolio beat the equal-weight benchmark after costs with any confidence. The production
   (gated) ensemble lost 0.3%/yr versus the benchmark. This is the central lesson the objective was designed to expose: rank
   accuracy across 9 heterogeneous assets mostly reflects persistent level differences (equities drift up, the rupee drifts
   down), which the benchmark-relative, risk-constrained portfolio cannot monetize after tracking-error penalties.
2. **The simplest model was the only candidate.** The shrunk historical mean is the only strategy whose IR confidence interval
   excludes zero ([0.02, 0.55]). After the multiple-testing haircut for 12 strategies, its Deflated Sharpe is 0.44: **not significant**.
3. **Costs are not the problem.** At 0× costs the gated ensemble still had negative excess (−0.3%/yr). Higher assumed costs
   *reduced* turnover and slightly *improved* results (gated IR −0.10 at 0× → +0.01 at 3×), a sign that most trades were noise.
4. **Regimes.** The ensemble added value only in bull/high-vol periods (+1.5%/yr, 1,733 days). It lost in bear/high-vol
   (−1.7%/yr), bear/low-vol (−1.5%/yr) and bull/low-vol (−0.5%/yr).
5. **Model drawdowns were deeper than the benchmark's** (≈ −33% vs −25%), because the models tilted toward equities
   and oil (beta ≈ 1.04).
6. **Ablation.** No single feature group materially changed LightGBM. Removing momentum, volatility or cross-sectional
   features lowered IR by 0.10–0.13. Liquidity features carried nothing (the proxies have no volume), as expected.
   Macro features (FRED, availability-lagged) had the largest IC contribution (−0.005 when removed), but the effect is small.
7. **Calibration held on real data.** Conformalized 80% intervals covered 79–80% out of sample for every model, through
   1998, 2000–02, 2008 and 2020. Probability forecasts again failed to beat climatology.
8. **Data-quality checks caught real events.** Validation flagged Brent's April 2020 collapse, the rupee's early-1990s peg
   (stale prints), and the weekly lag in Fed FX releases.

9. **The promotion gates were wrong, and were fixed.** All three real-data model versions (h = 5, 20, 60) initially
   passed the forecast-only gates (IC t 2.6–2.8, calibration, sample size). Because the objective is net economic value,
   an economic gate was added: the gated strategy's walk-forward net IR must exceed 0, and PSR(net edge > 0) must be at
   least 0.90. Re-evaluation (`bigboss models reevaluate`) demoted every version:

   | version | net IR | IR 90% CI | ann. excess | PSR(edge > 0) | gate |
   |---|---|---|---|---|---|
   | h = 5 | +0.17 | [−0.12, 0.44] | +0.30% | 0.82 | fail (PSR < 0.90) |
   | h = 20 | −0.12 | [−0.42, 0.19] | −0.32% | 0.26 | fail |
   | h = 60 | −0.19 | [−0.47, 0.09] | −0.37% | 0.15 | fail |

   The h = 20 economic check reproduces the full experiment's gated result exactly (IR −0.119), a useful consistency check.
   With no champion, the observed-data scanner shows forecasts **research only**, with every action INSUFFICIENT EVIDENCE.
10. **Drift.** Against the 1990–2021 training distribution, the last 126 days show extreme PSI for rate features (T-bill 8.3,
    10y yield 6.9), i.e. the post-2022 rate regime. These are market-state features: this is regime change, not a data bug,
    but it is another reason not to deploy.

**Bottom line.** On this real dataset BigBoss would mostly have recommended INSUFFICIENT EVIDENCE or small tilts, and the
gated system's realized edge was zero to slightly negative. The platform reports exactly that, which is its job. Whether
stock-level data (a much larger cross-section, plus fundamentals and filings) changes the picture is the next experiment.

---

# Part 1b - Real data: 20 US large caps, 2008-2023 (FMP, dividend-adjusted; experiment `20261022T124953-d909`)

| item | value |
|---|---|
| universe | AAPL AMZN BAC CSCO CVX DIS GOOGL INTC JNJ JPM KO META MSFT NVDA PEP T UNH V WMT XOM (the subset of the configured list that FMP's free plan serves) |
| benchmark | SPY (dividend-adjusted) |
| data | FMP daily bars 2005-2026, split- and dividend-adjusted; SEC XBRL fundamentals and filing events for all 20; FRED/Treasury/World Bank macro |
| OOS | **2008-01-04 → 2023-07-12**, 31 walk-forward folds; holdout from 2023-07-13 untouched |
| horizon | 20 days, fill at next close, cost model at base estimates |

**Read this caveat first: the universe is survivorship-biased.** These 20 names were chosen *in 2026* because they are
today's giants. Any strategy holding them looks good against SPY in hindsight, so the meaningful comparison is
against **equal weight of the same 20 names**, not SPY.

| strategy | ann. excess vs SPY | IR | IR 90% CI | turnover | Deflated SR |
|---|---|---|---|---|---|
| equal weight (the fair yardstick) | **+5.4%** | 0.91 | [0.60, 1.34] | 0.33 | 0.97 |
| 12-1 momentum top-k | +7.0% | 0.57 | [0.22, 1.01] | 2.81 | 0.72 |
| low-vol top-k | −0.7% | −0.14 | [−0.52, 0.27] | 2.07 | 0.01 |
| model: hist. mean | +5.7% | 0.95 | [0.59, 1.33] | 0.55 | 0.98 |
| model: ridge | +3.4% | 0.49 | [0.08, 0.83] | 2.73 | 0.60 |
| model: LightGBM | +3.9% | 0.51 | [0.13, 0.90] | 2.88 | 0.64 |
| model: XGBoost | +3.5% | 0.49 | [0.12, 0.91] | 2.87 | 0.61 |
| model: MLP | +5.0% | 0.67 | [0.33, 1.02] | 2.56 | 0.84 |
| ensemble (ungated) | +4.7% | 0.67 | [0.33, 1.02] | 2.27 | 0.84 |
| **ensemble (gated)** | **+1.9%** | 0.43 | [0.08, 0.78] | 1.25 | 0.51 |

Forecast quality (rank IC / t): hist. mean 0.072 / 3.04; MLP 0.041 / 1.86; ensemble 0.045 / 2.06; LightGBM 0.005 / 0.26;
momentum rule 0.003 / 0.11; XGBoost −0.001 / −0.03; ridge −0.001 / −0.03. 80% intervals covered 79-80%.

**Findings**

1. **No model beat equal weight of the same names.** Every positive number versus SPY is explained by the
   survivorship-biased universe. Equal weight (+5.4%/yr) beat the ensemble (+4.7%), and the gated production rule (+1.9%)
   gave up most of the tilt by abstaining or staying close to SPY.
2. **Tree models and the linear model had zero cross-sectional skill** (IC ≈ 0). The only signal came from the
   shrunk historical mean, which is the survivorship effect itself (persistent winners kept winning), and a weak MLP.
3. **SEC fundamentals did not help.** In the ablation, removing the fundamental group *raised* LightGBM's IR by 0.13.
   This was the first real test of the point-in-time XBRL features (Model B inputs), and it was negative. Removing
   momentum (+0.17) or market features (+0.14) also helped; only volatility features (−0.02 IR, −0.017 IC) and filing
   events (−0.01) carried anything.
4. **Costs barely mattered** (gated excess +1.9% at 0× → +1.7% at 3×). The problem is signal, not friction.
5. The gated ensemble's excess was positive in bear/high-vol (+3.4%), bull/high-vol (+4.0%) and ~0 in bull/low-vol (+0.3%).

**Verdict:** with 20 hand-picked survivors, there is no evidence that BigBoss's models add value. This experiment mainly
demonstrates why a point-in-time universe is required (Part 1c).

---

# Part 1c - Real data: point-in-time US stock universe (Massive; experiments `20261022T160534-c20a` h=5, `20261022T162144-28ee` h=20)

| item | value |
|---|---|
| data | Massive grouped-daily bars for **every** US-listed security, 500 sessions (2024-10-23 → 2026-10-21; the free plan's 2-year window), split-adjusted (no dividends) |
| universe | on each date, the top 500 **common stocks** by median dollar volume over the previous 60 sessions (price ≥ $5, ≥ 40 sessions of history), using only data known before that date |
| breadth | 701 distinct stocks were ever members, **87 of them later delisted, acquired or renamed** (kept in the test); 697 with SEC XBRL fundamentals; plus 24 ETFs as instruments; SPY benchmark |
| OOS | **2025-08-13 → 2026-07-07** (≈ 11 months, 11 monthly retrains); holdout from 2026-07-08 untouched |
| protocol | min. 200 training sessions, retrain every 21, 5-day embargo, purged labels; costs at base estimates with real ADV-based impact |

| strategy | h = 5: ann. excess | IR | h = 20: ann. excess | IR |
|---|---|---|---|---|
| equal weight of PIT universe | +10.7% | 0.89 | +10.7% | 0.89 |
| 12-1 momentum top-10 | +92.3% | 1.56 (DSR 0.43) | same | same |
| low-vol top-10 | −18.2% | −1.68 | same | same |
| model: ridge | +37.2% | 1.19 | +11.4% | 0.64 |
| model: LightGBM | −2.7% | −0.08 | +1.2% | 0.16 |
| model: XGBoost | +23.6% | 1.58 | +7.7% | 0.65 |
| model: MLP | +24.0% | 1.21 | +18.9% | 1.38 |
| ensemble (ungated) | +15.6% | 1.05 | +13.4% | 1.15 |
| **ensemble (gated)** | **−3.6%** | −1.15 | **+0.8%** | 0.30 |

Forecast skill (rank IC / t, h = 5 → h = 20): ensemble 0.015 / 0.62 → 0.026 / 0.67; LightGBM 0.005 → 0.026; XGBoost 0.013 → 0.021;
ridge 0.007 → 0.016; MLP 0.010 → 0.063 (t 1.71); momentum rule −0.073 (short-term reversal) → +0.142 (t 1.27).
80% intervals covered 76-81%. Every Brier score was worse than climatology.

**Findings**

1. **No statistically detectable forecast skill** at either horizon (all |IC t| < 2). The impressive-looking model
   returns are market exposure: the ungated ensemble (+13-16%/yr vs SPY) roughly matches equal weight of the same
   point-in-time universe (+10.7%), which beat SPY because mid/large caps beyond the mega caps outperformed in this window.
   The ungated models also ran betas above 1 against SPY (ensemble 1.39 at h = 5 and 1.07 at h = 20; ridge up to 1.96),
   so part of their excess return is simply leverage to a rising market.
2. **The gate behaved correctly.** With no demonstrated skill the gated ensemble mostly stayed in SPY (turnover 0.9-1.1)
   and ended near zero (−3.6% / +0.8%). The production rule refused to bet on noise.
3. **Survivorship bias, measured.** A first run had a bug: a global 260-day history filter silently dropped 57 of the 87
   delisted members. Fixing it lowered equal weight's excess from +13.2% to +10.7%/yr. That **2.5%/yr is the measured
   survivorship bonus** in this window, and it is why Part 1b's hand-picked universe cannot be trusted.
4. **Eleven months is too short.** Momentum top-10's +92% has a Deflated Sharpe of only 0.43, and the ablation swings by
   ±1 IR when single groups are removed (fundamentals: IC 0.026 → 0.003 when removed, the first hint they help, but at t < 1).
   No feature conclusions are warranted. Any model trained on this window would fail the sample-size and economic gates,
   and the scanner would show it research-only.
5. **Universe hygiene matters.** Three data bugs were found and fixed before these results: ETFs/ETNs leaking in through
   Massive's paginated reference data, delisted tickers missing from SEC's current ticker map (solved with Massive CIKs),
   and the history-filter survivorship leak above.

---

# Part 1d - Real data: 24 US ETFs, 2002-2022 (yfinance, dividend-adjusted; experiment `20261023T114106-fcbf`)

| item | value |
|---|---|
| universe | 24 liquid US ETFs (sectors, size, international, bonds, credit, REITs, gold, silver, commodities); SPY benchmark |
| data | yfinance from 1999, dividend-adjusted; holdout from 2022-08-17 untouched |
| OOS | **2002-01-08 → 2022-08-16** (20.6 years), h = 20, monthly retrains |

| strategy | ann. return | ann. excess vs SPY | IR | PSR | Deflated SR | max DD |
|---|---|---|---|---|---|---|
| SPY | 8.6% | – | – | – | – | −55.2% |
| equal weight | 8.0% | −0.6% | −0.20 | 0.18 | 0.01 | −41.9% |
| 12-1 momentum top-k | 9.5% | +0.9% | 0.02 | 0.54 | 0.06 | −32.5% |
| model: ridge | 10.4% | +1.8% | **0.32** | **0.93** | 0.42 | −51.7% |
| model: MLP | 9.3% | +0.7% | 0.15 | 0.75 | 0.16 | −53.9% |
| ensemble (ungated) | 9.1% | +0.5% | 0.13 | 0.72 | 0.14 | −53.8% |
| **ensemble (gated)** | 8.6% | **0.0%** | 0.03 | 0.55 | 0.06 | −54.1% |

Forecast skill (rank IC / t): hist. mean 0.054 / **2.42**, ridge 0.047 / **2.15**, MLP 0.074 / **2.96**, ensemble 0.052 / **2.21**,
LightGBM 0.011 / 0.46. 80% intervals covered 78-80%. Brier scores were again no better than climatology.

**Findings**

1. **The first statistically significant forecast skill on real data** (ensemble IC t 2.2, MLP 3.0, over 20 years
   and 24 assets). The skill is in *ranking* asset classes over the next month, mostly slow-moving carry/trend effects
   that the shrunk historical mean also captures.
2. **The skill barely survives as money.** The best strategy (ridge) earned +1.8%/yr over SPY with IR 0.32. Its PSR of
   0.93 passes the economic gate *in isolation*, but after correcting for the 12 strategies tested the Deflated Sharpe is
   0.42: not significant. The ensemble kept +0.5%/yr.
3. **The gate kept the production rule close to SPY.** The skill test (t ≥ 1 on prior data) passed in 25 of 42 folds
   (median t 1.27), but even then the per-asset disagreement/sign-agreement checks abstained on a third to a half of the
   24 ETFs, and abstained weight goes to the benchmark. Net: beta 1.00, excess 0.0%. That is the intended trade-off
   (no false positives) at the price of missing a weak, real signal; a looser gate is a candidate for a challenger test.
4. **Costs are not the issue here.** Moving from 0× to 3× costs changed the ensemble's excess return by only 0.1-0.3%/yr.
5. **Survivorship is limited** (ETFs rarely close and these 24 are the long-lived ones), but the universe is still
   chosen with hindsight.

---

# Part 1e - Real data: 10 Indian large caps, 2010-2023 (yfinance; experiment `20261023T123218-b200`)

| item | value |
|---|---|
| universe | RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, LT (NSE) |
| benchmark | ^NSEI (NIFTY 50 **price** index: excludes dividends, so every excess return below is flattered by ~1.2-1.5%/yr) |
| data | yfinance from 2007 (pre-2007 Indian adjustments are unreliable; see `DATA_SOURCES.md`) |
| OOS | **2010-02-01 → 2023-11-02**; holdout from 2023-11-03 untouched; India costs include 10 bp STT |

| strategy | ann. return | ann. excess vs ^NSEI | IR | max DD |
|---|---|---|---|---|
| ^NSEI (buy & hold) | 9.1% | – | – | −38.4% |
| **equal weight** | **17.8%** | **+7.4%** | **1.22** | −34.2% |
| 12-1 momentum top-k | 19.8% | +9.5% | 0.68 | −38.7% |
| low-vol top-k | 19.5% | +9.1% | 0.66 | −44.1% |
| model: ridge | 13.9% | +3.5% | 0.63 | −36.2% |
| ensemble (ungated) | 13.8% | +3.4% | 0.58 | −37.2% |
| **ensemble (gated)** | 10.2% | **−0.2%** | −0.06 | −37.0% |

Forecast skill: no model was significant (ensemble IC 0.019, t 0.72; momentum rule −0.046, t −1.46).

**Findings**

1. **Every strategy that "won" is explained by the universe, not by skill.** Equal weight of today's ten largest
   Indian companies beat the index by 7.4%/yr. That is survivorship (these are the ten that became giants), plus the
   dividends missing from the price-index benchmark. No model beat equal weight of the same names.
2. **No forecast skill**, so the gate abstained and the gated ensemble held the benchmark-like portfolio (−0.2%/yr).
3. **Not usable for decisions until** there is a point-in-time Indian universe (index constituents by date, including
   names that left) and a total-return benchmark (NIFTY 50 TRI).

# Part 1f - Real data: point-in-time US top-500, 2018-2026, incl. delisted names (DoltHub; experiment `20261023T144009-f932`)

**The longest survivorship-free real test so far.** 5.1 years out of sample, roughly 5x Part 1c.

| item | value |
|---|---|
| data | DoltHub `post-no-preference/stocks` raw bars, adjusted by BigBoss from DoltHub's split/dividend tables; 24 unrecorded splits corrected and 25 bogus split records dropped after verification (see `DATA_SOURCES.md`); total returns match yfinance (median daily corr 0.9997, mean gap −7 bp/yr) |
| universe | on each date from **2018-09-07**, the top 500 common stocks by trailing 60-day median raw dollar volume (raw price ≥ $5, ≥ 40 sessions), from information before that date |
| breadth | **1,330 distinct stocks** were members; **302 stopped trading** before the end (bankruptcies, acquisitions, renames); SPY benchmark |
| OOS | **2020-06-02 → 2025-07-21** (21 quarterly retrains, min. 2 years training); holdout from 2025-07-22 untouched |
| features | price, volume, cross-section, market and macro; **no fundamentals** (SEC coverage of delisted names would be uneven, which would itself bias results) |

| strategy | ann. return | ann. excess vs SPY | IR | beta | max DD | Deflated SR |
|---|---|---|---|---|---|---|
| SPY | 16.7% | – | – | 1.00 | −23.5% | – |
| equal weight of PIT universe | 18.5% | +1.5% | 0.21 | 1.08 | −29.5% | 0.12 |
| **12-1 momentum top-10** | **56.8%** | **+39.8%** | 0.89 | 1.64 | **−59.0%** | 0.64 |
| low-vol top-10 | 4.9% | −12.1% | −0.91 | – | −15.7% | 0.00 |
| model: ridge | 14.0% | −3.0% | −0.13 | 1.12 | −35.6% | 0.03 |
| model: LightGBM | 12.9% | −4.1% | −0.10 | – | −44.4% | 0.03 |
| model: XGBoost | 4.2% | −12.8% | −0.48 | – | −61.8% | 0.00 |
| model: MLP | 4.7% | −12.2% | −0.58 | – | −54.0% | 0.00 |
| ensemble (ungated) | 12.6% | −4.4% | −0.25 | 1.15 | −35.5% | 0.01 |
| **ensemble (gated)** | 16.7% | **−0.3%** | −0.16 | 0.98 | −23.5% | 0.03 |

Forecast skill (rank IC / t): every model negative: ensemble −0.029 / −1.05, ridge −0.037 / −1.75, MLP −0.020 / −0.58,
LightGBM −0.008 / −0.28. 80% intervals covered 79-80%. Brier scores worse than climatology.

**Findings**

1. **No forecast skill in US single stocks over five years, and the ML models lost money.** Every model ranked stocks
   slightly *worse* than chance; the ungated ensemble trailed SPY by 4.4%/yr and XGBoost/MLP by ~12%/yr with drawdowns
   above 50%. The positive Part 1c numbers (11 months, 2025-26) did not generalise.
2. **The gate did exactly its job.** The skill test (t ≥ 1 on prior data) passed on **0 of 50** rebalances, so the
   production rule abstained throughout, held SPY (turnover 0.10, beta 0.98) and finished at −0.3%/yr - avoiding the
   models' losses.
3. **12-1 momentum top-10 earned +39.8%/yr over SPY, from a handful of famous winners**: Carvana, Super Micro,
   MicroStrategy, Novavax, GameStop, Palantir, D-Wave, Tesla, AppLovin, Riot. All of these were checked against
   yfinance, so it is not a data error. But it is a concentrated speculative bet: beta 1.64, a −59% drawdown, a flat
   2023, and a Deflated Sharpe of 0.64 after 12 strategies. Not significant, and not something the decision engine
   should adopt on this evidence.
4. **Equal weight of the point-in-time universe beat SPY by only 1.5%/yr** (vs +10.7% in Part 1c's 11 months): it
   won in high-volatility bull markets (+15%/yr) and lost in calm bull markets (−7%/yr) as mega-caps led.
5. **Costs are irrelevant to the conclusion.** At 0x to 3x costs the ensemble's excess moved from −4.2% to −4.8%.

---

# Part 2 - Synthetic market (machinery validation)

**Read this first.** Every number in Part 2 comes from the **SYNTHETIC** market (`bigboss/data/synthetic.py`), not
from real securities. It has a *known, planted* signal (a slowly varying per-asset alpha that trailing momentum partially
reveals). These experiments test whether the machinery finds a real signal, stays calibrated, avoids leakage, and charges
costs correctly.

### Setup (experiment `20261022T044438-49d7`)

| item | value |
|---|---|
| universe | 30 synthetic assets + `SYNMKT` (daily-rebalanced equal-weight index, the benchmark) |
| history | 2,520 trading days (2014-01 → 2023-08); `alpha_scale = 3` |
| horizon | 20 trading days, decision at close *t*, fill at close *t+1* |
| walk-forward | 3-year minimum train, retrain every 126 days, 5-year rolling window, 5-day embargo, purged |
| OOS period | 2016-11-25 → 2022-03-18 (final 15% holdout from 2022-03-21 **not touched**) |
| portfolio | cost-aware active-risk mean-variance, λ = 4, max weight 10%, sector cap 30%, turnover cap 50%/rebalance, rebalance every 20 days |
| costs | base estimates from `configs/bigboss.toml` (≈9 bps round trip for these assets), square-root impact |
| strategies tested | 12 (the Deflated Sharpe uses this count) |

### Strategy results (net of estimated costs, OOS only)

| strategy | ann. return | ann. excess | Sharpe | IR | IR 90% CI | max DD | turnover/yr | Deflated SR |
|---|---|---|---|---|---|---|---|---|
| benchmark (buy & hold) | 14.4% | −0.1% | 0.93 | – | – | −19.3% | 0.09 | – |
| equal weight | 14.2% | −0.2% | 0.92 | −0.53 | [−1.16, 0.13] | −19.4% | 0.48 | 0.00 |
| **12-1 momentum top-k (transparent)** | **27.4%** | **13.0%** | **1.42** | **1.45** | [0.73, 2.11] | −15.6% | 2.23 | **0.96** |
| low-vol top-k | 13.5% | −0.9% | 0.86 | −0.11 | [−0.69, 0.57] | −17.9% | 1.97 | 0.03 |
| model: shrunk hist. mean | 20.9% | 6.5% | 1.17 | 1.04 | [0.48, 1.64] | −17.2% | 0.46 | 0.78 |
| model: momentum rule | 24.8% | 10.4% | 1.36 | 1.41 | [0.78, 2.11] | −14.8% | 1.72 | 0.95 |
| model: ridge (linear factor) | 22.4% | 8.0% | 1.19 | 1.08 | [0.48, 1.67] | −15.3% | 2.99 | 0.81 |
| model: LightGBM | 13.7% | −0.7% | 0.80 | −0.03 | [−0.67, 0.63] | −20.9% | 3.20 | 0.04 |
| model: XGBoost | 16.3% | 1.9% | 0.92 | 0.30 | [−0.36, 0.88] | −16.0% | 3.20 | 0.17 |
| model: MLP | 21.4% | 7.0% | 1.19 | 1.09 | [0.51, 1.95] | −17.9% | 2.62 | 0.82 |
| ensemble (ungated) | 22.4% | 8.0% | 1.22 | 1.22 | [0.58, 1.83] | −15.4% | 2.10 | 0.88 |
| ensemble (gated, the production rule) | 20.5% | 6.1% | 1.14 | 0.96 | [0.35, 1.63] | −15.6% | 2.13 | 0.72 |

### Forecast quality (OOS)

| model | rank IC | IC t (overlap-corrected) | direction acc. | 80% interval coverage | Brier | Brier (climatology) |
|---|---|---|---|---|---|---|
| zero excess | – | – | – | 80.6% | 0.2499 | 0.2497 |
| hist. mean | 0.031 | 1.39 | 51.2% | 80.6% | 0.2523 | 0.2497 |
| momentum rule | **0.092** | **3.86** | 54.4% | 80.6% | 0.2502 | 0.2497 |
| ridge | 0.063 | 2.86 | 52.7% | 80.6% | 0.2513 | 0.2497 |
| LightGBM | 0.016 | 0.75 | 50.5% | 80.3% | 0.2507 | 0.2497 |
| XGBoost | 0.020 | 0.95 | 50.8% | 80.4% | 0.2516 | 0.2497 |
| MLP | 0.074 | 3.32 | 52.2% | 80.1% | 0.2508 | 0.2497 |
| ensemble | 0.077 | 3.40 | 53.3% | 79.1% | 0.2510 | 0.2497 |

### What worked

* **Calibration.** Conformalized 80% intervals landed within 79–81% coverage for every model, out of sample.
* **The pipeline finds a planted signal, and only a planted one.** The ensemble's IC is 0.077 (t = 3.4) on the market with alpha.
  On a zero-alpha synthetic market (test `test_zero_signal_market_shows_no_skill`), the IC t-stat is statistically zero.
  Realistic IC magnitudes (well below 0.35) also act as a leakage alarm.
* **Cost handling.** Costs reduce excess return monotonically; the gated ensemble keeps 5.1% annual excess even at 3× base costs.
  The optimizer's cost term cuts turnover as costs rise (2.37 → 1.62 per year from 0× to 3×).
* **Regimes.** The gated ensemble's excess was positive in all four ex-ante regimes: bull/low-vol +2.1% (632 days),
  bull/high-vol +7.6% (494), bear/low-vol +8.8% (201), bear/high-vol +13.3% (only 58 days, too few to rely on).

### Negative results (kept on purpose)

1. **The transparent baseline won.** A plain 12-1 momentum top-k portfolio beat every model, including the ensemble
   (IR 1.45 vs 1.22), and is the only strategy besides the momentum rule with Deflated Sharpe ≥ 0.95. On this
   data-generating process (alpha revealed through momentum), extra modelling added cost and noise, not information.
2. **Gradient-boosted trees failed.** LightGBM (IC 0.016) and XGBoost (IC 0.020) were statistically indistinguishable from zero
   and produced the highest turnover. A small MLP and a ridge regression did far better.
3. **Ablation shows LightGBM overfits.** Removing almost any single feature group *improved* LightGBM, by up to
   ΔIR +0.98 (without momentum features). Only removing the volatility group hurt it. With an expanding training window
   instead of the 5-year rolling window, LightGBM's IC rose from 0.016 to 0.040 (first, inconsistent ablation run),
   so the default rolling window starves the tree models of data.
4. **Gating cost money here.** The skill gate and disagreement abstention reduced IR from 1.22 to 0.96. They are kept
   anyway: their job is to protect against the far more common real-world case where skill is absent, which this
   synthetic market (with a guaranteed signal) never tests.
5. **Probability forecasts have no value at single-name level.** Every model's Brier score is slightly *worse* than
   climatology (always predicting the base rate). Single-stock P(outperform) estimates hover around 50–55%. This is
   why the decision engine's default probability gate is 0.52 and why sizing relies on expected net value, not hit rate.
   Direction accuracy (≤ 54%) is reported only to show how little it says about economic value.
6. **Low-volatility top-k lost money net of costs** (−0.9%/yr), as the synthetic market plants no low-vol premium.
   Correctly, the system found no edge there.

## Bugs found by the tests during development (fixed)

* The look-ahead detector ignored features that were NaN without future data and present with it, which is exactly the
  leak signature. A deliberately leaky test feature exposed the gap.
* The simulator let costs drive cash negative (implicit leverage) and treated unknown ADV as unlimited liquidity.
* The synthetic benchmark compounded mean *log* returns (not an investable index), which made equal weight look like +4.5%/yr alpha.
* Ablation re-runs used a different training window than the main run.
* On real data, a label can be missing (e.g. no FX quote on the exit date) while its label-end date is known; one NaN label
  turned the conformal quantile into NaN. Calibrators now drop missing labels (synthetic data never had gaps).
* A per-name abstention rule based on σ/|μ| would have abstained on everything, because single-stock forecasts always have
  σ ≫ |μ|. It was replaced by epistemic criteria (model disagreement, sign agreement, OOS skill).

---

## Not yet done

* Stock-level real-data experiments (US large caps, ETFs, NSE) - needs a free Massive/FMP/Alpha Vantage key or Stooq/Yahoo access.
* Objective comparison across risk measures (variance vs CVaR vs drawdown) on real data.
* Hyperparameter search (Optuna + purged K-fold is wired in `backtest/splits.py` but deliberately not used, to limit
  multiple testing until real data is available).
* GPU benchmarks on the RTX 4060.
