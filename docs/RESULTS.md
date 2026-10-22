# Experiment log: results, negative results, failed approaches

Two experiments are reported:

1. **Real data (OBSERVED)**: 9 keyless cross-asset proxies built from FRED, 1990–2026, 28 years out of sample. This is the
   first evidence about real markets, with strong caveats (below).
2. **Synthetic data**: a market with a *known* planted signal, used to test the machinery (leakage, calibration, costs).

No stock-level real backtest exists yet: free stock prices need a key (Massive / FMP / Alpha Vantage) or Stooq/Yahoo,
which the development network blocks.

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
