# Experiment log: results, negative results, failed approaches

**Read this first.** Every model and backtest number below comes from the **SYNTHETIC** market
(`bigboss/data/synthetic.py`), not from real securities. The development container's network policy blocked every free
daily price source (Stooq, Yahoo, NSE/BSE) and FRED was down for maintenance, so no real-market backtest could be run here.
The synthetic market has a *known, planted* signal (a slowly varying per-asset alpha that trailing momentum partially
reveals). These experiments therefore test whether the **machinery** works: whether it finds a real signal, stays
calibrated, avoids leakage, and charges costs correctly. They say **nothing** about whether BigBoss can beat real markets.
Real-market results will almost certainly be weaker. Run `bigboss ingest && bigboss backtest --ablation` to produce them.

## Setup (experiment `20261022T044438-49d7`)

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

## Strategy results (net of estimated costs, OOS only)

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

## Forecast quality (OOS)

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

## What worked

* **Calibration.** Conformalized 80% intervals landed within 79–81% coverage for every model, out of sample.
* **The pipeline finds a planted signal, and only a planted one.** The ensemble's IC is 0.077 (t = 3.4) on the market with alpha.
  On a zero-alpha synthetic market (test `test_zero_signal_market_shows_no_skill`), the IC t-stat is statistically zero.
  Realistic IC magnitudes (well below 0.35) also act as a leakage alarm.
* **Cost handling.** Costs reduce excess return monotonically; the gated ensemble keeps 5.1% annual excess even at 3× base costs.
  The optimizer's cost term cuts turnover as costs rise (2.37 → 1.62 per year from 0× to 3×).
* **Regimes.** The gated ensemble's excess was positive in all four ex-ante regimes: bull/low-vol +2.1% (632 days),
  bull/high-vol +7.6% (494), bear/low-vol +8.8% (201), bear/high-vol +13.3% (only 58 days, too few to rely on).

## Negative results (kept on purpose)

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
* A per-name abstention rule based on σ/|μ| would have abstained on everything, because single-stock forecasts always have
  σ ≫ |μ|. It was replaced by epistemic criteria (model disagreement, sign agreement, OOS skill).

## Not yet done

* Real-data experiments (US large caps, ETFs, NSE), with survivorship-bias-aware universes.
* Objective comparison across risk measures (variance vs CVaR vs drawdown) on real data.
* Hyperparameter search (Optuna + purged K-fold is wired in `backtest/splits.py` but deliberately not used, to limit
  multiple testing until real data is available).
* GPU benchmarks on the RTX 4060.
