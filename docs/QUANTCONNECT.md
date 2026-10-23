# QuantConnect: replaying BigBoss decisions in an independent engine

**Purpose.** BigBoss's backtester and data are its own. QuantConnect's LEAN engine has independent survivorship-free US
equity data, its own corporate-action handling and its own fill and fee models. Replaying BigBoss's decisions there
answers one question: *do two independent engines agree on what these decisions would have earned?* If they agree,
the BigBoss backtest is not an artefact of its own data or simulator. If they disagree, one side has a bug.

It does **not** test whether the forecasts are any good; the walk-forward experiment does that. It is also backtest-only:
the algorithm configures no brokerage and must not be deployed live.

## 1. Export

```bash
bigboss backtest --horizon 20                    # any experiment; it saves weights for every strategy
bigboss export-qc --experiment <ID> --out qc_export/ [--strategy model_ensemble] [--model ensemble]
```

| file | content |
|---|---|
| `weights.csv` | `date,symbol,weight`: BigBoss's target weights at each decision date for `--strategy` (default `ensemble_gated`) |
| `signals.csv` | `date,symbol,mean,q10,q90,p_pos,disagreement,sign_agreement,skill_t`: every OOS forecast of `--model` |
| `main.py` | the LEAN algorithm (`bigboss/integrations/qc_lean_main.py`) |
| `manifest.json` | experiment id, dates, sources, BigBoss's own metrics for the strategy, SHA-256 of each file, skipped symbols |
| `README.md` | the steps below, filled in with this experiment's dates and parameters |

Only US equities and ETFs are exported. FRED proxies (`XA:*`), Indian symbols (`.NS`), indices (`^…`) and synthetic
symbols are skipped and listed in the manifest. Tickers are mapped to QuantConnect's convention (`BRK-B` → `BRK.B`).

**Timing contract.** A row dated D was decided with data up to the close of D. BigBoss fills it at the close of D+1.
The algorithm uses a row only on trading days strictly after D and places orders 30 minutes before that day's close
(hour bars by default; set `resolution=minute` and `minutes_before_close=5` to get closer to the close).

## 2. Run on QuantConnect (free plan)

1. Upload the data. Use the **Object Store** panel in the web IDE with keys `bigboss/weights.csv` and `bigboss/signals.csv`,
   or host the files at a public URL and set the `weights_url` / `signals_url` parameters. A big universe's
   `signals.csv` is tens of MB, and the Object Store is the reliable route for that.
2. New project → Python → replace `main.py` with the exported one.
3. Parameters (Project panel). Defaults are shown; the README in the export folder has this experiment's values.

| parameter | default | meaning |
|---|---|---|
| `mode` | `weights` | `weights`: replay BigBoss's exact targets. `signals`: rebuild a gated top-k portfolio from the forecasts |
| `start`, `end` | experiment OOS window | backtest window |
| `cash` | `1000000` | starting cash |
| `benchmark` | `SPY` | benchmark; in signals mode it also receives the uninvested weight |
| `costs` | `bigboss` | `bigboss`: 0.5 bp commission + 4 bp constant slippage (≈ BigBoss's US estimates). `qc`: QuantConnect defaults |
| `top_k`, `max_weight`, `rebalance_days`, `round_trip_cost` | 10, 0.10, 20, 0.0010 | signals mode portfolio |
| `min_skill_t`, `max_disagreement`, `min_sign_agreement` | 1.0, 0.03, 0.6 | signals mode gate (BigBoss's abstention rule) |

4. Run the backtest. Check the runtime statistics *Tickers unavailable* and *Targets not tradable* (the names are in
   the log). Untradable targets stay in cash, matching BigBoss's "unknown liquidity → no fill" rule.
5. Download the results (Backtest → Download → results JSON).

## 3. Compare

```bash
bigboss qc-compare --experiment <ID> --strategy ensemble_gated --qc-results backtest.json
```

The command prints both engines' annual return, excess return, Sharpe, IR, volatility and maximum drawdown over the
common days, plus:

* `daily_return_correlation`: should be > 0.95 for a weights-mode replay;
* `tracking_difference_ann`: mean annual return gap;
* `max_cumulative_gap`: largest relative gap between the two equity curves;
* `verdict`: `consistent` when correlation > 0.95 and the annual gap is under 2%; otherwise investigate.

**Expected small differences:** fill time (BigBoss: next close, LEAN: shortly before the next close); fee and slippage
models; QuantConnect's own dividend and split data; delisting handling (both liquidate, at slightly different prices);
LEAN trades whole shares. **Red flags:** large gaps around specific dates (corporate actions, delistings), or a much
lower QuantConnect return for a strategy that holds small, illiquid names.

## Choosing what to replay

* `ensemble_gated` when the gate mostly abstained replays almost only SPY. That is a valid but weak check (e.g. Part 1c's
  h = 20 PIT experiment held only SPY on 10 of 12 rebalances).
* `model_ensemble` or `equal_weight` of the point-in-time universe hold dozens of names, delisted ones included. These
  are the informative checks of data and simulator.
* `signals` mode is a rebuild, not a replay: it uses a simpler top-k rule than BigBoss's optimizer, so expect
  different returns. Use it to stress-test the forecasts in an independent engine and data set.
