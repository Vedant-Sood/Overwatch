"""QuantConnect bridge: export BigBoss walk-forward output for an independent LEAN backtest, and compare the results.

Why: QuantConnect's cloud has survivorship-free US equity data with its own fill/fee models. Replaying BigBoss's
decisions there checks BigBoss's backtester and data against an independent engine.

Export (``bigboss export-qc``) writes to an output directory:
  weights.csv     date,symbol,weight - BigBoss's target weights at each decision date (mode "weights")
  signals.csv     date,symbol,mean,q10,q90,p_pos,disagreement,sign_agreement,skill_t (mode "signals")
  manifest.json   provenance: experiment, strategy, dates, model, sources, SHA-256 of each file, BigBoss's own metrics
  main.py         the LEAN algorithm (copy of bigboss/integrations/qc_lean_main.py)
  README.md       step-by-step instructions

Timing contract (both engines): a row dated D was decided at the close of D and may only be acted on from the next
session. The LEAN algorithm enforces this by using only rows dated strictly before the current trading day.

Compare (``bigboss qc-compare``) reads QuantConnect's backtest-result JSON (web IDE: Backtest -> Download results) and
reports both engines' metrics over the common window plus the correlation of their daily returns.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bigboss.backtest.metrics import summarize
from bigboss.config import Config, market_of

LEAN_SOURCE = Path(__file__).with_name("qc_lean_main.py")
SIGNAL_COLUMNS = ["mean", "q10", "q90", "p_pos", "disagreement", "sign_agreement", "skill_t"]


def qc_ticker(symbol: str) -> str | None:
    """BigBoss symbol -> QuantConnect US equity ticker, or None when QuantConnect's US equity data cannot represent it."""
    s = symbol.upper()
    if s.startswith(("XA:", "^", "SYN")) or market_of(s) != "US":
        return None
    return s.replace("-", ".")  # e.g. BRK-B -> BRK.B (QC convention)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_experiment(cfg: Config, exp_id: str, out_dir: Path, strategy: str = "ensemble_gated", model: str = "ensemble") -> dict:
    exp = cfg.data_dir / "experiments" / exp_id
    if not (exp / "summary.json").exists():
        raise FileNotFoundError(f"experiment {exp_id} not found under {cfg.data_dir / 'experiments'}")
    summary = json.loads((exp / "summary.json").read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped: set[str] = set()

    wpath = exp / f"weights_{strategy}.parquet"
    if not wpath.exists():
        raise FileNotFoundError(f"no saved weights for strategy {strategy!r} in {exp_id} (re-run the experiment to save all strategies)")
    wide = pd.read_parquet(wpath)
    w = wide.stack().rename("weight").reset_index()
    w.columns = ["date", "symbol", "weight"]
    w = w[w["weight"].abs() > 1e-6]
    w["ticker"] = w["symbol"].map(qc_ticker)
    skipped |= set(w.loc[w["ticker"].isna(), "symbol"])
    w = w.dropna(subset=["ticker"])
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w[["date", "ticker", "weight"]].rename(columns={"ticker": "symbol"}).sort_values(["date", "symbol"]).to_csv(
        out_dir / "weights.csv", index=False, float_format="%.6f")

    oos = pd.read_parquet(exp / "oos.parquet")
    cols = {f"mean_{model}": "mean", f"q10_{model}": "q10", f"q90_{model}": "q90", f"p_pos_{model}": "p_pos"}
    missing = [c for c in cols if c not in oos]
    if missing:
        raise ValueError(f"model {model!r} not in experiment OOS output (missing {missing})")
    sig = oos[["date", "symbol"] + list(cols)].rename(columns=cols)
    for extra, src in (("disagreement", "disagreement"), ("sign_agreement", "sign_agreement"), ("skill_t", "ensemble_skill_t")):
        sig[extra] = oos[src] if model == "ensemble" and src in oos else np.nan
    sig["ticker"] = sig["symbol"].map(qc_ticker)
    skipped |= set(sig.loc[sig["ticker"].isna(), "symbol"])
    sig = sig.dropna(subset=["ticker", "mean"])
    sig["date"] = pd.to_datetime(sig["date"]).dt.strftime("%Y-%m-%d")
    sig = sig[["date", "ticker"] + SIGNAL_COLUMNS].rename(columns={"ticker": "symbol"}).sort_values(["date", "symbol"])
    sig.to_csv(out_dir / "signals.csv", index=False, float_format="%.6g")

    shutil.copyfile(LEAN_SOURCE, out_dir / "main.py")
    (out_dir / "README.md").write_text(_instructions(summary, strategy))
    strat = summary["strategies"].get(strategy, {})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "bigboss_experiment": exp_id, "strategy": strategy, "model": model,
        "data_mode": summary["data_mode"], "horizon_days": summary["horizon"], "benchmark": summary["benchmark"],
        "oos_start": summary["oos_start"], "oos_end": summary["oos_end"], "rebalance_every_days": summary["assumptions"]["rebalance_every_days"],
        "execution": summary["assumptions"]["execution"], "universe_size": summary["universe_size"], "sources": summary.get("sources"),
        "symbols_skipped_not_on_qc_us_equities": sorted(skipped),
        "n_weight_rows": int(len(w)), "n_decision_dates": int(w["date"].nunique()), "n_signal_rows": int(len(sig)),
        "bigboss_metrics": {k: strat.get(k) for k in ("ann_return", "ann_excess_return", "sharpe", "information_ratio",
                                                      "max_drawdown", "ann_turnover", "ann_cost_drag")},
        "files": {f: _sha256(out_dir / f) for f in ("weights.csv", "signals.csv", "main.py")},
        "timing_contract": "rows dated D are decided at the close of D and may only be traded from the next session",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def _instructions(summary: dict, strategy: str) -> str:
    return f"""# Running this BigBoss export on QuantConnect

Experiment `{summary['id']}` ({summary['data_mode']}), strategy `{strategy}`, OOS {summary['oos_start']} -> {summary['oos_end']}.

1. Host the data where the QuantConnect cloud can read it. Either
   - **Object Store** (recommended): in the QuantConnect web IDE open the Object Store panel and upload `weights.csv`
     (and `signals.csv` for signals mode) with keys `bigboss/weights.csv` and `bigboss/signals.csv`; or
   - **a public URL** (e.g. a GitHub raw link or a Dropbox link ending in `dl=1`); set the `weights_url` / `signals_url`
     parameters below.
2. Create a new Python project and replace `main.py` with the `main.py` in this folder.
3. Set project parameters (Project panel -> Parameters). Defaults reproduce BigBoss's setup:
   - `mode` = `weights` (replay BigBoss's exact targets) or `signals` (rebuild a simple gated top-k portfolio)
   - `start` = `{summary['oos_start']}`, `end` = `{summary['oos_end']}`, `cash` = `1000000`, `benchmark` = `{summary['benchmark']}`
   - `costs` = `bigboss` (≈ BigBoss's cost model) or `qc` (QuantConnect's default brokerage models)
   - signals mode only: `top_k` = `10`, `max_weight` = `0.10`, `rebalance_days` = `{summary['assumptions']['rebalance_every_days']}`,
     `min_skill_t` = `1.0`, `max_disagreement` = `0.03`, `min_sign_agreement` = `0.6`
4. Run a backtest, then **Download results** (JSON) and compare locally:
   `bigboss qc-compare --experiment {summary['id']} --strategy {strategy} --qc-results <downloaded.json>`

Expect small differences (fill timing: BigBoss fills at the next close, the algorithm fills shortly before the next
close on hourly bars; fees/slippage models; QuantConnect's own corporate-action data). Large differences point to a
data or backtester problem on one side - that is the purpose of this check.
"""


# ---------------- comparison ----------------

def _series_values(series: dict) -> pd.Series:
    """QuantConnect chart series -> pd.Series indexed by date. Handles [[t, v]], [[t, o, h, l, c]] and [{x, y}] formats."""
    vals = series.get("values") or series.get("Values") or []
    rows = []
    for v in vals:
        if isinstance(v, dict):
            t, y = v.get("x"), v.get("y")
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            t, y = v[0], v[-1]  # close for candlesticks
        else:
            continue
        if t is not None and y is not None:
            rows.append((pd.to_datetime(t, unit="s"), float(y)))
    s = pd.Series(dict(rows)).sort_index()
    return s.groupby(s.index.normalize()).last()


def parse_qc_results(path: Path) -> tuple[pd.Series, pd.Series | None]:
    """Return (equity curve, benchmark curve) from a QuantConnect backtest-result JSON."""
    data = json.loads(Path(path).read_text())
    charts = data.get("charts") or data.get("Charts") or data.get("backtest", {}).get("charts") or {}
    if not charts:
        raise ValueError("no 'charts' in QuantConnect result file - download the full backtest JSON")
    eq_chart = charts.get("Strategy Equity") or charts.get("strategy equity")
    if not eq_chart:
        raise ValueError("'Strategy Equity' chart missing from the result file")
    series = eq_chart.get("series") or eq_chart.get("Series")
    equity = _series_values(series.get("Equity") or next(iter(series.values())))
    bench = None
    bchart = charts.get("Benchmark")
    if bchart:
        bser = bchart.get("series") or bchart.get("Series")
        if bser:
            bench = _series_values(next(iter(bser.values())))
    return equity, bench


def compare(cfg: Config, exp_id: str, qc_results: Path, strategy: str = "ensemble_gated") -> dict:
    exp = cfg.data_dir / "experiments" / exp_id
    rets = pd.read_parquet(exp / "returns.parquet")
    bb = rets[strategy].dropna()
    bb_bench = rets["benchmark_index"].dropna()
    equity, qc_bench_curve = parse_qc_results(qc_results)
    qc = equity.pct_change().dropna()
    qc_bench = qc_bench_curve.pct_change().dropna() if qc_bench_curve is not None else bb_bench
    common = bb.index.intersection(qc.index)
    if len(common) < 20:
        raise ValueError(f"only {len(common)} overlapping days between BigBoss and QuantConnect results")
    a, b = bb.loc[common], qc.loc[common]
    m_bb = summarize(a, bb_bench.reindex(common).fillna(0))
    m_qc = summarize(b, qc_bench.reindex(common).fillna(0))
    keys = ("ann_return", "ann_excess_return", "sharpe", "information_ratio", "max_drawdown", "ann_vol")
    cum_a, cum_b = (1 + a).cumprod(), (1 + b).cumprod()
    return {
        "common_days": int(len(common)), "start": str(common[0].date()), "end": str(common[-1].date()),
        "daily_return_correlation": float(a.corr(b)),
        "tracking_difference_ann": float((a - b).mean() * 252),
        "max_cumulative_gap": float((cum_a / cum_b - 1).abs().max()),
        "bigboss": {k: m_bb.get(k) for k in keys}, "quantconnect": {k: m_qc.get(k) for k in keys},
        "verdict": ("consistent" if a.corr(b) > 0.95 and abs((a - b).mean() * 252) < 0.02 else
                    "inconsistent - investigate data, fills or corporate actions"),
    }
