"""BigBoss command line.

  bigboss ingest [--groups us_etfs us_equities] [--symbols AAPL MSFT] [--no-macro] [--no-sec]
  bigboss demo-data                      # load the labelled SYNTHETIC dataset (offline testing only)
  bigboss train --horizon 20             # walk-forward train + register a model version
  bigboss backtest --horizon 20 [--ablation] [--final-holdout]
  bigboss scan --horizon 20              # ranked recommendations from the champion model
  bigboss research "Is MSFT attractive?" # agent workflow, writes a report
  bigboss models [promote VID | rollback --horizon 20 | challenge VID]
  bigboss update                         # daily job: ingest -> forecast -> log predictions -> resolve outcomes
  bigboss schedule --every-hours 24      # run `update` on an interval (or use cron / Task Scheduler)
  bigboss serve                          # API + built dashboard on http://127.0.0.1:8000
Add --synthetic to any research command to use the SYNTHETIC store instead of observed data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from bigboss.config import get_config


def _bb(args):
    from bigboss.service import BigBoss

    return BigBoss(get_config(), "SYNTHETIC" if getattr(args, "synthetic", False) else "OBSERVED")


def cmd_ingest(args):
    from bigboss.data.ingest import Ingestor
    from bigboss.data.store import Store

    cfg = get_config()
    store = Store.for_mode(cfg.data_dir, "OBSERVED")
    syms = args.symbols or cfg.universe(args.groups)
    ing = Ingestor(cfg, store)
    out = {}
    if not args.no_prices:
        out["prices"] = ing.ingest_prices(syms)
    if not args.no_macro:
        out["macro"] = ing.ingest_macro()
    if not args.no_sec:
        out["sec"] = ing.ingest_sec(syms)
    for k, v in out.items():
        s = pd.Series(v).value_counts().to_dict()
        print(f"{k}: {s}")
        failed = [x for x, st in v.items() if st == "FAILED"]
        if failed:
            print(f"  failed: {', '.join(failed[:30])}{' ...' if len(failed) > 30 else ''}")
    print(f"run_id={ing.run_id}; details: `bigboss status`")


def cmd_status(args):
    bb = _bb(args)
    st = bb.data_status()
    pr = pd.DataFrame(st["prices"])
    print(f"data_mode={st['data_mode']}  symbols={len(pr)}  events={st['n_events']}")
    if not pr.empty:
        print(pr[["symbol", "first_date", "last_date", "n_rows", "source", "adjustment", "age_days", "stale"]].to_string(index=False))
    iss = pd.DataFrame(st["validation_issues"])
    if not iss.empty:
        print("\nvalidation issues (latest run):")
        print(iss[["symbol", "severity", "check_name", "detail"]].to_string(index=False))
    mac = pd.DataFrame(st["macro"])
    if not mac.empty:
        print("\nmacro:")
        print(mac.to_string(index=False))


def cmd_demo(args):
    from bigboss.data.synthetic import load_into_store
    from bigboss.service import BigBoss

    bb = BigBoss(get_config(), "SYNTHETIC")
    load_into_store(bb.store, n_assets=args.n_assets, n_days=args.n_days, seed=args.seed, alpha_scale=args.alpha_scale)
    print(f"Loaded SYNTHETIC market ({args.n_assets} assets, {args.n_days} days) into {bb.store.path}. "
          "This is simulated data for testing the platform; it says nothing about real markets.")


def cmd_train(args):
    from bigboss.models.zoo import available_models

    bb = _bb(args)
    models = args.models or available_models()
    for h in args.horizon:
        t = time.time()
        vid = bb.registry.train(bb.dataset(), h, models, min_train=args.min_train)
        m = bb.registry.meta(vid)
        print(f"\nversion {vid} (h={h}, {time.time() - t:.0f}s)  ensemble OOS rank-IC t={m['ensemble_skill_t']:.2f}  gates passed={m['gates']['passed']}")
        print(pd.DataFrame(m["forecast_quality"])[["model", "rank_ic_mean", "rank_ic_t", "interval80_coverage", "brier", "brier_climatology", "direction_accuracy"]].round(4).to_string(index=False))
        for k, c in m["gates"]["checks"].items():
            print(f"  gate {k}: {'PASS' if c['pass'] else 'FAIL'}  {json.dumps({x: y for x, y in c.items() if x != 'pass'}, default=str)}")
        print(f"  champion for h={h}: {bb.registry.champion(h)}")


def cmd_backtest(args):
    from bigboss.backtest.experiment import run_experiment

    bb = _bb(args)
    if args.final_holdout:
        ans = input("You are about to evaluate on the LOCKED FINAL HOLDOUT. Each access is logged and reduces its value.\n"
                    "Type 'unlock holdout' to continue: ")
        if ans.strip() != "unlock holdout":
            print("aborted")
            return
    s = run_experiment(bb.store, bb.cfg, args.horizon, args.models, include_holdout=args.final_holdout, ablation=args.ablation,
                       ds=bb.dataset(), progress=lambda f, m: print(f"  [{f:4.0%}] {m}", file=sys.stderr))
    tab = pd.DataFrame(s["strategies"]).T
    cols = ["ann_return", "ann_excess_return", "sharpe", "sortino", "information_ratio", "max_drawdown", "calmar", "ann_turnover", "ann_cost_drag", "psr_excess", "deflated_sharpe_excess"]
    print(f"\nexperiment {s['id']}  data_mode={s['data_mode']}  OOS {s['oos_start']} -> {s['oos_end']}  universe={s['universe_size']}  strategies tested={s['n_strategies_tested']}")
    print(tab[cols].astype(float).round(3).to_string())
    print("\nforecast quality:")
    print(pd.DataFrame(s["forecast_quality"])[["model", "rank_ic_mean", "rank_ic_t", "interval80_coverage", "brier", "brier_climatology"]].round(4).to_string(index=False))
    if s["cost_sensitivity"]:
        print("\ncost sensitivity:")
        print(pd.DataFrame(s["cost_sensitivity"]).round(3).to_string(index=False))
    if s["ablation"]:
        print("\nablation (LightGBM):")
        print(pd.DataFrame(s["ablation"]).round(4).to_string(index=False))
    print(f"\nsaved to {bb.cfg.data_dir / 'experiments' / s['id']}")


def cmd_scan(args):
    bb = _bb(args)
    s = bb.scan(args.horizon, log=args.log)
    print(f"data_mode={s['data_mode']} as_of={s['as_of']} horizon={s['horizon']}d model={s['model_version']} skill_t={s['skill_t']:.2f} counts={s['counts']}")
    c = pd.DataFrame(s["candidates"])
    cols = ["symbol", "action", "mean", "q10", "q90", "cost_total_round_trip_bps", "expected_net_excess", "p_net_positive", "suggested_weight"]
    print(c[cols].head(args.top).round(4).to_string(index=False))


def cmd_research(args):
    from bigboss.agents.workflow import run_research

    r = run_research(_bb(args), args.question, args.symbols)
    print(r["narrative"]["text"])
    print(f"\nreport {r['report_id']} ({r['narrative']['generator']})")


def cmd_models(args):
    bb = _bb(args)
    if args.action == "promote":
        print(json.dumps(bb.registry.promote(args.version, args.force_reason), indent=1, default=str))
    elif args.action == "rollback":
        print(bb.registry.rollback(args.horizon))
    elif args.action == "challenge":
        print(json.dumps(bb.registry.challenge(args.version), indent=1, default=str))
    else:
        st = bb.registry.state()
        print("champions:", st["champions"])
        for m in bb.registry.versions()[:20]:
            print(f"{m['version']}  h={m['horizon']}  skill_t={m['ensemble_skill_t']:.2f}  gates={m['gates']['passed']}  oos={m['oos_start']}..{m['oos_end']}")


def cmd_update(args):
    """Daily maintenance job."""
    from bigboss.monitoring import resolve_outcomes

    cfg = get_config()
    if not args.synthetic:
        cmd_ingest(argparse.Namespace(groups=None, symbols=None, no_prices=False, no_macro=False, no_sec=args.skip_sec))
    bb = _bb(args)
    bb.dataset(refresh=True)
    for h in cfg.get("forecast.horizons"):
        if bb.registry.champion(h):
            s = bb.scan(h, log=True)
            print(f"logged {len(s['candidates'])} predictions for h={h} as_of={s['as_of']}")
    print(f"resolved outcomes: {resolve_outcomes(bb.store, cfg.get('forecast.execution_delay_days', 1))}")


def cmd_schedule(args):
    while True:
        t = time.time()
        try:
            cmd_update(args)
        except Exception as e:  # keep the scheduler alive; the error is printed for the operator
            print(f"update failed: {e}", file=sys.stderr)
        time.sleep(max(60, args.every_hours * 3600 - (time.time() - t)))


def cmd_serve(args):
    import uvicorn

    uvicorn.run("bigboss.api.app:app", host=args.host, port=args.port, reload=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bigboss", description="BigBoss investment research platform (research and paper trading only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, synthetic=True):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if synthetic:
            sp.add_argument("--synthetic", action="store_true", help="use the SYNTHETIC store")
        return sp

    sp = add("ingest", cmd_ingest, synthetic=False)
    sp.add_argument("--groups", nargs="*")
    sp.add_argument("--symbols", nargs="*")
    sp.add_argument("--no-prices", action="store_true")
    sp.add_argument("--no-macro", action="store_true")
    sp.add_argument("--no-sec", action="store_true")
    add("status", cmd_status)
    sp = add("demo-data", cmd_demo, synthetic=False)
    sp.add_argument("--n-assets", type=int, default=30)
    sp.add_argument("--n-days", type=int, default=2520)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--alpha-scale", type=float, default=2.0)
    sp = add("train", cmd_train)
    sp.add_argument("--horizon", type=int, nargs="+", default=[20])
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--min-train", type=int, default=756)
    sp = add("backtest", cmd_backtest)
    sp.add_argument("--horizon", type=int, default=20)
    sp.add_argument("--models", nargs="*")
    sp.add_argument("--ablation", action="store_true")
    sp.add_argument("--final-holdout", action="store_true")
    sp = add("scan", cmd_scan)
    sp.add_argument("--horizon", type=int, default=20)
    sp.add_argument("--top", type=int, default=25)
    sp.add_argument("--log", action="store_true", help="log predictions for later outcome evaluation")
    sp = add("research", cmd_research)
    sp.add_argument("question")
    sp.add_argument("--symbols", nargs="*")
    sp = add("models", cmd_models)
    sp.add_argument("action", nargs="?", choices=["list", "promote", "rollback", "challenge"], default="list")
    sp.add_argument("version", nargs="?")
    sp.add_argument("--horizon", type=int, default=20)
    sp.add_argument("--force-reason")
    sp = add("update", cmd_update)
    sp.add_argument("--skip-sec", action="store_true")
    sp = add("schedule", cmd_schedule)
    sp.add_argument("--every-hours", type=float, default=24)
    sp.add_argument("--skip-sec", action="store_true")
    sp = add("serve", cmd_serve, synthetic=False)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
