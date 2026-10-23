"""QuantConnect bridge: export format, LEAN algorithm helpers (run outside LEAN via a stub AlgorithmImports), comparison."""

import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bigboss.integrations import quantconnect as qc


def _lean_module():
    stub = types.ModuleType("AlgorithmImports")
    for name in ("QCAlgorithm", "FeeModel"):
        setattr(stub, name, type(name, (), {}))
    sys.modules["AlgorithmImports"] = stub
    spec = importlib.util.spec_from_file_location("qc_lean_main_under_test", qc.LEAN_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_experiment(tmp_path):
    exp_id = "20260101T000000-test"
    exp = tmp_path / "experiments" / exp_id
    exp.mkdir(parents=True)
    days = pd.bdate_range("2025-01-02", periods=60)
    dec = days[::20]
    pd.DataFrame({"AAPL": [0.4, 0.5, 0.0], "BRK-B": [0.3, 0.0, 0.2], "XA:GOLD": [0.3, 0.5, 0.0]},
                 index=dec).to_parquet(exp / "weights_ensemble_gated.parquet")
    rows = [{"date": d, "symbol": s, "mean_ensemble": 0.01, "q10_ensemble": -0.05, "q90_ensemble": 0.07,
             "p_pos_ensemble": 0.55, "disagreement": 0.01, "sign_agreement": 0.8, "ensemble_skill_t": 1.5}
            for d in dec for s in ("AAPL", "BRK-B", "RELIANCE.NS")]
    pd.DataFrame(rows).to_parquet(exp / "oos.parquet")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, len(days)), index=days)
    pd.DataFrame({"ensemble_gated": r, "benchmark_index": r * 0.8}).to_parquet(exp / "returns.parquet")
    summary = {"id": exp_id, "data_mode": "OBSERVED", "horizon": 20, "benchmark": "SPY", "oos_start": "2025-01-02",
               "oos_end": "2025-03-26", "universe_size": 3, "sources": ["test"],
               "assumptions": {"rebalance_every_days": 20, "execution": "decision at close t, fill at close t+1"},
               "strategies": {"ensemble_gated": {"ann_return": 0.1, "information_ratio": 0.2}}}
    (exp / "summary.json").write_text(json.dumps(summary))
    return SimpleNamespace(data_dir=tmp_path), exp_id, r


def test_ticker_mapping():
    assert qc.qc_ticker("BRK-B") == "BRK.B"
    assert qc.qc_ticker("aapl") == "AAPL"
    for s in ("XA:GOLD", "RELIANCE.NS", "^NSEI", "SYN001"):
        assert qc.qc_ticker(s) is None


def test_export_writes_us_only_files_and_manifest(tmp_path):
    cfg, exp_id, _ = _fake_experiment(tmp_path)
    out = tmp_path / "qc"
    m = qc.export_experiment(cfg, exp_id, out)
    w = pd.read_csv(out / "weights.csv")
    assert set(w["symbol"]) == {"AAPL", "BRK.B"}
    assert (w["weight"] > 0).all()  # zero weights dropped
    s = pd.read_csv(out / "signals.csv")
    assert list(s.columns) == ["date", "symbol"] + qc.SIGNAL_COLUMNS and set(s["symbol"]) == {"AAPL", "BRK.B"}
    assert m["symbols_skipped_not_on_qc_us_equities"] == ["RELIANCE.NS", "XA:GOLD"]
    assert set(m["files"]) == {"weights.csv", "signals.csv", "main.py"} and (out / "README.md").exists()
    assert (out / "main.py").read_text() == qc.LEAN_SOURCE.read_text()


def test_export_rejects_missing_strategy(tmp_path):
    cfg, exp_id, _ = _fake_experiment(tmp_path)
    with pytest.raises(FileNotFoundError):
        qc.export_experiment(cfg, exp_id, tmp_path / "qc", strategy="model_mlp")


def test_lean_helpers_timing_gate_and_targets():
    lean = _lean_module()
    table = lean.parse_table("date,symbol,weight\n2025-01-02,AAPL,0.6\n2025-01-02,msft,0.6\n2025-02-03,AAPL,nan\n")
    assert table["2025-01-02"]["MSFT"]["weight"] == 0.6 and table["2025-02-03"]["AAPL"]["weight"] is None
    dates = sorted(table)
    # a decision dated D is only usable strictly after D
    assert lean.latest_before(dates, "2025-01-02") is None
    assert lean.latest_before(dates, "2025-01-03") == "2025-01-02"
    assert lean.latest_before(dates, "2025-02-04") == "2025-02-03"
    t = lean.normalize_targets({"AAPL": 0.6, "MSFT": 0.6})
    assert abs(sum(t.values()) - 1.0) < 1e-12  # gross capped at 1, no leverage
    assert lean.normalize_targets({"AAPL": 0.3}, benchmark="SPY") == {"AAPL": 0.3, "SPY": 0.7}
    rows = {"A": {"mean": 0.02, "skill_t": 2.0, "disagreement": 0.01, "sign_agreement": 0.9},
            "B": {"mean": 0.03, "skill_t": 0.5, "disagreement": 0.01, "sign_agreement": 0.9},   # fails skill gate
            "C": {"mean": 0.04, "skill_t": 2.0, "disagreement": 0.05, "sign_agreement": 0.9},   # too much disagreement
            "D": {"mean": 0.0005, "skill_t": 2.0, "disagreement": 0.01, "sign_agreement": 0.9},  # edge < cost
            "E": {"mean": 0.015, "skill_t": None, "disagreement": None, "sign_agreement": None}}
    picks = lean.select_signals(rows, top_k=10, max_weight=0.1, round_trip_cost=0.001,
                                min_skill_t=1.0, max_disagreement=0.03, min_sign_agreement=0.6)
    assert picks == {"A": 0.1, "E": 0.1}


def _qc_json(path, equity, bench, fmt):
    def vals(s):
        ts = [int(pd.Timestamp(d).timestamp()) + 16 * 3600 for d in s.index]
        if fmt == "xy":
            return [{"x": t, "y": float(v)} for t, v in zip(ts, s)]
        return [[t, float(v), float(v), float(v), float(v)] for t, v in zip(ts, s)]
    path.write_text(json.dumps({"charts": {"Strategy Equity": {"series": {"Equity": {"values": vals(equity)}}},
                                           "Benchmark": {"series": {"Benchmark": {"values": vals(bench)}}}}}))


@pytest.mark.parametrize("fmt", ["xy", "ohlc"])
def test_compare_identical_replay_is_consistent(tmp_path, fmt):
    cfg, exp_id, r = _fake_experiment(tmp_path)
    equity = 1e6 * (1 + r).cumprod()
    bench = 100 * (1 + r * 0.8).cumprod()
    # QuantConnect's first equity point is the starting cash, one day before the first return
    equity = pd.concat([pd.Series([1e6], index=[r.index[0] - pd.offsets.BDay()]), equity])
    bench = pd.concat([pd.Series([100.0], index=[r.index[0] - pd.offsets.BDay()]), bench])
    p = tmp_path / "res.json"
    _qc_json(p, equity, bench, fmt)
    out = qc.compare(cfg, exp_id, p)
    assert out["common_days"] == len(r)
    assert out["daily_return_correlation"] > 0.9999 and out["verdict"] == "consistent"
    assert abs(out["bigboss"]["ann_return"] - out["quantconnect"]["ann_return"]) < 1e-9


def test_compare_flags_divergent_replay(tmp_path):
    cfg, exp_id, r = _fake_experiment(tmp_path)
    other = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(r)), index=r.index)
    p = tmp_path / "res.json"
    _qc_json(p, 1e6 * (1 + other).cumprod(), 100 * (1 + other).cumprod(), "xy")
    assert qc.compare(cfg, exp_id, p)["verdict"].startswith("inconsistent")
