import { useEffect, useState } from "react";
import { api, pct, signCls, waitJob } from "../api";
import type { Ctx } from "../App";
import { Plot, seriesColor } from "../components/Plot";
import { DataBadge, ErrorBox, ModeBadge, Progress, SimBanner, Table, Tabs } from "../components/ui";

const KEY = ["benchmark", "equal_weight", "momentum_12_1_topk", "ensemble_gated"];

export default function StrategyLab({ ctx }: { ctx: Ctx }) {
  const [list, setList] = useState<any[]>([]);
  const [sel, setSel] = useState<string>("");
  const [e, setE] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [job, setJob] = useState<any>(null);
  const [tab, setTab] = useState("Performance");
  const [opts, setOpts] = useState({ horizon: 20, ablation: false });
  const load = () => api.get(`/api/backtests?mode=${ctx.mode}`).then((l) => { setList(l); if (l.length && !sel) setSel(l[0].id); }).catch(setErr);
  useEffect(() => { setSel(""); setE(null); load(); }, [ctx.mode]);
  useEffect(() => { if (sel) api.get(`/api/backtests/${sel}`).then(setE).catch(setErr); }, [sel]);

  const run = async () => {
    setErr(null);
    try {
      const { job_id } = await api.post("/api/jobs/experiment", { mode: ctx.mode, horizon: opts.horizon, ablation: opts.ablation });
      const j = await waitJob(job_id, setJob);
      await load();
      setSel(j.result.id);
    } catch (x) { setErr(x); }
  };

  const strategies = e ? Object.entries(e.strategies).map(([k, v]: any) => ({ strategy: k, ...v })) : [];
  return (
    <div>
      <div className="page-head"><h1>Strategy Laboratory</h1><span className="row">{ctx.mode === "observed" && <ModeBadge mode={ctx.mode} />}<DataBadge kind="simulated" /></span></div>
      <SimBanner mode={ctx.mode} />
      <div className="card">
        <div className="row" style={{ alignItems: "end" }}>
          <label className="f">Experiment<select value={sel} onChange={(x) => setSel(x.target.value)}>{list.map((x) => <option key={x.id} value={x.id}>{x.id} · h={x.horizon} · IR {x.ensemble_gated_ir?.toFixed(2) ?? "–"}</option>)}</select></label>
          <label className="f">Horizon<select value={opts.horizon} onChange={(x) => setOpts({ ...opts, horizon: +x.target.value })}>{(ctx.health?.horizons || [20]).map((h: number) => <option key={h}>{h}</option>)}</select></label>
          <label className="row" style={{ gap: 4 }}><input type="checkbox" checked={opts.ablation} onChange={(x) => setOpts({ ...opts, ablation: x.target.checked })} />ablation study (slower)</label>
          <button className="btn primary" onClick={run} disabled={job?.status === "running"}>Run new walk-forward experiment</button>
          <Progress job={job?.status === "running" || job?.status === "queued" ? job : null} />
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>Backtests exclude the locked final holdout. Unlocking it requires the CLI (`bigboss backtest --final-holdout`) and every access is logged.</div>
      </div>
      <ErrorBox error={err} />
      {!e && !list.length && <div className="muted" style={{ marginTop: 12 }}>No experiments yet.</div>}
      {e && <>
        <div className="banner warn" style={{ marginTop: 14 }}>
          <strong>Simulated historical results.</strong> OOS {e.oos_start} → {e.oos_end} · {e.universe_size} assets · {e.n_strategies_tested} strategies tested (deflated Sharpe accounts for this) ·
          holdout starts {e.holdout_starts}{e.include_holdout ? " (HOLDOUT INCLUDED)" : " (excluded)"}. {e.assumptions?.survivorship}. A strong backtest is not evidence of future profit.
        </div>
        <Tabs tabs={["Performance", "Forecast quality", "Costs & regimes", "Ablation", "Folds & assumptions"]} value={tab} onChange={setTab} />
        {tab === "Performance" && <>
          <div className="card"><h3>Growth of 1 (net of costs) <DataBadge kind="simulated" /></h3>
            <Plot theme={ctx.theme} height={380} data={KEY.filter((k) => e.equity_curves[k]).map((k, i) => ({ x: e.equity_curves[k].dates, y: e.equity_curves[k].values, type: "scatter", mode: "lines", name: k, line: { width: 2, color: seriesColor(i) } }))} layout={{ yaxis: { title: { text: "growth of 1" } } }} />
          </div>
          <div className="card" style={{ marginTop: 14 }}><h3>Drawdowns</h3>
            <Plot theme={ctx.theme} height={220} data={Object.entries(e.drawdowns).map(([k, v]: any) => ({ x: e.equity_curves[k].dates, y: v, type: "scatter", mode: "lines", name: k, line: { width: 2, color: seriesColor(KEY.indexOf(k)) } }))} layout={{ yaxis: { tickformat: ".0%" } }} />
          </div>
          <div className="card" style={{ marginTop: 14 }}><h3>All strategies (net of estimated costs)</h3>
            <Table rows={strategies} initialSort="information_ratio" cols={[
              { key: "strategy", label: "Strategy", left: true },
              ...[["ann_return", "Ann. ret"], ["ann_excess_return", "Ann. excess"], ["ann_vol", "Vol"], ["max_drawdown", "Max DD"], ["tracking_error", "TE"], ["ann_cost_drag", "Cost drag"]].map(([k, l]) => ({ key: k, label: l, render: (r: any) => <span className={k.includes("excess") ? signCls(r[k]) : ""}>{pct(r[k], 1)}</span> })),
              ...[["sharpe", "Sharpe"], ["sortino", "Sortino"], ["calmar", "Calmar"], ["information_ratio", "IR"], ["beta", "Beta"], ["profit_factor_excess", "PF (excess)"], ["ann_turnover", "Turnover"]].map(([k, l]) => ({ key: k, label: l, render: (r: any) => r[k]?.toFixed(2) ?? "–" })),
              { key: "ir_ci90", label: "IR 90% CI", render: (r) => r.ir_ci90 ? `${r.ir_ci90[0]?.toFixed(2)} … ${r.ir_ci90[1]?.toFixed(2)}` : "–", sort: (r) => r.ir_ci90?.[0] },
              { key: "deflated_sharpe_excess", label: "Deflated SR", render: (r) => pct(r.deflated_sharpe_excess, 0) },
            ]} />
            <div className="muted" style={{ fontSize: 12 }}>Deflated SR = probability the excess-return Sharpe is above what the best of {e.n_strategies_tested} skill-less strategies would show. IR CI from a block bootstrap.</div>
          </div>
        </>}
        {tab === "Forecast quality" && <div className="grid g2">
          <div className="card"><h3>Out-of-sample forecast quality <DataBadge kind="prediction" /></h3>
            <Table rows={e.forecast_quality} cols={[{ key: "model", label: "Model", left: true }, { key: "rank_ic_mean", label: "Rank IC", render: (r) => r.rank_ic_mean?.toFixed(4) ?? "–" },
              { key: "rank_ic_t", label: "IC t", render: (r) => r.rank_ic_t?.toFixed(2) }, { key: "direction_accuracy", label: "Dir. acc.", render: (r) => pct(r.direction_accuracy, 1) },
              { key: "interval80_coverage", label: "80% cov.", render: (r) => pct(r.interval80_coverage, 1) }, { key: "brier", label: "Brier", render: (r) => r.brier?.toFixed(4) },
              { key: "brier_climatology", label: "Brier (clim.)", render: (r) => r.brier_climatology?.toFixed(4) }]} />
            <div className="muted" style={{ fontSize: 12 }}>Direction accuracy is shown for reference only; BigBoss optimizes net economic value, not hit rate.</div>
          </div>
          <div className="card"><h3>Probability calibration (ensemble)</h3>
            {(() => { const c = e.forecast_quality.find((f: any) => f.model === "ensemble")?.calibration || []; return (
              <Plot theme={ctx.theme} data={[
                { x: [0.3, 0.7], y: [0.3, 0.7], type: "scatter", mode: "lines", name: "perfect calibration", line: { width: 1, dash: "dot", color: seriesColor(7) } },
                { x: c.map((b: any) => b.mean_pred), y: c.map((b: any) => b.freq), type: "scatter", mode: "lines+markers", name: "ensemble", marker: { size: 8 }, line: { width: 2, color: seriesColor(0) }, customdata: c.map((b: any) => b.n), hovertemplate: "pred %{x:.2f} → observed %{y:.2f} (n=%{customdata})<extra></extra>" },
              ]} layout={{ xaxis: { title: { text: "predicted P(excess>0)" } }, yaxis: { title: { text: "observed frequency" } }, hovermode: "closest" }} />); })()}
          </div>
          <div className="card"><h3>Top LightGBM features (mean gain share)</h3>
            <Plot theme={ctx.theme} height={420} data={[{ type: "bar", orientation: "h", y: Object.keys(e.feature_importance), x: Object.values(e.feature_importance), marker: { color: seriesColor(0) } }]}
              layout={{ yaxis: { autorange: "reversed" }, margin: { l: 140, r: 10, t: 6, b: 30 }, hovermode: "closest", showlegend: false }} /></div>
          <div className="card"><h3>Ensemble weights per fold (fit only on prior OOS)</h3>
            <Table rows={e.ensemble_weights} cols={Object.keys(e.ensemble_weights[0] || {}).map((k) => ({ key: k, label: k, render: (r: any) => typeof r[k] === "number" && k !== "fold" && k !== "fitted_on_rows" ? r[k].toFixed(2) : String(r[k]) }))} /></div>
        </div>}
        {tab === "Costs & regimes" && <div className="grid g2">
          <div className="card"><h3>Sensitivity to trading costs</h3>
            <Plot theme={ctx.theme} data={["ensemble_gated", "model_ensemble", "equal_weight"].map((s, i) => { const r = e.cost_sensitivity.filter((x: any) => x.strategy === s); return { x: r.map((x: any) => x.cost_multiplier), y: r.map((x: any) => x.ann_excess_return), type: "scatter", mode: "lines+markers", name: s, line: { width: 2, color: seriesColor(i) }, marker: { size: 8 } }; })}
              layout={{ xaxis: { title: { text: "cost multiplier (1 = base estimate)" } }, yaxis: { title: { text: "annual excess return" }, tickformat: ".1%" } }} /></div>
          <div className="card"><h3>Performance by market regime (ensemble_gated)</h3>
            <Table rows={e.regimes.ensemble_gated || []} cols={[{ key: "regime", label: "Regime", left: true }, { key: "days", label: "Days" },
              { key: "ann_return", label: "Ann. ret", render: (r) => pct(r.ann_return, 1) }, { key: "ann_benchmark", label: "Benchmark", render: (r) => pct(r.ann_benchmark, 1) },
              { key: "ann_excess", label: "Excess", render: (r) => <span className={signCls(r.ann_excess)}>{pct(r.ann_excess, 1)}</span> }, { key: "information_ratio", label: "IR", render: (r) => r.information_ratio?.toFixed(2) ?? "–" }]} /></div>
        </div>}
        {tab === "Ablation" && <div className="card"><h3>Ablation: LightGBM without each feature group</h3>
          {!e.ablation?.length ? <div className="muted">This experiment ran without ablation. Tick "ablation study" and run a new experiment.</div> :
            <Table rows={e.ablation} cols={[{ key: "variant", label: "Variant", left: true }, { key: "n_features", label: "Features" },
              { key: "rank_ic_mean", label: "Rank IC", render: (r) => r.rank_ic_mean?.toFixed(4) }, { key: "delta_ic_vs_all", label: "Δ IC", render: (r) => <span className={signCls(-r.delta_ic_vs_all)}>{r.delta_ic_vs_all?.toFixed(4)}</span> },
              { key: "information_ratio", label: "IR", render: (r) => r.information_ratio?.toFixed(2) }, { key: "delta_ir_vs_all", label: "Δ IR", render: (r) => r.delta_ir_vs_all?.toFixed(2) }]} />}
          <div className="muted" style={{ fontSize: 12 }}>A group "helps" if removing it lowers IC/IR. Positive Δ means the model did better without that group.</div></div>}
        {tab === "Folds & assumptions" && <div className="grid g2">
          <div className="card"><h3>Walk-forward folds</h3><Table rows={e.folds} cols={Object.keys(e.folds[0] || {}).map((k) => ({ key: k, label: k }))} /></div>
          <div className="card"><h3>Assumptions</h3><pre className="mono">{JSON.stringify(e.assumptions, null, 1)}</pre><div className="muted">Sources: {e.sources?.join(", ")}</div></div>
        </div>}
      </>}
    </div>
  );
}
