import { useEffect, useMemo, useState } from "react";
import { api, pct, signCls } from "../api";
import type { Ctx } from "../App";
import { Plot, cssVar } from "../components/Plot";
import { Action, DataBadge, ErrorBox, ModeBadge, SimBanner, Stat, Table } from "../components/ui";

export function Thesis({ t }: { t: any }) {
  if (!t) return null;
  return (
    <div style={{ fontSize: 13 }}>
      <p style={{ marginTop: 0 }}><strong>Thesis.</strong> {t.summary}</p>
      <div className="grid g3">
        <div><strong>Supporting evidence</strong><ul className="tight">{t.evidence?.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul></div>
        <div><strong>Counterarguments</strong><ul className="tight">{t.counterarguments?.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul></div>
        <div><strong>Invalidation conditions</strong><ul className="tight">{t.invalidation?.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul></div>
      </div>
      <p className="muted" style={{ marginBottom: 0 }}>{t.disclaimer}</p>
    </div>
  );
}

export default function ScannerPage({ ctx }: { ctx: Ctx }) {
  const [h, setH] = useState<number>(ctx.health?.primary_horizon || 20);
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [filter, setFilter] = useState("ALL");
  const [sel, setSel] = useState<any>(null);
  useEffect(() => { setD(null); setErr(null); setSel(null); api.get(`/api/scanner?mode=${ctx.mode}&horizon=${h}`).then(setD).catch(setErr); }, [ctx.mode, h]);
  const rows = useMemo(() => (d?.candidates || []).filter((c: any) => filter === "ALL" || c.action === filter), [d, filter]);
  const champions = Object.keys(ctx.health?.champions?.[ctx.mode] || {}).map(Number).sort((a, b) => a - b);

  return (
    <div>
      <div className="page-head">
        <h1>AI Opportunity Scanner</h1>
        <div className="row">
          <label className="f">Horizon<select value={h} onChange={(e) => setH(+e.target.value)}>{(champions.length ? champions : [h]).map((x) => <option key={x} value={x}>{x} days</option>)}</select></label>
          <label className="f">Action<select value={filter} onChange={(e) => setFilter(e.target.value)}>{["ALL", "BUY", "HOLD", "REDUCE", "SELL", "AVOID", "INSUFFICIENT_EVIDENCE"].map((x) => <option key={x}>{x}</option>)}</select></label>
        </div>
      </div>
      <SimBanner mode={ctx.mode} />
      <ErrorBox error={err} hint={<span>Train a model in <a href="#/data">Data & Setup</a>.</span>} />
      {d && <>
        <div className="grid g4">
          <Stat label="As of / model" value={d.as_of} sub={d.model_version} />
          <Stat label="OOS skill (rank-IC t-stat)" value={d.skill_t?.toFixed(2)} sub={d.skill_t >= d.skill_gate ? `✓ above gate ${d.skill_gate}` : `✖ below gate ${d.skill_gate} - ensemble abstains`} cls={d.skill_t >= d.skill_gate ? "pos" : "neg"} />
          <Stat label="Decisions" value={Object.entries(d.counts).map(([k, v]) => `${k === "INSUFFICIENT_EVIDENCE" ? "INSUFF." : k} ${v}`).join(" · ")} />
          <Stat label={`Benchmark (${d.benchmark}) assumed ${d.horizon}d return`} value={pct(d.bench_expected_h)} sub="trailing 10y geometric mean - an assumption" />
        </div>
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Expected net excess return vs. uncertainty <DataBadge kind="prediction" /></h3>
          <Plot theme={ctx.theme} height={340} data={[{
            type: "scatter", mode: "markers+text", x: d.candidates.map((c: any) => c.sigma_h), y: d.candidates.map((c: any) => c.expected_net_excess),
            text: d.candidates.map((c: any) => (c.action === "BUY" ? c.symbol : "")), textposition: "top center",
            marker: { size: 10, line: { width: 2, color: cssVar("--surface-1") },
              color: d.candidates.map((c: any) => c.action === "BUY" ? cssVar("--series-1") : c.action === "INSUFFICIENT_EVIDENCE" ? cssVar("--axis") : cssVar("--text-muted")) },
            customdata: d.candidates.map((c: any) => [c.symbol, c.action, c.p_net_positive]),
            hovertemplate: "%{customdata[0]} · %{customdata[1]}<br>net excess %{y:.2%}<br>σ(h) %{x:.2%}<br>P(net>0) %{customdata[2]:.0%}<extra></extra>",
          }]} layout={{ hovermode: "closest", xaxis: { title: { text: `forecast σ over ${d.horizon} days` }, tickformat: ".1%" }, yaxis: { title: { text: "expected net excess" }, tickformat: ".1%" }, showlegend: false }} />
          <div className="muted" style={{ fontSize: 12 }}>Blue = BUY, gray = AVOID/HOLD, light = insufficient evidence. Labels only on BUY candidates.</div>
        </div>
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Candidates ranked by risk-adjusted net excess return <DataBadge kind="prediction" /> <span className="muted" style={{ fontWeight: 400 }}>click a row for the thesis</span></h3>
          <Table rows={rows} onRow={setSel} cols={[
            { key: "symbol", label: "Symbol", left: true },
            { key: "action", label: "Action", left: true, render: (c) => <Action a={c.action} /> },
            { key: "mean", label: "E[excess]", render: (c) => <span className={signCls(c.mean)}>{pct(c.mean)}</span> },
            { key: "q10", label: "80% interval", render: (c) => `${pct(c.q10, 1)} … ${pct(c.q90, 1)}`, sort: (c) => c.q10 },
            { key: "cost_total_round_trip_bps", label: "Cost RT (bps)", render: (c) => c.cost_total_round_trip_bps?.toFixed(1) },
            { key: "expected_net_excess", label: "E[net excess]", render: (c) => <span className={signCls(c.expected_net_excess)}>{pct(c.expected_net_excess)}</span> },
            { key: "p_net_positive", label: "P(net>0)", render: (c) => pct(c.p_net_positive, 0) },
            { key: "es10", label: "ES10", render: (c) => pct(c.es10, 1) },
            { key: "disagreement", label: "Model σ", render: (c) => pct(c.disagreement, 2) },
            { key: "suggested_weight", label: "Size", render: (c) => pct(c.suggested_weight, 1) },
            { key: "risk_adjusted_score", label: "Score", render: (c) => c.risk_adjusted_score?.toFixed(3) },
          ]} />
        </div>
        {sel && (
          <div className="card" style={{ marginTop: 14 }}>
            <h3>{sel.symbol} <Action a={sel.action} /> <DataBadge kind="prediction" /> <button className="btn" onClick={() => ctx.go("research", sel.symbol)}>Open research →</button></h3>
            <div className="grid g4" style={{ marginBottom: 10 }}>
              <Stat label="1 · Expected gross return" value={pct(sel.expected_gross)} />
              <Stat label="2 · Expected excess vs benchmark" value={pct(sel.expected_excess)} />
              <Stat label="3–6 · Costs (comm / spread / slip / impact)" value={`${sel.cost_total_round_trip_bps?.toFixed(1)} bps RT`}
                sub={`${sel.cost_commission_bps?.toFixed(1)} / ${sel.cost_half_spread_bps?.toFixed(1)} / ${sel.cost_slippage_bps?.toFixed(1)} / ${sel.cost_impact_bps?.toFixed(2)} bps one-way${sel.cost_tax_bps ? ` + tax ${sel.cost_tax_bps}` : ""}`} />
              <Stat label="7 · Expected net excess" value={pct(sel.expected_net_excess)} cls={signCls(sel.expected_net_excess)} />
              <Stat label="8 · Volatility / ES10 (horizon)" value={`${pct(sel.sigma_h, 1)} / ${pct(sel.es10, 1)}`} sub={`E[downside] ${pct(sel.expected_downside)}`} />
              <Stat label="9 · Liquidity" value={sel.cost_liquidity_ok ? "✓ within limit" : "✖ exceeds limit"} sub={`participation ${pct(sel.cost_participation, 4)} of ADV`} />
              <Stat label="10 · Active risk per 1% weight" value={pct(sel.te_per_1pct, 3)} sub="tracking error over horizon" />
              <Stat label="11 · P(net excess > 0)" value={pct(sel.p_net_positive, 0)} sub={`P(outperform) ${pct(sel.p_outperform, 0)}`} />
              <Stat label="12 · Forecast uncertainty" value={`${pct(sel.q10, 1)} … ${pct(sel.q90, 1)}`} sub={`model σ ${pct(sel.disagreement)} · ${pct(sel.sign_agreement, 0)} agree on sign`} />
              <Stat label="Suggested position" value={pct(sel.suggested_weight, 1)} sub={`current ${pct(sel.current_weight, 1)}`} />
            </div>
            <Thesis t={sel.thesis} />
          </div>
        )}
        <div className="muted" style={{ marginTop: 10, fontSize: 12 }}><ModeBadge mode={d.data_mode} /> Forecasts are calibrated model estimates with wide uncertainty. They are not guarantees.</div>
      </>}
    </div>
  );
}
